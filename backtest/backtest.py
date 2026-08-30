#!/usr/bin/env python3
"""
Full-season backtest.

Replays a whole season gameweek by gameweek: pick a squad, take transfers, choose
a starting XI and captain, score against what actually happened.

Two corrections relative to the previous version. Prices are taken from the panel
at the gameweek being simulated rather than from today's bootstrap, so a player who
rose from 4.5 to 6.5 is not affordable at 6.5 in October. And identity runs on the
stable player `code`, so a squad carried across a season boundary stays the same
set of players.

Per-gameweek manager picks are only served for the current season, so a completed
season can only be compared against its final total, not week by week.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd

import features
import ml_model
from optimizer import optimize_squad
from visualize import plot_worm_graph

STARTING_BUDGET = 1000
MAX_FREE_TRANSFERS = 5  # FPL banks up to five
FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
POSITION_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}


# ──────────────────────────── Season tables ────────────────────────────

def season_tables(panel, season):
    """
    Per-gameweek lookups for one season.

    Prices come from the panel so every decision is priced as at the gameweek it
    was taken, which is the whole point of a backtest.
    """
    rows = panel[panel["season"] == season]
    points = {(int(r.code), int(r.GW)): float(r.total_points) for r in rows.itertuples()}
    minutes = {(int(r.code), int(r.GW)): float(r.minutes) for r in rows.itertuples()}
    prices = {(int(r.code), int(r.GW)): float(r.value) for r in rows.itertuples()}
    positions = dict(zip(rows["code"].astype(int), rows["position_id"].astype(int)))
    teams = dict(zip(rows["code"].astype(int), rows["team_id"].astype(int)))
    names = dict(zip(rows["code"].astype(int), rows["name"]))
    return points, minutes, prices, positions, teams, names


def price_at(prices, code, gw, fallback=50.0):
    """Price at a gameweek, falling back to the most recent earlier price."""
    for candidate in range(gw, 0, -1):
        if (code, candidate) in prices:
            return prices[(code, candidate)]
    return fallback


# ──────────────────────────── Scoring ────────────────────────────

def apply_autosubs(starters, bench, minutes, positions, gw):
    """
    Replace starters who did not play with bench players who did.

    FPL substitutes automatically in bench order, accepting a substitution only if
    the resulting formation stays legal. Ignoring this understates every strategy,
    since a blanked starter is silently worth zero.
    """
    played = [c for c in starters if minutes.get((c, gw), 0) > 0]
    blanked = [c for c in starters if minutes.get((c, gw), 0) == 0]
    if not blanked:
        return starters

    final = list(played)
    available = [c for c in bench if minutes.get((c, gw), 0) > 0]

    for out_code in blanked:
        for i, in_code in enumerate(available):
            candidate = final + [in_code]
            if is_legal_xi(candidate, positions):
                final.append(in_code)
                available.pop(i)
                break
        else:
            # No legal substitute: the slot stays empty and scores nothing.
            continue

    return final


def is_legal_xi(codes, positions):
    """A starting XI is legal at up to 11 players with one keeper and a valid shape."""
    if len(codes) > 11:
        return False
    counts = {p: 0 for p in POSITION_QUOTA}
    for code in codes:
        counts[positions.get(code, 3)] += 1
    if counts[1] > 1:
        return False
    # Only enforce the minimums once the XI is full.
    if len(codes) == 11:
        return all(counts[p] >= minimum for p, minimum in FORMATION_MIN.items())
    return True


def score_gameweek(starters, bench, captain, gw, points, minutes, positions, chip=None):
    """Points scored by a squad in one gameweek, after autosubs and captaincy."""
    if chip == "BB":
        scoring = list(starters) + list(bench)
    else:
        scoring = apply_autosubs(starters, bench, minutes, positions, gw)

    total = 0.0
    for code in scoring:
        scored = points.get((code, gw), 0.0)
        if code == captain:
            # Captaincy only doubles if the captain actually played; otherwise the
            # armband passes to the vice, which we approximate as no multiplier.
            multiplier = 3 if chip == "TC" else 2
            if minutes.get((code, gw), 0) > 0:
                scored *= multiplier
        total += scored
    return total


# ──────────────────────────── Simulation ────────────────────────────

def train_for_gameweek(labelled, feature_cols, season_index, gw, kind):
    """
    Fit on rows whose outcome was known before this gameweek kicked off.

    A row at GW t-2 is labelled with GW t-1's points, so that is the newest row
    that can legitimately be trained on when predicting GW t.
    """
    train = labelled[
        (labelled["season_index"] < season_index)
        | ((labelled["season_index"] == season_index) & (labelled["GW"] <= gw - 2))
    ]
    if len(train) < 500:
        return None
    return ml_model.train_model(train, feature_cols, kind=kind)


def predictions_for_gameweek(panel, feature_cols, model, season, gw):
    """
    Predicted points for every player, from their most recent row before `gw`.

    The next-gameweek fixture columns on that row describe `gw` itself, which is
    published in advance and so is legitimately available.
    """
    rows = panel[(panel["season"] == season) & (panel["GW"] == gw - 1)]
    rows = rows.dropna(subset=feature_cols)
    if rows.empty:
        return {}
    predicted = model.predict(rows[feature_cols])
    return dict(zip(rows["code"].astype(int), predicted))


def build_player_frame(codes, pred_map, prices, positions, teams, gw):
    """The optimizer's input for one gameweek, priced as at that gameweek."""
    records = []
    for code in codes:
        records.append(
            {
                "element_id": int(code),
                "predicted_points": float(pred_map.get(code, 0.0)),
                "value": price_at(prices, code, gw),
                "position": positions.get(code, 3),
                "team": teams.get(code, 0),
            }
        )
    return pd.DataFrame(records)


def simulate(panel, labelled, feature_cols, season, first_gw, last_gw, kind,
             strategy="model", retrain_every=4, max_transfers=2, verbose=True):
    """
    Play a season under one strategy.

    strategy 'model' re-optimizes every gameweek; 'hold' picks an opening squad
    and never transfers again, which isolates how much the transfer engine is
    actually worth.
    """
    points, minutes, prices, positions, teams, names = season_tables(panel, season)
    season_index = int(panel.loc[panel["season"] == season, "season_index"].iloc[0])
    codes = sorted({c for c, _ in points.keys()})

    squad = None
    bank = 0.0
    free_transfers = 1
    gw_scores = {}
    model = None
    transfer_log = []

    for gw in range(first_gw, last_gw + 1):
        if model is None or (gw - first_gw) % retrain_every == 0:
            fresh = train_for_gameweek(labelled, feature_cols, season_index, gw, kind)
            if fresh is not None:
                model = fresh
        if model is None:
            continue

        pred_map = predictions_for_gameweek(panel, feature_cols, model, season, gw)
        if not pred_map:
            continue

        frame = build_player_frame(codes, pred_map, prices, positions, teams, gw)

        if squad is None:
            # Opening squad: a full rebuild on the starting budget.
            result = optimize_squad(frame, free_transfers=15, current_squad_ids=None,
                                    budget=STARTING_BUDGET, n=1)
            transfers_made, hit = 0, 0
        elif strategy == "hold":
            result = pick_xi_only(squad, frame)
            transfers_made, hit = 0, 0
        else:
            budget = sum(price_at(prices, c, gw) for c in squad) + bank
            result = optimize_squad(frame, free_transfers=free_transfers,
                                    current_squad_ids=list(squad), budget=budget, n=1,
                                    hard_max_transfers=max_transfers)
            transfers_made, hit = 0, 0

        if not result:
            gw_scores[gw] = 0.0
            continue
        chosen = result[0]

        new_squad = [int(c) for c in chosen["element_id"]]
        if squad is not None and strategy != "hold":
            outgoing = set(squad) - set(new_squad)
            transfers_made = len(outgoing)
            hit = max(0, transfers_made - free_transfers) * 4
            if transfers_made:
                incoming = set(new_squad) - set(squad)
                transfer_log.append(
                    (gw, [names.get(c, c) for c in outgoing], [names.get(c, c) for c in incoming])
                )
            spent_before = sum(price_at(prices, c, gw) for c in squad) + bank
            bank = spent_before - sum(price_at(prices, c, gw) for c in new_squad)

        squad = new_squad
        starters = [int(c) for c in chosen[chosen["is_starter"]]["element_id"]]
        bench = [c for c in squad if c not in starters]
        captains = [int(c) for c in chosen[chosen["is_captain"]]["element_id"]]
        captain = captains[0] if captains else None

        raw = score_gameweek(starters, bench, captain, gw, points, minutes, positions)
        gw_scores[gw] = raw - hit

        if strategy != "hold":
            free_transfers = (
                min(MAX_FREE_TRANSFERS, free_transfers + 1) if transfers_made == 0 else 1
            )

        if verbose:
            note = f" {transfers_made}T" if transfers_made else ""
            note += f" (-{hit})" if hit else ""
            print(f"    GW{gw:2d}: {gw_scores[gw]:6.1f}{note}")

    return gw_scores, transfer_log


def pick_xi_only(squad, frame):
    """
    Choose the best legal XI and captain from a fixed squad.

    Used by the hold strategy: no transfers, but the manager still picks a team.
    """
    held = frame[frame["element_id"].isin(squad)].copy()
    if len(held) < 11:
        return []
    budget = float(held["value"].sum()) + 1e6
    return optimize_squad(held, free_transfers=0, current_squad_ids=list(squad),
                          budget=budget, n=1, hard_max_transfers=0)


# ──────────────────────────── Main ────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full-season FPL backtest.")
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--train-seasons", nargs="+", default=["2024-25", "2025-26"])
    parser.add_argument("--first-gw", type=int, default=1)
    parser.add_argument("--last-gw", type=int, default=38)
    parser.add_argument("--model", default=ml_model.DEFAULT_MODEL, choices=sorted(ml_model.MODELS))
    parser.add_argument("--retrain-every", type=int, default=4)
    parser.add_argument("--max-transfers", type=int, default=2)
    parser.add_argument("--actual-total", type=int, default=2151,
                        help="manager's real total for the season, for comparison")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FPL backtest - {args.season} GW{args.first_gw}-{args.last_gw}")
    print("=" * 60)

    labelled, feature_cols, panel = features.prepare(args.train_seasons)
    print(f"  {len(labelled):,} labelled rows, {len(feature_cols)} features")

    runs = {}
    for strategy, label in [("model", "Model (weekly transfers)"), ("hold", "Hold (no transfers)")]:
        print(f"\n  {label}")
        scores, log = simulate(
            panel, labelled, feature_cols, args.season,
            args.first_gw, args.last_gw, args.model,
            strategy=strategy, retrain_every=args.retrain_every,
            max_transfers=args.max_transfers, verbose=False,
        )
        runs[label] = scores
        total = sum(scores.values())
        print(f"    total {total:.0f} over {len(scores)} gameweeks")
        if strategy == "model":
            print(f"    {len(log)} gameweeks with transfers")

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Strategy':<30}{'Total':>8}{'vs actual':>12}")
    print(f"  {'-' * 50}")
    print(f"  {'Actual (your 2025-26)':<30}{args.actual_total:>8}{'':>12}")
    for label, scores in runs.items():
        total = sum(scores.values())
        print(f"  {label:<30}{total:>8.0f}{total - args.actual_total:>+12.0f}")

    output = os.path.join(SCRIPT_DIR, "output", "worm_graph.png")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plot_worm_graph(runs, output, args.first_gw, args.last_gw, args.actual_total)
    print("\n  done")


if __name__ == "__main__":
    main()
