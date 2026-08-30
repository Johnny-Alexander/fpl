"""
Weekly FPL transfer and squad recommendation.

Predictions are computed on stable player `code` and resolved to this season's
element ids only at the point of talking to the API, so a season rollover cannot
silently attach one player's form to another's name.
"""

import argparse

import pandas as pd

import data_fetcher
import features
import identity
import ml_model
from optimizer import optimize_squad

DEFAULT_TEAM_ID = 7246903
TRAINING_SEASONS = ["2024-25", "2025-26", "2026-27"]


def build_player_table(bootstrap, panel, model, feature_cols, season, min_chance, gameweek):
    """
    Join model predictions onto the live player list.

    The join runs through `code`, and unmatched players are reported rather than
    silently zero-filled -- a large unmatched count means the historical data has
    fallen behind the live season.
    """
    elements = pd.DataFrame(bootstrap["elements"])
    elements["element_id"] = elements["id"].astype(int)
    elements["code"] = elements["code"].astype(int)
    elements["position"] = elements["element_type"].astype(int)
    elements["value"] = elements["now_cost"].astype(int)

    latest = features.latest_rows(panel, season)
    if latest.empty:
        raise RuntimeError(
            f"No rows for season {season} in the historical data. "
            "Update data/historical (git pull) before running."
        )

    # Form should run through the gameweek immediately before the one we plan
    # for. Anything older means the recommendation is built on stale form.
    form_gw = int(latest["GW"].max())
    if form_gw < gameweek - 1:
        print(f"  WARNING: form data ends at GW{form_gw} but planning GW{gameweek} - "
              f"{gameweek - 1 - form_gw} gameweek(s) stale")

    # The upcoming gameweek's fixtures are published, so fill that context from
    # the live list rather than leaving it null on the season's last row.
    fixtures = data_fetcher.get_fixtures()
    latest = features.apply_upcoming_fixtures(latest, fixtures, gameweek)
    blanks = int((latest["next_fixture_count"] == 0).sum())
    doubles = int((latest["next_fixture_count"] >= 2).sum())
    print(f"  GW{gameweek} fixtures: {blanks} players blank, {doubles} on a double")

    latest = ml_model.predict(model, latest, feature_cols)

    carry = [
        "code",
        "predicted_points",
        "total_points_rolling_3",
        "minutes_rolling_3",
        "start_rate_5",
        "career_gws",
        "GW",
    ]
    merged = elements.merge(
        latest[carry].rename(columns={"GW": "form_through_gw"}), on="code", how="left"
    )

    matched = merged["predicted_points"].notna().sum()
    for col in ["predicted_points", "total_points_rolling_3", "minutes_rolling_3",
                "start_rate_5", "career_gws"]:
        merged[col] = merged[col].fillna(0.0)

    availability = data_fetcher.availability_frame(bootstrap, min_chance=min_chance)
    merged = merged.merge(
        availability[["element_id", "is_available", "status_label", "availability_factor"]],
        on="element_id",
        how="left",
    )
    merged["is_available"] = merged["is_available"].fillna(False)

    return merged, matched


def apply_availability_gate(players, current_squad_ids):
    """
    Remove players who cannot play the upcoming gameweek.

    Players already in the squad are retained regardless of status, so the
    optimizer can decide whether an injured asset is worth transferring out
    rather than being forced to sell.
    """
    keep = players["is_available"] | players["element_id"].isin(current_squad_ids or [])
    gated = players[keep].copy()

    # An unavailable player already in the squad is worth zero this week.
    unavailable_held = ~gated["is_available"]
    gated.loc[unavailable_held, "predicted_points"] = 0.0
    return gated, int((~keep).sum())


def apply_evidence_gate(players, current_squad_ids, min_evidence):
    """
    Refuse to transfer *in* a player the model has barely seen.

    This is a policy constraint on the action, not a correction to the
    prediction. Shrinking the features toward a prior was tried first and made no
    measurable difference (see features.DEFAULT_SHRINK); the failure it targeted
    is not really a bad estimate, it is acting on an estimate built from one
    appearance. Constraining the decision addresses that directly and leaves the
    predictions honest.

    Players already held are exempt -- the gate governs buying, not keeping.
    """
    if min_evidence <= 0:
        return players, 0

    held = set(current_squad_ids or [])
    thin = (players["career_gws"] < min_evidence) & (~players["element_id"].isin(held))
    return players[~thin].copy(), int(thin.sum())


def pair_transfers(out_ids, in_ids, players):
    """
    Pair outgoing with incoming players by position.

    A valid transfer set preserves the 2/5/5/3 squad shape, so the positions
    leaving match the positions arriving and pairing within position is
    well-defined. Zipping the raw id sets, as the previous version did, paired
    players in arbitrary hash order and produced rationales that described
    swaps that were not being proposed.
    """
    indexed = players.set_index("element_id")
    pairs = []
    for position in sorted(indexed.loc[list(out_ids), "position"].unique()):
        outs = [i for i in out_ids if indexed.at[i, "position"] == position]
        ins = [i for i in in_ids if indexed.at[i, "position"] == position]
        outs.sort(key=lambda i: indexed.at[i, "predicted_points"])
        ins.sort(key=lambda i: indexed.at[i, "predicted_points"])
        pairs.extend(zip(outs, ins))
    return pairs


def describe_transfers(option_index, squad, current_squad_ids, players, free_transfers):
    new_ids = set(squad["element_id"])
    old_ids = set(current_squad_ids)
    out_ids, in_ids = old_ids - new_ids, new_ids - old_ids

    if not out_ids:
        print(f"\nOption {option_index}: no transfer improves on the current squad.")
        return

    n_transfers = len(out_ids)
    hits = max(0, n_transfers - free_transfers)
    cost = hits * 4

    print(f"\n--- Option {option_index} ({n_transfers} transfer{'s' if n_transfers > 1 else ''}) ---")
    if cost:
        print(f"  Point hit: -{cost} ({hits} beyond the {free_transfers} free)")
    else:
        print("  Free transfer, no hit")

    indexed = players.set_index("element_id")
    total_gain = 0.0
    for out_id, in_id in pair_transfers(out_ids, in_ids, players):
        out_p, in_p = indexed.loc[out_id], indexed.loc[in_id]
        gain = in_p["predicted_points"] - out_p["predicted_points"]
        total_gain += gain

        print(f"  OUT {out_p['web_name']:<16} £{out_p['value'] / 10:>4.1f}m  "
              f"pred {out_p['predicted_points']:>5.2f}  [{out_p['status_label']}]")
        print(f"  IN  {in_p['web_name']:<16} £{in_p['value'] / 10:>4.1f}m  "
              f"pred {in_p['predicted_points']:>5.2f}  [{in_p['status_label']}]")
        print(f"      net {gain:+.2f} pts | form(3gw) "
              f"{out_p['total_points_rolling_3']:.1f} -> {in_p['total_points_rolling_3']:.1f} pts, "
              f"mins {out_p['minutes_rolling_3']:.0f} -> {in_p['minutes_rolling_3']:.0f}")

    print(f"  Expected gain {total_gain:+.2f}, after hit {total_gain - cost:+.2f} pts")


def print_squad(squad):
    captain = squad[squad["is_captain"]]
    starters = squad[squad["is_starter"]].sort_values("position")
    bench = squad[~squad["is_starter"]].sort_values("position")

    print("\n  Starting XI:")
    for _, p in starters.iterrows():
        mark = " (C)" if bool(p["is_captain"]) else ""
        print(f"    [{identity.ID_TO_POSITION[p['position']]}] {p['web_name']:<16}"
              f"£{p['value'] / 10:>4.1f}m  pred {p['predicted_points']:>5.2f}{mark}")
    if not bench.empty:
        print("  Bench:")
        for _, p in bench.iterrows():
            print(f"    [{identity.ID_TO_POSITION[p['position']]}] {p['web_name']:<16}"
                  f"£{p['value'] / 10:>4.1f}m  pred {p['predicted_points']:>5.2f}")
    if not captain.empty:
        print(f"\n  Captain: {captain.iloc[0]['web_name']} "
              f"({captain.iloc[0]['predicted_points']:.2f} pred -> "
              f"{captain.iloc[0]['predicted_points'] * 2:.2f} doubled)")


def main():
    parser = argparse.ArgumentParser(description="FPL transfer recommendations.")
    parser.add_argument("--team-id", type=int, default=DEFAULT_TEAM_ID)
    parser.add_argument("--free-transfers", type=int, default=1)
    parser.add_argument("--max-transfers", type=int, default=3)
    parser.add_argument("--options", type=int, default=3)
    parser.add_argument("--model", default=ml_model.DEFAULT_MODEL, choices=sorted(ml_model.MODELS))
    parser.add_argument("--min-chance", type=int, default=75,
                        help="exclude doubtful players below this %% chance of playing")
    parser.add_argument("--min-evidence", type=int, default=0,
                        help="minimum career gameweeks before a player can be bought. "
                             "Off by default: it looks sensible but measured worse over "
                             "the 2025-26 backtest at every threshold tried "
                             "(1934 ungated vs 1865 at 5, 1921 at 10 and 20)")
    parser.add_argument("--wildcard", action="store_true", help="ignore current squad and rebuild")
    args = parser.parse_args()

    bootstrap = data_fetcher.get_bootstrap_static()
    season = identity.current_season_label(bootstrap)
    current_gw = data_fetcher.get_current_gameweek(bootstrap)
    print(f"Season {season}, planning GW{current_gw}")

    squad_ids, budget = [], 1000
    entry = data_fetcher.get_user_team(args.team_id, current_gw - 1) if current_gw > 1 else None
    if entry:
        squad_ids = [p["element"] for p in entry["picks"]]
        bank = entry["entry_history"]["bank"]
        elements = pd.DataFrame(bootstrap["elements"])
        held = elements[elements["id"].isin(squad_ids)]
        budget = int(held["now_cost"].sum()) + bank
        print(f"Squad loaded from GW{current_gw - 1}. Budget £{budget / 10:.1f}m "
              f"(bank £{bank / 10:.1f}m)")
        print("  note: budget uses current prices; true selling price may be lower "
              "on risen players")
    else:
        print("No squad found for the previous gameweek - running wildcard mode.")

    if args.wildcard:
        squad_ids = []

    print(f"\nTraining {args.model} on {', '.join(TRAINING_SEASONS)}...")
    fixtures = data_fetcher.get_fixtures()
    labelled, feature_cols, panel = features.prepare(
        TRAINING_SEASONS, bootstrap=bootstrap, fixtures=fixtures
    )
    model = ml_model.train_model(labelled, feature_cols, kind=args.model)
    print(f"  {len(labelled):,} training rows, {len(feature_cols)} features")

    players, matched = build_player_table(
        bootstrap, panel, model, feature_cols, season, args.min_chance, current_gw
    )
    print(f"  matched {matched}/{len(players)} live players to historical form by code")

    players, dropped = apply_availability_gate(players, squad_ids)
    print(f"  availability gate removed {dropped} unavailable players")

    players, thin = apply_evidence_gate(players, squad_ids, args.min_evidence)
    if args.min_evidence > 0:
        print(f"  evidence gate removed {thin} players with under "
              f"{args.min_evidence} career gameweeks")

    if squad_ids:
        squads = optimize_squad(
            players,
            free_transfers=args.free_transfers,
            current_squad_ids=squad_ids,
            budget=budget,
            n=args.options,
            hard_max_transfers=args.max_transfers,
        )
        if not squads:
            print("\nNo feasible squad found.")
            return
        print(f"\nTop {len(squads)} transfer plans:")
        for index, squad in enumerate(squads, 1):
            describe_transfers(index, squad, squad_ids, players, args.free_transfers)
        print("\nRecommended squad (option 1):")
        print_squad(squads[0])
    else:
        squads = optimize_squad(players, free_transfers=15, current_squad_ids=None, budget=budget)
        if not squads:
            print("\nNo feasible squad found.")
            return
        print("\nOptimal wildcard squad:")
        print_squad(squads[0])


if __name__ == "__main__":
    main()
