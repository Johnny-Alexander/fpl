"""
Prediction ledger for the live 2026-27 experiment.

The point of running this season is not the weekly email but the record it
leaves. Each recommendation is appended before its deadline, so what the model
said is fixed before the gameweek is played and cannot be revised afterwards.
Scoring happens later, against what actually occurred.

The question this is built to answer: the model averaged the 70th percentile
across four backtested seasons, and its one strong season (2025-26, 96th) was
also the first with eight chips, when humans were still adapting to the rule
change. Whether the edge survives a season where they are not is not something a
backtest can settle.
"""

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "backtest"))

import data_fetcher

LEDGER_DIR = os.path.join(PROJECT_DIR, "tracking")
LEDGER = os.path.join(LEDGER_DIR, "ledger.jsonl")


def record(recommendation, path=LEDGER):
    """
    Append a recommendation, refusing to overwrite one already made.

    A gameweek may only be recorded once. Re-running before a deadline is
    harmless and common -- but silently replacing an earlier prediction with a
    later one would let the record drift toward whatever was known latest, which
    is exactly the bias the ledger exists to prevent.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {e["gameweek"] for e in entries(path)
                if e.get("season") == recommendation.get("season")}
    if recommendation["gameweek"] in existing:
        return False

    with open(path, "a") as fh:
        fh.write(json.dumps(recommendation, sort_keys=True) + "\n")
    return True


def entries(path=LEDGER):
    """Every recorded recommendation, oldest first."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return sorted(out, key=lambda e: (e.get("season", ""), e.get("gameweek", 0)))


def actual_points(gameweek):
    """{element_id: (points, minutes)} for a finished gameweek."""
    live = data_fetcher.get_live_gameweek(gameweek, finished=True)
    if not live or "elements" not in live:
        return {}
    return {
        int(e["id"]): (
            float(e["stats"].get("total_points") or 0),
            float(e["stats"].get("minutes") or 0),
        )
        for e in live["elements"]
    }


def score_entry(entry, results=None):
    """
    What the recommended team actually scored, against what the manager scored.

    Autosubs are applied, so a starter who did not play is replaced by a bench
    player who did, exactly as FPL would. Scoring the XI as picked would
    understate every recommendation.
    """
    from backtest import apply_autosubs  # noqa: E402  (path set at import time)

    results = results if results is not None else actual_points(entry["gameweek"])
    if not results:
        return None

    squad = entry.get("squad") or []
    if not squad:
        return None

    positions = {p["element_id"]: p["position"] for p in squad}
    minutes = {(p["element_id"], entry["gameweek"]): results.get(p["element_id"], (0, 0))[1]
               for p in squad}
    starters = [p["element_id"] for p in squad if p.get("is_starter")]
    bench = [p["element_id"] for p in squad if not p.get("is_starter")]
    captain = next((p["element_id"] for p in squad if p.get("is_captain")), None)

    playing = apply_autosubs(starters, bench, minutes, positions, entry["gameweek"])

    total = 0.0
    for element_id in playing:
        points, played = results.get(element_id, (0.0, 0.0))
        if element_id == captain and played > 0:
            points *= 2
        total += points

    hit = entry["options"][0]["hit_points"] if entry.get("options") else 0
    return {
        "gameweek": entry["gameweek"],
        "model_points": round(total - hit, 1),
        "model_hit": hit,
        "predicted": entry.get("predicted_xi_points"),
        "captain": (entry.get("captain") or {}).get("name"),
        "captain_points": round(results.get(captain, (0.0, 0.0))[0], 1) if captain else None,
    }


def manager_points(team_id):
    """{gameweek: points} actually scored by the manager this season."""
    history = data_fetcher.get_entry_history(team_id)
    return {int(g["event"]): float(g["points"]) for g in history.get("current", [])}


def history(path=LEDGER, team_id=None):
    """
    Scored ledger entries with running totals for both sides.

    Only gameweeks that have finished are scored; a recommendation made for an
    upcoming gameweek is carried with `scored: False`.
    """
    bootstrap = data_fetcher.get_bootstrap_static()
    finished = {int(e["id"]) for e in bootstrap["events"] if e.get("finished")}

    recorded = entries(path)
    if team_id is None and recorded:
        team_id = recorded[0].get("team_id")
    actual = manager_points(team_id) if team_id else {}

    rows, model_total, human_total = [], 0.0, 0.0
    for entry in recorded:
        gameweek = entry["gameweek"]
        row = {"gameweek": gameweek, "scored": gameweek in finished,
               "predicted": entry.get("predicted_xi_points")}
        if row["scored"]:
            scored = score_entry(entry)
            if scored:
                model_total += scored["model_points"]
                human_total += actual.get(gameweek, 0.0)
                row.update(scored)
                row["human_points"] = actual.get(gameweek)
                row["model_cumulative"] = round(model_total, 1)
                row["human_cumulative"] = round(human_total, 1)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Inspect the prediction ledger.")
    parser.add_argument("--team-id", type=int, default=None)
    args = parser.parse_args()

    rows = history(team_id=args.team_id)
    if not rows:
        print("Ledger is empty. Run weekly.py to record a recommendation.")
        return

    print(f"  {'GW':>3}{'pred':>7}{'model':>7}{'you':>6}{'diff':>7}"
          f"{'model cum':>11}{'your cum':>10}  captain")
    for row in rows:
        if not row.get("scored") or "model_points" not in row:
            print(f"  {row['gameweek']:>3}{row.get('predicted') or 0:>7.1f}"
                  f"{'—':>7}{'—':>6}{'—':>7}{'':>11}{'':>10}  (not yet played)")
            continue
        human = row.get("human_points")
        diff = row["model_points"] - human if human is not None else None
        print(f"  {row['gameweek']:>3}{row['predicted'] or 0:>7.1f}"
              f"{row['model_points']:>7.1f}{human if human is not None else 0:>6.0f}"
              f"{diff if diff is not None else 0:>+7.1f}"
              f"{row['model_cumulative']:>11.0f}{row['human_cumulative']:>10.0f}"
              f"  {row.get('captain') or ''}")


if __name__ == "__main__":
    main()
