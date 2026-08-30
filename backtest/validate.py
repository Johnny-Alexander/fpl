#!/usr/bin/env python3
"""
Replicate the backtest across several seasons.

Every headline figure this project has produced came from a single season, and
single-season results have repeatedly failed to survive replication -- the chip
gain alone varied fourfold between the first two seasons tested. This runs the
same three strategies over each season in turn, each trained on the season before
it, so the spread across seasons is visible rather than assumed.

Each season uses the chip allocation actually in force that year: FPL doubled it
from five chips to eight in 2025-26, and granting an older season the modern set
would credit it with chips it never had.
"""

import argparse
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import chips
import features
from backtest import simulate

DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

ARMS = [
    ("hold", "hold", False),
    ("transfers", "model", False),
    ("transfers+chips", "model", True),
]


def prior_season(season):
    """The season immediately before, used as training history."""
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def run_season(season, model_kind, retrain_every, max_transfers):
    """Play one season under each arm; returns {arm: total}."""
    train_seasons = [prior_season(season), season]
    for label in train_seasons:
        path = os.path.join(features.identity.season_dir(label), "gws", "merged_gw.csv")
        if not os.path.exists(path):
            print(f"  {season}: missing {label}, skipping")
            return None

    labelled, feature_cols, panel = features.prepare(train_seasons)
    windows = chips.windows_for_season(season)

    totals, chip_logs = {}, {}
    for arm, strategy, use_chips in ARMS:
        scores, _, chip_log = simulate(
            panel, labelled, feature_cols, season, 1, 38, model_kind,
            strategy=strategy, retrain_every=retrain_every,
            max_transfers=max_transfers, verbose=False,
            use_chips=use_chips, chip_windows=windows,
        )
        totals[arm] = sum(scores.values())
        chip_logs[arm] = chip_log
        print(f"    {arm:<16}{totals[arm]:>8.0f}", flush=True)

    played = chip_logs.get("transfers+chips") or []
    if played:
        print(f"    chips: {', '.join(f'{c}{g}' for g, c in played)}")
    return totals


def main():
    parser = argparse.ArgumentParser(description="Multi-season backtest replication.")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--model", default="gbr")
    parser.add_argument("--retrain-every", type=int, default=4)
    parser.add_argument("--max-transfers", type=int, default=2)
    args = parser.parse_args()

    print("=" * 66)
    print("  Multi-season replication")
    print("=" * 66)

    results = {}
    for season in args.seasons:
        allocation = "8 chips" if season >= chips.DOUBLED_FROM else "5 chips"
        print(f"\n  {season}  (trained on {prior_season(season)}, {allocation})")
        totals = run_season(season, args.model, args.retrain_every, args.max_transfers)
        if totals:
            results[season] = totals

    if not results:
        print("\n  no seasons ran")
        return

    print(f"\n{'=' * 66}")
    print("  SUMMARY")
    print(f"{'=' * 66}")
    print(f"  {'season':<12}{'hold':>8}{'transfers':>11}{'+chips':>9}"
          f"{'transfer Δ':>12}{'chip Δ':>9}")
    print(f"  {'-' * 60}")

    transfer_gains, chip_gains = [], []
    for season, totals in results.items():
        transfer_gain = totals["transfers"] - totals["hold"]
        chip_gain = totals["transfers+chips"] - totals["transfers"]
        transfer_gains.append(transfer_gain)
        chip_gains.append(chip_gain)
        print(f"  {season:<12}{totals['hold']:>8.0f}{totals['transfers']:>11.0f}"
              f"{totals['transfers+chips']:>9.0f}{transfer_gain:>+12.0f}{chip_gain:>+9.0f}")

    print(f"  {'-' * 60}")
    summarise("transfers vs hold", transfer_gains)
    summarise("chips vs no chips", chip_gains)


def summarise(label, values):
    if not values:
        return
    mean = statistics.mean(values)
    spread = f"{min(values):+.0f} to {max(values):+.0f}"
    if len(values) > 1:
        sd = statistics.stdev(values)
        print(f"  {label:<22} mean {mean:+.0f}   sd {sd:.0f}   range {spread}")
    else:
        print(f"  {label:<22} mean {mean:+.0f}   range {spread}")


if __name__ == "__main__":
    main()
