"""
Population benchmarks for the model's season totals.

Every result so far has been measured against one manager's one season, which
says little about whether a total is actually good. This samples real managers at
random and reports a score as a percentile of the playing population.

The mapping is built from FPL's own `rank_percentage`, which each manager's
history reports alongside their season total. Using it sidesteps the hard part:
a percentile computed from sampled totals alone needs the size of the playing
population as a denominator, and that is not published for a completed season --
`total_players` describes the season in progress, which is a different and
smaller number. It also makes the result robust to *which* managers get drawn,
since every sampled manager states their own true percentile; the sample only has
to cover the range of scores, not match the population's density.

That matters, because a naive draw is biased. Entry ids are issued in
registration order, so sampling up to the current season's id ceiling
systematically omits the newest managers, who score below average. On a first
run that pushed the estimate for a known score eight percentiles too high.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import requests

import data_fetcher

CACHE = os.path.join(data_fetcher.CACHE_DIR, "manager_history.json")
DEFAULT_SAMPLE = 900
REQUEST_DELAY = 0.12  # polite spacing between calls

# Ids are issued in registration order and a completed season's ceiling is not
# published, so the draw reaches beyond the current count. Ids above the real
# ceiling simply return nothing and drop out.
ID_HEADROOM = 1.35


def _load_cache():
    if os.path.exists(CACHE):
        try:
            with open(CACHE) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(payload):
    os.makedirs(data_fetcher.CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE, "w") as fh:
            json.dump(payload, fh)
    except OSError:
        pass


def sample_managers(n=DEFAULT_SAMPLE, seed=7, verbose=True):
    """
    Past-season records for `n` randomly drawn managers.

    Returns {season_label: [(total, rank, top_percent), ...]}. Cached per entry
    id, so raising `n` only fetches the additional managers.
    """
    cache = _load_cache()
    bootstrap = data_fetcher.get_bootstrap_static()
    ceiling = int((bootstrap.get("total_players") or 10_000_000) * ID_HEADROOM)

    rng = random.Random(seed)
    wanted = [rng.randint(1, ceiling) for _ in range(n)]
    missing = [i for i in wanted if str(i) not in cache]

    if missing and verbose:
        print(f"  fetching {len(missing)} managers ({len(wanted) - len(missing)} cached)...")

    for index, entry_id in enumerate(missing, 1):
        try:
            response = requests.get(
                f"{data_fetcher.BASE_URL}/entry/{entry_id}/history/",
                headers=data_fetcher.HEADERS,
                timeout=15,
            )
            payload = response.json() if response.status_code == 200 else {}
        except (requests.RequestException, ValueError):
            payload = {}

        cache[str(entry_id)] = [
            {
                "season": row.get("season_name"),
                "total": row.get("total_points"),
                "rank": row.get("rank"),
                "top_percent": row.get("rank_percentage"),
            }
            for row in payload.get("past", [])
        ]
        if verbose and index % 150 == 0:
            print(f"    {index}/{len(missing)}", flush=True)
        time.sleep(REQUEST_DELAY)

    if missing:
        _save_cache(cache)

    by_season = {}
    for entry_id in wanted:
        for row in cache.get(str(entry_id), []):
            if row.get("total") is None or row.get("top_percent") in (None, ""):
                continue
            try:
                record = (int(row["total"]), int(row["rank"]), float(row["top_percent"]))
            except (TypeError, ValueError):
                continue
            by_season.setdefault(row["season"], []).append(record)
    return by_season


def percentile_curve(records):
    """
    Sorted (total, percentile) pairs, where percentile is the share of managers
    the total beats. FPL reports the complement, so it is inverted here.
    """
    points = sorted((total, 100.0 - top) for total, _, top in records)
    totals = np.array([p[0] for p in points], dtype=float)
    percentiles = np.array([p[1] for p in points], dtype=float)
    # Percentile is monotone in points; enforce it so interpolation is sane
    # despite the rounding in FPL's reported figure.
    percentiles = np.maximum.accumulate(percentiles)
    return totals, percentiles


def locate(total, records, population=None):
    """
    Percentile for a season total, and the rank that implies.

    Rank is derived from the percentile rather than interpolated from sampled
    ranks directly: rank falls as points rise, and interpolating a decreasing
    series invites exactly the sign error that produced ranks improving as scores
    got worse.
    """
    totals, percentiles = percentile_curve(records)
    percentile = float(np.interp(total, totals, percentiles))
    if population is None:
        population = population_size(records)
    rank = population * (1.0 - percentile / 100.0)
    return percentile, rank


def population_size(records):
    """
    Managers who played that season, implied by rank and reported percentile.

    A manager ranked r in the top p% implies a population of about r / (p/100).
    Averaged over the sample, away from the extremes where rounding dominates.
    """
    estimates = [
        rank / (top / 100.0)
        for _, rank, top in records
        if 2.0 <= top <= 98.0 and rank > 0
    ]
    return float(np.median(estimates)) if estimates else float("nan")


# Totals from the four-season replication in backtest/validate.py. Kept here so
# the percentile view can be reproduced without re-running an hour of backtests;
# regenerate them with `python3 backtest/validate.py`.
BACKTEST_TOTALS = {
    "2022-23": {"hold": 1742, "transfers": 1995, "chips": 2112},
    "2023-24": {"hold": 1532, "transfers": 1983, "chips": 2075},
    "2024-25": {"hold": 1899, "transfers": 1956, "chips": 2030},
    "2025-26": {"hold": 1806, "transfers": 1996, "chips": 2250},
}

ARM_LABELS = {"hold": "hold", "transfers": "transfers", "chips": "+chips"}


def report_backtest(by_season, totals=None):
    """
    Place each season's backtest totals in that season's population.

    Raw totals are not comparable across seasons -- the median manager scored 2210
    in 2022-23 and 2003 in 2025-26 -- so a percentile is the only fair way to read
    them side by side.
    """
    totals = totals or BACKTEST_TOTALS
    header = f"{'season':<10}{'n':>5}{'median':>8}   "
    header += "".join(f"{ARM_LABELS[a]:>18}" for a in ("hold", "transfers", "chips"))
    print(header)
    print("-" * len(header))

    collected = {arm: [] for arm in ("hold", "transfers", "chips")}
    for season, arms in totals.items():
        records = by_season.get(season.replace("-", "/"))
        if not records:
            print(f"{season:<10}  no sampled managers")
            continue
        population = population_size(records)
        median = np.median([t for t, _, _ in records])
        cells = []
        for arm in ("hold", "transfers", "chips"):
            percentile, _ = locate(arms[arm], records, population)
            collected[arm].append(percentile)
            cells.append(f"{arms[arm]} ({percentile:.0f}th)")
        print(f"{season:<10}{len(records):>5}{median:>8.0f}   "
              + "".join(f"{c:>18}" for c in cells))

    print("-" * len(header))
    means = "".join(f"{np.mean(collected[a]):>16.0f}th" for a in ("hold", "transfers", "chips"))
    print(f"{'mean':<10}{'':>13}   {means}")
    return {arm: float(np.mean(values)) for arm, values in collected.items() if values}


def main():
    parser = argparse.ArgumentParser(description="Where a season total sits in the population.")
    parser.add_argument("--backtest", action="store_true",
                        help="place the four-season backtest totals in their populations")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--season", default="2025/26")
    parser.add_argument("--scores", type=int, nargs="*",
                        default=[1806, 1996, 2151, 2250])
    parser.add_argument("--check-total", type=int, default=2151)
    parser.add_argument("--check-top-percent", type=float, default=12.0)
    args = parser.parse_args()

    print("=" * 66)
    print(f"  Manager population, {args.season}")
    print("=" * 66)

    by_season = sample_managers(args.sample)

    if args.backtest:
        print()
        report_backtest(by_season)
        return

    records = by_season.get(args.season)
    if not records:
        print(f"  no sampled managers played {args.season}")
        return

    totals = np.array([t for t, _, _ in records], dtype=float)
    print(f"\n  {len(records)} sampled managers played {args.season}")
    print(f"    min {totals.min():.0f}   p10 {np.percentile(totals, 10):.0f}"
          f"   median {np.median(totals):.0f}   p90 {np.percentile(totals, 90):.0f}"
          f"   max {totals.max():.0f}")
    print(f"    implied population: {population_size(records):,.0f} managers")

    population = population_size(records)
    estimated, _ = locate(args.check_total, records, population)
    truth = 100.0 - args.check_top_percent
    print(f"\n  sanity check")
    print(f"    FPL reports {args.check_total} as top {args.check_top_percent:.0f}%"
          f" -> {truth:.0f}th percentile")
    print(f"    this sample estimates {estimated:.1f}th percentile"
          f"  ({estimated - truth:+.1f})")
    print(f"    sample looks {'usable' if abs(estimated - truth) <= 3 else 'BIASED'}")

    print(f"\n  {'total':>7}{'percentile':>13}{'top':>9}{'approx rank':>14}")
    for score in sorted(args.scores):
        percentile, rank = locate(score, records, population)
        print(f"  {score:>7}{percentile:>12.1f}%{100 - percentile:>8.1f}%{rank:>14,.0f}")


if __name__ == "__main__":
    main()
