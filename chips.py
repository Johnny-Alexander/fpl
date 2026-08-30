"""
Chip strategy.

FPL grants eight chips a season -- two each of wildcard, free hit, bench boost and
triple captain -- one of each usable in the first half and one in the second. An
unused chip expires worth nothing, so the decision is never "is this a good week"
in isolation but "is this good enough given how many weeks remain to use it".

That shape is handled with a threshold that decays to zero at the end of each
chip's window: early on a chip is only spent on a standout week, and by the final
gameweek of the window it is spent regardless, because holding it is worth less
than any positive return.

Windows come from the live API (`bootstrap['chips']`) where available, since FPL
has changed the chip allocation between seasons -- it doubled from five chips to
eight in 2025-26 -- and hardcoding the old rules would silently under-use them.
"""

# Optimizer codes for each FPL chip name.
CHIP_CODES = {
    "wildcard": "WC",
    "freehit": "FH",
    "bboost": "BB",
    "3xc": "TC",
}

# The 2025-26 onward allocation: two of each, split at the halfway point. Used
# when the live chip schedule is unavailable (e.g. backtesting a past season).
DEFAULT_WINDOWS = [
    ("wildcard", 2, 19), ("wildcard", 20, 38),
    ("freehit", 2, 19), ("freehit", 20, 38),
    ("bboost", 1, 19), ("bboost", 20, 38),
    ("3xc", 1, 19), ("3xc", 20, 38),
]

# Before 2025-26: two wildcards split by half, and a single free hit, bench boost
# and triple captain each playable in any gameweek. Backtesting an older season
# with the current allocation would hand it three chips it never had.
LEGACY_WINDOWS = [
    ("wildcard", 2, 19), ("wildcard", 20, 38),
    ("freehit", 1, 38),
    ("bboost", 1, 38),
    ("3xc", 1, 38),
]

# The season the allocation doubled.
DOUBLED_FROM = "2025-26"


def windows_for_season(season):
    """Chip allocation in force for a given season label, e.g. '2023-24'."""
    return DEFAULT_WINDOWS if season >= DOUBLED_FROM else LEGACY_WINDOWS

# Minimum expected gain, in points, to spend a chip at the *start* of its window.
# Each decays linearly to zero at the window's final gameweek.
#
# These are 1.5x an initial hand-picked set, adopted because that multiplier beat
# the original on *both* seasons tested rather than on the one being reported:
# 2025-26 2250 vs 2198, and 2024-25 2030 vs 1975. Higher bars mean chips are held
# for genuinely standout weeks instead of being spent on the first decent one.
# Two seasons is weak evidence; treat these as a reasonable prior, not a fit.
DEFAULT_THRESHOLDS = {
    "TC": 12.0,  # extra points from tripling rather than doubling the captain
    "BB": 21.0,  # points the four bench players are expected to contribute
    "WC": 18.0,  # gain of an unconstrained rebuild over the best normal move
    "FH": 24.0,  # same, but for one week only, so it must clear a higher bar
}


def windows_from_bootstrap(bootstrap):
    """
    Chip windows for the current season, from the API.

    Returns None when the payload has no usable chip schedule, so callers can
    fall back to DEFAULT_WINDOWS.
    """
    entries = (bootstrap or {}).get("chips") or []
    windows = []
    for entry in entries:
        name = entry.get("name")
        start, stop = entry.get("start_event"), entry.get("stop_event")
        if name in CHIP_CODES and start and stop:
            windows.append((name, int(start), int(stop)))
    return windows or None


class ChipState:
    """Tracks which chips remain and when each may be played."""

    def __init__(self, windows=None, thresholds=None):
        self.windows = list(windows or DEFAULT_WINDOWS)
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.used = [False] * len(self.windows)

    def available(self, gw):
        """(index, optimizer code, gameweeks left in window) for each playable chip."""
        out = []
        for index, (name, start, stop) in enumerate(self.windows):
            if not self.used[index] and start <= gw <= stop:
                out.append((index, CHIP_CODES[name], stop - gw))
        return out

    def threshold(self, code, gw):
        """
        Bar the chip's expected gain must clear this week.

        Decays linearly across the window so an unspent chip is always played on
        the last gameweek it is legal, rather than expiring for nothing.
        """
        base = self.thresholds.get(code, 0.0)
        spans = [
            (start, stop)
            for index, (name, start, stop) in enumerate(self.windows)
            if CHIP_CODES[name] == code and not self.used[index] and start <= gw <= stop
        ]
        if not spans:
            return float("inf")
        start, stop = min(spans, key=lambda s: s[1])
        if stop <= start:
            return 0.0
        remaining = (stop - gw) / (stop - start)
        return base * max(0.0, min(1.0, remaining))

    def play(self, index):
        self.used[index] = True

    def mark_played(self, code, gw):
        """Consume the soonest-expiring chip of this type that is legal now."""
        candidates = [
            (index, stop)
            for index, (name, start, stop) in enumerate(self.windows)
            if CHIP_CODES[name] == code and not self.used[index] and start <= gw <= stop
        ]
        if candidates:
            self.play(min(candidates, key=lambda c: c[1])[0])
            return True
        return False

    @property
    def remaining(self):
        return sum(1 for used in self.used if not used)


def triple_captain_gain(squad, pred_map):
    """
    Extra points from tripling rather than doubling the captain.

    The captain already scores double, so the chip is worth one further copy of
    the best player's expected points.
    """
    if not squad:
        return 0.0
    return max((pred_map.get(code, 0.0) for code in squad), default=0.0)


def bench_boost_gain(squad, pred_map):
    """
    Points the bench is expected to contribute.

    Ranking the squad by prediction approximates the eleven that would start, so
    the gain is whatever the remaining four are worth.
    """
    if len(squad) <= 11:
        return 0.0
    ranked = sorted(squad, key=lambda code: pred_map.get(code, 0.0), reverse=True)
    return sum(pred_map.get(code, 0.0) for code in ranked[11:])


def expected_from_squad(squad, pred_map):
    """Expected points of the best eleven in a squad, with the captain doubled."""
    if not squad:
        return 0.0
    ranked = sorted(squad, key=lambda code: pred_map.get(code, 0.0), reverse=True)
    eleven = ranked[:11]
    return sum(pred_map.get(code, 0.0) for code in eleven) + pred_map.get(eleven[0], 0.0)
