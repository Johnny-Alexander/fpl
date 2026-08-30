"""
Feature pipeline for FPL points prediction.

Builds a dense player-gameweek panel keyed on stable player `code`, then derives
backward-looking form features and a next-gameweek target.

Three things this fixes relative to a naive `shift(-1)` over the raw gameweek files:

  Double gameweeks. A player with two fixtures in one gameweek has two rows, so
  `shift(-1)` returns their second fixture of the *same* gameweek as the target.
  Rows are aggregated to one per player-gameweek first.

  Blank gameweeks. A player whose team has no fixture simply has no row, so
  `shift(-1)` silently skips over the blank and returns the gameweek after it.
  The model therefore never learns that blanks score zero. The panel is reindexed
  to a dense grid so blanks are explicit.

  Season boundaries. Rolling form should carry across seasons, but the target must
  not: the last gameweek of one season would otherwise be labelled with the first
  gameweek of the next.
"""

import numpy as np
import pandas as pd

import identity

# Stats that accumulate within a gameweek and so must be summed across a double.
SUM_STATS = [
    "minutes",
    "total_points",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "starts",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
]

# Point-in-time attributes: take the last value within a gameweek.
LAST_STATS = ["value", "selected", "transfers_balance"]

# Features rolled over recent gameweeks.
ROLLING_STATS = [
    "minutes",
    "total_points",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "clean_sheets",
    "goals_conceded",
    "saves",
]

ROLLING_WINDOWS = (3, 5)
CUMULATIVE_STATS = ["goals_scored", "assists", "total_points", "minutes"]


def _numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    return df


def load_season(season):
    """Load one season's gameweek data with stable identity attached."""
    path = f"{identity.season_dir(season)}/gws/merged_gw.csv"
    raw = pd.read_csv(path)

    # 2024-25 introduced Assistant Manager elements, which occupy their own slot
    # rather than the 15-man squad and have no position in the 2/5/5/3 shape.
    # They appear only in that season (322 rows) and must not enter the panel:
    # left in, they carry a null position that propagates into the optimizer.
    if "position" in raw.columns:
        raw = raw[raw["position"] != "AM"]

    df = identity.attach_identity(raw, season)
    df = _numeric(df, SUM_STATS + LAST_STATS)
    df["GW"] = df["GW"].astype(int)

    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(bool)
    else:
        df["was_home"] = False

    # Fixture difficulty from that season's fixture list.
    df = _attach_fixture_difficulty(df, season)
    return df


def _attach_fixture_difficulty(df, season):
    """Attach the FPL difficulty rating for the fixture each row describes."""
    try:
        fixtures = pd.read_csv(f"{identity.season_dir(season)}/fixtures.csv")
        cols = fixtures[["id", "team_h_difficulty", "team_a_difficulty"]]
        df = df.merge(cols, left_on="fixture", right_on="id", how="left")
    except (FileNotFoundError, KeyError):
        df["team_h_difficulty"] = 3
        df["team_a_difficulty"] = 3

    df["match_difficulty"] = np.where(
        df["was_home"], df["team_h_difficulty"], df["team_a_difficulty"]
    )
    df["match_difficulty"] = df["match_difficulty"].fillna(3).astype(float)
    return df


def collapse_doubles(df):
    """
    Reduce to one row per player-gameweek, summing counting stats across the
    fixtures of a double gameweek.
    """
    keys = ["season", "code", "GW"]

    agg = {stat: "sum" for stat in SUM_STATS}
    agg.update({stat: "last" for stat in LAST_STATS})
    agg["team_id"] = "first"
    agg["position_id"] = "first"
    agg["name"] = "first"
    agg["element"] = "first"
    agg["was_home"] = "mean"
    agg["match_difficulty"] = "mean"

    out = df.groupby(keys, as_index=False).agg(agg)
    out["fixture_count"] = (
        df.groupby(keys, as_index=False).size().rename(columns={"size": "n"})["n"]
    )
    return out


def expand_blanks(df):
    """
    Reindex to a dense player-gameweek grid so blank gameweeks are explicit
    zero-point rows rather than gaps.

    Each player's grid spans only the gameweeks between their first and last
    appearance in that season, so we don't invent rows for a January signing
    before they arrived.
    """
    frames = []
    for (season, code), group in df.groupby(["season", "code"], sort=False):
        lo, hi = int(group["GW"].min()), int(group["GW"].max())
        full = pd.DataFrame({"GW": range(lo, hi + 1)})
        full["season"] = season
        full["code"] = code

        merged = full.merge(group, on=["season", "code", "GW"], how="left")

        # A missing row means no fixture: zero returns, zero minutes.
        merged["fixture_count"] = merged["fixture_count"].fillna(0)
        merged["is_blank"] = (merged["fixture_count"] == 0).astype(int)
        for stat in SUM_STATS:
            merged[stat] = merged[stat].fillna(0.0)

        # Carry price, ownership and identity through the blank.
        for col in LAST_STATS + ["team_id", "position_id", "name", "element"]:
            merged[col] = merged[col].ffill().bfill()

        merged["was_home"] = merged["was_home"].fillna(0.0)
        merged["match_difficulty"] = merged["match_difficulty"].fillna(0.0)
        frames.append(merged)

    return pd.concat(frames, ignore_index=True)


def live_gameweek_rows(bootstrap, fixtures, gw, season):
    """
    Build panel rows for one finished gameweek from the live API.

    The vendored dataset is a third-party mirror and typically lags the live
    season by a gameweek or more. Rather than silently planning on stale form,
    finished gameweeks missing from disk are reconstructed here. The live endpoint
    already aggregates a player's fixtures within the gameweek, so these rows are
    equivalent to a collapsed row.
    """
    live = None
    try:
        import data_fetcher

        live = data_fetcher.get_live_gameweek(gw, finished=True)
    except Exception:
        return None
    if not live or "elements" not in live:
        return None

    elements = {int(e["id"]): e for e in bootstrap["elements"]}
    total_players = max(int(bootstrap.get("total_players") or 0), 1)
    context = upcoming_fixture_context(fixtures, gw)

    rows = []
    for entry in live["elements"]:
        element_id = int(entry["id"])
        meta = elements.get(element_id)
        if meta is None:
            continue
        stats = entry.get("stats", {})
        team_id = int(meta["team"])
        count, difficulty, home = context.get(team_id, (0, 0.0, 0.0))

        row = {
            "season": season,
            "code": int(meta["code"]),
            "GW": int(gw),
            "team_id": team_id,
            "position_id": int(meta["element_type"]),
            "name": meta["web_name"],
            "element": element_id,
            "value": float(meta.get("now_cost") or 0),
            "selected": float(meta.get("selected_by_percent") or 0) / 100.0 * total_players,
            "transfers_balance": float(meta.get("transfers_in_event") or 0)
            - float(meta.get("transfers_out_event") or 0),
            "was_home": home,
            "match_difficulty": difficulty,
            "fixture_count": count,
        }
        for stat in SUM_STATS:
            row[stat] = float(stats.get(stat) or 0)
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def build_panel(seasons, bootstrap=None, fixtures=None):
    """
    Load, clean and stack seasons into a dense player-gameweek panel.

    When `bootstrap` and `fixtures` are supplied, any finished gameweek of the
    current season that is missing from disk is pulled from the live API.
    """
    frames = []
    for season in seasons:
        df = load_season(season)
        df = collapse_doubles(df)
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)

    if bootstrap is not None and fixtures is not None:
        import identity as _identity

        current = _identity.current_season_label(bootstrap)
        if current in seasons:
            on_disk = set(panel.loc[panel["season"] == current, "GW"].unique())
            finished = [e["id"] for e in bootstrap["events"] if e.get("finished")]
            missing = [gw for gw in finished if gw not in on_disk]
            for gw in missing:
                rows = live_gameweek_rows(bootstrap, fixtures, gw, current)
                if rows is not None:
                    panel = pd.concat([panel, rows], ignore_index=True)
                    print(f"  filled GW{gw} for {current} from the live API "
                          f"({len(rows)} players)")

    panel = expand_blanks(panel)

    # Order seasons chronologically so rolling windows and shifts run forwards.
    season_order = {s: i for i, s in enumerate(sorted(panel["season"].unique()))}
    panel["season_index"] = panel["season"].map(season_order)
    panel = panel.sort_values(["code", "season_index", "GW"]).reset_index(drop=True)
    return panel


# Stats whose rolling means are shrunk. Price is excluded: it is observed
# exactly, so there is nothing to regularise -- shrinking a £15.5m striker toward
# a positional average would simply be wrong.
SHRINK_STATS = [s for s in ROLLING_STATS if s != "value"]

# Pseudo-observations of the prior. K_CAREER governs how fast a rolling window
# escapes the player's own career level; K_POSITION how fast a career average
# escapes the positional baseline.
DEFAULT_K_CAREER = 2.0
DEFAULT_K_POSITION = 6.0

# Off by default -- measured, not assumed. Walk-forward over 2025-26 GW4-38,
# comparing the top-15 predicted players per gameweek and pairing the arms fold
# by fold: mean realised 4.676 unshrunk vs 4.703 shrunk, better in only 16 of 35
# gameweeks, paired t-test p=0.86, 95% CI [-0.27, +0.32]. MAE was ~1% worse in
# every evidence segment. A pooled top-50 across folds *did* look like a large
# gain, but that metric is dominated by a handful of gameweeks and did not
# survive pairing.
#
# The likely reason it does nothing: from GW4 onward almost every player has
# ample history, so there is little for a small-sample correction to fix. The
# case it was built for -- the opening gameweeks of a season, where a newcomer
# has one appearance -- is barely present in any test window we can construct,
# so this stays available but unused rather than shipped on a hunch.
DEFAULT_SHRINK = False


def positional_priors(df, stats):
    """
    Per-position mean of each stat, for each season, computed from *earlier*
    seasons only.

    Using the whole panel would leak: the prior for 2025-26 would be informed by
    how 2025-26 turned out. The oldest season has no earlier data and falls back
    to its own means, which only ever affects training rows.
    """
    priors = {}
    season_indices = sorted(df["season_index"].unique())
    for season_index in season_indices:
        earlier = df[df["season_index"] < season_index]
        source = earlier if len(earlier) else df[df["season_index"] == season_index]
        priors[season_index] = source.groupby("position_id")[stats].mean()
    return priors


def apply_shrinkage(df, k_position=DEFAULT_K_POSITION, k_career=DEFAULT_K_CAREER):
    """
    Shrink rolling form toward what is actually known about the player.

    Two levels, because one is not enough. Shrinking a rolling mean straight to a
    league baseline would punish an established player's hot streak exactly as
    hard as it punishes a newcomer's single lucky afternoon, which is the opposite
    of what the evidence supports.

        career mean   <- shrunk toward the positional baseline, weighted by how
                         many gameweeks the player has ever played
        rolling mean  <- shrunk toward that career mean, weighted by the number
                         of observations in the window

    So a player with 76 gameweeks keeps almost all of their own signal and their
    hot streak is judged against their own level; a player with one appearance is
    pulled most of the way back to what a typical player in their position does.
    """
    priors = positional_priors(df, SHRINK_STATS)

    # Positional baseline for each row, taken from earlier seasons.
    prior_frame = pd.DataFrame(index=df.index, columns=SHRINK_STATS, dtype=float)
    for season_index, table in priors.items():
        mask = df["season_index"] == season_index
        if not mask.any():
            continue
        positions = df.loc[mask, "position_id"]
        for stat in SHRINK_STATS:
            prior_frame.loc[mask, stat] = positions.map(table[stat]).to_numpy()
    prior_frame = prior_frame.fillna(0.0)

    grouped = df.groupby("code", sort=False)
    career_n = df["career_gws"].to_numpy(dtype=float)

    for stat in SHRINK_STATS:
        career_sum = grouped[stat].cumsum().to_numpy(dtype=float)
        prior = prior_frame[stat].to_numpy(dtype=float)

        # Level 1: career average, regularised toward the positional baseline.
        career_mean = (career_sum + k_position * prior) / (career_n + k_position)

        # Level 2: each rolling window, regularised toward that career average.
        for window in ROLLING_WINDOWS:
            column = f"{stat}_rolling_{window}"
            observed = df[column].to_numpy(dtype=float)
            n = np.minimum(career_n, window)
            df[column] = (n * observed + k_career * career_mean) / (n + k_career)

    return df


def add_features(panel, shrink=DEFAULT_SHRINK, k_position=DEFAULT_K_POSITION,
                 k_career=DEFAULT_K_CAREER):
    """
    Derive backward-looking form features and the next-gameweek target.

    Rolling windows include the current row, which is correct: the target is the
    *following* gameweek, so nothing from the target window leaks in.
    """
    df = panel.copy()
    by_code = df.groupby("code", sort=False)

    # Target: points in the next gameweek, within the same season. Minutes are
    # carried alongside so a model can separate "will they play" from "how much
    # will they score if they do".
    df["target_points"] = by_code["total_points"].shift(-1)
    df["target_minutes"] = by_code["minutes"].shift(-1)
    next_season = by_code["season_index"].shift(-1)
    df.loc[next_season != df["season_index"], ["target_points", "target_minutes"]] = np.nan

    # Next-gameweek fixture context. All of this is published in advance, so it
    # is legitimately known at prediction time.
    df["next_difficulty"] = by_code["match_difficulty"].shift(-1)
    df["next_fixture_count"] = by_code["fixture_count"].shift(-1)
    df["next_was_home"] = by_code["was_home"].shift(-1)
    df.loc[next_season != df["season_index"], ["next_difficulty", "next_fixture_count", "next_was_home"]] = np.nan

    # How much evidence exists for this player, counted across seasons. Used both
    # as a feature and as the weight in the shrinkage below.
    df["career_gws"] = df.groupby("code", sort=False).cumcount() + 1
    df["log_career_gws"] = np.log1p(df["career_gws"])

    for stat in ROLLING_STATS:
        grouped = df.groupby("code", sort=False)[stat]
        for window in ROLLING_WINDOWS:
            df[f"{stat}_rolling_{window}"] = grouped.transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )

    if shrink:
        df = apply_shrinkage(df, k_position=k_position, k_career=k_career)

    # Season-to-date totals, reset each season.
    by_season = df.groupby(["code", "season_index"], sort=False)
    for stat in CUMULATIVE_STATS:
        df[f"{stat}_cumsum"] = by_season[stat].cumsum()

    df["pts_per_min"] = np.where(
        df["minutes_cumsum"] > 0,
        df["total_points_cumsum"] / df["minutes_cumsum"] * 90,
        0.0,
    )

    # Share of recent gameweeks actually started -- the strongest minutes signal.
    df["start_rate_5"] = df.groupby("code", sort=False)["starts"].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )

    df["log_selected"] = np.log1p(df["selected"].clip(lower=0))

    return df


def feature_columns():
    """The model's input columns."""
    cols = []
    for stat in ROLLING_STATS:
        for window in ROLLING_WINDOWS:
            cols.append(f"{stat}_rolling_{window}")
    cols += [f"{stat}_cumsum" for stat in ["goals_scored", "assists"]]
    cols += [
        "pts_per_min",
        "start_rate_5",
        "log_selected",
        # log_career_gws is computed but deliberately not a feature. Handing the
        # model an explicit evidence count cost 62 points over the 2025-26
        # backtest (1996 -> 1934); `log_selected` already proxies for how
        # established a player is, and the extra split appears to do more harm
        # than good. The column stays available for the evidence gate.
        "position_id",
        "next_difficulty",
        "next_fixture_count",
        "next_was_home",
    ]
    return cols


def prepare(seasons, bootstrap=None, fixtures=None, shrink=DEFAULT_SHRINK,
            k_position=DEFAULT_K_POSITION, k_career=DEFAULT_K_CAREER):
    """
    Build the panel and split it into (labelled training rows, feature names,
    full panel). The full panel retains unlabelled rows, which is what prediction
    for the upcoming gameweek needs.
    """
    panel = build_panel(seasons, bootstrap=bootstrap, fixtures=fixtures)
    panel = add_features(panel, shrink=shrink, k_position=k_position,
                         k_career=k_career)
    cols = feature_columns()
    labelled = panel.dropna(subset=["target_points"] + cols).copy()
    return labelled, cols, panel


def latest_rows(panel, season):
    """
    The most recent row per player for a given season -- the basis for predicting
    the upcoming gameweek.
    """
    season_rows = panel[panel["season"] == season]
    return season_rows.sort_values(["code", "GW"]).drop_duplicates("code", keep="last").copy()


def upcoming_fixture_context(fixtures, gameweek):
    """
    Per-team fixture context for an upcoming gameweek, from the live fixture list.

    Returns team_id -> (count, mean difficulty, home share). A team absent from
    the result has a blank gameweek. Counting fixtures rather than overwriting
    per team matters for doubles, which the previous version collapsed to
    whichever fixture happened to be last in the list.
    """
    context = {}
    for fixture in fixtures:
        if fixture.get("event") != gameweek:
            continue
        for team_key, difficulty_key, is_home in (
            ("team_h", "team_h_difficulty", True),
            ("team_a", "team_a_difficulty", False),
        ):
            team = fixture.get(team_key)
            if team is None:
                continue
            entry = context.setdefault(int(team), {"n": 0, "difficulty": [], "home": []})
            entry["n"] += 1
            entry["difficulty"].append(fixture.get(difficulty_key) or 3)
            entry["home"].append(1.0 if is_home else 0.0)

    return {
        team: (
            entry["n"],
            float(np.mean(entry["difficulty"])),
            float(np.mean(entry["home"])),
        )
        for team, entry in context.items()
    }


def apply_upcoming_fixtures(latest, fixtures, gameweek):
    """
    Fill the next-gameweek fixture features for prediction rows.

    On the most recent row of a season there is no following row to shift from,
    so these columns are null. They are not unknown, though -- the fixture list is
    published in advance. Teams with no fixture are marked blank (count 0) rather
    than defaulted to an average difficulty, which is what previously let the
    optimizer buy into a blank gameweek.
    """
    context = upcoming_fixture_context(fixtures, gameweek)
    out = latest.copy()

    counts, difficulties, homes = [], [], []
    for team in out["team_id"]:
        n, difficulty, home = context.get(int(team), (0, 0.0, 0.0))
        counts.append(n)
        difficulties.append(difficulty)
        homes.append(home)

    out["next_fixture_count"] = counts
    out["next_difficulty"] = difficulties
    out["next_was_home"] = homes
    return out


if __name__ == "__main__":
    seasons = ["2024-25", "2025-26", "2026-27"]
    labelled, cols, panel = prepare(seasons)
    print(f"panel rows      : {len(panel):,}")
    print(f"labelled rows   : {len(labelled):,}")
    print(f"features        : {len(cols)}")
    print(f"blank rows      : {int(panel['is_blank'].sum()):,}")
    print(f"double gws      : {int((panel['fixture_count'] == 2).sum()):,}")
    print("\ntarget distribution:")
    print(labelled["target_points"].describe().to_string())
