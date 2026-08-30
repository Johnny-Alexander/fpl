"""
Stable player and team identity across FPL seasons.

FPL reassigns `element` ids every season -- 462 of the 467 players who appeared in
both 2025-26 and 2026-27 changed id. Joining last season's data onto this season's
API on `element` therefore attaches every prediction to the wrong player, silently.

The `code` field is the stable identity: it is assigned once per player and never
reused. Everything upstream of the optimizer keys on `code`; `element` ids are
resolved back only at the boundary where we talk to the live API.

Team ids are also per-season, and the gameweek CSVs store team *names* rather than
ids, so team identity is resolved through each season's teams.csv.
"""

import functools
import os

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "historical", "data")


def season_dir(season):
    return os.path.join(DATA_DIR, season)


def available_seasons():
    """Seasons present on disk that have gameweek data, oldest first."""
    if not os.path.isdir(DATA_DIR):
        return []
    seasons = []
    for name in sorted(os.listdir(DATA_DIR)):
        if len(name) == 7 and name[4] == "-":
            if os.path.exists(os.path.join(season_dir(name), "gws", "merged_gw.csv")):
                seasons.append(name)
    return seasons


@functools.lru_cache(maxsize=None)
def season_players(season):
    """
    Per-season player registry: element id, stable code, name, position.

    Sourced from players_raw.csv, which covers every element appearing in that
    season's gameweek data.
    """
    path = os.path.join(season_dir(season), "players_raw.csv")
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "element": df["id"].astype(int),
            "code": df["code"].astype(int),
            "web_name": df["web_name"],
            "element_type": df["element_type"].astype(int),
        }
    )
    return out


@functools.lru_cache(maxsize=None)
def element_to_code(season):
    p = season_players(season)
    return dict(zip(p["element"], p["code"]))


@functools.lru_cache(maxsize=None)
def code_to_element(season):
    p = season_players(season)
    return dict(zip(p["code"], p["element"]))


@functools.lru_cache(maxsize=None)
def season_teams(season):
    """Per-season team registry: id, name, short_name, and strength ratings."""
    path = os.path.join(season_dir(season), "teams.csv")
    df = pd.read_csv(path)
    keep = ["id", "name", "short_name"]
    for col in [
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]:
        if col in df.columns:
            keep.append(col)
    return df[keep].copy()


@functools.lru_cache(maxsize=None)
def team_name_to_id(season):
    """
    Map the team *name* used in merged_gw.csv to that season's team id.

    Both the full name and short name are registered, since the gameweek files
    have not been consistent about which they use.
    """
    teams = season_teams(season)
    mapping = {}
    for _, row in teams.iterrows():
        mapping[row["name"]] = int(row["id"])
        mapping[row["short_name"]] = int(row["id"])
    return mapping


def attach_identity(gw_df, season):
    """
    Add stable `code`, a real integer `team_id`, and an integer `position_id`
    to a raw merged_gw frame.

    The raw frame carries a season-local `element`, a team *name*, and a string
    position ('GK'/'DEF'/'MID'/'FWD'); none of those are usable for joining.
    """
    df = gw_df.copy()

    df["code"] = df["element"].astype(int).map(element_to_code(season))

    name_map = team_name_to_id(season)
    df["team_id"] = df["team"].map(name_map)

    df["position_id"] = df["position"].map(POSITION_TO_ID)

    df["season"] = season
    return df


POSITION_TO_ID = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
ID_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def live_code_map(bootstrap):
    """
    From a live bootstrap-static payload, build the current season's code -> element
    map and its inverse. This is the only place season-local ids re-enter the system.
    """
    elements = bootstrap["elements"]
    code_to_id = {int(e["code"]): int(e["id"]) for e in elements}
    id_to_code = {int(e["id"]): int(e["code"]) for e in elements}
    return code_to_id, id_to_code


def current_season_label(bootstrap):
    """
    Infer the season label (e.g. '2026-27') from bootstrap fixture scheduling.

    FPL seasons start in August, so the first event's deadline year is the
    opening year of the season.
    """
    events = bootstrap.get("events", [])
    for event in events:
        deadline = event.get("deadline_time")
        if deadline:
            year = int(deadline[:4])
            month = int(deadline[5:7])
            start = year if month >= 7 else year - 1
            return f"{start}-{str(start + 1)[-2:]}"
    return None
