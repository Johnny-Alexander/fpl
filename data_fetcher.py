"""
FPL API client.

Adds a small disk cache on top of the public endpoints. The backtest hits the API
once per simulated gameweek, which without caching means hundreds of identical
requests; finished gameweeks are immutable so they are cached indefinitely, while
bootstrap-static (prices, injuries) is cached for an hour.
"""

import json
import os
import time

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_DIR, ".cache")

# The API rejects requests without a plausible user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fpl-optimiser/1.0)"}

BOOTSTRAP_TTL = 3600  # prices and injury flags move daily
DEFAULT_TTL = 3600
FOREVER = None  # finished gameweeks never change


def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{key}.json")


def _read_cache(key, ttl):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    if ttl is not None and time.time() - os.path.getmtime(path) > ttl:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(key, payload):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _cache_path(key) + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _cache_path(key))
    except OSError:
        pass  # a failed cache write should never break a fetch


def _get(endpoint, cache_key=None, ttl=DEFAULT_TTL, allow_404=False):
    if cache_key:
        cached = _read_cache(cache_key, ttl)
        if cached is not None:
            return cached

    response = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, timeout=30)
    if allow_404 and response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()

    if cache_key:
        _write_cache(cache_key, payload)
    return payload


def get_bootstrap_static():
    """All static data: players (elements), teams, events (gameweeks)."""
    return _get("bootstrap-static/", "bootstrap", BOOTSTRAP_TTL)


def get_fixtures():
    """All fixtures for the current season."""
    return _get("fixtures/", "fixtures", DEFAULT_TTL)


def get_player_detailed_data(element_id):
    """Per-player fixture history and upcoming fixtures."""
    return _get(f"element-summary/{element_id}/", f"element_{element_id}", DEFAULT_TTL)


def get_user_team(team_id, gameweek):
    """
    A manager's picks for a given gameweek.

    Returns None when the gameweek has not started or the entry did not exist,
    which the API reports as a 404.
    """
    return _get(
        f"entry/{team_id}/event/{gameweek}/picks/",
        f"picks_{team_id}_{gameweek}",
        DEFAULT_TTL,
        allow_404=True,
    )


def get_entry_history(team_id):
    """A manager's gameweek-by-gameweek history for the current season."""
    return _get(f"entry/{team_id}/history/", f"history_{team_id}", DEFAULT_TTL)


def get_live_gameweek(gw, finished=False):
    """
    Live per-player stats for a gameweek.

    Finished gameweeks are cached permanently; in-flight ones expire normally.
    """
    return _get(
        f"event/{gw}/live/",
        f"live_{gw}",
        FOREVER if finished else DEFAULT_TTL,
        allow_404=True,
    )


def get_current_gameweek(bootstrap_data=None):
    """
    The gameweek to plan for: the next unfinished one.

    `is_next` is the right anchor -- `is_current` points at a gameweek that has
    already kicked off, which is too late to transfer into.
    """
    if not bootstrap_data:
        bootstrap_data = get_bootstrap_static()

    events = bootstrap_data["events"]
    for event in events:
        if event.get("is_next"):
            return event["id"]
    for event in events:
        if event.get("is_current"):
            return event["id"]
    for event in events:
        if not event.get("finished"):
            return event["id"]
    return 1


def finished_gameweeks(bootstrap_data):
    """Ids of gameweeks whose data is complete."""
    return [e["id"] for e in bootstrap_data["events"] if e.get("finished")]


# ──────────────────────────── Availability ────────────────────────────

# FPL element status codes.
#   a = available   d = doubtful   i = injured
#   s = suspended   u = unavailable (left club, ineligible)
STATUS_AVAILABLE = "a"
STATUS_DOUBTFUL = "d"

STATUS_LABELS = {
    "a": "available",
    "d": "doubtful",
    "i": "injured",
    "s": "suspended",
    "u": "unavailable",
}


def availability_frame(bootstrap_data, min_chance=75):
    """
    Per-player availability for the upcoming gameweek.

    `chance_of_playing_next_round` is null for the majority of players, which means
    "no news" rather than "no chance", so a null is treated as fully available.
    Players flagged doubtful below `min_chance` are excluded, as are all injured,
    suspended and unavailable players.
    """
    elements = pd.DataFrame(bootstrap_data["elements"])
    out = pd.DataFrame(
        {
            "element_id": elements["id"].astype(int),
            "code": elements["code"].astype(int),
            "status": elements["status"],
            "chance_next": elements["chance_of_playing_next_round"],
        }
    )

    # Null chance means no injury news has been filed.
    chance = out["chance_next"].fillna(100.0)

    out["is_available"] = (out["status"] == STATUS_AVAILABLE) | (
        (out["status"] == STATUS_DOUBTFUL) & (chance >= min_chance)
    )
    out["availability_factor"] = chance / 100.0
    out.loc[~out["is_available"], "availability_factor"] = 0.0
    out["status_label"] = out["status"].map(STATUS_LABELS).fillna("unknown")
    return out


if __name__ == "__main__":
    data = get_bootstrap_static()
    current_gw = get_current_gameweek(data)
    print(f"Planning gameweek: GW{current_gw}")
    print(f"Finished gameweeks: {len(finished_gameweeks(data))}")

    avail = availability_frame(data)
    print(f"\nSquad availability across {len(avail)} players:")
    print(avail["status_label"].value_counts().to_string())
    print(f"\nSelectable: {int(avail['is_available'].sum())}")
