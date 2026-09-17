#!/usr/bin/env python3
"""
Is a report due? Answered using nothing but the standard library.

On a laptop this hardly matters, but a cloud runner pays the setup cost on every
wake -- checkout, Python, pip install, a 118MB dataset clone -- and almost every
wake has nothing to do. Deciding first, with no third-party imports, turns a
no-op run from about four minutes into about fifteen seconds.

The window constants live here and are imported by weekly.py, so the gate and the
job can never disagree about when a send is due.
"""

import argparse
import datetime as dt
import json
import os
import ssl
import sys
import urllib.request

BASE_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; fpl-optimiser/1.0)"}

# Send on the first run inside this many hours of the deadline. A band with both
# a floor and a ceiling can be stepped over when the sampling period exceeds the
# band's width; a ceiling plus the ledger's refusal to record a gameweek twice
# cannot miss, whatever the schedule.
MAX_HOURS = 30.0
MIN_HOURS = 1.5  # below this the report lands too late to act on

LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "tracking", "ledger.jsonl")


def _tls_context():
    """
    Python from python.org on macOS ships no CA bundle and ignores the system
    keychain, so the default context trusts nothing. CI runners are fine; certifi
    covers the laptop. Optional, so this file stays runnable with no installs.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(path):
    request = urllib.request.Request(f"{BASE_URL}/{path}", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=20, context=_tls_context()) as response:
        return json.loads(response.read().decode())


def current_gameweek(bootstrap):
    """
    The gameweek to plan for: the next unfinished one.

    Order matters and is not obvious. `is_next` comes first because `is_current`
    points at a gameweek that has already kicked off, which is too late to
    transfer into -- checking is_current first returns a finished gameweek, which
    the ledger then reports as already recorded, and nothing is ever sent again.
    This mirrors data_fetcher.get_current_gameweek exactly; the two must agree or
    the pre-flight gate and the job disagree about what week it is.
    """
    events = bootstrap["events"]
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


def season_label(bootstrap):
    for event in bootstrap.get("events", []):
        deadline = event.get("deadline_time")
        if deadline:
            year, month = int(deadline[:4]), int(deadline[5:7])
            start = year if month >= 7 else year - 1
            return f"{start}-{str(start + 1)[-2:]}"
    return None


def recorded(path=LEDGER):
    """(season, gameweek) pairs already in the ledger."""
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                out.add((entry.get("season"), entry.get("gameweek")))
            except json.JSONDecodeError:
                continue
    return out


def status(ledger=LEDGER):
    """(is_due, gameweek, hours_to_deadline, reason)."""
    bootstrap = _fetch("bootstrap-static/")
    gameweek = current_gameweek(bootstrap)
    season = season_label(bootstrap)

    deadline = None
    for event in bootstrap["events"]:
        if int(event["id"]) == int(gameweek):
            deadline = event.get("deadline_time")
            break
    if not deadline:
        return False, gameweek, None, "no deadline listed"

    moment = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    hours = (moment - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600

    if (season, gameweek) in recorded(ledger):
        return False, gameweek, hours, "already recorded"
    if hours > MAX_HOURS:
        return False, gameweek, hours, f"still {hours:.1f}h out, ceiling {MAX_HOURS:.0f}h"
    if hours < MIN_HOURS:
        return False, gameweek, hours, "deadline too close or passed"
    return True, gameweek, hours, f"due, {hours:.1f}h before deadline"


def main():
    parser = argparse.ArgumentParser(description="Decide whether a report is due.")
    parser.add_argument("--github-output", action="store_true",
                        help="append due/gameweek to $GITHUB_OUTPUT for Actions")
    args = parser.parse_args()

    try:
        due, gameweek, hours, reason = status()
    except Exception as error:               # noqa: BLE001 - a failed check is not due
        print(f"check failed: {type(error).__name__}: {error}", file=sys.stderr)
        due, gameweek, hours, reason = False, 0, None, "check failed"

    print(f"GW{gameweek}: {reason}")

    if args.github_output and os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(f"due={'true' if due else 'false'}\n")
            fh.write(f"gameweek={gameweek}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
