#!/usr/bin/env python3
"""
The weekly job: decide whether a deadline is close, build the report, record it,
and email it.

Run daily rather than weekly. FPL deadlines this season sit anywhere from two to
twenty-one days apart -- midweek rounds, then a three-week international break --
so a fixed weekly schedule would miss most of them. This checks each day whether
the next deadline falls inside the send window and does nothing otherwise, which
makes the schedule robust to the fixture calendar rather than dependent on it.

Credentials come from the environment or a gitignored .env beside this file:

    FPL_SMTP_USER=you@example.com
    FPL_SMTP_PASSWORD=<app password, not your account password>
    FPL_EMAIL_TO=you@example.com
    FPL_SMTP_HOST=smtp-mail.outlook.com   # optional
    FPL_SMTP_PORT=587                     # optional
"""

import argparse
import base64
import datetime as dt
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from email.utils import make_msgid

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import data_fetcher
import main as recommender
import report
import tracker

ENV_FILE = os.path.join(PROJECT_DIR, ".env")
# Send on the first run that falls inside this many hours of the deadline.
#
# A window with both a floor and a ceiling looks natural and is a trap: sampling
# every 24 hours moves the time-to-deadline by exactly 24 each run, so a window
# narrower than the sampling period can be stepped straight over -- 30 hours out
# on one run, 6 on the next, never inside a 20-28 band. A ceiling plus the
# ledger's refusal to record a gameweek twice cannot miss, whatever the schedule.
# Paired with the 6-hourly job this fires between 24 and 30 hours before.
DEFAULT_MAX_HOURS = 30.0
MIN_HOURS = 1.5  # below this the report arrives too late to act on


def load_env(path=ENV_FILE):
    """Read KEY=VALUE lines into the environment without overriding what is set."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def hours_to_deadline(bootstrap, gameweek):
    deadline = recommender.deadline_for(bootstrap, gameweek)
    if not deadline:
        return None, None
    moment = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    return (moment - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600, moment


def build(team_id, free_transfers, chips_used, quiet=True):
    """Recommendation, ledger history, and the rendered charts."""
    log = (lambda *a, **k: None) if quiet else print
    recommendation = recommender.build_recommendation(
        team_id=team_id, free_transfers=free_transfers,
        chips_used=chips_used, log=log,
    )

    images = {}
    if recommendation.get("squad"):
        bootstrap = data_fetcher.get_bootstrap_static()
        fixtures = data_fetcher.get_fixtures()
        teams = {p["team"] for p in recommendation["squad"] if p.get("team")}
        outlook = report.fixture_outlook(
            fixtures, bootstrap, teams, recommendation["gameweek"], 5
        )
        images["fixtures"] = report.chart_fixture_grid(
            recommendation["squad"], outlook, recommendation["gameweek"], 5
        )

    history = tracker.history(team_id=team_id)
    tracker_png = report.chart_season_tracker(history)
    if tracker_png:
        images["tracker"] = tracker_png

    return recommendation, history, images


def compose(recommendation, history, images, to_address, from_address):
    message = EmailMessage()
    message["Subject"] = (
        f"FPL GW{recommendation['gameweek']}: "
        + (recommendation["options"][0]["moves"][0]["in"]["name"]
           if recommendation.get("options") and recommendation["options"][0]["moves"]
           else "no transfer")
        + f", captain {(recommendation.get('captain') or {}).get('name', '?')}"
    )
    message["From"] = from_address
    message["To"] = to_address

    cids = {name: make_msgid(idstring="fpl")[1:-1] for name in images}
    html = report.render_html(recommendation, history, images)
    for name, cid in cids.items():
        html = html.replace(f"cid:{name}", f"cid:{cid}")

    message.set_content(
        "This report is formatted as HTML. "
        f"GW{recommendation['gameweek']} captain: "
        f"{(recommendation.get('captain') or {}).get('name', '?')}."
    )
    message.add_alternative(html, subtype="html")

    payload = message.get_payload()[-1]
    for name, data in images.items():
        payload.add_related(data, maintype="image", subtype="png",
                            cid=f"<{cids[name]}>", filename=f"{name}.png")
    return message


def send(message, host, port, user, password):
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls(context=context)
        server.login(user, password)
        server.send_message(message)


def main():
    parser = argparse.ArgumentParser(description="Deadline-aware weekly FPL report.")
    parser.add_argument("--team-id", type=int, default=recommender.DEFAULT_TEAM_ID)
    parser.add_argument("--free-transfers", type=int, default=1)
    parser.add_argument("--chips-used", nargs="*", default=[], metavar="CODE")
    parser.add_argument("--force", action="store_true",
                        help="ignore the deadline window")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and save the report but do not send or record")
    parser.add_argument("--out", default=None, help="also write the HTML here")
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS,
                        help="send on the first run within this many hours of the deadline")
    args = parser.parse_args()

    load_env()
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    bootstrap = data_fetcher.get_bootstrap_static()
    gameweek = data_fetcher.get_current_gameweek(bootstrap)
    hours, moment = hours_to_deadline(bootstrap, gameweek)

    if hours is None:
        print(f"[{stamp}] no deadline listed for GW{gameweek}; nothing to do")
        return 0

    if not args.force and not (MIN_HOURS <= hours <= args.max_hours):
        detail = "already passed" if hours < MIN_HOURS else "still too far out"
        print(f"[{stamp}] GW{gameweek} deadline is {hours:.1f}h away "
              f"({detail}, ceiling {args.max_hours:.0f}h); nothing to do")
        return 0

    already = {e["gameweek"] for e in tracker.entries()
               if e.get("season") == recommender.identity.current_season_label(bootstrap)}
    if gameweek in already and not args.force:
        print(f"[{stamp}] GW{gameweek} already recorded; nothing to do")
        return 0

    print(f"[{stamp}] GW{gameweek} deadline {moment:%a %d %b %H:%M} UTC "
          f"({hours:.1f}h) - building report")

    recommendation, history, images = build(
        args.team_id, args.free_transfers, args.chips_used
    )

    to_address = os.environ.get("FPL_EMAIL_TO")
    user = os.environ.get("FPL_SMTP_USER")
    password = os.environ.get("FPL_SMTP_PASSWORD")
    host = os.environ.get("FPL_SMTP_HOST", "smtp-mail.outlook.com")
    port = int(os.environ.get("FPL_SMTP_PORT", "587"))

    message = compose(recommendation, history, images,
                      to_address or "unset@localhost", user or "unset@localhost")

    if args.out:
        # Written for a browser rather than a mail client, so the charts are
        # inlined as data URIs. Email gets the cid: form instead, which is the
        # only one most clients render without the reader loading images.
        html = report.render_html(recommendation, history, images)
        for name, data in images.items():
            encoded = base64.b64encode(data).decode("ascii")
            html = html.replace(f"cid:{name}", f"data:image/png;base64,{encoded}")
        with open(args.out, "w") as fh:
            fh.write("<!doctype html><meta charset=\"utf-8\">"
                     f"<title>FPL GW{recommendation['gameweek']}</title>"
                     f"<body style=\"margin:0;background:#f9f9f7\">{html}</body>")
        print(f"  preview written to {args.out}")

    if args.dry_run:
        print("  dry run - not sent, not recorded")
        return 0

    if not (to_address and user and password):
        print("  SMTP credentials not set (FPL_SMTP_USER / FPL_SMTP_PASSWORD / "
              "FPL_EMAIL_TO) - report built but not sent")
        # Still record: the ledger is the experiment, and it must not depend on
        # whether the email happened to go out.
        tracker.record(recommendation)
        print(f"  recorded GW{recommendation['gameweek']} in the ledger")
        return 1

    try:
        send(message, host, port, user, password)
        print(f"  sent to {to_address}")
    except Exception as error:                      # noqa: BLE001 - reported, not raised
        print(f"  SEND FAILED: {type(error).__name__}: {error}")
        tracker.record(recommendation)
        print(f"  recorded GW{recommendation['gameweek']} anyway")
        return 1

    tracker.record(recommendation)
    print(f"  recorded GW{recommendation['gameweek']} in the ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
