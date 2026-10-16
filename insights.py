"""
The weekly email's context beyond the recommendation: who the model rates
across the whole game and why, the strongest options in each position, and how
last gameweek's recommended team actually scored.

None of this is recorded. The ledger holds what the model decided; this is for
reading that decision, rebuilt from the same run each week, so it can grow or
change without touching the experiment record.
"""

import attribution
import report
import tracker

TOP_N = 20
PER_POSITION = 5
FIXTURE_WEEKS = 5


def explain_players(context):
    """
    {code: {factor: points}} for every player whose prediction it can explain.

    Only predictions the model made unaltered are explained. A held player the
    availability gate zeroed has a model prediction that is not the one shown,
    and splitting a number the reader never sees would be misleading.
    """
    rows = context["feature_rows"].dropna(subset=context["feature_cols"])
    result = attribution.explain(context["model"], rows, context["feature_cols"])
    if result is None:
        return None, {}

    base, frame = result
    shown = dict(zip(context["players"]["code"], context["players"]["predicted_points"]))
    explained = {}
    for code, contributions in zip(rows["code"].astype(int), frame.to_dict("records")):
        total = base + sum(contributions.values())
        if code in shown and abs(total - shown[code]) < 1e-6:
            explained[code] = {k: round(float(v), 2) for k, v in contributions.items()}
    return base, explained


def _player(row, shorts, owned, buying, factors):
    element_id = int(row["element_id"])
    return {
        "element_id": element_id,
        "name": str(row["web_name"]),
        "team": int(row["team"]),
        "team_short": shorts.get(int(row["team"]), "?"),
        "position": int(row["position"]),
        "price": float(row["value"]) / 10.0,
        "owned_pct": float(row.get("selected_by_percent") or 0.0),
        "predicted": round(float(row["predicted_points"]), 2),
        "status": "owned" if element_id in owned else "buy" if element_id in buying else None,
        "factors": factors.get(int(row["code"])),
    }


def availability_notes(context, element_ids):
    """
    {element_id: {"status", "news", "if_fit"}} for players shown at zero
    because the availability gate set them there.

    The gate keeps an unavailable player the manager already owns, at zero, so
    the optimizer can decide whether to sell. Shown bare, that zero reads as the
    model's verdict on the player. FPL's news line says why it is zero, and the
    model's ungated number says what is being given up while they are out.
    """
    players = context["players"].set_index("element_id")
    rows = context["feature_rows"]
    ungated = dict(zip(rows["code"].astype(int), rows["predicted_points"]))
    notes = {}
    for element_id in element_ids:
        if element_id not in players.index or bool(players.at[element_id, "is_available"]):
            continue
        if_fit = ungated.get(int(players.at[element_id, "code"]))
        notes[element_id] = {
            "status": str(players.at[element_id, "status_label"] or ""),
            "news": str(players.at[element_id, "news"] or "").strip(),
            "if_fit": round(float(if_fit), 2) if if_fit else None,
        }
    return notes


def last_review(history_rows, season):
    """
    The most recent scored gameweek: what the recommended team was predicted,
    what each player actually scored, and the two totals.
    """
    scored = [r for r in history_rows if r.get("scored") and "model_points" in r]
    if not scored:
        return None
    row = scored[-1]
    entry = next((e for e in tracker.entries()
                  if e["gameweek"] == row["gameweek"] and e.get("season") == season), None)
    results = tracker.actual_points(row["gameweek"])
    if entry is None or not results:
        return None

    players = []
    for player in entry.get("squad") or []:
        points, minutes = results.get(player["element_id"], (0.0, 0.0))
        players.append({
            "name": player["name"],
            "position": player["position"],
            "predicted": player["predicted"],
            "actual": points,
            "minutes": minutes,
            "is_starter": bool(player.get("is_starter")),
            "is_captain": bool(player.get("is_captain")),
        })
    return {
        "gameweek": row["gameweek"],
        "predicted_total": entry.get("predicted_xi_points"),
        "model_points": row["model_points"],
        "model_hit": row.get("model_hit", 0),
        "human_points": row.get("human_points"),
        "players": players,
    }


def build(context, recommendation, history_rows=()):
    """
    Plain data for the report's extra sections.

    {
      "base": average player-gameweek the factors start from (None if the
              model cannot be explained),
      "top": the TOP_N highest predictions among available players,
      "by_position": {position: the PER_POSITION highest in it},
      "squad_factors": {element_id: {factor: points} or None} for the
                       recommended squad and the players it transfers out,
      "availability": availability_notes(...) for the same players,
      "fixture_weeks": [gameweek, ...],
      "fixtures": {team: {gameweek: [(opponent, difficulty, is_home), ...]}},
      "review": last_review(...) or None,
    }
    """
    players = context["players"]
    bootstrap = context["bootstrap"]
    gameweek = context["gameweek"]
    shorts = {int(t["id"]): t["short_name"] for t in bootstrap["teams"]}

    base, factors = explain_players(context)

    owned = set(recommendation.get("squad_before") or [])
    options = recommendation.get("options") or []
    moves = options[0]["moves"] if options else []
    buying = {move["in"]["element_id"] for move in moves}

    available = players[players["is_available"]].sort_values(
        "predicted_points", ascending=False
    )
    top = [_player(row, shorts, owned, buying, factors)
           for _, row in available.head(TOP_N).iterrows()]

    by_position = {}
    for position in sorted(available["position"].unique()):
        best = available[available["position"] == position].head(PER_POSITION)
        by_position[int(position)] = [_player(row, shorts, owned, buying, factors)
                                      for _, row in best.iterrows()]

    # The players being sold sit beside the squad in the email, so the swap can
    # be read factor by factor.
    shown = ([p["element_id"] for p in recommendation.get("squad") or []]
             + [move["out"]["element_id"] for move in moves])
    codes = dict(zip(players["element_id"], players["code"]))
    squad_factors = {i: factors.get(int(codes.get(i, -1))) for i in shown}

    last_event = max(int(e["id"]) for e in bootstrap["events"])
    weeks = max(0, min(FIXTURE_WEEKS, last_event - gameweek + 1))
    teams = {p["team"] for group in by_position.values() for p in group}
    fixtures = report.fixture_outlook(context["fixtures"], bootstrap, teams, gameweek, weeks)

    return {
        "base": None if base is None else round(float(base), 2),
        "top": top,
        "by_position": by_position,
        "squad_factors": squad_factors,
        "availability": availability_notes(context, shown),
        "fixture_weeks": list(range(gameweek, gameweek + weeks)),
        "fixtures": fixtures,
        "review": last_review(history_rows, recommendation.get("season")),
    }

