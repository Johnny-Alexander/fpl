"""
Squad selection as a mixed-integer program.

Maximises expected points for the upcoming gameweek subject to the FPL rules:
15 players in a 2/5/5/3 shape, a legal starting XI, at most 3 per club, the budget,
and a 4-point charge per transfer beyond the free allowance.
"""

import pulp

POSITION_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
FORMATION_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
SQUAD_SIZE = 15
STARTERS = 11
MAX_PER_TEAM = 3
TRANSFER_COST = 4
MAX_FREE_TRANSFERS = 5  # FPL banks up to five

# Weight on bench players in the objective. A benched player only scores through
# an autosub, so they are worth a small fraction of their expected points -- enough
# to break ties toward a stronger bench without distorting the starting XI.
BENCH_WEIGHT = 0.1


def optimize_squad(
    player_data,
    free_transfers=1,
    current_squad_ids=None,
    budget=1000,
    n=1,
    hard_max_transfers=SQUAD_SIZE,
    chip_active=None,
):
    """
    Return up to `n` squads, best first, as a list of DataFrames.

    Each carries `is_starter` and `is_captain` flags. An empty list means no
    feasible squad exists under the constraints.

    chip_active: None, 'WC' (wildcard), 'FH' (free hit), 'BB' (bench boost)
    or 'TC' (triple captain).
    """
    players = player_data[player_data["position"].isin(POSITION_QUOTA)].copy()
    players = players.drop_duplicates(subset=["element_id"]).reset_index(drop=True)
    if players.empty:
        return []

    ids = list(players["element_id"])
    # Dict lookups keep model construction linear; scanning the frame per player
    # made this quadratic, which the backtest pays for once per gameweek.
    points = dict(zip(players["element_id"], players["predicted_points"]))
    values = dict(zip(players["element_id"], players["value"]))
    positions = dict(zip(players["element_id"], players["position"]))
    teams = dict(zip(players["element_id"], players["team"]))

    by_position = {p: [i for i in ids if positions[i] == p] for p in POSITION_QUOTA}
    by_team = {}
    for i in ids:
        by_team.setdefault(teams[i], []).append(i)

    prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")
    extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat="Integer")

    # ── Objective ──
    if chip_active == "BB":
        # Bench boost: every one of the 15 scores.
        objective = pulp.lpSum(points[i] * squad[i] for i in ids)
    else:
        captain_multiplier = 2 if chip_active == "TC" else 1
        objective = (
            pulp.lpSum(points[i] * start[i] for i in ids)
            + pulp.lpSum(points[i] * captain_multiplier * captain[i] for i in ids)
            + pulp.lpSum(points[i] * BENCH_WEIGHT * (squad[i] - start[i]) for i in ids)
        )

    # ── Squad structure ──
    prob += pulp.lpSum(values[i] * squad[i] for i in ids) <= budget
    prob += pulp.lpSum(squad[i] for i in ids) == SQUAD_SIZE
    for position, quota in POSITION_QUOTA.items():
        prob += pulp.lpSum(squad[i] for i in by_position[position]) == quota

    # ── Starting XI ──
    prob += pulp.lpSum(start[i] for i in ids) == STARTERS
    prob += pulp.lpSum(captain[i] for i in ids) == 1
    for position, minimum in FORMATION_MIN.items():
        prob += pulp.lpSum(start[i] for i in by_position[position]) >= minimum
    prob += pulp.lpSum(start[i] for i in by_position[1]) == 1  # exactly one keeper

    for i in ids:
        prob += start[i] <= squad[i]
        prob += captain[i] <= start[i]

    for team_ids in by_team.values():
        prob += pulp.lpSum(squad[i] for i in team_ids) <= MAX_PER_TEAM

    # ── Transfers ──
    held = set(current_squad_ids or [])
    free_hit_or_wildcard = chip_active in ("WC", "FH")
    if held and not free_hit_or_wildcard:
        kept = pulp.lpSum(squad[i] for i in held if i in squad)
        transfers_made = SQUAD_SIZE - kept
        prob += extra_transfers >= transfers_made - free_transfers
        prob += transfers_made <= hard_max_transfers
        objective = objective - TRANSFER_COST * extra_transfers
    else:
        prob += extra_transfers == 0

    prob += objective

    # ── Solve, then re-solve for alternatives ──
    squads = []
    for _ in range(n):
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if prob.status != pulp.LpStatusOptimal:
            break

        chosen = [i for i in ids if squad[i].value() and squad[i].value() > 0.5]
        picked = players[players["element_id"].isin(chosen)].copy()
        picked["is_starter"] = picked["element_id"].map(
            lambda i: chip_active == "BB" or bool(start[i].value() and start[i].value() > 0.5)
        )
        picked["is_captain"] = picked["element_id"].map(
            lambda i: bool(captain[i].value() and captain[i].value() > 0.5)
        )
        squads.append(picked)

        # Exclude this solution from the next solve. When we are proposing
        # transfers, forbidding the *transfer set* rather than the exact 15 keeps
        # the alternatives genuinely distinct -- otherwise each further option
        # differs by a single bench player and reads as the same plan.
        incoming = [i for i in chosen if i not in held]
        if held and incoming:
            prob += pulp.lpSum(squad[i] for i in incoming) <= len(incoming) - 1
        else:
            prob += pulp.lpSum(squad[i] for i in chosen) <= SQUAD_SIZE - 1

    return squads


DEFAULT_HORIZON = 5
DEFAULT_DECAY = 0.85  # weight on gameweek t is DECAY**t
DEFAULT_POOL = 220    # players considered, on top of whatever is already held

# A free transfer is not free: keeping it banked buys flexibility to react to
# next week's injury news. Without a small charge the solver is indifferent
# between holding and making a zero-gain swap, and churns the squad for nothing.
# Set far below any real difference in predicted points, so it only breaks ties.
IDLE_TRANSFER_PENALTY = 0.05


def _prune_pool(players, points_by_gw, held, pool_size):
    """
    Keep the strongest candidates plus everything currently held.

    A horizon of five gameweeks over six hundred players is tens of thousands of
    binaries, which CBC will not solve in reasonable time. Ranking by a player's
    best gameweek in the horizon keeps anyone with a standout fixture, which is
    precisely who a multi-week plan is looking for -- ranking by the mean would
    discard a player with one huge week.
    """
    best = {}
    for per_player in points_by_gw.values():
        for code, value in per_player.items():
            if value > best.get(code, float("-inf")):
                best[code] = value

    ranked = sorted(best, key=lambda c: -best[c])[:pool_size]
    keep = set(ranked) | set(held)
    return players[players["element_id"].isin(keep)].copy()


def optimize_horizon(
    player_data,
    points_by_gw,
    current_squad_ids,
    budget,
    free_transfers=1,
    horizon=DEFAULT_HORIZON,
    decay=DEFAULT_DECAY,
    max_transfers_per_gw=3,
    pool_size=DEFAULT_POOL,
    time_limit=60,
):
    """
    Plan transfers over several gameweeks at once.

    The single-gameweek objective is myopic: it cannot buy into a fixture run,
    bank a transfer toward a double gameweek, or accept a weaker week now for a
    stronger one later. Here squad membership is indexed by gameweek and linked by
    a transfer chain, so the solver commits to a plan rather than a move.

    Later gameweeks are discounted by `decay` per week, both because predictions
    degrade with distance and because a plan will be re-solved next week anyway --
    only the first move is actually played.

    MEASURED, AND NOT WORTH USING YET. Replicated over four seasons with chips
    off, against the single-gameweek objective:

        horizon 3:  -41, -45, -110, +77   mean -30, better in 1 of 4 seasons
        horizon 5:  +17, -65,  +56, +83   mean +23, better in 3 of 4 (p~0.53)

    The mechanism is sound -- the unit tests show it banking transfers and timing
    a purchase to the week a fixture run starts -- but there is nothing for it to
    plan around. Across a five-week horizon a player's predicted points vary by
    0.096 on average, against 1.325 of spread between players: a ratio of 0.07.
    The mean rank change from week one to week five is 43 places out of 780.

    The cause is upstream. Form features are held constant across the horizon by
    construction, so the only inputs that vary are next_difficulty,
    next_fixture_count and next_was_home, which together carry under a tenth of
    the model's feature importance. Five near-identical rankings make the planner
    an expensive way to solve the same week five times.

    Making this pay needs a more fixture-sensitive model -- opponent defensive
    strength rather than FPL's coarse 1-5 rating, per-90 rates, home/away splits
    -- not a better optimiser. Revisit once predictions actually distinguish one
    gameweek from another.

    points_by_gw: {gameweek: {element_id: predicted_points}}, in play order.
    Returns (first_gameweek_squad, plan) where plan lists the transfers per week.
    """
    gameweeks = sorted(points_by_gw)[:horizon]
    if not gameweeks:
        return [], []

    players = player_data[player_data["position"].isin(POSITION_QUOTA)].copy()
    players = players.drop_duplicates(subset=["element_id"]).reset_index(drop=True)
    held = set(current_squad_ids or [])
    players = _prune_pool(players, points_by_gw, held, pool_size)
    if players.empty:
        return [], []

    ids = list(players["element_id"])
    values = dict(zip(players["element_id"], players["value"]))
    positions = dict(zip(players["element_id"], players["position"]))
    teams = dict(zip(players["element_id"], players["team"]))

    by_position = {p: [i for i in ids if positions[i] == p] for p in POSITION_QUOTA}
    by_team = {}
    for i in ids:
        by_team.setdefault(teams[i], []).append(i)

    def points(i, gw):
        return points_by_gw.get(gw, {}).get(i, 0.0)

    prob = pulp.LpProblem("FPL_Horizon", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", (ids, gameweeks), cat="Binary")
    start = pulp.LpVariable.dicts("start", (ids, gameweeks), cat="Binary")
    captain = pulp.LpVariable.dicts("captain", (ids, gameweeks), cat="Binary")

    # Transfers out are driven up by the chain constraint and down by the hit
    # penalty, so they settle at max(0, was_in - is_in) without needing to be
    # binary -- which keeps the problem tractable.
    sold = pulp.LpVariable.dicts("sold", (ids, gameweeks), lowBound=0, upBound=1)
    hits = pulp.LpVariable.dicts("hits", gameweeks, lowBound=0)
    # Free transfers carried into the *next* gameweek, and the unused count it is
    # derived from. `spare` is max(0, available - made), which needs the indicator
    # below: writing banked <= available - made + 1 directly makes the problem
    # infeasible whenever a hit is taken, silently forbidding a legal move.
    banked = pulp.LpVariable.dicts("banked", gameweeks, lowBound=0,
                                   upBound=MAX_FREE_TRANSFERS)
    spare = pulp.LpVariable.dicts("spare", gameweeks, lowBound=0,
                                  upBound=MAX_FREE_TRANSFERS)
    overspent = pulp.LpVariable.dicts("overspent", gameweeks, cat="Binary")

    objective = []
    for index, gw in enumerate(gameweeks):
        weight = decay ** index

        objective.append(
            weight * pulp.lpSum(points(i, gw) * start[i][gw] for i in ids)
            + weight * pulp.lpSum(points(i, gw) * captain[i][gw] for i in ids)
            + weight * BENCH_WEIGHT * pulp.lpSum(
                points(i, gw) * (squad[i][gw] - start[i][gw]) for i in ids
            )
            - weight * TRANSFER_COST * hits[gw]
            - IDLE_TRANSFER_PENALTY * pulp.lpSum(sold[i][gw] for i in ids)
        )

        prob += pulp.lpSum(squad[i][gw] for i in ids) == SQUAD_SIZE
        for position, quota in POSITION_QUOTA.items():
            prob += pulp.lpSum(squad[i][gw] for i in by_position[position]) == quota

        prob += pulp.lpSum(start[i][gw] for i in ids) == STARTERS
        prob += pulp.lpSum(captain[i][gw] for i in ids) == 1
        prob += pulp.lpSum(start[i][gw] for i in by_position[1]) == 1
        for position, minimum in FORMATION_MIN.items():
            prob += pulp.lpSum(start[i][gw] for i in by_position[position]) >= minimum

        for i in ids:
            prob += start[i][gw] <= squad[i][gw]
            prob += captain[i][gw] <= start[i][gw]

        for team_ids in by_team.values():
            prob += pulp.lpSum(squad[i][gw] for i in team_ids) <= MAX_PER_TEAM

        # Prices are held constant across the horizon: future price changes are
        # not forecastable, and pretending otherwise would let the plan spend
        # money it has not earned.
        prob += pulp.lpSum(values[i] * squad[i][gw] for i in ids) <= budget

    # ── Transfer chain ──
    for index, gw in enumerate(gameweeks):
        previous = gameweeks[index - 1] if index else None
        for i in ids:
            was_in = squad[i][previous] if previous else (1 if i in held else 0)
            prob += sold[i][gw] >= was_in - squad[i][gw]

        made = pulp.lpSum(sold[i][gw] for i in ids)
        prob += made <= max_transfers_per_gw

        available = banked[previous] if previous else free_transfers
        prob += hits[gw] >= made - available

        # spare = max(0, available - made), exactly.
        big_m = MAX_FREE_TRANSFERS + max_transfers_per_gw
        prob += spare[gw] >= available - made
        prob += spare[gw] <= available - made + big_m * overspent[gw]
        prob += spare[gw] <= big_m * (1 - overspent[gw])

        # One free transfer is granted each week, and the bank is capped.
        prob += banked[gw] <= spare[gw] + 1
        prob += banked[gw] <= MAX_FREE_TRANSFERS

    prob += pulp.lpSum(objective)
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))

    if prob.status not in (pulp.LpStatusOptimal,):
        return [], []

    def chosen(gw):
        return [i for i in ids if squad[i][gw].value() and squad[i][gw].value() > 0.5]

    first = gameweeks[0]
    picked = players[players["element_id"].isin(chosen(first))].copy()
    picked["is_starter"] = picked["element_id"].map(
        lambda i: bool(start[i][first].value() and start[i][first].value() > 0.5)
    )
    picked["is_captain"] = picked["element_id"].map(
        lambda i: bool(captain[i][first].value() and captain[i][first].value() > 0.5)
    )

    plan, previous_squad = [], held
    for gw in gameweeks:
        current = set(chosen(gw))
        plan.append(
            {
                "gameweek": gw,
                "out": sorted(previous_squad - current),
                "in": sorted(current - previous_squad),
                "hits": round(hits[gw].value() or 0.0),
            }
        )
        previous_squad = current

    return [picked], plan


if __name__ == "__main__":
    import pandas as pd

    data = pd.DataFrame(
        {
            "element_id": range(1, 31),
            "predicted_points": [5] * 30,
            "value": [50] * 30,
            "position": [1] * 4 + [2] * 10 + [3] * 10 + [4] * 6,
            "team": [1, 2, 3, 4, 5, 6] * 5,
        }
    )
    data.loc[0, "predicted_points"] = 10
    data.loc[1, "predicted_points"] = 8
    data.loc[4, "predicted_points"] = 2

    result = optimize_squad(data, budget=1000)
    assert result, "expected a feasible squad"
    picked = result[0]
    assert len(picked) == SQUAD_SIZE, len(picked)
    assert picked["is_starter"].sum() == STARTERS
    assert picked["is_captain"].sum() == 1
    print(f"squad {len(picked)}, starters {int(picked['is_starter'].sum())}, "
          f"captain {picked[picked['is_captain']]['element_id'].tolist()}")
    print("optimizer self-test passed")
