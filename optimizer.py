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
