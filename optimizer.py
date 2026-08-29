import pulp
import pandas as pd

def optimize_squad(player_data, free_transfers=1, current_squad_ids=None, budget=1000, n=1, hard_max_transfers=15, chip_active=None):
    """
    player_data: DataFrame containing 'element_id', 'predicted_points', 'value', 'position', 'team'
    free_transfers: Number of free transfers available.
    current_squad_ids: List of element_ids currently in the squad (for transfer penalty calculation and constraints).
    budget: Maximum total value of the squad.
    n: Number of top alternative squads to return.
    hard_max_transfers: The absolute maximum number of transfers to consider (useful to limit if not on wildcard).
    chip_active: String ('WC', 'FH', 'BB', 'TC') or None.
    """
    prob = pulp.LpProblem("FPL_Optimization", pulp.LpMaximize)
    players = player_data[player_data['position'].isin([1, 2, 3, 4])].copy()
    players = players.drop_duplicates(subset=['element_id']).reset_index(drop=True)
    player_vars = pulp.LpVariable.dicts("player", players['element_id'], cat='Binary')
    starter_vars = pulp.LpVariable.dicts("starter", players['element_id'], cat='Binary')
    captain_vars = pulp.LpVariable.dicts("captain", players['element_id'], cat='Binary')
    
    extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0, cat='Integer')
    
    # 1. Base Objective formulation
    expected_points = 0
    if chip_active == 'BB':
        # Bench Boost: all 15 players score fully, no captain bonus
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * player_vars[i] for i in players['element_id']])
    elif chip_active == 'TC':
        # Triple Captain: starters + 2 * captain + 0.1 * bench
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * starter_vars[i] for i in players['element_id']])
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * 2 * captain_vars[i] for i in players['element_id']])
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * 0.1 * (player_vars[i] - starter_vars[i]) for i in players['element_id']])
    else:
        # Standard: starters + 1 * captain + 0.1 * bench
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * starter_vars[i] for i in players['element_id']])
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * captain_vars[i] for i in players['element_id']])
        expected_points += pulp.lpSum([players.loc[players['element_id'] == i, 'predicted_points'].values[0] * 0.1 * (player_vars[i] - starter_vars[i]) for i in players['element_id']])
    
    # Budget tracking
    prob += pulp.lpSum([players.loc[players['element_id'] == i, 'value'].values[0] * player_vars[i] for i in players['element_id']]) <= budget
    
    # 15 Player Constraints
    prob += pulp.lpSum([player_vars[i] for i in players['element_id']]) == 15
    prob += pulp.lpSum([player_vars[i] for i in players[players['position'] == 1]['element_id']]) == 2
    prob += pulp.lpSum([player_vars[i] for i in players[players['position'] == 2]['element_id']]) == 5
    prob += pulp.lpSum([player_vars[i] for i in players[players['position'] == 3]['element_id']]) == 5
    prob += pulp.lpSum([player_vars[i] for i in players[players['position'] == 4]['element_id']]) == 3
    
    # 11 Starter / Captain Constraints
    if chip_active != 'BB':
        prob += pulp.lpSum([starter_vars[i] for i in players['element_id']]) == 11
        prob += pulp.lpSum([captain_vars[i] for i in players['element_id']]) == 1
        
        # Valid Formations (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)
        prob += pulp.lpSum([starter_vars[i] for i in players[players['position'] == 1]['element_id']]) == 1
        prob += pulp.lpSum([starter_vars[i] for i in players[players['position'] == 2]['element_id']]) >= 3
        prob += pulp.lpSum([starter_vars[i] for i in players[players['position'] == 3]['element_id']]) >= 2
        prob += pulp.lpSum([starter_vars[i] for i in players[players['position'] == 4]['element_id']]) >= 1
    
    # Link variables
    for i in players['element_id']:
        prob += starter_vars[i] <= player_vars[i]
        prob += captain_vars[i] <= starter_vars[i]
    
    teams = players['team'].unique()
    for team in teams:
        prob += pulp.lpSum([player_vars[i] for i in players[players['team'] == team]['element_id']]) <= 3
        
    if current_squad_ids is not None and chip_active not in ['WC', 'FH']:
        kept_players = [player_vars[i] for i in current_squad_ids if i in player_vars]
        transfers_made = 15 - pulp.lpSum(kept_players)
        prob += extra_transfers >= transfers_made - free_transfers
        prob += transfers_made <= hard_max_transfers
        prob += expected_points - 4 * extra_transfers

    else:
        prob += expected_points
        prob += extra_transfers == 0
        
    squads = []
    for _ in range(n):
        # Suppress solver output message for cleaner CLI
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        if prob.status != pulp.LpStatusOptimal:
            break
            
        selected_ids = [i for i in player_vars if player_vars[i].value() == 1.0]
        selected_players = players[players['element_id'].isin(selected_ids)].copy()
        
        # Populate starter and captain flags
        selected_players['is_starter'] = selected_players['element_id'].map(
            lambda i: True if (chip_active == 'BB' or starter_vars[i].value() == 1.0) else False
        )
        selected_players['is_captain'] = selected_players['element_id'].map(
            lambda i: True if (chip_active != 'BB' and captain_vars[i].value() == 1.0) else False
        )
        squads.append(selected_players)
        
        # Add constraint to forbid this exact combination of 15 players
        prob += pulp.lpSum([player_vars[i] for i in selected_ids]) <= 14
        
    if not squads:
        print("Warning: Optimization could not find an optimal solution.")
        return None
        
    return squads if n > 1 else squads[0]

if __name__ == "__main__":
    # Test with dummy data
    data = pd.DataFrame({
        'element_id': range(1, 31),
        'predicted_points': [5]*30,
        'value': [50]*30,
        'position': [1]*4 + [2]*10 + [3]*10 + [4]*6,
        'team': [1, 2, 3, 4, 5, 6]*5
    })
    data.loc[0, 'predicted_points'] = 10 # make player 1 better (likely captain)
    data.loc[1, 'predicted_points'] = 8  # decent starter
    data.loc[4, 'predicted_points'] = 2  # bench fodder
    
    optim = optimize_squad(data, budget=1000)
    if optim is not None:
        print("Optimization test complete. Squad size:", len(optim))
        print(f"Captain: {optim[optim['is_captain']]['element_id'].values}")
        print(f"Starters: {len(optim[optim['is_starter']])}")
    else:
        print("Optimization failed.")
