import pandas as pd
from data_fetcher import get_bootstrap_static, get_user_team, get_current_gameweek, get_fixtures
from ml_model import prepare_data, train_model, get_latest_features
from optimizer import optimize_squad

def main():
    print("Fetching static data from FPL...")
    bootstrap = get_bootstrap_static()
    elements = pd.DataFrame(bootstrap['elements'])
    
    # Map API columns to optimizer expected columns
    elements['element_id'] = elements['id']
    elements['position'] = elements['element_type']
    elements['value'] = elements['now_cost']
    
    current_gw = get_current_gameweek(bootstrap)
    print(f"Current Gameweek is basically {current_gw}")
    
    # We want to predict for the current GW if it's upcoming, or next if it's started
    # The API 'is_next' gives the one to predict.
    
    team_id = 8936155
    print(f"Fetching user team {team_id}...")
    
    # Fetch team from previous/current GW to get the squad
    # Usually if GW has not started, entry API might not allow fetching upcoming picks.
    # So we fetch for current_gw - 1 or current_gw. We'll try current_gw, then fall back.
    user_team_data = get_user_team(team_id, current_gw)
    if not user_team_data and current_gw > 1:
         user_team_data = get_user_team(team_id, current_gw - 1)
         
    current_squad_ids = []
    if user_team_data:
        current_squad_ids = [pick['element'] for pick in user_team_data['picks']]
        bank = user_team_data['entry_history']['bank']
        squad_value = sum(elements[elements['element_id'].isin(current_squad_ids)]['value'])
        budget = squad_value + bank
        print(f"Found current squad. Total Budget available: {budget/10:.1f}m")
    else:
        print("Could not fetch user team Picks. Defaulting to 100.0m Wildcard.")
        budget = 1000
    
    print("\nTraining ML Model on historical data... (this may take a few seconds)")
    file_paths = [
        "data/historical/data/2024-25/gws/merged_gw.csv",
        "data/historical/data/2025-26/gws/merged_gw.csv"
    ]
    model_data, feature_cols, full_data = prepare_data(file_paths)
    model = train_model(model_data, feature_cols)
    
    # Build fixture difficulty map for the upcoming gameweek from live API data
    print("\nFetching live fixture data for difficulty ratings...")
    team_next_difficulty_map = {}
    try:
        fixtures = get_fixtures()
        next_gw = current_gw  # current_gw is the upcoming GW to predict for
        next_gw_fixtures = [f for f in fixtures if f.get('event') == next_gw]
        for fix in next_gw_fixtures:
            team_next_difficulty_map[fix['team_h']] = fix['team_h_difficulty']
            team_next_difficulty_map[fix['team_a']] = fix['team_a_difficulty']
        print(f"  Mapped difficulty for {len(team_next_difficulty_map)} teams in GW{next_gw}")
    except Exception as e:
        print(f"  Warning: Could not fetch live fixtures: {e}. Using default difficulty.")
    
    print("\nGenerating predictions for next Gameweek...")
    latest_features = get_latest_features(full_data, feature_cols, team_next_difficulty_map=team_next_difficulty_map)
    latest_features['predicted_points'] = model.predict(latest_features[feature_cols])
    
    if 'element' in latest_features.columns:
        latest_features['element_id'] = latest_features['element']
    
    feature_cols_to_merge = ['element_id', 'predicted_points', 'total_points_rolling_3', 'minutes_rolling_3']
    for col in feature_cols_to_merge:
        if col not in latest_features.columns and col != 'element_id':
            latest_features[col] = 0.0
            
    merged_data = pd.merge(elements, latest_features[feature_cols_to_merge], on='element_id', how='left')
    merged_data['predicted_points'] = merged_data['predicted_points'].fillna(0)
    merged_data['total_points_rolling_3'] = merged_data['total_points_rolling_3'].fillna(0)
    merged_data['minutes_rolling_3'] = merged_data['minutes_rolling_3'].fillna(0)
    
    print("\n--- Running Optimization ---")
    free_transfers = 1  # Default to 1 free transfer; could be fetched from API in future
    if current_squad_ids:
        n_options = 3
        hard_max = 4  # Allow up to 4 transfers if the point gain justifies the hit
        print(f"Finding top {n_options} optimal transfer plans (up to {hard_max} transfers, {free_transfers} free)...")
        optimal_squads = optimize_squad(
            merged_data,
            free_transfers=free_transfers,
            current_squad_ids=current_squad_ids,
            budget=budget,
            n=n_options,
            hard_max_transfers=hard_max
        )
        
        if optimal_squads:
            print(f"\n🏆 Top {len(optimal_squads)} Suggested Transfer Options:")
            for idx, optimal_squad in enumerate(optimal_squads, 1):
                new_squad_ids = set(optimal_squad['element_id'])
                old_squad_ids = set(current_squad_ids)
                
                transfers_out = old_squad_ids - new_squad_ids
                transfers_in = new_squad_ids - old_squad_ids
                num_transfers = len(transfers_out)
                extra_hits = max(0, num_transfers - free_transfers)
                point_cost = extra_hits * 4
                
                if not transfers_out and not transfers_in:
                    print(f"\nOption {idx}: Your current squad is mathematically optimal! No transfers recommended.")
                    continue
                    
                print(f"\n--- Option {idx} ({num_transfers} transfer{'s' if num_transfers > 1 else ''}) ---")
                if point_cost > 0:
                    print(f"  ⚠️  POINT HIT: -{point_cost} points ({extra_hits} extra transfer{'s' if extra_hits > 1 else ''} beyond free)")
                else:
                    print(f"  ✅ FREE transfer - no point hit")
                    
                for tid_out, tid_in in zip(transfers_out, transfers_in):
                    p_out = merged_data[merged_data['element_id'] == tid_out].iloc[0]
                    p_in = merged_data[merged_data['element_id'] == tid_in].iloc[0]
                    
                    print(f"  🔻 OUT: {p_out['web_name']} (£{p_out['value']/10:.1f}m)")
                    print(f"  🔺 IN:  {p_in['web_name']} (£{p_in['value']/10:.1f}m)")
                    
                    point_diff = p_in['predicted_points'] - p_out['predicted_points']
                    print(f"  💡 WHY? -> Gains an estimated {point_diff:+.2f} points next gameweek.")
                    print(f"     📊 Form Comparison (Avg per game over last 3 GWs):")
                    print(f"        {p_out['web_name']}: {p_out['total_points_rolling_3']:.1f} pts (Played {p_out['minutes_rolling_3']:.0f} mins/gw)")
                    print(f"        {p_in['web_name']}: {p_in['total_points_rolling_3']:.1f} pts (Played {p_in['minutes_rolling_3']:.0f} mins/gw)")
                
                if point_cost > 0:
                    total_gain = sum(
                        merged_data[merged_data['element_id'] == tid_in].iloc[0]['predicted_points'] -
                        merged_data[merged_data['element_id'] == tid_out].iloc[0]['predicted_points']
                        for tid_out, tid_in in zip(transfers_out, transfers_in)
                    )
                    net_gain = total_gain - point_cost
                    print(f"  📈 NET GAIN after -{point_cost}pt hit: {net_gain:+.2f} points")
    else:
        print("Running Wildcard optimization...")
        optimal_squad = optimize_squad(merged_data, free_transfers=15, current_squad_ids=None, budget=budget)
        if optimal_squad is not None:
            print("\n⭐️ Optimal Wildcard Squad:")
            for _, p in optimal_squad.sort_values(by='position').iterrows():
                print(f"[{p['position']}] {p['web_name']} - £{p['value']/10:.1f}m | Pred: {p['predicted_points']:.2f}")

if __name__ == "__main__":
    main()
