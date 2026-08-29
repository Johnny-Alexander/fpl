import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

def prepare_data(file_paths):
    dfs = []
    for i, fp in enumerate(file_paths):
        try:
            df_gw = pd.read_csv(fp)
            df_gw['season_index'] = i
            
            # Load corresponding fixtures.csv
            fixture_fp = fp.replace('gws/merged_gw.csv', 'fixtures.csv')
            try:
                df_fixtures = pd.read_csv(fixture_fp)[['id', 'team_h_difficulty', 'team_a_difficulty']]
                df_gw = pd.merge(df_gw, df_fixtures, left_on='fixture', right_on='id', how='left')
            except Exception as e:
                print(f"Warning: Could not load fixtures for {fp}: {e}")
                df_gw['team_h_difficulty'] = 3
                df_gw['team_a_difficulty'] = 3
            
            dfs.append(df_gw)
        except Exception as e:
            print(f"Warning: Could not read {fp}: {e}")
    
    if not dfs:
        raise ValueError("No data loaded")

    data = pd.concat(dfs, ignore_index=True)
    
    # Filter out managers if they exist, to prevent ID collision with new players
    if 'position' in data.columns:
        data = data[data['position'] != 'AM']
    
    data = data.sort_values(by=['name', 'season_index', 'GW'])
    
    # Target: total_points in the next GW
    data['target_points'] = data.groupby('name')['total_points'].shift(-1)
    
    # Calculate match difficulty
    if 'team_h_difficulty' in data.columns and 'was_home' in data.columns:
        data['match_difficulty'] = np.where(data['was_home'], data['team_h_difficulty'], data['team_a_difficulty'])
    else:
        data['match_difficulty'] = 3 # fallback
        
    data['next_difficulty'] = data.groupby('name')['match_difficulty'].shift(-1)
    
    # Core rolling features
    rolling_features = ['minutes', 'total_points', 'bps', 'influence', 'creativity',
                        'threat', 'ict_index', 'value']
    # Attacking features (key for identifying explosive returners)
    attacking_features = ['goals_scored', 'assists', 'expected_goals', 'expected_assists',
                          'expected_goal_involvements', 'clean_sheets', 'goals_conceded', 'saves']
    
    all_rolling = rolling_features + attacking_features
    
    for f in all_rolling:
        data[f] = pd.to_numeric(data[f], errors='coerce').fillna(0)
        data[f'{f}_rolling_3'] = data.groupby('name')[f].transform(lambda x: x.rolling(3, min_periods=1).mean())
        data[f'{f}_rolling_5'] = data.groupby('name')[f].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # Cumulative season stats (captures absolute quality)
    for f in ['goals_scored', 'assists', 'total_points', 'minutes']:
        data[f'{f}_cumsum'] = data.groupby('name')[f].cumsum()
    
    # Points per minute (efficiency metric, avoids div by zero)
    data['pts_per_min'] = np.where(data['minutes_cumsum'] > 0,
                                    data['total_points_cumsum'] / data['minutes_cumsum'] * 90,
                                    0)
    
    # Selection popularity (wisdom of the crowd)
    if 'selected' in data.columns:
        data['selected'] = pd.to_numeric(data['selected'], errors='coerce').fillna(0)
        data['log_selected'] = np.log1p(data['selected'])
    else:
        data['log_selected'] = 0
    
    feature_cols = ([f'{f}_rolling_3' for f in all_rolling] +
                    [f'{f}_rolling_5' for f in all_rolling] +
                    ['goals_scored_cumsum', 'assists_cumsum', 'pts_per_min', 'log_selected',
                     'next_difficulty'])
    
    model_data = data.dropna(subset=['target_points'] + feature_cols)
    return model_data, feature_cols, data

def train_model(model_data, feature_cols):
    X = model_data[feature_cols]
    y = model_data['target_points']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Model trained. Mean Absolute Error on test set: {mae:.2f} points")
    
    return model

def get_latest_features(data, feature_cols, team_next_difficulty_map=None):
    """
    Get the most recent row for each player to predict their NEXT gameweek points.
    """
    latest_data = data.drop_duplicates(subset=['name'], keep='last').copy()
    
    # Populate next_difficulty manually for the latest row since shift(-1) leaves it NaN
    if 'next_difficulty' in feature_cols:
        if team_next_difficulty_map and 'team' in latest_data.columns:
            latest_data['next_difficulty'] = latest_data['team'].map(team_next_difficulty_map).fillna(3)
        else:
            latest_data['next_difficulty'] = latest_data['next_difficulty'].fillna(3)
    
    latest_data = latest_data.dropna(subset=feature_cols)
    
    # Filter out players who only existed in older seasons to prevent ID collisions
    if 'season_index' in latest_data.columns:
        max_season = data['season_index'].max()
        latest_data = latest_data[latest_data['season_index'] == max_season]
    
    # We need player_id (element) to map to the optimizer. In Vaastav's data it's usually 'element'.
    if 'element' not in latest_data.columns:
        print("Warning: 'element' column not found, FPL API integration might need ID mapping.")
        
    return latest_data

if __name__ == "__main__":
    file_paths = [
        "data/historical/data/2024-25/gws/merged_gw.csv",
        "data/historical/data/2025-26/gws/merged_gw.csv"
    ]
    print("Preparing data...")
    model_data, feature_cols, full_data = prepare_data(file_paths)
    print(f"Data prepared. Training on {len(model_data)} rows...")
    model = train_model(model_data, feature_cols)
    
    print("Extracting latest features for prediction...")
    latest_data = get_latest_features(full_data, feature_cols)
    
    # Predict for the upcoming GW
    latest_data['predicted_points'] = model.predict(latest_data[feature_cols])
    
    top_predicted = latest_data.sort_values(by='predicted_points', ascending=False)
    print("\nTop 10 Predictions for next Gameweek:")
    print(top_predicted[['name', 'predicted_points', 'value']].head(10))
