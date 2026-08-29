#!/usr/bin/env python3
"""
FPL Backtesting Framework
Simulates model-recommended transfers from various switch points and compares
cumulative points against actual performance via a worm graph.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor
from ml_model import prepare_data
from optimizer import optimize_squad
from visualize import plot_worm_graph

TEAM_ID = 8936155
BASE_URL = "https://fantasy.premierleague.com/api"
SWITCH_GWS = [10, 20, 30]
CURRENT_GW = 31


# ──────────────────────────── Data Loading ────────────────────────────

def fetch_bootstrap():
    return requests.get(f"{BASE_URL}/bootstrap-static/").json()


def get_user_history():
    """Returns {gw: points} for the user's actual season."""
    data = requests.get(f"{BASE_URL}/entry/{TEAM_ID}/history/").json()
    return {gw['event']: gw['points'] for gw in data['current']}


def get_user_squad_at_gw(gw):
    """Returns (squad_ids, bank) for the user's team at a given GW."""
    resp = requests.get(f"{BASE_URL}/entry/{TEAM_ID}/event/{gw}/picks/")
    if resp.status_code == 200:
        data = resp.json()
        return [p['element'] for p in data['picks']], data['entry_history']['bank']
    return None, None


def load_player_gw_points():
    """Build lookup: (element_id, gw) -> actual total_points."""
    csv_path = os.path.join(PROJECT_DIR, 'data/historical/data/2025-26/gws/merged_gw.csv')
    df = pd.read_csv(csv_path)
    points = {}
    for _, row in df.iterrows():
        points[(int(row['element']), int(row['GW']))] = int(row['total_points'])

    max_csv_gw = int(df['GW'].max())
    for gw in range(max_csv_gw + 1, CURRENT_GW + 1):
        print(f"  Fetching GW{gw} live data from API...")
        resp = requests.get(f"{BASE_URL}/event/{gw}/live/")
        if resp.status_code == 200:
            for elem in resp.json()['elements']:
                points[(elem['id'], gw)] = elem['stats']['total_points']
    return points


def build_elements_df(bootstrap):
    """Build the elements DataFrame used by the optimizer."""
    df = pd.DataFrame(bootstrap['elements'])
    df['element_id'] = df['id']
    df['position'] = df['element_type']
    df['value'] = df['now_cost']
    return df


# ──────────────────────────── Model ────────────────────────────

def train_model_at_gw(full_data, feature_cols, cutoff_gw):
    """Train model using only data available before cutoff_gw."""
    train_data = full_data[
        (full_data['season_index'] == 0) |
        ((full_data['season_index'] == 1) & (full_data['GW'] <= cutoff_gw - 2))
    ].copy()
    train_data = train_data.dropna(subset=['target_points'] + feature_cols)

    if len(train_data) < 100:
        print(f"    Warning: Only {len(train_data)} training rows")

    X = train_data[feature_cols]
    y = train_data['target_points']
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X, y)
    return model


def get_predictions(model, full_data, feature_cols, cutoff_gw):
    """Get element_id -> predicted_points for all players at a GW cutoff."""
    data_up_to = full_data[
        (full_data['season_index'] == 0) |
        ((full_data['season_index'] == 1) & (full_data['GW'] < cutoff_gw))
    ].copy()

    latest = data_up_to.drop_duplicates(subset=['name'], keep='last').copy()

    # Only keep current-season players
    if 'season_index' in latest.columns:
        latest = latest[latest['season_index'] == full_data['season_index'].max()]

    if 'next_difficulty' in feature_cols:
        latest['next_difficulty'] = latest['next_difficulty'].fillna(3)

    latest = latest.dropna(subset=feature_cols)
    if len(latest) == 0:
        return {}

    latest['predicted_points'] = model.predict(latest[feature_cols])

    if 'element' in latest.columns:
        return dict(zip(latest['element'].astype(int), latest['predicted_points']))
    return {}


def calc_gw_score(starting_11, captain, gw, player_gw_points):
    """Sum of starting 11 actual points, captain doubled."""
    total = 0
    for pid in starting_11:
        pts = player_gw_points.get((pid, gw), 0)
        total += pts * 2 if pid == captain else pts
    return total


# ──────────────────────────── Simulation ────────────────────────────

def simulate_path(switch_gw, full_data, feature_cols, player_gw_points,
                  actual_points, elements_df, positions):
    """Simulate one path: use model from switch_gw onward."""
    print(f"\n{'='*50}")
    print(f"  Simulating: switch to model at GW{switch_gw}")
    print(f"{'='*50}")

    # Get squad from the GW before the switch (the team going into switch_gw)
    squad_ids, bank = get_user_squad_at_gw(switch_gw - 1)
    if squad_ids is None and switch_gw > 1:
        squad_ids, bank = get_user_squad_at_gw(switch_gw)
    if squad_ids is None:
        print(f"  ❌ Could not fetch squad for GW{switch_gw}")
        return None

    budget = sum(elements_df.set_index('element_id')['value'].to_dict().get(pid, 50)
                 for pid in squad_ids) + (bank or 0)
    free_transfers = 1
    
    # Chip tracking
    chips = {
        'WC1': False, 'WC2': False, 'FH': False, 'BB': False, 'TC': False
    }
    
    # Track pre-Free Hit squad
    fh_saved_squad = None
    fh_saved_budget = None

    gw_scores = {}

    # Use actual points for GWs before switch
    for gw in range(1, switch_gw):
        gw_scores[gw] = actual_points.get(gw, 0)

    values = elements_df.set_index('element_id')['value'].to_dict()

    for gw in range(switch_gw, CURRENT_GW + 1):
        print(f"    Training model on data up to GW{gw}...")
        model = train_model_at_gw(full_data, feature_cols, gw)
        pred_map = get_predictions(model, full_data, feature_cols, gw)

        # Restore squad if Free Hit was played last week
        if fh_saved_squad is not None:
            squad_ids = fh_saved_squad
            budget = fh_saved_budget
            fh_saved_squad = None
            fh_saved_budget = None

        # Determine if we should play a chip this week
        chip_active = None
        
        # 1. Triple Captain Heuristic: Captain predicted > 11 pts
        if not chips['TC']:
            best_pred = max([pred_map.get(pid, 0) for pid in squad_ids] + [0])
            if best_pred > 11.0:
                chip_active = 'TC'
                chips['TC'] = True
        
        # 2. Bench Boost Heuristic: Bench predicted > 14 pts
        if not chip_active and not chips['BB']:
            # Sort squad by predicted points
            sorted_squad = sorted(squad_ids, key=lambda pid: pred_map.get(pid, 0), reverse=True)
            bench_pred_sum = sum(pred_map.get(pid, 0) for pid in sorted_squad[11:])
            if bench_pred_sum > 14.0:
                chip_active = 'BB'
                chips['BB'] = True
                
        # 3. Wildcard Heuristic: 
        if not chip_active:
            if not chips['WC1'] and gw <= 19 and gw >= switch_gw + 3:
                chip_active = 'WC'
                chips['WC1'] = True
            elif not chips['WC2'] and gw > 19 and gw >= 28:
                chip_active = 'WC'
                chips['WC2'] = True
                
        # 4. Free Hit Heuristic: If expected points of current squad is very low (< 35)
        if not chip_active and not chips['FH']:
            current_expected = sum(pred_map.get(pid, 0) for pid in sorted(squad_ids, key=lambda p: pred_map.get(p, 0), reverse=True)[:11])
            if current_expected < 35.0:
                chip_active = 'FH'
                chips['FH'] = True
                fh_saved_squad = list(squad_ids)
                fh_saved_budget = budget

        if chip_active:
            print(f"    🌟 PLAYING CHIP: {chip_active}")

        # Build optimizer input
        player_data = elements_df.copy()
        player_data['predicted_points'] = player_data['element_id'].map(pred_map).fillna(0)

        # Run optimizer
        transfers_made = 0
        penalty = 0
        
        starting_11 = []
        captain = None
        
        try:
            # For WC and FH, allow 15 transfers freely
            opt_free_transfers = 15 if chip_active in ['WC', 'FH'] else free_transfers
            opt_hard_max = 15 if chip_active in ['WC', 'FH'] else 3
            
            result = optimize_squad(
                player_data,
                free_transfers=opt_free_transfers,
                current_squad_ids=squad_ids,
                budget=budget,
                n=1,
                hard_max_transfers=opt_hard_max,
                chip_active=chip_active
            )
            if result is not None:
                new_ids = list(result['element_id'])
                transfers_made = len(set(squad_ids) - set(new_ids))
                
                # Penalties don't apply on WC or FH
                if chip_active in ['WC', 'FH']:
                    extra = 0
                else:
                    extra = max(0, transfers_made - free_transfers)
                    
                penalty = extra * 4
                squad_ids = new_ids
                budget = sum(values.get(pid, 50) for pid in squad_ids) + (bank or 0)
                
                starting_11 = result[result['is_starter']]['element_id'].tolist()
                caps = result[result['is_captain']]['element_id'].tolist()
                captain = caps[0] if caps else None
        except Exception as e:
            print(f"    ⚠️ Optimizer error at GW{gw}: {e}")

        # Update free transfers
        if chip_active in ['WC', 'FH']:
            free_transfers = 1 # Reverts to 1 after chip
        elif transfers_made == 0:
            free_transfers = min(2, free_transfers + 1)
        else:
            free_transfers = 1

        # Fallback if optimization failed completely
        if not starting_11:
            sorted_squad = sorted(squad_ids, key=lambda pid: pred_map.get(pid, 0), reverse=True)
            starting_11 = sorted_squad[:11]
            captain = sorted_squad[0]

        # Calculate score
        total = 0
        
        if chip_active == 'BB':
            # Everyone scores
            for pid in squad_ids:
                total += player_gw_points.get((pid, gw), 0)
        else:
            for pid in starting_11:
                pts = player_gw_points.get((pid, gw), 0)
                if pid == captain:
                    if chip_active == 'TC':
                        total += pts * 3
                    else:
                        total += pts * 2
                else:
                    total += pts
                    
        score = total - penalty

        gw_scores[gw] = score
        t_str = f"{transfers_made}T" if transfers_made else "0T"
        h_str = f" (-{penalty}pt hit)" if penalty > 0 else ""
        print(f"    GW{gw:2d}: {score:3d} pts | {t_str}{h_str}")

    total = sum(gw_scores.get(gw, 0) for gw in range(1, CURRENT_GW + 1))
    print(f"  📊 Total: {total} pts")
    return gw_scores


# ──────────────────────────── Main ────────────────────────────

def main():
    print("=" * 60)
    print("  FPL Model Backtesting Framework")
    print("=" * 60)

    # Load all data
    print("\n📊 Loading data...")
    bootstrap = fetch_bootstrap()
    elements_df = build_elements_df(bootstrap)
    actual_points = get_user_history()
    player_gw_points = load_player_gw_points()
    positions = dict(zip(elements_df['element_id'], elements_df['position']))

    # Prepare ML features (computed once on full data)
    print("\n🔧 Preparing ML feature data...")
    file_paths = [
        os.path.join(PROJECT_DIR, 'data/historical/data/2024-25/gws/merged_gw.csv'),
        os.path.join(PROJECT_DIR, 'data/historical/data/2025-26/gws/merged_gw.csv')
    ]
    _, feature_cols, full_data = prepare_data(file_paths)

    # Run simulations
    model_paths = {}
    for switch_gw in SWITCH_GWS:
        gw_scores = simulate_path(
            switch_gw, full_data, feature_cols, player_gw_points,
            actual_points, elements_df, positions
        )
        if gw_scores:
            model_paths[switch_gw] = gw_scores

    # Summary table
    actual_total = sum(actual_points.get(gw, 0) for gw in range(1, CURRENT_GW + 1))
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Path':<25} {'Total Pts':>10} {'vs Actual':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Actual':<25} {actual_total:>10}")
    for sw_gw in sorted(model_paths.keys()):
        total = sum(model_paths[sw_gw].get(gw, 0) for gw in range(1, CURRENT_GW + 1))
        diff = total - actual_total
        print(f"  {'Model from GW' + str(sw_gw):<25} {total:>10} {diff:>+10}")

    # Generate worm graph
    output_path = os.path.join(SCRIPT_DIR, 'output', 'worm_graph.png')
    plot_worm_graph(actual_points, model_paths, output_path, CURRENT_GW)

    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
