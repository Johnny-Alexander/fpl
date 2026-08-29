import requests
import pandas as pd
import json

BASE_URL = "https://fantasy.premierleague.com/api"

def get_bootstrap_static():
    """Fetches all static data: players (elements), teams, events (gameweeks)"""
    url = f"{BASE_URL}/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_fixtures():
    """Fetches all fixtures for the current season"""
    url = f"{BASE_URL}/fixtures/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_player_detailed_data(element_id):
    """Fetches detailed data for a specific player"""
    url = f"{BASE_URL}/element-summary/{element_id}/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def get_user_team(team_id, gameweek):
    """Fetches the user's team picks for a given gameweek"""
    url = f"{BASE_URL}/entry/{team_id}/event/{gameweek}/picks/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None  # Gameweek might not have started or team didn't exist
    response.raise_for_status()

def get_current_gameweek(bootstrap_data=None):
    """Parses bootstrap data to find the current active gameweek"""
    if not bootstrap_data:
        bootstrap_data = get_bootstrap_static()
    
    events = bootstrap_data['events']
    for event in events:
        if event['is_current']:
            return event['id']
    
    # If no current gameweek (e.g., before season starts), find the next one
    for event in events:
        if event['is_next']:
            return event['id']
    
    return 1

if __name__ == "__main__":
    # Quick test
    print("Fetching static data...")
    data = get_bootstrap_static()
    current_gw = get_current_gameweek(data)
    print(f"Current Gameweek: {current_gw}")
    
    team_id = 8936155
    print(f"Fetching team picks for Team {team_id} in GW {current_gw}...")
    team_data = get_user_team(team_id, current_gw)
    if team_data:
        print(f"Found {len(team_data['picks'])} picks.")
    else:
        print("Could not fetch team picks.")
