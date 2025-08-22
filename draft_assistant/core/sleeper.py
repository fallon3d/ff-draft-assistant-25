"""
Sleeper API integration (read-only).
"""
import requests
from functools import lru_cache

API_BASE = "https://api.sleeper.app/v1"

@lru_cache(maxsize=8)
def get_league_info(league_id):
    """
    Get league information, including draft_id and total rosters.
    """
    try:
        resp = requests.get(f"{API_BASE}/league/{league_id}")
        resp.raise_for_status()
        data = resp.json()
        return {
            "league_id": data.get("league_id"),
            "total_rosters": data.get("total_rosters"),
            "draft_id": data.get("draft_id"),
            "name": data.get("name"),
            "settings": data.get("settings")
        }
    except Exception as e:
        print(f"Error fetching league info: {e}")
        return None

@lru_cache(maxsize=8)
def get_draft(draft_id):
    """
    Get draft settings for a given draft_id (e.g., rounds, teams).
    """
    try:
        resp = requests.get(f"{API_BASE}/draft/{draft_id}")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching draft: {e}")
        return None

@lru_cache(maxsize=8)
def get_drafts_for_league(league_id):
    """
    Get all drafts for a league.
    """
    try:
        resp = requests.get(f"{API_BASE}/league/{league_id}/drafts")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching drafts for league: {e}")
        return None

@lru_cache(maxsize=8)
def get_picks(draft_id):
    """
    Get all picks made so far in the draft.
    """
    try:
        resp = requests.get(f"{API_BASE}/draft/{draft_id}/picks")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching draft picks: {e}")
        return []

@lru_cache(maxsize=8)
def get_users(league_id):
    """
    Get users in a league (including roster IDs and team display names).
    """
    try:
        resp = requests.get(f"{API_BASE}/league/{league_id}/users")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching league users: {e}")
        return []
