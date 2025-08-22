"""
Sleeper API helpers, including players and mock parsing.
Adds K and DST support (maps DEF -> DST).
"""
import re
import time
import requests
import pandas as pd

API_BASE = "https://api.sleeper.app/v1"
TIMEOUT = 12
_cache = {}

def _cache_get(key, ttl=5):
    data = _cache.get(key)
    if not data: return None
    ts, val = data
    if time.time() - ts > ttl: return None
    return val

def _cache_set(key, val): _cache[key] = (time.time(), val)

def _get_json(url):
    try:
        r = requests.get(url, timeout=TIMEOUT); r.raise_for_status()
        return r.json()
    except Exception:
        return None

def get_league_info(league_id: str):
    key = f"league:{league_id}"; c = _cache_get(key, ttl=30)
    if c is not None: return c
    data = _get_json(f"{API_BASE}/league/{league_id}")
    if not data: return None
    out = {
        "league_id": data.get("league_id"),
        "name": data.get("name"),
        "draft_id": data.get("draft_id"),
        "total_rosters": data.get("total_rosters"),
        "settings": data.get("settings"),
    }
    _cache_set(key, out); return out

def get_drafts_for_league(league_id: str):
    key = f"drafts:{league_id}"; c = _cache_get(key, ttl=30)
    if c is not None: return c
    data = _get_json(f"{API_BASE}/league/{league_id}/drafts")
    if data is None: return []
    _cache_set(key, data); return data

def get_draft(draft_id: str):
    key = f"draft:{draft_id}"; c = _cache_get(key, ttl=30)
    if c is not None: return c
    data = _get_json(f"{API_BASE}/draft/{draft_id}")
    if data is None: return {}
    _cache_set(key, data); return data

def get_picks(draft_id: str):
    key = f"picks:{draft_id}"; c = _cache_get(key, ttl=5)
    if c is not None: return c
    data = _get_json(f"{API_BASE}/draft/{draft_id}/picks")
    if data is None: return []
    _cache_set(key, data); return data

def get_users(league_id: str):
    key = f"users:{league_id}"; c = _cache_get(key, ttl=60)
    if c is not None: return c
    data = _get_json(f"{API_BASE}/league/{league_id}/users")
    if data is None: return []
    _cache_set(key, data); return data

def get_players_nfl():
    key = "players:nfl"; c = _cache_get(key, ttl=3600)
    if c is not None: return c
    data = _get_json(f"{API_BASE}/players/nfl")
    if data is None: return {}
    _cache_set(key, data); return data

def players_df_from_dict(players_map: dict) -> pd.DataFrame:
    rows = []
    for _, p in (players_map or {}).items():
        pos = p.get("position") or (p.get("fantasy_positions") or [None])[0]
        if pos == "DEF":  # map Sleeper DEF to DST
            pos = "DST"
        if pos not in ("QB","RB","WR","TE","K","DST"): 
            continue
        full = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if not full: 
            continue
        team = p.get("team") or p.get("active_team") or "FA"
        bye = p.get("bye_week") or 0
        rookie = 1 if (p.get("rookie") or p.get("years_exp") in (0,1)) else 0
        rows.append({
            "RK": 999, "TIERS": "", "PLAYER": full, "TEAM": team, "POS": pos,
            "BYE": int(bye) if str(bye).isdigit() else 0, "SOS": 0, "ADP": 0, "ROOKIE": rookie,
        })
    return pd.DataFrame(rows)

def parse_draft_id_from_url(url: str) -> str | None:
    if not url: return None
    m = re.search(r"/draft/[^/]+/([A-Za-z0-9_-]+)", url) or re.search(r"/draft/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None

def picked_player_names(picks: list, players_map: dict) -> list[str]:
    out = []
    for p in picks or []:
        pid = p.get("player_id"); meta = p.get("metadata") or {}
        nm = ""
        if pid and players_map and pid in players_map:
            info = players_map[pid]
            nm = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
        if not nm:
            nm = f"{meta.get('first_name','')} {meta.get('last_name','')}".strip() or meta.get("name","")
        if nm: out.append(nm)
    return out

def picks_to_internal_log(picks: list, players_map: dict) -> list[dict]:
    out = []
    for p in picks or []:
        meta = p.get("metadata") or {}
        pid = p.get("player_id")
        first, last, pos = "", "", meta.get("position")
        if pid and players_map and pid in players_map:
            info = players_map[pid]
            full = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
            first = full; last = ""; 
            pos = "DST" if (pos == "DEF") else (pos or info.get("position"))
            if pos == "DEF": pos = "DST"
        else:
            first = f"{meta.get('first_name','')} {meta.get('last_name','')}".strip() or meta.get("name","")
            last = ""
        out.append({
            "round": p.get("round"),
            "pick_no": p.get("pick_no") or p.get("pick", 0),
            "team": f"Roster {p.get('roster_id','?')}",
            "metadata": {"first_name": first, "last_name": last, "position": pos},
        })
    return out
