"""
Sleeper API (read-only, polite).
"""
import re
import time
import requests
import pandas as pd

API_BASE = "https://api.sleeper.app/v1"
TIMEOUT = 12

# Simple in-memory TTL cache
_cache = {}

def _cache_get(key, ttl=5):
    data = _cache.get(key)
    if not data:
        return None
    ts, val = data
    if time.time() - ts > ttl:
        return None
    return val

def _cache_set(key, val):
    _cache[key] = (time.time(), val)

def _get_json(url):
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---------------- League/Draft/Picks ----------------
def get_league_info(league_id: str):
    key = f"league:{league_id}"
    cached = _cache_get(key, ttl=30)
    if cached is not None:
        return cached
    data = _get_json(f"{API_BASE}/league/{league_id}")
    if not data:
        return None
    out = {
        "league_id": data.get("league_id"),
        "name": data.get("name"),
        "draft_id": data.get("draft_id"),
        "total_rosters": data.get("total_rosters"),
        "settings": data.get("settings"),
    }
    _cache_set(key, out)
    return out

def get_drafts_for_league(league_id: str):
    key = f"drafts:{league_id}"
    cached = _cache_get(key, ttl=30)
    if cached is not None:
        return cached
    data = _get_json(f"{API_BASE}/league/{league_id}/drafts")
    if data is None:
        return []
    _cache_set(key, data)
    return data

def get_draft(draft_id: str):
    key = f"draft:{draft_id}"
    cached = _cache_get(key, ttl=30)
    if cached is not None:
        return cached
    data = _get_json(f"{API_BASE}/draft/{draft_id}")
    if data is None:
        return {}
    _cache_set(key, data)
    return data

def get_picks(draft_id: str):
    key = f"picks:{draft_id}"
    cached = _cache_get(key, ttl=5)
    if cached is not None:
        return cached
    data = _get_json(f"{API_BASE}/draft/{draft_id}/picks")
    if data is None:
        return []
    _cache_set(key, data)
    return data

def get_users(league_id: str):
    key = f"users:{league_id}"
    cached = _cache_get(key, ttl=60)
    if cached is not None:
        return cached
    data = _get_json(f"{API_BASE}/league/{league_id}/users")
    if data is None:
        return []
    _cache_set(key, data)
    return data

# ---------------- Players ----------------
def get_players_nfl():
    """
    Return Sleeper /players/nfl map (player_id -> info) with long TTL cache.
    """
    key = "players:nfl"
    cached = _cache_get(key, ttl=3600)  # 1 hour
    if cached is not None:
        return cached
    data = _get_json(f"{API_BASE}/players/nfl")
    if data is None:
        return {}
    _cache_set(key, data)
    return data

def players_df_from_dict(players_map: dict) -> pd.DataFrame:
    """
    Convert Sleeper players map to a DF with our canonical columns.
    Only keep positions QB/RB/WR/TE. Rank low (RK=999) so uploads take precedence.
    """
    rows = []
    for pid, p in (players_map or {}).items():
        # Determine position
        pos = p.get("position")
        if not pos:
            fps = p.get("fantasy_positions") or []
            pos = fps[0] if fps else None
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        # Name
        full = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
        if not full:
            continue
        team = p.get("team") or p.get("active_team") or "FA"
        bye = p.get("bye_week") or 0
        rows.append({
            "RK": 999,              # rank low so CSVs stay on top
            "TIERS": "",
            "PLAYER": full,
            "TEAM": team,
            "POS": pos,
            "BYE": int(bye) if isinstance(bye, (int, float, str)) and str(bye).isdigit() else 0,
            "SOS": 0,
            "ADP": 0,
        })
    df = pd.DataFrame(rows)
    return df

# ---------------- Helpers for mock sync ----------------
def parse_draft_id_from_url(url: str) -> str | None:
    """
    Extract draft_id from a Sleeper mock or draft URL like:
      https://sleeper.com/draft/nfl/123456789012345678
      https://sleeper.app/draft/nfl/123456...
    """
    if not url:
        return None
    m = re.search(r"/draft/[^/]+/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    # fallback: /draft/<id>
    m = re.search(r"/draft/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None

def picked_player_names(picks: list, players_map: dict) -> list[str]:
    """
    Build a list of player names from picks using player_id if present,
    else metadata.first_name/last_name.
    """
    out = []
    for p in picks or []:
        pid = p.get("player_id")
        meta = p.get("metadata") or {}
        nm = ""
        if pid and players_map and pid in players_map:
            info = players_map[pid]
            nm = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
        if not nm:
            nm = f"{meta.get('first_name','')} {meta.get('last_name','')}".strip() or meta.get("name","")
        if nm:
            out.append(nm)
    return out

def picks_to_internal_log(picks: list, players_map: dict) -> list[dict]:
    """
    Translate Sleeper picks into our internal pick log format used by the app.
    """
    out = []
    for p in picks or []:
        meta = p.get("metadata") or {}
        pid = p.get("player_id")
        first, last, pos = "", "", meta.get("position")
        if pid and players_map and pid in players_map:
            info = players_map[pid]
            full = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
            first = full
            last = ""
            pos = pos or info.get("position")
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
