"""
Sleeper public API helpers (read-only)
"""

from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Optional
import requests

BASE = "https://api.sleeper.app/v1"
HTTP_TIMEOUT = 10.0

# Naive in-memory cache (path -> (timestamp, data))
_CACHE: Dict[str, tuple[float, Any]] = {}
TTL_SECONDS = 30.0

def _get(path: str) -> Any:
    now = time.time()
    if path in _CACHE and now - _CACHE[path][0] < TTL_SECONDS:
        return _CACHE[path][1]
    url = f"{BASE}{path}"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        _CACHE[path] = (now, data)
        return data
    except Exception:
        return None

def get_league_info(league_id: str) -> Optional[dict]:
    return _get(f"/league/{league_id}")

def get_drafts_for_league(league_id: str) -> Optional[List[dict]]:
    return _get(f"/league/{league_id}/drafts") or []

def get_draft(draft_id: str) -> Optional[dict]:
    return _get(f"/draft/{draft_id}")

def get_picks(draft_id: str) -> Optional[List[dict]]:
    return _get(f"/draft/{draft_id}/picks") or []

def get_users(league_id: str) -> Optional[List[dict]]:
    return _get(f"/league/{league_id}/users") or []

def get_players_nfl() -> Optional[dict]:
    # Big payload – cache for longer
    path = "/players/nfl"
    global TTL_SECONDS
    old_ttl = TTL_SECONDS
    TTL_SECONDS = 3600.0
    data = _get(path)
    TTL_SECONDS = old_ttl
    return data or {}

def picked_player_names(picks: List[dict], players_map: dict) -> set[str]:
    """
    Convert Sleeper player_ids in picks to a set of display names using players_map.
    Falls back to metadata names if needed.
    """
    out = set()
    for p in picks or []:
        pid = p.get("player_id")
        meta = p.get("metadata") or {}
        if pid and players_map and pid in players_map:
            pm = players_map[pid]
            dn = (pm.get("full_name") or f"{pm.get('first_name','')} {pm.get('last_name','')}".strip()).strip()
            if dn: out.add(dn)
        else:
            # fallback
            name = (meta.get("full_name") or f"{meta.get('first_name','')} {meta.get('last_name','')}".strip()).strip()
            if name: out.add(name)
    return out

def parse_draft_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"/draft/([A-Za-z0-9_]+)", url)
    return m.group(1) if m else None

def picks_to_internal_log(picks: List[dict], players_map: dict) -> List[dict]:
    """
    Convert Sleeper picks to our simple structure:
    {round, pick_no, team, metadata:{first_name,last_name,position}}
    """
    out = []
    for p in picks or []:
        meta = p.get("metadata") or {}
        pid = p.get("player_id")
        first, last = meta.get("first_name",""), meta.get("last_name","")
        if (not first and not last) and pid and pid in players_map:
            pm = players_map[pid]
            first, last = pm.get("first_name",""), pm.get("last_name","")
            meta["position"] = pm.get("position")
        out.append({
            "round": int(p.get("round", 0)),
            "pick_no": int(p.get("pick_no", p.get("pick", 0))),
            "team": p.get("picked_by") or "",
            "metadata": {"first_name": first, "last_name": last, "position": meta.get("position")},
        })
    return out
