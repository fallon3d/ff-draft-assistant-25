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

# ---- Robust draft_id parsing for mock URLs ----
def parse_draft_id_from_url(url_or_id: str) -> Optional[str]:
    """
    Accepts:
      - Raw draft_id (all digits/letters)
      - URLs like:
        https://sleeper.com/draft/123...
        https://sleeper.com/draft/nfl/123...
        https://sleeper.com/draft/board/123...
    Strategy: Prefer the longest 10–24 char alnum token at the end or anywhere in path.
    """
    s = str(url_or_id).strip()
    # Raw id?
    if re.fullmatch(r"[A-Za-z0-9_]{10,24}", s):
        return s

    # Common shapes
    m = re.search(r"/draft/(?:nfl/|board/)?([A-Za-z0-9_]{10,24})", s)
    if m:
        return m.group(1)

    # Fallback: longest plausible id-looking token
    candidates = re.findall(r"([A-Za-z0-9_]{10,24})", s)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    return None

# ---- Defensive picks -> internal log conversion ----
def picks_to_internal_log(picks: List[dict], players_map: dict, teams: int | None = None) -> List[dict]:
    """
    Convert Sleeper picks to our simple structure:
      {round, pick_no, team, metadata:{first_name,last_name,position}}
    Tries multiple fields and computes round/pick_no from 'pick' if needed (and teams provided).
    """
    out = []
    for p in picks or []:
        meta = p.get("metadata") or {}
        pid = p.get("player_id")
        # Name
        first, last = meta.get("first_name",""), meta.get("last_name","")
        if (not first and not last) and pid and pid in players_map:
            pm = players_map[pid]
            first, last = pm.get("first_name",""), pm.get("last_name","")
            meta["position"] = meta.get("position") or pm.get("position")
        # Position fallback from players_map
        if not meta.get("position") and pid and pid in players_map:
            meta["position"] = players_map[pid].get("position")

        # Round / pick_no
        rnd = p.get("round")
        pick_no = p.get("pick_no")
        overall = p.get("pick")  # Sleeper sometimes provides this

        if (rnd is None or pick_no is None) and overall and teams:
            try:
                overall_i = int(overall)
                teams_i = int(teams)
                rnd = (overall_i - 1) // teams_i + 1
                pick_no = (overall_i - 1) % teams_i + 1
            except Exception:
                pass

        try:
            rnd = int(rnd) if rnd is not None else 0
        except Exception:
            rnd = 0
        try:
            pick_no = int(pick_no) if pick_no is not None else 0
        except Exception:
            pick_no = 0

        out.append({
            "round": rnd,
            "pick_no": pick_no,
            "team": p.get("picked_by") or "",
            "metadata": {"first_name": first, "last_name": last, "position": meta.get("position")},
        })
    return out
