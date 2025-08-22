"""
Build team rosters grouped by position from pick data.
"""

from __future__ import annotations
from typing import List, Dict

def build_rosters(picks: List[dict], users: List[dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    picks: list of {roster_id, metadata:{first_name,last_name,position}}
    users: from Sleeper users endpoint (contains roster_id + display name)
    returns: {team_display_name: {POS: [players...]}}
    """
    roster_id_to_name = {}
    for u in users or []:
        rid = int(u.get("roster_id", 0))
        nm = u.get("metadata", {}).get("team_name") or u.get("display_name") or f"Slot {rid}"
        roster_id_to_name[rid] = nm

    team_maps: Dict[str, Dict[str, List[str]]] = {}
    for p in picks or []:
        rid = int(p.get("roster_id", 0))
        team = roster_id_to_name.get(rid, f"Slot {rid}")
        meta = p.get("metadata") or {}
        pos = str(meta.get("position") or "").upper()
        if pos in ("DEF","D/ST","D-ST","TEAM D","TEAM DEF","DEFENSE"): pos = "DST"
        nm = f"{meta.get('first_name','')} {meta.get('last_name','')}".strip() or meta.get("name","")
        team_maps.setdefault(team, {})
        team_maps[team].setdefault(pos, [])
        if nm: team_maps[team][pos].append(nm)
    return team_maps
