"""
Roster and draft board management.
"""
def build_rosters(picks, users):
    """
    Given list of pick dicts and list of user dicts (with 'roster_id' and 'display_name'),
    return a dict mapping team names to roster lists by position.
    """
    # Build roster_id to team name mapping
    roster_map = {}
    for user in users:
        name = user.get("display_name") or user.get("username") or f"Team {user.get('roster_id', '')}"
        roster_id = user.get("roster_id")
        roster_map[str(roster_id)] = name
    
    # Initialize team rosters
    teams = {}
    for rid, name in roster_map.items():
        teams[name] = {"QB": [], "RB": [], "WR": [], "TE": [], "Other": []}
    
    # Fill rosters based on picks
    for pick in picks:
        rid = str(pick.get("roster_id"))
        team_name = roster_map.get(rid, f"Team {rid}")
        position = pick.get("metadata", {}).get("position")
        first = pick.get("metadata", {}).get("first_name", "")
        last = pick.get("metadata", {}).get("last_name", "")
        player_name = f"{first} {last}".strip()
        if not player_name:
            player_name = pick.get("metadata", {}).get("name", "")
        if position in teams.get(team_name, {}):
            teams[team_name][position].append(player_name)
        else:
            teams[team_name]["Other"].append(player_name)
    return teams
