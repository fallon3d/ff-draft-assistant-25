"""
Player evaluation and Value-Based Drafting (VBD) calculations.
"""
def evaluate_players(df, config, user_team=None):
    """
    Evaluate players with custom scoring, VBD, schedule, and coaching modifiers.
    Returns DataFrame with additional columns: value, vbd, notes.
    """
    import pandas as pd
    
    teams = config.get("draft", {}).get("teams", 12)
    # Assume starters: 1 QB, 2 RB, 3 WR, 1 TE per team by default
    baselines = {
        "QB": teams,
        "RB": teams * 2,
        "WR": teams * 3,
        "TE": teams
    }
    # Load schedule and coaching modifiers if available
    try:
        schedule_df = pd.read_csv("data/sample_schedule.csv")
        sched_dict = dict(zip(schedule_df["TEAM"], schedule_df["STARS"]))
    except Exception:
        sched_dict = {}
    import json
    try:
        with open("data/coaching_modifiers.json") as f:
            coach_mod = json.load(f)
    except Exception:
        coach_mod = {}

    df = df.copy()
    # Base value: inverse of RK (assuming lower RK is better, e.g., 1 is top)
    if "RK" in df.columns:
        df["base_points"] = df["RK"].apply(lambda x: max(0, 100 - float(x)))
    else:
        # fallback to index or ECR
        df["base_points"] = 50
    
    # Compute baseline points by position
    baseline_points = {}
    for pos, count in baselines.items():
        subset = df[df["POS"] == pos].sort_values("base_points", ascending=False)
        if len(subset) >= count:
            baseline_points[pos] = float(subset.iloc[count-1]["base_points"])
        else:
            baseline_points[pos] = 0.0
    
    values = []
    vbds = []
    notes_list = []
    for idx, player in df.iterrows():
        pos = player.get("POS")
        team = player.get("TEAM")
        base = player.get("base_points", 0)
        # Schedule adjustment
        stars = sched_dict.get(team, None)
        sched_multiplier = 1.0
        note = ""
        if stars:
            # Assume stars 1-5, 3 is neutral
            diff = stars - 3
            sched_multiplier += diff * 0.05  # 5% per star away from 3
            if diff > 0:
                note += f"Favorable schedule (+{diff*5:.0f}%), "
            elif diff < 0:
                note += f"Tough schedule ({diff*5:.0f}%), "
        # Coaching adjustment
        coach = coach_mod.get(team, {})
        coach_multiplier = 1.0
        if coach:
            if pos == "RB":
                coach_multiplier *= coach.get("rush_rate", 1.0)
            if pos in ["WR", "QB", "TE"]:
                coach_multiplier *= coach.get("pass_rate", 1.0)
            coach_multiplier *= coach.get("pace", 1.0)
        total_points = base * sched_multiplier * coach_multiplier
        # Calculate VBD
        baseline = baseline_points.get(pos, 0)
        vbd = total_points - baseline
        # Assemble notes
        if player.get("TIERS"):
            note += f"Tier {player.get('TIERS')}, "
        if player.get("BYE"):
            note += f"Bye wk {player.get('BYE')}, "
        # Trim trailing comma
        note = note.rstrip(", ")
        values.append(total_points)
        vbds.append(vbd)
        notes_list.append(note)
    df["value"] = values
    df["vbd"] = vbds
    df["notes"] = notes_list
    # Sort by value descending
    df = df.sort_values("value", ascending=False)
    return df
