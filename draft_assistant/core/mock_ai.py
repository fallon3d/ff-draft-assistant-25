"""
AI strategies for mock draft opponents.
"""
import random

STRATEGIES = ["Balanced", "Zero-RB", "RB-Heavy", "Early-QB", "Late-QB", "TE-Early", "Rookie"]

def pick_for_team(available_df, strategy, round_number):
    """
    Choose a player for the team based on strategy.
    available_df should have 'POS' and 'value'.
    Returns the index of selected player.
    """
    if available_df.empty:
        return None
    # Sort by value (descending)
    df = available_df.sort_values("value", ascending=False).copy()
    # Strategy conditions
    if strategy == "Zero-RB" and round_number <= 3:
        df = df[df["POS"] != "RB"]
    if strategy == "RB-Heavy" and round_number <= 3:
        df_rb = df[df["POS"] == "RB"]
        if not df_rb.empty:
            df = df_rb
    if strategy == "Early-QB" and round_number <= 4:
        df_qb = df[df["POS"] == "QB"]
        if not df_qb.empty:
            df = df_qb
    if strategy == "Late-QB" and round_number < 6:
        df = df[df["POS"] != "QB"]
    if strategy == "TE-Early" and round_number <= 2:
        df_te = df[df["POS"] == "TE"]
        if not df_te.empty:
            df = df_te
    if strategy == "Rookie":
        # pretend rookies are flagged in TIERS == 'R', else random small chance to pick lower tier
        rookies = df[df["TIERS"] == "R"]
        if not rookies.empty and random.random() < 0.3:
            df = rookies
    # Default: Balanced or fallback
    if df.empty:
        # if filtered out all, use original
        df = available_df.sort_values("value", ascending=False).copy()
    # Add randomness: choose from top 3
    top_n = df.head(3)
    if not top_n.empty:
        chosen = top_n.sample(n=1)
    else:
        chosen = df.head(1)
    if chosen.empty:
        return None
    return chosen.index[0]
