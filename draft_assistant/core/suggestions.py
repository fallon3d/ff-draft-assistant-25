"""
Suggestion engine for draft picks.
"""
import pandas as pd

def top_suggestions(available_df, user_roster=None, count=5):
    """
    Suggest top players to pick based on value and roster needs.
    """
    if available_df is None or available_df.empty:
        return pd.DataFrame()
    # Simplest: top value
    return available_df.sort_values("value", ascending=False).head(count)

def likely_gone(available_df, picks_left=3):
    """
    Players likely to be gone before next turn (top picks).
    """
    if available_df is None or available_df.empty:
        return []
    top = available_df.sort_values("value", ascending=False).head(picks_left * 2)
    return top["PLAYER"].tolist()

def detect_runs(pick_log, recent=5):
    """
    Detect position runs from the last few picks.
    """
    if not pick_log:
        return ""
    last_picks = pick_log[-recent:]
    positions = [p.get("metadata", {}).get("position") for p in last_picks]
    if len(positions) >= 3 and positions.count(positions[0]) == len(positions):
        return f"{positions[0]} run"
    return ""
