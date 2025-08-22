"""
Mock AI pickers for practice mode.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def pick_for_team(available: pd.DataFrame, archetype: str, round_number: int) -> int | None:
    """
    Very light AI: pick from top N, with archetype flavor.
    Returns DataFrame index to draft.
    """
    if available is None or available.empty:
        return None
    df = available.copy()

    # Archetype filters
    if archetype == "Zero-RB" and round_number <= 3:
        df = df[df["POS"] != "RB"]
    elif archetype == "RB-Heavy" and round_number <= 4:
        df = df[df["POS"].isin(["RB","WR"])]

    df = df.sort_values(["score" if "score" in df.columns else "value","value"], ascending=False)
    top_n = min(7, len(df))
    weights = np.linspace(1.0, 0.25, num=top_n)
    weights = weights / weights.sum()
    idx_choices = df.iloc[:top_n].index.to_list()
    if not idx_choices:
        return None
    choice = np.random.choice(idx_choices, p=weights)
    return int(choice)
