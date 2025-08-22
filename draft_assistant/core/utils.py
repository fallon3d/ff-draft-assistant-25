"""
Utility helpers.
"""
import os
import toml
import pandas as pd

def read_config(path: str = os.path.join(os.path.dirname(__file__), "..", "config.toml")) -> dict:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}
    return toml.load(path)

def save_config(config: dict, path: str = os.path.join(os.path.dirname(__file__), "..", "config.toml")) -> None:
    path = os.path.abspath(path)
    with open(path, "w") as f:
        toml.dump(config, f)

def snake_position(overall_pick: int, teams: int):
    """
    Given overall pick number (1-indexed) and number of teams,
    return (round_number, pick_in_round, draft_slot) with snake ordering.
    """
    if teams <= 0 or overall_pick <= 0:
        return (0, 0, 0)
    round_num = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    if round_num % 2 == 0:
        draft_slot = teams - pick_in_round + 1
    else:
        draft_slot = pick_in_round
    return (round_num, pick_in_round, draft_slot)

def slot_to_display_name(slot: int, users: list) -> str:
    """
    Map Sleeper roster_id/slot to a display name if available.
    """
    for u in users or []:
        if str(u.get("roster_id")) == str(slot):
            return u.get("display_name") or u.get("username")
    return ""

# ---- Data helpers ----
HEADER_ALIASES = {
    "PLAYER": ["PLAYER", "PLAYER NAME", "NAME"],
    "TEAM": ["TEAM", "NFL TEAM"],
    "POS": ["POS", "POSITION"],
    "BYE": ["BYE", "BYE WEEK"],
    "RK": ["RK", "RANK", "ECR"],
    "TIERS": ["TIERS", "TIER"],
    "SOS": ["SOS", "SOS SEASON", "STARS"],
    "ADP": ["ADP"],
}

def normalize_player_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common fantasy CSV headers and guarantee required columns.
    Returns a dataframe with at least: RK, TIERS, PLAYER, TEAM, POS, BYE, SOS, ADP
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["RK", "TIERS", "PLAYER", "TEAM", "POS", "BYE", "SOS", "ADP"])
    # strip/standardize column names
    df = df.rename(columns=lambda c: str(c).strip())
    # map known aliases to canonical names
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                df = df.rename(columns={a: canon})
                break
    # ensure required columns exist
    for c in ["RK", "TIERS", "PLAYER", "TEAM", "POS", "BYE"]:
        if c not in df.columns:
            df[c] = "" if c == "PLAYER" else 0
    # helpful optional columns
    for c in ["SOS", "ADP"]:
        if c not in df.columns:
            df[c] = 0
    # coercions
    for c in ["RK", "BYE", "ADP", "SOS"]:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass
    return df
