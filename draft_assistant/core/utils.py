"""
Utility helpers.
"""
import os
import re
import toml
import pandas as pd
from typing import List, Optional

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

# ---- Header normalization ----
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
    Returns: RK, TIERS, PLAYER, TEAM, POS, BYE, SOS, ADP (at least)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["RK", "TIERS", "PLAYER", "TEAM", "POS", "BYE", "SOS", "ADP"])
    df = df.rename(columns=lambda c: str(c).strip())
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                df = df.rename(columns={a: canon})
                break
    for c in ["RK", "TIERS", "PLAYER", "TEAM", "POS", "BYE"]:
        if c not in df.columns:
            df[c] = "" if c == "PLAYER" else 0
    for c in ["SOS", "ADP"]:
        if c not in df.columns:
            df[c] = 0
    for c in ["RK", "BYE", "ADP", "SOS"]:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass
    return df

# ---- Name normalization & merging ----
_SUFFIXES = (" jr", " sr", " iii", " ii", " iv")

def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)  # remove punctuation
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"\s+", " ", s).strip()
    return s

def merge_player_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge and de-duplicate by (norm_name, TEAM, POS).
    Prefer earlier frames (left-most wins).
    """
    frames = [normalize_player_headers(f) for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["RK", "TIERS", "PLAYER", "TEAM", "POS", "BYE", "SOS", "ADP"])
    out = pd.concat(frames, ignore_index=True)
    out["__norm_name__"] = out["PLAYER"].apply(normalize_name)
    out["TEAM"] = out["TEAM"].fillna("").astype(str)
    out["POS"] = out["POS"].fillna("").astype(str)
    # Keep first occurrence per key
    out = out.drop_duplicates(subset=["__norm_name__", "TEAM", "POS"], keep="first").drop(columns="__norm_name__")
    return out.reset_index(drop=True)

def remove_players_by_name(df: pd.DataFrame, picked_names: List[str]) -> pd.DataFrame:
    """
    Remove players from df if their normalized names appear in picked_names.
    """
    if df is None or df.empty or not picked_names:
        return df
    picked_norm = {normalize_name(n) for n in picked_names if n}
    df = df.copy()
    df["__norm_name__"] = df["PLAYER"].apply(normalize_name)
    out = df[~df["__norm_name__"].isin(picked_norm)].drop(columns="__norm_name__", errors="ignore")
    return out.reset_index(drop=True)

def _read_csv_if_exists(path: str) -> pd.DataFrame:
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def load_combined_player_pool(
    include_sleeper: bool = True,
    base_path: Optional[str] = None,
    extra_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a combined pool from:
      - base CSV (sample_players.csv)
      - up to 3 extra CSVs
      - Sleeper live players (optional)
    Sleeper rows are ranked low (RK=999) so they don't override your ranks.
    """
    from . import sleeper as _sleeper

    frames = []
    if base_path:
        frames.append(_read_csv_if_exists(base_path))
    for p in (extra_paths or []):
        frames.append(_read_csv_if_exists(p))

    if include_sleeper:
        sdict = _sleeper.get_players_nfl() or {}
        s_df = _sleeper.players_df_from_dict(sdict)
        frames.append(s_df)

    merged = merge_player_frames([f for f in frames if f is not None and not f.empty])
    return merged
