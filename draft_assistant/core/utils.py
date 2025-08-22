"""
Utility helpers: config, snake math, header normalization, combined pool,
Sleeper user helpers, and exact distance to next pick in a snake draft.
"""
import os
import re
import toml
import pandas as pd
from typing import List, Optional, Tuple

def read_config(path: str = os.path.join(os.path.dirname(__file__), "..", "config.toml")) -> dict:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        return {}
    return toml.load(path)

def save_config(config: dict, path: str = os.path.join(os.path.dirname(__file__), "..", "config.toml")) -> None:
    path = os.path.abspath(path)
    with open(path, "w") as f:
        toml.dump(config, f)

# ---------- Snake draft math ----------
def snake_position(overall_pick: int, teams: int) -> Tuple[int,int,int]:
    if teams <= 0 or overall_pick <= 0:
        return (0, 0, 0)
    round_num = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    draft_slot = teams - pick_in_round + 1 if round_num % 2 == 0 else pick_in_round
    return (round_num, pick_in_round, draft_slot)

def slot_for_overall(overall_pick: int, teams: int) -> int:
    return snake_position(overall_pick, teams)[2]

def _pick_index_for_slot(round_num: int, teams: int, user_slot: int) -> int:
    """1-indexed pick index within a round for a given slot."""
    if round_num % 2 == 1:  # odd rounds: 1..teams
        return user_slot
    else:  # even rounds: teams..1
        return teams - user_slot + 1

def next_pick_overall(current_overall: int, teams: int, user_slot: int) -> int:
    """Return the next overall number when the given slot will pick again."""
    r, pick_in_round, _ = snake_position(current_overall, teams)
    user_idx_now = _pick_index_for_slot(r, teams, user_slot)
    if pick_in_round < user_idx_now:
        # later this round
        return (r - 1) * teams + user_idx_now
    else:
        # next round
        r_next = r + 1
        user_idx_next = _pick_index_for_slot(r_next, teams, user_slot)
        return (r) * teams + user_idx_next

def picks_until_next_turn(current_overall: int, teams: int, user_slot: int) -> int:
    """How many picks until the user's next pick (0 if now)."""
    nxt = next_pick_overall(current_overall, teams, user_slot)
    return max(0, nxt - current_overall)

# ---------- Sleeper helpers ----------
def slot_to_display_name(slot: int, users: list) -> str:
    for u in users or []:
        if str(u.get("roster_id")) == str(slot):
            return u.get("display_name") or u.get("username") or f"Slot {slot}"
    return f"Slot {slot}"

def user_roster_id(users: list, username: str) -> int | None:
    if not users or not username:
        return None
    for u in users:
        names = [(u.get("display_name") or "").lower(), (u.get("username") or "").lower()]
        if username.lower() in names:
            try:
                return int(u.get("roster_id"))
            except Exception:
                return None
    return None

def roster_display_name(users: list, roster_id: int) -> str:
    for u in users or []:
        if str(u.get("roster_id")) == str(roster_id):
            return u.get("display_name") or u.get("username") or f"Slot {roster_id}"
    return f"Slot {roster_id}"

# ---------- Header normalization & merging ----------
HEADER_ALIASES = {
    "PLAYER": ["PLAYER", "PLAYER NAME", "NAME"],
    "TEAM": ["TEAM", "NFL TEAM"],
    "POS": ["POS", "POSITION"],
    "BYE": ["BYE", "BYE WEEK"],
    "RK": ["RK", "RANK", "ECR", "Overall Rank"],
    "TIERS": ["TIERS", "TIER"],
    "SOS": ["SOS", "SOS SEASON", "STARS", "Strength of Schedule"],
    "ADP": ["ADP", "Avg. Draft Position", "Average Draft Position"],
    "ROOKIE": ["ROOKIE"],  # optional hint
}

def _normalize_pos_value(x: str) -> str:
    s = str(x).strip().upper()
    if s in ("DEF", "DST", "D/ST", "D-ST", "TEAM DEF", "TEAM D", "DEFENSE"):
        return "DST"
    return s

def normalize_player_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["RK","TIERS","PLAYER","TEAM","POS","BYE","SOS","ADP","ROOKIE"])
    df = df.rename(columns=lambda c: str(c).strip())
    for canon, aliases in HEADER_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                df = df.rename(columns={a: canon})
                break
    for c in ["RK","TIERS","PLAYER","TEAM","POS","BYE"]:
        if c not in df.columns:
            df[c] = "" if c == "PLAYER" else 0
    for c in ["SOS","ADP","ROOKIE"]:
        if c not in df.columns:
            df[c] = 0
    for c in ["RK","BYE","ADP","SOS"]:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        except Exception:
            pass
    # Normalize DEF/DST naming
    df["POS"] = df["POS"].apply(_normalize_pos_value)
    return df

# ---- Name normalization & merging ----
_SUFFIXES = (" jr", " sr", " iii", " ii", " iv")
def normalize_name(name: str) -> str:
    if not name: return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"\s+", " ", s).strip()
    return s

def merge_player_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [normalize_player_headers(f) for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=["RK","TIERS","PLAYER","TEAM","POS","BYE","SOS","ADP","ROOKIE"])
    out = pd.concat(frames, ignore_index=True)
    out["__norm_name__"] = out["PLAYER"].apply(normalize_name)
    out["TEAM"] = out["TEAM"].fillna("").astype(str)
    out["POS"] = out["POS"].fillna("").astype(str)
    out = out.drop_duplicates(subset=["__norm_name__","TEAM","POS"], keep="first").drop(columns="__norm_name__")
    return out.reset_index(drop=True)

def remove_players_by_name(df: pd.DataFrame, picked_names: List[str]) -> pd.DataFrame:
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

def load_combined_player_pool(include_sleeper: bool = True, base_path: Optional[str] = None, extra_paths: Optional[List[str]] = None) -> pd.DataFrame:
    from . import sleeper as _sleeper
    frames = []
    if base_path: frames.append(_read_csv_if_exists(base_path))
    for p in (extra_paths or []): frames.append(_read_csv_if_exists(p))
    if include_sleeper:
        sdict = _sleeper.get_players_nfl() or {}
        s_df = _sleeper.players_df_from_dict(sdict)
        frames.append(s_df)
    return merge_player_frames([f for f in frames if f is not None and not f.empty])
