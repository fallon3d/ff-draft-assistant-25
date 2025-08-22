"""
Utilities: config, CSV ingest/normalize, name helpers, and snake math.
"""

from __future__ import annotations
import os
import io
import toml
import pandas as pd
from typing import Dict, Any, Iterable, List, Set

ROOT = os.path.dirname(os.path.dirname(__file__))
CONF_PATH = os.path.join(ROOT, "config.toml")

# ---------------- Config ----------------
def read_config() -> Dict[str, Any]:
    try:
        with open(CONF_PATH, "r", encoding="utf-8") as f:
            return toml.load(f)
    except Exception:
        return {}

def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONF_PATH, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)

# ---------------- CSV/XLSX ingest ----------------
_HEADER_MAP = {
    "player": "PLAYER",
    "player name": "PLAYER",
    "name": "PLAYER",
    "team": "TEAM",
    "pos": "POS",
    "position": "POS",
    "rk": "RK",
    "rank": "RK",
    "tiers": "TIERS",
    "tier": "TIERS",
    "bye": "BYE",
    "bye week": "BYE",
    "ecr vs. adp": "ECR VS. ADP",
    "ecr_vs_adp": "ECR VS. ADP",
    "injury": "INJURY_RISK",
    "injury risk": "INJURY_RISK",
    "volatility": "VOLATILITY",
    "oc": "OC",
    "offense coordinator": "OC",
    "offensive coordinator": "OC",
    "hc": "HC",
    "head coach": "HC",
    "sos season": "SOS SEASON",
    "sos": "SOS SEASON",
    "proj_pts": "PROJ_PTS",
    "proj pass yds": "PROJ_PASS_YDS",
    "proj pass td": "PROJ_PASS_TD",
    "proj rush yds": "PROJ_RUSH_YDS",
    "proj rush td": "PROJ_RUSH_TD",
    "proj rec": "PROJ_REC",
    "proj rec yds": "PROJ_REC_YDS",
    "proj rec td": "PROJ_REC_TD",
    "proj fg": "PROJ_FG",
    "proj xp": "PROJ_XP",
    "proj sacks": "PROJ_SACKS",
    "proj turnovers": "PROJ_TURNOVERS",
    "proj points allowed": "PROJ_POINTS_ALLOWED",
    "handcuff to": "HANDCUFF_TO",
}

def _read_any_table(path_or_buf) -> pd.DataFrame:
    if isinstance(path_or_buf, (str, os.PathLike)):
        p = str(path_or_buf)
        if p.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(p)
        return pd.read_csv(p)
    # Uploaded file-like (bytes)
    data = path_or_buf
    try:
        return pd.read_excel(io.BytesIO(data))
    except Exception:
        return pd.read_csv(io.BytesIO(data))

def normalize_player_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        key = str(c).strip().lower()
        cols.append(_HEADER_MAP.get(key, str(c).strip().upper()))
    df = df.copy()
    df.columns = cols
    # Ensure expected columns exist
    for c in ["PLAYER","TEAM","POS","RK","TIERS","BYE","ECR VS. ADP","INJURY_RISK","VOLATILITY","OC","HC","SOS SEASON",
              "PROJ_PTS","PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_RUSH_YDS","PROJ_RUSH_TD","PROJ_REC","PROJ_REC_YDS",
              "PROJ_REC_TD","PROJ_FG","PROJ_XP","PROJ_SACKS","PROJ_TURNOVERS","PROJ_POINTS_ALLOWED","HANDCUFF_TO"]:
        if c not in df.columns:
            df[c] = pd.NA
    # Canonicalize positions (DST synonyms)
    df["POS"] = df["POS"].astype(str).str.upper().str.replace("DEFENSE","DST").str.replace("DEF","DST").str.replace("D/ST","DST")
    # BYE to int-ish
    df["BYE"] = pd.to_numeric(df["BYE"], errors="coerce").fillna(0).astype(int)
    return df

def read_player_table(path_or_file) -> pd.DataFrame:
    try:
        df = _read_any_table(path_or_file)
    except Exception:
        return pd.DataFrame(columns=["PLAYER","TEAM","POS"])
    return normalize_player_headers(df)

def remove_players_by_name(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty: return df
    s = {str(n).strip().lower() for n in names if n}
    return df[~df["PLAYER"].astype(str).str.lower().isin(s)].reset_index(drop=True)

def lookup_bye_weeks(universe: pd.DataFrame, picked_names: Iterable[str]) -> Set[int]:
    if universe is None or universe.empty: return set()
    m = universe.set_index("PLAYER")["BYE"].to_dict()
    out = set()
    for n in picked_names:
        try:
            v = int(m.get(n, 0))
            if v: out.add(v)
        except Exception:
            continue
    return out

# ---------------- Sleeper helpers ----------------
def user_roster_id(users: List[dict], username: str) -> int | None:
    if not users: return None
    uname = str(username or "").strip().lower()
    for u in users:
        dn = str(u.get("display_name") or "").strip().lower()
        if dn == uname:
            rid = u.get("roster_id") or u.get("draft_slot")
            try:
                return int(rid)
            except Exception:
                return None
    return None

def slot_to_display_name(slot: int, users: List[dict]) -> str:
    if not users:
        return f"Team {slot}"
    for u in users:
        rid = u.get("roster_id") or u.get("draft_slot")
        try:
            if int(rid) == int(slot):
                return str(u.get("display_name") or f"Team {slot}")
        except Exception:
            continue
    return f"Team {slot}"

# ---------------- Snake draft math ----------------
def snake_position(overall: int, teams: int) -> tuple[int,int,int]:
    """Return (round, pick_in_round, slot)."""
    overall = int(overall); teams = int(teams)
    rnd = (overall - 1) // teams + 1
    pick_in_round = (overall - 1) % teams + 1
    if rnd % 2 == 1:
        slot = pick_in_round
    else:
        slot = teams - pick_in_round + 1
    return rnd, pick_in_round, slot

def slot_for_overall(overall: int, teams: int) -> int:
    return snake_position(int(overall), int(teams))[2]

def slot_for_round_pick(round_number: int, pick_in_round: int, teams: int) -> int:
    if round_number % 2 == 1:
        return int(pick_in_round)
    return int(teams) - int(pick_in_round) + 1

def next_pick_overall(current_overall: int, teams: int, user_slot: int) -> int:
    """
    Find the next overall pick number for 'user_slot' strictly AFTER current_overall.
    """
    teams = int(teams); user_slot = int(user_slot)
    # iterate forward rounds until we find the next slot occurrence
    r0, _, _ = snake_position(current_overall, teams)
    for r in range(r0, r0 + 200):  # hard cap
        pr = user_slot if r % 2 == 1 else teams - user_slot + 1
        overall = (r - 1) * teams + pr
        if overall > current_overall:
            return overall
    return current_overall + teams  # fallback

def picks_until_next_turn(current_overall: int, teams: int, user_slot: int) -> int:
    nxt = next_pick_overall(current_overall, teams, user_slot)
    return max(0, int(nxt) - int(current_overall) - 1)
