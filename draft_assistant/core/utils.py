"""
Utilities: config I/O, player table normalization, draft math, simple helpers
"""

from __future__ import annotations
import os
import re
import toml
import pandas as pd

# Try project-root config.toml first; fallback to draft_assistant/config.toml
_CORE_DIR = os.path.dirname(__file__)
_DA_DIR = os.path.abspath(os.path.join(_CORE_DIR, ".."))
_ROOT_DIR = os.path.abspath(os.path.join(_DA_DIR, ".."))
CONFIG_PATHS = [
    os.path.join(_ROOT_DIR, "config.toml"),
    os.path.join(_DA_DIR, "config.toml"),
]

def _pick_config_path() -> str:
    for p in CONFIG_PATHS:
        if os.path.exists(p):
            return p
    # default to project-root path
    return CONFIG_PATHS[0]

# -------- Config --------
def read_config() -> dict:
    path = _pick_config_path()
    try:
        return toml.load(path)
    except Exception:
        return {}

def save_config(cfg: dict) -> None:
    path = _pick_config_path()
    with open(path, "w") as f:
        toml.dump(cfg, f)

# -------- Headers --------
def normalize_player_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from user CSV/XLSX to the app schema.
    Supports:
      - HEAD COACH -> COACH
      - OFFENSE COORDINATOR -> OC
      - BYE WEEK/BYE mirrored
    Uppercases TEAM/POS, maps DEF variants to DST, coerces ROOKIE->0/1.
    """
    if df is None or df.empty:
        return df

    def canon(s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"[\s\-_/]+", " ", s)
        return s

    alias_map = {
        "RK": {"rk", "overall rank", "rank", "overall"},
        "TIERS": {"tier", "tiers", "tier rank"},
        "PLAYER": {"player", "player name", "name", "player_name", "playername"},
        "TEAM": {"team", "nfl team", "tm"},
        "POS": {"pos", "position"},
        "BYE WEEK": {"bye week", "bye wk", "byeweek"},
        "BYE": {"bye", "bye wk", "byeweek"},
        "ECR": {"ecr", "consensus rank"},
        "ADP": {"adp", "avg draft pos", "average draft position"},
        "ECR VS. ADP": {"ecr vs. adp", "ecr-adp", "ecr_vs_adp", "ecr delta"},
        "SOS SEASON": {"sos season", "sos", "strength of schedule"},
        "PROJ_PTS": {"proj pts", "projected points", "points proj", "projection"},
        "PROJ_PASS_YDS": {"proj pass yds", "pass yds proj", "pass yards"},
        "PROJ_PASS_TD": {"proj pass td", "pass td proj", "pass tds"},
        "PROJ_RUSH_YDS": {"proj rush yds", "rush yds proj", "rush yards"},
        "PROJ_RUSH_TD": {"proj rush td", "rush td proj", "rush tds"},
        "PROJ_REC": {"proj rec", "receptions proj", "receptions"},
        "PROJ_REC_YDS": {"proj rec yds", "receiving yds proj", "rec yards"},
        "PROJ_REC_TD": {"proj rec td", "receiving td proj", "rec tds"},
        "PROJ_FG": {"proj fg", "field goals proj"},
        "PROJ_XP": {"proj xp", "extra points proj"},
        "PROJ_SACKS": {"proj sacks", "sacks proj"},
        "PROJ_TURNOVERS": {"proj turnovers", "turnovers proj", "takeaways"},
        "PROJ_POINTS_ALLOWED": {"proj points allowed", "points allowed proj"},
        "AGE": {"age"},
        "ROOKIE": {"rookie", "is rookie"},
        "INJURY_RISK": {"injury risk", "injury", "injury_grade"},
        "VOLATILITY": {"volatility", "consistency"},
        "HANDCUFF_TO": {"handcuff to", "handcuff", "backs up"},
        "TEAM_TENDENCY": {"team tendency", "tendency", "offense tendency"},
        "COACH": {"coach", "head coach", "hc"},
        "OC": {"offense coordinator", "offensive coordinator", "oc", "off coord", "offense coord"},
        "STATUS": {"status"},
        "NOTES": {"notes", "comment"},
    }

    def build_map():
        canon_to_std = {}
        for std, aliases in alias_map.items():
            for a in aliases:
                canon_to_std[canon(a)] = std
            canon_to_std[canon(std)] = std
        return canon_to_std

    canon_to_std = build_map()
    rename_map = {}
    for col in list(df.columns):
        c = canon(col)
        if c in canon_to_std:
            rename_map[col] = canon_to_std[c]
        elif c == "player name":
            rename_map[col] = "PLAYER"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Mirror BYE/BYE WEEK
    if "BYE WEEK" in df.columns and "BYE" not in df.columns:
        df["BYE"] = df["BYE WEEK"]
    elif "BYE" in df.columns and "BYE WEEK" not in df.columns:
        df["BYE WEEK"] = df["BYE"]

    # Normalize POS/TEAM
    if "POS" in df.columns:
        df["POS"] = (
            df["POS"].astype(str).str.upper()
              .str.replace(r"^\s*D[/\-]?ST\s*$", "DST", regex=True)
              .str.replace(r"^\s*DEF(ENSE)?\s*$", "DST", regex=True)
              .str.strip()
        )
    if "TEAM" in df.columns:
        df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip()

    # ROOKIE -> 0/1
    if "ROOKIE" in df.columns:
        def _rook(v):
            s = str(v).strip().lower()
            if s in {"1","true","yes","y"}: return 1
            if s in {"0","false","no","n",""}: return 0
            return 1 if "rook" in s else 0
        df["ROOKIE"] = df["ROOKIE"].map(_rook)

    # Ensure essential columns exist
    for col in ["PLAYER","TEAM","POS"]:
        if col not in df.columns: df[col] = None
    for col in ["BYE","BYE WEEK"]:
        if col not in df.columns: df[col] = None

    preferred = [
        "RK","TIERS","PLAYER","TEAM","POS","BYE WEEK","BYE",
        "ECR","ADP","ECR VS. ADP","SOS SEASON","PROJ_PTS",
        "PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_RUSH_YDS","PROJ_RUSH_TD",
        "PROJ_REC","PROJ_REC_YDS","PROJ_REC_TD",
        "PROJ_FG","PROJ_XP","PROJ_SACKS","PROJ_TURNOVERS","PROJ_POINTS_ALLOWED",
        "AGE","ROOKIE","INJURY_RISK","VOLATILITY","HANDCUFF_TO",
        "TEAM_TENDENCY","COACH","OC","STATUS","NOTES",
    ]
    ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[ordered]

# -------- Loader (CSV/XLSX) --------
def read_player_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = normalize_player_headers(df)

    numeric_cols = [
        "RK","TIERS","BYE","BYE WEEK","ECR","ADP","ECR VS. ADP","SOS SEASON","PROJ_PTS",
        "PROJ_PASS_YDS","PROJ_PASS_TD","PROJ_RUSH_YDS","PROJ_RUSH_TD",
        "PROJ_REC","PROJ_REC_YDS","PROJ_REC_TD","PROJ_FG","PROJ_XP",
        "PROJ_SACKS","PROJ_TURNOVERS","PROJ_POINTS_ALLOWED","AGE","ROOKIE",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "PLAYER" in df.columns:
        df["PLAYER"] = df["PLAYER"].astype(str).str.strip()

    return df

# -------- Draft math --------
def snake_position(overall_pick: int, teams: int) -> tuple[int,int,int]:
    r = (overall_pick - 1) // teams + 1
    pos_in_round = (overall_pick - 1) % teams + 1
    slot = pos_in_round if (r % 2 == 1) else (teams - pos_in_round + 1)
    return r, pos_in_round, slot

def slot_for_overall(overall_pick: int, teams: int) -> int:
    return snake_position(overall_pick, teams)[2]

def picks_until_next_turn(current_overall: int, teams: int, user_slot: int) -> int:
    r, _, slot = snake_position(current_overall, teams)
    # If you pick later in this round:
    if (r % 2 == 1 and slot < user_slot) or (r % 2 == 0 and slot > user_slot):
        return abs(user_slot - slot)
    # Otherwise finish round, then count to your next turn
    to_end = teams - slot
    to_next = (user_slot - 1) if r % 2 == 1 else (teams - user_slot)
    return to_end + to_next + 1

def next_pick_overall(start_overall_after_current: int, teams: int, user_slot: int) -> int:
    n1 = start_overall_after_current
    return n1 + picks_until_next_turn(n1, teams, user_slot) + 1

def user_roster_id(users: list[dict], username: str) -> int | None:
    if not users: return None
    for u in users:
        if str(u.get("display_name","")).strip().lower() == str(username).strip().lower():
            return int(u.get("roster_id") or 1)
    return None

def slot_to_display_name(slot: int, users: list[dict]) -> str:
    if not users: return f"Slot {slot}"
    for u in users:
        if int(u.get("roster_id", -1)) == int(slot):
            nm = u.get("metadata", {}).get("team_name") or u.get("display_name") or f"Slot {slot}"
            return str(nm)
    return f"Slot {slot}"

def remove_players_by_name(df: pd.DataFrame, names: set[str]) -> pd.DataFrame:
    if not names or df is None or df.empty:
        return df
    mask = ~df["PLAYER"].isin(names)
    return df.loc[mask].reset_index(drop=True)

def lookup_bye_weeks(full_df: pd.DataFrame, names: list[str]) -> set[int]:
    if not names or full_df is None or full_df.empty or "BYE" not in full_df.columns:
        return set()
    m = full_df.set_index("PLAYER")["BYE"].to_dict()
    out = set()
    for n in names:
        v = m.get(n)
        if pd.notna(v):
            try: out.add(int(v))
            except Exception: pass
    return out
