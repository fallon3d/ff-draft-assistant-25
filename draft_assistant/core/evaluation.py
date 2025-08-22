"""
Evaluation: compute projections, apply modifiers, compute VBD, and return
a player table with:
  - 'value' (final adjusted projection proxy)
  - 'vbd'   (value-based-drafting vs positional baseline)
  - tags (for downstream 'why' text)
"""

from __future__ import annotations
from typing import Dict, Any
import math
import pandas as pd

DEFAULT_STARTERS = {"QB":1, "RB":2, "WR":3, "TE":1, "K":1, "DST":1}

# Default scoring weights (season-long)
DEFAULT_SCORING = {
    "ppr": 1.0,
    "pass_yd": 0.04, "pass_td": 4.0, "int": -2.0,      # interceptions not in sheet; ignore unless provided later
    "rush_yd": 0.10, "rush_td": 6.0,
    "rec": 1.0, "rec_yd": 0.10, "rec_td": 6.0,
    "fg": 3.0, "xp": 1.0,
    "sack": 1.0, "turnover": 2.0,                      # D/ST takeaways
    # Points allowed buckets not modeled exactly; we use linear proxy via PROJ_POINTS_ALLOWED (downward)
}

SOS_MULT = {1: 0.92, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.08}  # ±8%

INJURY_PEN = {"low": 0.0, "medium": -0.03, "high": -0.06}
VOLATILITY_BONUS_LATE = {"low": 0.00, "medium": 0.02, "high": 0.04}

def _coalesce(*vals, default=0.0) -> float:
    for v in vals:
        try:
            if pd.notna(v):
                return float(v)
        except Exception:
            pass
    return float(default)

def _proj_points_row(row: pd.Series, scoring: Dict[str, float]) -> float:
    """Compute projection from PROJ_* fields; fall back to given PROJ_PTS."""
    # QB/pass
    pts = 0.0
    pts += _coalesce(row.get("PROJ_PASS_YDS")) * scoring["pass_yd"]
    pts += _coalesce(row.get("PROJ_PASS_TD")) * scoring["pass_td"]
    # rush
    pts += _coalesce(row.get("PROJ_RUSH_YDS")) * scoring["rush_yd"]
    pts += _coalesce(row.get("PROJ_RUSH_TD")) * scoring["rush_td"]
    # rec
    pts += _coalesce(row.get("PROJ_REC")) * scoring["rec"]
    pts += _coalesce(row.get("PROJ_REC_YDS")) * scoring["rec_yd"]
    pts += _coalesce(row.get("PROJ_REC_TD")) * scoring["rec_td"]
    # K
    pts += _coalesce(row.get("PROJ_FG")) * scoring["fg"]
    pts += _coalesce(row.get("PROJ_XP")) * scoring["xp"]
    # D/ST
    pts += _coalesce(row.get("PROJ_SACKS")) * scoring["sack"]
    pts += _coalesce(row.get("PROJ_TURNOVERS")) * scoring["turnover"]
    # Penalize points allowed linearly (not bucketed)
    pts += -0.02 * _coalesce(row.get("PROJ_POINTS_ALLOWED"))  # mild penalty
    # If sheet already has PROJ_PTS, blend toward it in evaluation() using weight_proj
    return pts

def _baseline_index(pos: str, teams: int, starters: Dict[str,int]) -> int:
    return max(1, teams * int(starters.get(pos, 0)))

def _compute_vbd(df: pd.DataFrame, teams: int, starters: Dict[str,int]) -> pd.Series:
    vbd = pd.Series([0.0]*len(df), index=df.index, dtype=float)
    for pos in ("QB","RB","WR","TE","K","DST"):
        sub = df[df["POS"] == pos].sort_values("value_raw", ascending=False)
        if sub.empty:
            continue
        idx = _baseline_index(pos, teams, starters)  # e.g., RB baseline = teams*2
        idx = min(idx, len(sub))
        baseline_val = float(sub.iloc[idx-1]["value_raw"])
        vbd.loc[sub.index] = sub["value_raw"] - baseline_val
    return vbd

def evaluate_players(
    df_in: pd.DataFrame,
    config: Dict[str, Any],
    teams: int,
    rounds: int,
    weight_proj: float = 0.65
) -> pd.DataFrame:
    """
    Return df with computed columns:
      - value_raw: projection from components (or PROJ_PTS)
      - value: blended (weight_proj * calc_from_components + (1-weight)*PROJ_PTS) then modifiers
      - vbd: value - baseline per position
      - tags: lightweight list for downstream reasons
    """
    if df_in is None or df_in.empty:
        return df_in

    df = df_in.copy()

    scoring_cfg = dict(DEFAULT_SCORING)
    scoring_cfg["ppr"] = float(config.get("scoring", {}).get("ppr", scoring_cfg["ppr"]))

    # raw projection from components
    df["calc_proj"] = df.apply(lambda r: _proj_points_row(r, scoring_cfg), axis=1)
    df["proj_pts"] = df["PROJ_PTS"] if "PROJ_PTS" in df.columns else 0.0
    df["value_raw"] = df["calc_proj"]  # before modifiers

    # Blend with provided PROJ_PTS if present
    if "PROJ_PTS" in df.columns:
        df["value_blend"] = weight_proj * df["calc_proj"] + (1.0 - weight_proj) * df["PROJ_PTS"]
    else:
        df["value_blend"] = df["calc_proj"]

    # Apply schedule, injury, small coach/OC nudge
    sos_mult = df.get("SOS SEASON")
    df["sos_mult"] = sos_mult.map(lambda s: SOS_MULT.get(int(s), 1.0) if pd.notna(s) else 1.0) if sos_mult is not None else 1.0

    def _inj_pen(v):
        s = str(v or "").strip().lower()
        return INJURY_PEN.get(s, 0.0)
    df["inj_mult"] = df.get("INJURY_RISK", pd.Series([""]*len(df))).map(_inj_pen).fillna(0.0)

    # small coach/OC heuristic: if OC present, tiny +1% (data-informed confidence)
    df["coach_mult"] = 0.01 * ((~df.get("OC", pd.Series([None]*len(df))).isna()).astype(float))

    # Final value before VBD
    df["value"] = df["value_blend"] * df["sos_mult"] * (1.0 + df["coach_mult"]) * (1.0 + df["inj_mult"])

    # Compute VBD (per position baseline)
    starters_cfg = dict(DEFAULT_STARTERS)
    starters_cfg.update(config.get("starters", {}))  # allow overrides in config if present
    df["vbd"] = _compute_vbd(df, teams=teams, starters=starters_cfg)

    # Clean columns for downstream
    for col in ["calc_proj","value_blend","sos_mult","inj_mult","coach_mult"]:
        df[col] = df[col].astype(float)

    return df
