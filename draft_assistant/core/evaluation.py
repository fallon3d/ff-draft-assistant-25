"""
Evaluation: compute projections, apply modifiers, compute VBD.
"""

from __future__ import annotations
from typing import Dict, Any
import pandas as pd

DEFAULT_STARTERS = {"QB":1, "RB":2, "WR":3, "TE":1, "K":1, "DST":1}

DEFAULT_SCORING = {
    "ppr": 1.0,
    "pass_yd": 0.04, "pass_td": 4.0,
    "rush_yd": 0.10, "rush_td": 6.0,
    "rec": 1.0, "rec_yd": 0.10, "rec_td": 6.0,
    "fg": 3.0, "xp": 1.0,
    "sack": 1.0, "turnover": 2.0,
}

SOS_MULT = {1: 0.92, 2: 0.96, 3: 1.00, 4: 1.04, 5: 1.08}  # ±8%
INJURY_PEN = {"low": 0.0, "medium": -0.03, "high": -0.06}

def _coalesce(*vals, default=0.0) -> float:
    for v in vals:
        try:
            if pd.notna(v): return float(v)
        except Exception:
            pass
    return float(default)

def _proj_points_row(row: pd.Series, scoring: Dict[str, float]) -> float:
    pts = 0.0
    pts += _coalesce(row.get("PROJ_PASS_YDS")) * scoring["pass_yd"]
    pts += _coalesce(row.get("PROJ_PASS_TD")) * scoring["pass_td"]
    pts += _coalesce(row.get("PROJ_RUSH_YDS")) * scoring["rush_yd"]
    pts += _coalesce(row.get("PROJ_RUSH_TD")) * scoring["rush_td"]
    pts += _coalesce(row.get("PROJ_REC")) * scoring["rec"]
    pts += _coalesce(row.get("PROJ_REC_YDS")) * scoring["rec_yd"]
    pts += _coalesce(row.get("PROJ_REC_TD")) * scoring["rec_td"]
    pts += _coalesce(row.get("PROJ_FG")) * scoring["fg"]
    pts += _coalesce(row.get("PROJ_XP")) * scoring["xp"]
    pts += _coalesce(row.get("PROJ_SACKS")) * scoring["sack"]
    pts += _coalesce(row.get("PROJ_TURNOVERS")) * scoring["turnover"]
    pts += -0.02 * _coalesce(row.get("PROJ_POINTS_ALLOWED"))
    return pts

def _baseline_index(pos: str, teams: int, starters: Dict[str,int]) -> int:
    # Baseline = last starter at the position (e.g., RB24 in 12-team 2-RB starters)
    return max(1, teams * int(starters.get(pos, 0)))

def _compute_vbd(df: pd.DataFrame, teams: int, starters: Dict[str,int]) -> pd.Series:
    """
    Compute VBD against the baseline *after* all value modifiers.
    """
    vbd = pd.Series(0.0, index=df.index, dtype=float)
    # Ensure the 'value' column exists
    vals = df["value"] if "value" in df.columns else df.get("value_blend", pd.Series(0.0, index=df.index))
    tmp = df.assign(_val=vals)

    for pos in ("QB","RB","WR","TE","K","DST"):
        sub = tmp[tmp["POS"] == pos].sort_values("_val", ascending=False)
        if sub.empty:
            continue
        idx = min(_baseline_index(pos, teams, starters), len(sub))
        baseline = float(sub.iloc[idx-1]["_val"])
        # <- this was the bug; keep only the bracketed assignment:
        vbd.loc[sub.index] = sub["_val"] - baseline

    return vbd

def evaluate_players(
    df_in: pd.DataFrame,
    config: Dict[str, Any],
    teams: int,
    rounds: int,
    weight_proj: float = 0.65
) -> pd.DataFrame:
    if df_in is None or df_in.empty: return df_in
    df = df_in.copy()

    scoring_cfg = dict(DEFAULT_SCORING)
    scoring_cfg["ppr"] = float(config.get("scoring", {}).get("ppr", scoring_cfg["ppr"]))

    # Base projections and blended value (if PROJ_PTS provided)
    df["calc_proj"] = df.apply(lambda r: _proj_points_row(r, scoring_cfg), axis=1)
    df["proj_pts"] = df["PROJ_PTS"] if "PROJ_PTS" in df.columns else 0.0
    df["value_raw"] = df["calc_proj"]

    if "PROJ_PTS" in df.columns:
        df["value_blend"] = weight_proj * df["calc_proj"] + (1.0 - weight_proj) * df["PROJ_PTS"]
    else:
        df["value_blend"] = df["calc_proj"]

    # Multipliers
    if "SOS SEASON" in df.columns:
        df["sos_mult"] = df["SOS SEASON"].map(lambda s: SOS_MULT.get(int(s), 1.0) if pd.notna(s) else 1.0)
    else:
        df["sos_mult"] = 1.0

    def _inj_pen(v):
        s = str(v or "").strip().lower()
        return INJURY_PEN.get(s, 0.0)
    df["inj_mult"] = df.get("INJURY_RISK", pd.Series([""]*len(df))).map(_inj_pen).fillna(0.0)

    # Tiny OC confidence bump when OC present
    df["coach_mult"] = 0.01 * ((~df.get("OC", pd.Series([None]*len(df))).isna()).astype(float))

    # Final value before VBD
    df["value"] = df["value_blend"] * df["sos_mult"] * (1.0 + df["coach_mult"]) * (1.0 + df["inj_mult"])

    # Starters config + VBD
    starters_cfg = dict(DEFAULT_STARTERS)
    starters_cfg.update(config.get("starters", {}))
    df["vbd"] = _compute_vbd(df, teams=teams, starters=starters_cfg)

    # Ensure numeric types
    for col in ["calc_proj","value_blend","sos_mult","inj_mult","coach_mult","value","vbd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
