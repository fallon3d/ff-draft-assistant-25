"""
Player evaluation and VBD scoring.

- Safe handling of pd.NA / NaN everywhere (no truthiness on NA).
- Robust numeric parsing for messy CSVs.
- VBD computed per-position using league size & starters.
- Final 'value' blends projections and adjustments (SOS, injury) with weight.
"""

from __future__ import annotations
from typing import Dict, Any
import math
import pandas as pd
import numpy as np

# Default starters; can be overridden via config["starters"]
DEFAULT_STARTERS: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1}

# Strength of schedule (1-5 stars) -> simple multiplier
SOS_MULT: Dict[int, float] = {1: 0.97, 2: 0.99, 3: 1.00, 4: 1.01, 5: 1.03}

# -------------------- safe parsing helpers --------------------
def _is_na(x) -> bool:
    try:
        return pd.isna(x)
    except Exception:
        return False

def _to_float(x, default: float = 0.0) -> float:
    if _is_na(x):
        return float(default)
    s = str(x).strip().replace(",", "").replace("%", "")
    if s == "" or s.lower() in {"na", "n/a", "none", "null", "-"}:
        return float(default)
    try:
        return float(s)
    except Exception:
        # fallback: extract first number
        import re
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else float(default)

def _to_int(x, default: int = 0) -> int:
    try:
        return int(round(_to_float(x, default)))
    except Exception:
        return int(default)

def _safe_lower_str(x) -> str:
    if _is_na(x):
        return ""
    try:
        return str(x).strip().lower()
    except Exception:
        return ""

# -------------------- small mappers --------------------
def _inj_pen(v) -> float:
    """
    Map injury risk to penalty factor (0.0 = none, 0.5 = medium, 1.0 = high)
    NOTE: This returns a penalty *factor*, not a multiplier.
    """
    s = _safe_lower_str(v)
    if s in {"high", "h"}:
        return 1.0
    if s in {"medium", "med", "m"}:
        return 0.5
    # treat empty/low/unknown as no penalty
    return 0.0

def _sos_mult(v) -> float:
    val = _to_int(v, 3)
    return SOS_MULT.get(val, 1.0)

# -------------------- VBD helpers --------------------
def _baseline_index(pos: str, teams: int, starters: Dict[str, int]) -> int:
    """
    Which positional rank is the baseline replacement player.
    e.g., in 12-team:
      QB baseline ~ QB12
      RB baseline ~ RB24
      WR baseline ~ WR36
      TE baseline ~ TE12
      K baseline ~ K12
      DST baseline ~ DST12
    """
    pos = (pos or "").upper().strip()
    base = starters.get(pos, 0) * teams
    return max(1, base)

def _compute_vbd(df: pd.DataFrame, teams: int, starters: Dict[str, int]) -> pd.Series:
    """
    Compute VBD per position: value_raw - baseline(value_raw at baseline_index).
    """
    if df.empty:
        return pd.Series(dtype=float)

    vbd = pd.Series(0.0, index=df.index)
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        sub = df[df["POS"] == pos].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("value_raw", ascending=False)
        idx = min(_baseline_index(pos, teams, starters), len(sub))
        baseline = float(_to_float(sub.iloc[idx - 1]["value_raw"], 0.0))
        # assign safely
        diff = sub["value_raw"].astype(float) - baseline
        vbd.loc[sub.index] = diff.values
    return vbd

# -------------------- main API --------------------
def evaluate_players(
    df_raw: pd.DataFrame,
    config: Dict[str, Any],
    teams: int = 12,
    rounds: int = 15,
    weight_proj: float = 0.65,
) -> pd.DataFrame:
    """
    Returns a new DataFrame with at least: PLAYER, TEAM, POS, value, vbd.
    Uses PROJ_PTS if available; otherwise creates a rank-based proxy.
    Applies small SOS and injury adjustments; 'weight_proj' blends projection
    with the (slightly) VBD-influenced value.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["PLAYER", "TEAM", "POS", "value", "vbd"])

    df = df_raw.copy()

    # Normalize essential columns
    for col in ["PLAYER", "TEAM", "POS"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    # Ensure RK and BYE exist
    if "RK" not in df.columns:
        df["RK"] = np.arange(1, len(df) + 1)
    df["RK"] = df["RK"].map(_to_int)

    if "BYE" not in df.columns and "BYE WEEK" in df.columns:
        df["BYE"] = df["BYE WEEK"]
    if "BYE" not in df.columns:
        df["BYE"] = ""
    df["BYE"] = df["BYE"].map(lambda x: _to_int(x, 0))

    # Projections vs rank proxy
    proj = df.get("PROJ_PTS", pd.Series([np.nan] * len(df)))
    proj = proj.map(lambda x: np.nan if _is_na(x) else _to_float(x, np.nan))

    # rank-based proxy (higher for better ranks)
    # scale: simple convex curve so early ranks separate more
    rk = df["RK"].astype(float).clip(lower=1)
    rank_proxy = (300.0 / np.sqrt(rk)).astype(float)

    calc_proj = proj.fillna(rank_proxy)
    df["calc_proj"] = calc_proj

    # value_raw starts from projection proxy
    df["value_raw"] = df["calc_proj"].astype(float)

    # SOS & INJ adjustments
    df["sos_mult"] = df.get("SOS SEASON", pd.Series([np.nan] * len(df))).map(_sos_mult).fillna(1.0)
    df["inj_pen"] = df.get("INJURY_RISK", pd.Series([""] * len(df))).map(_inj_pen).fillna(0.0)

    # Apply adjustments to raw value
    # Injury penalty: shave a small % per penalty unit (e.g., 6% per unit)
    INJ_PCT_PER_UNIT = float(config.get("scoring", {}).get("w_injury_pen", 0.06))
    df["value_adj"] = df["value_raw"] * df["sos_mult"] * (1.0 - INJ_PCT_PER_UNIT * df["inj_pen"])

    # VBD
    starters_cfg = dict(DEFAULT_STARTERS)
    starters_cfg.update(config.get("starters", {}))
    df["vbd"] = _compute_vbd(df, teams=teams, starters=starters_cfg)

    # Blend projection with a little VBD for final 'value'
    # (weight_proj biases toward projections)
    df["value_blend"] = df["value_adj"] + 0.25 * df["vbd"]
    df["value"] = (weight_proj * df["value_adj"]) + ((1.0 - weight_proj) * df["value_blend"])

    # Clean numeric columns (avoid object dtypes downstream)
    for col in ["calc_proj", "value_raw", "value_adj", "value_blend", "value", "vbd", "sos_mult", "inj_pen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep expected columns + passthrough
    keep = set([
        "PLAYER","TEAM","POS","RK","TIERS","BYE","ECR VS. ADP","INJURY_RISK","VOLATILITY","HANDCUFF_TO",
        "calc_proj","value_raw","vbd","value","sos_mult","inj_pen"
    ])
    cols = [c for c in df.columns if c in keep] + [c for c in df.columns if c not in keep]
    out = df[cols].copy()

    # Sort by our computed value descending
    out = out.sort_values(["value","vbd","calc_proj"], ascending=False).reset_index(drop=True)
    return out
