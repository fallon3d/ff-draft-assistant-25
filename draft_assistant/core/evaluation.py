"""
Player evaluation with simple VBD-like scoring + schedule/coaching modifiers.
Now includes K and DST baselines.
"""
import json
import os
import pandas as pd

def _load_schedule_stars() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_schedule.csv")
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        col_stars = "STARS" if "STARS" in df.columns else ("SOS" if "SOS" in df.columns else None)
        if col_stars is None:
            return {}
        return dict(zip(df["TEAM"], df[col_stars]))
    except Exception:
        return {}

def _load_coaching_mods() -> dict:
    path = os.path.join(os.path.dirname(__file__), "..", "data", "coaching_modifiers.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def evaluate_players(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Returns dataframe with added columns: base_points, value, vbd, notes
    Supports POS in {QB,RB,WR,TE,K,DST}
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["PLAYER", "POS", "TEAM", "value", "vbd", "notes"])

    teams = int(config.get("draft", {}).get("teams", 12))
    # Baselines (startable pool size): tweak as needed
    baselines = {"QB": teams, "RB": teams * 2, "WR": teams * 3, "TE": teams, "K": teams, "DST": teams}

    schedule = _load_schedule_stars()
    coach = _load_coaching_mods()

    out = df.copy()

    # Base points proxy from rank: rank 1 = 100 pts, rank 100 = 0 pts (clipped)
    if "RK" in out.columns:
        out["base_points"] = out["RK"].apply(lambda r: max(0.0, 100.0 - float(r)))
    else:
        out["base_points"] = 50.0

    # Baseline at each position (value at replacement)
    baseline_points = {}
    for pos, count in baselines.items():
        pos_df = out[out["POS"] == pos].sort_values("base_points", ascending=False)
        baseline_points[pos] = float(pos_df.iloc[count - 1]["base_points"]) if len(pos_df) >= count else 0.0

    values, vbds, notes = [], [], []
    for _, row in out.iterrows():
        pos = row.get("POS", "")
        team = row.get("TEAM", "")
        base = float(row.get("base_points", 0.0))

        # Schedule adjustment (1–5 stars, 3 neutral)
        stars = schedule.get(team)
        sched_mult = 1.0
        note = ""
        if stars is not None:
            diff = float(stars) - 3.0
            sched_mult += diff * 0.05
            if diff > 0:
                note += f"Favorable schedule (+{int(diff*5)}%), "
            elif diff < 0:
                note += f"Tough schedule ({int(diff*5)}%), "

        # Coaching tendencies (small)
        cmod = coach.get(team, {})
        coach_mult = 1.0
        if pos == "RB":
            coach_mult *= float(cmod.get("rush_rate", 1.0))
        if pos in ("QB", "WR", "TE"):
            coach_mult *= float(cmod.get("pass_rate", 1.0))
        coach_mult *= float(cmod.get("pace", 1.0))

        total = base * sched_mult * coach_mult
        baseline = baseline_points.get(pos, 0.0)
        vbd = total - baseline

        if str(row.get("TIERS", "")).strip():
            note += f"Tier {row.get('TIERS')}, "
        if str(row.get("BYE", "")).strip() and pos != "DST":
            note += f"Bye wk {row.get('BYE')}, "

        values.append(total)
        vbds.append(vbd)
        notes.append(note.rstrip(", "))

    out["value"] = values
    out["vbd"] = vbds
    out["notes"] = notes

    out = out.sort_values(["value", "vbd"], ascending=[False, False]).reset_index(drop=True)
    return out
