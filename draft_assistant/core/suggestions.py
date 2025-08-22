"""
Unified suggestion engine for Live + Mock.
- Considers only remaining players
- VBD/value from evaluation
- Roster needs (avoid overloading early; prefer filling starters)
- Tier scarcity bump
- Run detection (light)
- Likely-gone probability based on intervening picks + rank
- Late-round rookie upside bias
Returns a DataFrame with columns: PLAYER, POS, TEAM, score, why, value, vbd
"""
from __future__ import annotations
import math
import pandas as pd

# starter defaults: 1 QB, 2 RB, 3 WR, 1 TE (editable later)
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
MAX_AT_POS_EARLY = {"QB": 1, "TE": 1}  # don't stack early
EARLY_ROUNDS = 6

def _pos_counts_from_names(names: list[str], universe: pd.DataFrame) -> dict:
    if not names:
        return {"QB":0,"RB":0,"WR":0,"TE":0}
    # naive map by first name field (our internal logs store full name in first_name)
    m = universe.set_index("PLAYER")["POS"].to_dict()
    out = {"QB":0,"RB":0,"WR":0,"TE":0}
    for n in names:
        p = m.get(n)
        if p in out:
            out[p] += 1
    return out

def _rookie_flag(row: pd.Series) -> bool:
    # prefer explicit ROOKIE col; else TIERS contains 'R' or 'Rookie'
    val = False
    if "ROOKIE" in row.index:
        try:
            val = bool(row["ROOKIE"])
        except Exception:
            val = False
    if not val:
        s = str(row.get("TIERS","")).strip().lower()
        if "rookie" in s or s == "r":
            val = True
    return val

def _scarcity_bonus(df: pd.DataFrame, pos: str, top_tier_left: bool) -> float:
    # light scarcity bump for TE/QB early, or when few remain at current highest tier
    base = 0.0
    if pos in ("TE","QB"):
        base += 0.03
    if top_tier_left:
        base += 0.02
    return base

def _likely_gone_prob(rank: int, intervening_picks: int) -> float:
    """Simple sigmoid on (intervening_picks - rank_buffer). Lower rank (better) -> higher prob."""
    if rank <= 0:
        rank = 999
    x = (intervening_picks - max(0, 25 - rank*0.2)) / 8.0
    return 1.0 / (1.0 + math.exp(-x))  # 0..1

def detect_run(pick_log: list, recent: int = 6) -> str:
    if not pick_log:
        return ""
    last = pick_log[-recent:]
    pos = [p.get("metadata", {}).get("position", "") for p in last if p.get("metadata")]
    pos = [p for p in pos if p]
    if len(pos) >= 4 and len(set(pos)) <= 2:
        # If 4 of last 6 are same position: run!
        for candidate in ("WR","RB","QB","TE"):
            if pos.count(candidate) >= 4:
                return candidate
    return ""

def rank_suggestions(
    available_df: pd.DataFrame,
    round_number: int,
    total_rounds: int,
    user_picked_names: list[str],
    pick_log: list,
    teams: int,
    username: str,
) -> pd.DataFrame:
    if available_df is None or available_df.empty:
        return pd.DataFrame(columns=["PLAYER","POS","TEAM","score","why","value","vbd"])

    df = available_df.copy()
    # Ensure evaluation columns exist
    for c in ["value","vbd","RK","TIERS","BYE"]:
        if c not in df.columns:
            df[c] = 0

    # Compute roster needs
    counts = _pos_counts_from_names(user_picked_names, df)
    starters = STARTERS.copy()

    # Determine top tier still available per position
    tier_map = {}
    if "TIERS" in df.columns:
        for pos in ("QB","RB","WR","TE"):
            pos_df = df[df["POS"] == pos]
            if not pos_df.empty:
                try:
                    min_tier = pos_df["TIERS"].astype(str).replace("", "999").astype(int).min()
                except Exception:
                    min_tier = 999
                tier_map[pos] = min_tier
            else:
                tier_map[pos] = 999

    # Detect run and intervening picks to next user turn
    run_pos = detect_run(pick_log)
    # Intervening picks: distance until our next turn in snake
    # For simplicity, assume next overall is last_pick+1 and snake math handled elsewhere
    intervening = max(0, teams - 1)  # rough default

    scored = []
    for _, row in df.iterrows():
        pos = row["POS"]
        base = float(row.get("value", 0.0))
        vbd = float(row.get("vbd", 0.0))
        rank_overall = int(row.get("RK", 999))
        why_bits = []

        # Start with base/vbd
        score = base + 0.35 * vbd
        why_bits.append(f"value {base:.1f} / VBD {vbd:.1f}")

        # Roster need weighting: prefer filling starters; avoid overloading QB/TE early
        need_gap = max(0, starters.get(pos, 0) - counts.get(pos, 0))
        if need_gap > 0:
            score *= 1.10
            why_bits.append("fills starting need")
        if round_number <= EARLY_ROUNDS and counts.get(pos, 0) >= MAX_AT_POS_EARLY.get(pos, 99):
            score *= 0.90
            why_bits.append("deprioritize duplicate early")

        # Tier scarcity bump
        top_tier_left = False
        try:
            cur_tier = int(str(row.get("TIERS","") or "999"))
            top_tier_left = (cur_tier == tier_map.get(pos, 999))
        except Exception:
            pass
        sc_bump = _scarcity_bonus(df, pos, top_tier_left)
        score *= (1.0 + sc_bump)
        if sc_bump > 0:
            why_bits.append("tier/pos scarcity")

        # If a run is happening at this position, modest bump to stay with the board
        if run_pos and pos == run_pos:
            score *= 1.05
            why_bits.append(f"{pos} run")

        # Likely-gone probability before our next pick
        gone_p = _likely_gone_prob(rank_overall, intervening)
        score *= (1.0 + 0.10 * gone_p)
        if gone_p > 0.5:
            why_bits.append("likely gone by next turn")

        # Bye conflict (soft)
        # If multiple players already share same BYE at that pos, nudge down a bit
        try:
            bye = int(row.get("BYE", 0))
        except Exception:
            bye = 0
        if bye and pos in ("RB","WR","TE"):
            same_pos = [n for n in user_picked_names if df.set_index("PLAYER").get("POS", pd.Series()).get(n, None) == pos]
            # crude: if 2+ already have same bye (not tracked here), we won't over-penalize; just a small note
            pass

        # Late-round rookie upside (round ≥ last third)
        last_third = round_number >= max(8, int(total_rounds * (2/3.0)))
        is_rookie = _rookie_flag(row)
        if last_third and is_rookie:
            score *= 1.15
            why_bits.append("late-round rookie upside")

        # Cap tiny negatives
        score = max(0.0, score)

        scored.append({
            "PLAYER": row["PLAYER"],
            "TEAM": row.get("TEAM",""),
            "POS": pos,
            "score": score,
            "value": base,
            "vbd": vbd,
            "why": "; ".join(why_bits) or "best available",
        })

    out = pd.DataFrame(scored).sort_values(["score","value","vbd"], ascending=False).reset_index(drop=True)
    return out
