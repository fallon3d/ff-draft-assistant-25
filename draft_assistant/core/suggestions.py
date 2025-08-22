"""
Unified suggestion engine (Live + Mock) with:
- Tunable Elite TE / Hero RB thresholds (for Round-1 strategy)
- K/DST timing switch: only last round (or last 1–2)
- "Might make it" / "Likely gone" highlight
- Positional needs summary helper (now supports explicit user_pos_counts)
- Default-strategy nudge (e.g., Hero RB + WR Flood)
"""

import math
from typing import List, Dict, Any, Optional
import pandas as pd

from . import utils

# Starter defaults (edit if your league differs)
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1}
MAX_AT_POS_EARLY = {"QB": 1, "TE": 1}  # avoid doubling early
EARLY_ROUNDS = 6

def _pos_counts_from_names(names: List[str], universe: pd.DataFrame) -> dict:
    """Fallback counting by matching PLAYER names to POS (kept for safety)."""
    if not names or universe is None or universe.empty:
        return {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DST":0}
    m = universe.set_index("PLAYER")["POS"].to_dict()
    out = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DST":0}
    for n in names:
        p = m.get(n)
        if p in out:
            out[p] += 1
    return out

def needs_summary(
    available_df,
    user_picked_names: List[str],
    user_pos_counts: Optional[Dict[str, int]] = None
) -> str:
    """Return a human-readable needs string like 'You still need: 1 TE, 2 RB'."""
    counts = (
        user_pos_counts.copy()
        if user_pos_counts is not None
        else _pos_counts_from_names(user_picked_names, available_df if available_df is not None else pd.DataFrame(columns=["PLAYER","POS"]))
    )
    for k in ("QB","RB","WR","TE","K","DST"):
        counts.setdefault(k, 0)
    needs = {pos: max(0, STARTERS.get(pos, 0) - counts.get(pos, 0)) for pos in STARTERS}
    parts = [f"{v} {k}" for k, v in needs.items() if v > 0]
    return f"You still need: {', '.join(parts)}" if parts else "Starters filled — build depth and upside."

def _rookie_flag(row: pd.Series) -> bool:
    try:
        if "ROOKIE" in row.index and int(row["ROOKIE"]) == 1:
            return True
    except Exception:
        pass
    s = str(row.get("TIERS","")).strip().lower()
    return ("rookie" in s) or (s == "r")

def _scarcity_bonus(pos: str, top_tier_left: bool) -> float:
    base = 0.0
    if pos in ("TE","QB"):
        base += 0.03
    if top_tier_left:
        base += 0.02
    return base

def _likely_gone_prob(rank: int, intervening_picks: int) -> float:
    if rank <= 0:
        rank = 999
    x = (intervening_picks - max(0, 25 - rank*0.2)) / 8.0
    return 1.0 / (1.0 + math.exp(-x))  # 0..1

def _tier_map(df: pd.DataFrame) -> Dict[str, int]:
    tm = {}
    for pos in ("QB","RB","WR","TE","K","DST"):
        pos_df = df[df["POS"] == pos]
        if not pos_df.empty:
            try:
                min_tier = pos_df["TIERS"].astype(str).replace("", "999").astype(int).min()
            except Exception:
                min_tier = 999
            tm[pos] = min_tier
        else:
            tm[pos] = 999
    return tm

def detect_run(pick_log: List[dict], recent: int = 6) -> str:
    if not pick_log:
        return ""
    last = pick_log[-recent:]
    pos = [p.get("metadata", {}).get("position", "") for p in last if p.get("metadata")]
    pos = [p for p in pos if p]
    if len(pos) >= 4 and len(set(pos)) <= 2:
        for candidate in ("WR","RB","QB","TE","K","DST"):
            if pos.count(candidate) >= 4:
                return candidate
    return ""

def rank_suggestions(
    available_df,
    round_number: int,
    total_rounds: int,
    user_picked_names: List[str],
    pick_log: List[dict],
    teams: int,
    username: str,
    current_overall: Optional[int] = None,
    user_slot: Optional[int] = None,
    kd_only_last_round: bool = True,
    user_pos_counts: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    if available_df is None or getattr(available_df, "empty", True):
        return pd.DataFrame(columns=["PLAYER","POS","TEAM","score","why","value","vbd","next_turn_tag"])

    df = available_df.copy()
    for c in ["value","vbd","RK","TIERS","BYE","POS","TEAM","PLAYER"]:
        if c not in df.columns:
            df[c] = 0

    # >>> Use explicit counts if provided (fixes "You still need")
    if user_pos_counts is not None:
        counts = {k: int(user_pos_counts.get(k, 0)) for k in ("QB","RB","WR","TE","K","DST")}
    else:
        counts = _pos_counts_from_names(user_picked_names, df)

    tm = _tier_map(df)
    run_pos = detect_run(pick_log)

    # Intervening picks to your next turn (precise if we know slot)
    if current_overall and user_slot:
        intervening = utils.picks_until_next_turn(current_overall, int(teams), int(user_slot))
    else:
        intervening = max(0, teams - 1)

    last_third = round_number >= max(8, int(total_rounds * (2/3.0)))
    late_k_dst_round = total_rounds if kd_only_last_round else max(total_rounds - 1, total_rounds - 2)

    scored = []
    for _, row in df.iterrows():
        pos = row["POS"]
        base = float(row.get("value", 0.0))
        vbd = float(row.get("vbd", 0.0))
        rank_overall = int(row.get("RK", 999))
        why_bits = []

        # Base + VBD
        score = base + 0.35 * vbd
        why_bits.append(f"value {base:.1f} / VBD {vbd:.1f}")

        # Roster needs
        need_gap = max(0, STARTERS.get(pos, 0) - counts.get(pos, 0))
        if need_gap > 0:
            score *= 1.10
            why_bits.append("fills starting need")
        if round_number <= EARLY_ROUNDS and counts.get(pos, 0) >= MAX_AT_POS_EARLY.get(pos, 99):
            score *= 0.90
            why_bits.append("deprioritize duplicate early")

        # Tier scarcity
        try:
            cur_tier = int(str(row.get("TIERS","") or "999"))
            top_tier_left = (cur_tier == tm.get(pos, 999))
        except Exception:
            top_tier_left = False
        sc_bump = _scarcity_bonus(pos, top_tier_left)
        score *= (1.0 + sc_bump)
        if sc_bump > 0:
            why_bits.append("tier/pos scarcity")

        # Position run bump
        if run_pos and pos == run_pos:
            score *= 1.05
            why_bits.append(f"{pos} run")

        # K/DST timing
        if pos in ("K","DST"):
            if round_number < late_k_dst_round:
                score *= 0.65
                why_bits.append("delay K/DST until late")
            else:
                score *= 1.10
                why_bits.append("late-round K/DST timing")

        # Likely-gone before next pick
        gone_p = _likely_gone_prob(rank_overall, intervening)
        score *= (1.0 + 0.10 * gone_p)
        next_turn_tag = "🔥 likely gone" if gone_p >= 0.65 else ("⏳ might make it" if gone_p <= 0.35 else "")

        # Late-round rookie upside
        is_rookie = _rookie_flag(row)
        if last_third and is_rookie and pos in ("RB","WR","TE"):
            score *= 1.15
            why_bits.append("late-round rookie upside")

        scored.append({
            "PLAYER": row["PLAYER"],
            "TEAM": row.get("TEAM",""),
            "POS": pos,
            "score": max(0.0, score),
            "value": base,
            "vbd": vbd,
            "why": "; ".join(why_bits) or "best available",
            "next_turn_tag": next_turn_tag,
        })

    out = pd.DataFrame(scored).sort_values(["score","value","vbd"], ascending=False).reset_index(drop=True)
    return out

# ---------- Strategy picker (unchanged below except for signature) ----------
STRATS = {
    "Anchor WR + Early Elite TE": {
        "name": "Anchor WR + Early Elite TE (default)",
        "plan": [["WR"], ["TE","WR"], ["WR"]],
        "why": "Bank elite PPR volume at WR and lock an edge at TE early.",
    },
    "Modified Zero RB": {
        "name": "Modified Zero RB (triple WR + TE, then attack RBs)",
        "plan": [["WR"], ["WR"], ["WR","TE"]],
        "why": "Crush weekly WR/FLEX with receptions, scoop RB volume later.",
    },
    "Hero RB + WR Flood": {
        "name": "Hero RB + WR Flood (optional early TE)",
        "plan": [["RB"], ["WR","TE"], ["WR"]],
        "why": "Secure bell-cow RB floor, then flood WR/TE value.",
    },
}

def _simulate_board(df: pd.DataFrame, picks_to_remove: int) -> pd.DataFrame:
    if picks_to_remove <= 0 or df.empty:
        return df
    return df.sort_values("value", ascending=False).iloc[picks_to_remove:].reset_index(drop=True)

def _best_for_positions(df: pd.DataFrame, pos_choices: List[str]) -> float:
    sub = df[df["POS"].isin(pos_choices)].sort_values("value", ascending=False)
    return float(sub.iloc[0]["value"]) if not sub.empty else 0.0

def _top_value_at_pos(df: pd.DataFrame, pos: str) -> float:
    sub = df[df["POS"] == pos].sort_values("value", ascending=False)
    return float(sub.iloc[0]["value"]) if not sub.empty else 0.0

def choose_strategy(
    available_df,
    current_overall: int,
    user_slot: int,
    teams: int,
    total_rounds: int,
    elite_te_value: float = 78.0,
    hero_rb_value: float = 85.0,
    preferred_name: Optional[str] = None,
    prefer_margin: float = 0.05,
) -> Dict[str, Any]:
    if available_df is None or getattr(available_df, "empty", True):
        return {"name":"Anchor WR + Early Elite TE (default)","why":"Empty board fallback.","score":0.0}

    df = available_df.copy()
    if "value" not in df.columns:
        df["value"] = 50.0

    picks_to_next = utils.picks_until_next_turn(current_overall, teams, user_slot)
    next_overall_1 = current_overall  # on the clock now
    next_overall_2 = utils.next_pick_overall(next_overall_1 + 1 + picks_to_next, teams, user_slot)
    picks_between_second = max(0, next_overall_2 - (next_overall_1 + 1 + picks_to_next))

    # Detect thresholds now / after
    top_te_now = _top_value_at_pos(df, "TE")
    elite_te_now = top_te_now >= elite_te_value
    board_after_next = _simulate_board(df, picks_to_next)
    elite_te_wont_last = elite_te_now and (_top_value_at_pos(board_after_next, "TE") < elite_te_value)

    top_rb_now = _top_value_at_pos(df, "RB")
    hero_rb_now = top_rb_now >= hero_rb_value

    # Score each strategy (with dynamic plan tweaks)
    scores = []
    for key, meta in STRATS.items():
        plan = [p[:] for p in meta["plan"]]
        if key == "Anchor WR + Early Elite TE" and not (elite_te_now or elite_te_wont_last):
            plan[1] = ["WR"]
        if key == "Hero RB + WR Flood" and not hero_rb_now:
            plan[0] = ["WR"]

        v_total = 0.0
        board_now = df.sort_values("value", ascending=False).reset_index(drop=True)
        # Pick 1
        v_total += _best_for_positions(board_now, plan[0])
        board_after1 = board_now.drop(board_now[board_now["POS"].isin(plan[0])].head(1).index).reset_index(drop=True)
        # To next turn
        board_at_turn2 = _simulate_board(board_after1, picks_to_next)
        # Pick 2
        v_total += _best_for_positions(board_at_turn2, plan[1])
        board_after2 = board_at_turn2.drop(board_at_turn2[board_at_turn2["POS"].isin(plan[1])].head(1).index).reset_index(drop=True)
        # To third pick
        board_at_turn3 = _simulate_board(board_after2, picks_between_second)
        want3 = plan[2] if len(plan) > 2 else ["WR","RB","TE"]
        v_total += _best_for_positions(board_at_turn3, want3)

        why_extra = []
        if elite_te_now: why_extra.append("elite TE available")
        if elite_te_wont_last: why_extra.append("elite TE unlikely to last")
        if hero_rb_now: why_extra.append("hero RB available")
        why = meta["why"] + ((" (" + ", ".join(why_extra) + ")") if why_extra else "")

        scores.append((key, v_total, why))

    # Choose best (with preference nudge)
    scores.sort(key=lambda x: x[1], reverse=True)
    best_key, best_score, best_why = scores[0]

    if preferred_name:
        pref_tuple = next((t for t in scores if STRATS[t[0]]["name"] == preferred_name), None)
        if pref_tuple:
            _, pref_score, pref_why = pref_tuple
            pref_adj = pref_score * (1.0 + float(prefer_margin))
            if pref_adj >= best_score * 0.995:  # tiny tie-breaker tolerance
                return {"name": preferred_name, "why": pref_why + " (honoring your default)", "score": pref_adj}

    return {"name": STRATS[best_key]["name"], "why": best_why, "score": best_score}
