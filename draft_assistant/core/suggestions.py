"""
Suggestion engine (strategy-locked, no BPA, strict QB suppression, safe parsing).

Key changes:
- REMOVED BPA strategy.
- New `top_strategies(...)` returns the best 3 strategies to show on your first pick.
- `rank_suggestions(..., selected_strategy_name=...)` biases scores to match the locked strategy.
- Strong QB suppression when you already roster enough QBs (esp. with Late-QB/TE).
- Robust numeric parsing for messy spreadsheet fields.
"""

from __future__ import annotations
import math
from typing import List, Dict, Any, Optional
import pandas as pd

from . import utils

# --- global knobs (some also configurable via config.strategy) ---
STARTERS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "K": 1, "DST": 1}
MAX_AT_POS_EARLY = {"QB": 1, "TE": 1}   # avoid duplicates early
EARLY_ROUNDS = 6
DEFAULT_EARLY_QB_ROUND = 7              # soft deferral after hard gate
DEFAULT_QB_FORBID_BEFORE = 8            # HARD GATE: don't take QB before this round
DEFAULT_EARLY_TE_ROUND = 6
DEFAULT_ELITE_QB_POS_RANK = 3
DEFAULT_STACK_BONUS = 0.06
HANDCUFF_LATE_ROUND = 9


# ---------------- safe numeric parsing ----------------
def _num(x, default=0.0):
    if x is None:
        return float(default)
    try:
        if pd.isna(x):
            return float(default)
    except Exception:
        pass
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "n/a", "—", "-", "null", "none"}:
        return float(default)
    s = s.replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        import re
        m = re.search(r"[-+]?(\d+(\.\d+)?)", s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return float(default)
        return float(default)

def _int(x, default=0):
    try:
        return int(round(_num(x, default)))
    except Exception:
        return int(default)


# ---------------- needs summary ----------------
def _pos_counts_from_names(names: List[str], universe: pd.DataFrame) -> dict:
    if not names or universe is None or universe.empty:
        return {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DST":0}
    m = universe.set_index("PLAYER")["POS"].to_dict()
    out = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DST":0}
    for n in names:
        p = m.get(n)
        if p in out: out[p] += 1
    return out

def needs_summary(available_df, user_picked_names: List[str], user_pos_counts: Optional[Dict[str,int]]=None) -> str:
    counts = (
        {k:int(user_pos_counts.get(k,0)) for k in ("QB","RB","WR","TE","K","DST")}
        if user_pos_counts is not None else _pos_counts_from_names(user_picked_names, available_df or pd.DataFrame(columns=["PLAYER","POS"]))
    )
    needs = {pos: max(0, STARTERS.get(pos, 0) - counts.get(pos, 0)) for pos in STARTERS}
    parts = [f"{v} {k}" for k, v in needs.items() if v > 0]
    return f"You still need: {', '.join(parts)}" if parts else "Starters filled — build depth and upside."


# ---------------- misc helpers ----------------
def _rookie_flag(row: pd.Series) -> bool:
    try:
        if "ROOKIE" in row.index and _int(row["ROOKIE"]) == 1: 
            return True
    except Exception:
        pass
    s = str(row.get("TIERS","")).strip().lower()
    return ("rookie" in s) or (s == "r")

def _likely_gone_prob(rank: int, intervening_picks: int) -> float:
    if rank <= 0: rank = 999
    x = (intervening_picks - max(0, 25 - rank*0.2)) / 8.0
    return 1.0 / (1.0 + math.exp(-x))

def _tier_map(df: pd.DataFrame) -> Dict[str, int]:
    tm = {}
    for pos in ("QB","RB","WR","TE","K","DST"):
        pos_df = df[df["POS"] == pos]
        if not pos_df.empty:
            try:
                tiers = pos_df["TIERS"].astype(str).str.strip().replace({"": "999"})
                tiers = tiers.map(lambda x: str(_int(x, 999)))
                min_tier = tiers.astype(int).min()
            except Exception:
                min_tier = 999
            tm[pos] = min_tier
        else:
            tm[pos] = 999
    return tm

def detect_run(pick_log: List[dict], recent: int = 6) -> str:
    if not pick_log: return ""
    last = pick_log[-recent:]
    pos = [p.get("metadata", {}).get("position", "") for p in last if p.get("metadata")]
    pos = [p for p in pos if p]
    if len(pos) >= 4 and len(set(pos)) <= 2:
        for candidate in ("WR","RB","QB","TE","K","DST"):
            if pos.count(candidate) >= 4: return candidate
    return ""

def _scarcity_bonus(pos: str, remaining_in_top_tier: int) -> float:
    if remaining_in_top_tier <= 1:
        return 0.07
    if remaining_in_top_tier <= 2:
        return 0.05
    if remaining_in_top_tier <= 4:
        return 0.03
    return 0.0

def _pos_rank_map(df: pd.DataFrame) -> Dict[int, int]:
    pos_rank = {}
    for pos in ("QB","RB","WR","TE","K","DST"):
        sub = df[df["POS"] == pos].sort_values("value", ascending=False).reset_index()
        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            try:
                pos_rank[int(row["index"])] = i
            except Exception:
                continue
    return pos_rank

def _build_stack_state(user_picked_names: List[str], universe_df: pd.DataFrame) -> Dict[str, set]:
    if universe_df is None or universe_df.empty:
        return {"QB": set(), "WR": set(), "TE": set()}
    m_team = universe_df.set_index("PLAYER")["TEAM"].to_dict()
    m_pos = universe_df.set_index("PLAYER")["POS"].to_dict()
    have = {"QB": set(), "WR": set(), "TE": set()}
    for n in user_picked_names:
        t = m_team.get(n); p = m_pos.get(n)
        if t and p in have: have[p].add(t)
    return have


# ---------------- strategy catalog (NO BPA) ----------------
STRATS = {
    "Anchor WR + Early Elite TE": {
        "name": "Anchor WR + Early Elite TE (default)",
        "plan": [["WR"], ["TE","WR"], ["WR"]],
        "why": "Bank elite PPR volume at WR and lock an edge at TE early.",
    },
    "Modified Zero RB": {
        "name": "Modified Zero RB (triple WR + TE, then attack RBs)",
        "plan": [["WR"], ["WR"], ["WR","TE"]],
        "why": "Crush weekly WR/FLEX with receptions; scoop RB volume later.",
    },
    "Hero RB + WR Flood": {
        "name": "Hero RB + WR Flood (optional early TE)",
        "plan": [["RB"], ["WR","TE"], ["WR"]],
        "why": "Secure bell-cow RB floor, then flood WR/TE value.",
    },
    "Zero WR": {
        "name": "Zero WR (RB/TE early, fill WR later)",
        "plan": [["RB","TE"], ["RB","TE"], ["RB"]],
        "why": "Exploit RB scarcity; WR depth later in half/standard scoring.",
    },
    "Late QB/TE": {
        "name": "Late-Round QB/TE (streaming posture)",
        "plan": [["RB","WR"], ["WR","RB"], ["WR","RB"]],
        "why": "Force capital into RB/WR early; wait on QB/TE.",
    },
}

def _simulate_board(df: pd.DataFrame, picks_to_remove: int) -> pd.DataFrame:
    if picks_to_remove <= 0 or df.empty: return df
    return df.sort_values("value", ascending=False).iloc[picks_to_remove:].reset_index(drop=True)

def _best_for_positions(df: pd.DataFrame, pos_choices: List[str]) -> float:
    sub = df[df["POS"].isin(pos_choices)].sort_values("value", ascending=False)
    return float(sub.iloc[0]["value"]) if not sub.empty else 0.0

def _top_value_at_pos(df: pd.DataFrame, pos: str) -> float:
    sub = df[df["POS"] == pos].sort_values("value", ascending=False)
    return float(sub.iloc[0]["value"]) if not sub.empty else 0.0


# ---------------- public: top strategies (for first-pick UI) ----------------
def top_strategies(
    available_df,
    current_overall: int,
    user_slot: int,
    teams: int,
    total_rounds: int,
    elite_te_value: float = 78.0,
    hero_rb_value: float = 85.0,
    k: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Return the top-k strategies by simple 3-step lookahead (no QB before hard gate).
    """
    if available_df is None or getattr(available_df, "empty", True):
        return [{"name": STRATS["Anchor WR + Early Elite TE"]["name"], "why": "Empty board fallback.", "score": 0.0}]

    df = available_df.copy()
    if "value" not in df.columns:
        df["value"] = 50.0

    strat_cfg = (config or {}).get("strategy", {}) if config else {}
    qb_forbid_before = int(strat_cfg.get("qb_forbid_before_round", DEFAULT_QB_FORBID_BEFORE))

    rd, _, _ = utils.snake_position(current_overall, teams)

    # sanitize helper: no QB before the hard gate
    def _sanitize(choices: List[str], round_num: int) -> List[str]:
        if round_num < qb_forbid_before and "QB" in choices:
            out = [c for c in choices if c != "QB"]
            return out or ["WR","RB","TE"]
        return choices

    picks_to_next = utils.picks_until_next_turn(current_overall, teams, user_slot)
    next_overall_1 = current_overall
    next_overall_2 = utils.next_pick_overall(next_overall_1 + 1 + picks_to_next, teams, user_slot)
    picks_between_second = max(0, next_overall_2 - (next_overall_1 + 1 + picks_to_next))

    top_te_now = _top_value_at_pos(df, "TE")
    elite_te_now = top_te_now >= elite_te_value
    board_after_next = _simulate_board(df, picks_to_next)
    elite_te_wont_last = elite_te_now and (_top_value_at_pos(board_after_next, "TE") < elite_te_value)

    top_rb_now = _top_value_at_pos(df, "RB")
    hero_rb_now = top_rb_now >= hero_rb_value

    scores = []
    for key, meta in STRATS.items():
        plan = [p[:] for p in meta["plan"]]
        # nudges
        if key == "Anchor WR + Early Elite TE" and not (elite_te_now or elite_te_wont_last):
            plan[1] = ["WR"]
        if key == "Hero RB + WR Flood" and not hero_rb_now:
            plan[0] = ["WR"]

        p0 = _sanitize(plan[0], rd)
        p1 = _sanitize(plan[1], rd + 1) if len(plan) > 1 else ["WR","RB","TE"]
        p2 = _sanitize(plan[2] if len(plan) > 2 else ["WR","RB","TE"], rd + 2)

        v_total = 0.0
        board_now = df.sort_values("value", ascending=False).reset_index(drop=True)
        v_total += _best_for_positions(board_now, p0)

        board_after1 = board_now.drop(board_now[board_now["POS"].isin(p0)].head(1).index).reset_index(drop=True)
        board_at_turn2 = _simulate_board(board_after1, picks_to_next)
        v_total += _best_for_positions(board_at_turn2, p1)

        board_after2 = board_at_turn2.drop(board_at_turn2[board_at_turn2["POS"].isin(p1)].head(1).index).reset_index(drop=True)
        board_at_turn3 = _simulate_board(board_after2, picks_between_second)
        v_total += _best_for_positions(board_at_turn3, p2)

        scores.append({"key": key, "name": meta["name"], "why": meta["why"], "score": v_total})

    scores.sort(key=lambda d: d["score"], reverse=True)
    return scores[:max(1, int(k))]


# ---------------- main: rank suggestions under a locked strategy ----------------
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
    user_bye_weeks: Optional[set] = None,
    weights: Optional[Dict[str,float]] = None,
    universe_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    selected_strategy_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    score = base(value) + w_vbd * vbd + w_ecr * (ECR VS. ADP)
    Then bias by the LOCKED strategy's position plan, strong QB deferral/caps, scarcity, runs, bye, etc.
    """
    if available_df is None or getattr(available_df, "empty", True):
        return pd.DataFrame(columns=["PLAYER","POS","TEAM","score","why","value","vbd","next_turn_tag"])

    df = available_df.copy()
    for c in ["value","vbd","RK","TIERS","BYE","POS","TEAM","PLAYER","ECR VS. ADP","INJURY_RISK","VOLATILITY","HANDCUFF_TO"]:
        if c not in df.columns: df[c] = 0

    # config knobs
    strat_cfg = (config or {}).get("strategy", {}) if config else {}
    early_qb_round_soft = int(strat_cfg.get("early_qb_round", DEFAULT_EARLY_QB_ROUND))
    qb_forbid_before = int(strat_cfg.get("qb_forbid_before_round", DEFAULT_QB_FORBID_BEFORE))
    early_te_round = int(strat_cfg.get("early_te_round", DEFAULT_EARLY_TE_ROUND))
    elite_qb_pos_rank = int(strat_cfg.get("elite_qb_pos_rank", DEFAULT_ELITE_QB_POS_RANK))
    stack_bonus = float(strat_cfg.get("stack_bonus", DEFAULT_STACK_BONUS))
    punt_positions = [str(x).upper() for x in strat_cfg.get("punt_positions", [])]
    kd_only_last_round = bool(strat_cfg.get("kd_only_last_round", kd_only_last_round))

    w_vbd = float(weights.get("w_vbd", 0.35)) if weights else 0.35
    w_ecr = float(weights.get("w_ecr_delta", 0.12)) if weights else 0.12
    w_inj = float(weights.get("w_injury", 0.06)) if weights else 0.06
    w_vol = float(weights.get("w_vol", 0.05)) if weights else 0.05

    counts = {k: int(user_pos_counts.get(k, 0)) for k in ("QB","RB","WR","TE","K","DST")} if user_pos_counts is not None else _pos_counts_from_names(user_picked_names, df)
    tm = _tier_map(df)
    run_pos = detect_run(pick_log)

    remain_top_tier = {}
    temp_tiers = df["TIERS"].astype(str).str.strip().replace({"": "999"})
    tiers_int = temp_tiers.map(lambda x: _int(x, 999))
    for pos in ("QB","RB","WR","TE","K","DST"):
        mask = (df["POS"] == pos) & (tiers_int == (tiers_int[df["POS"] == pos].min() if (tiers_int[df["POS"] == pos]).size else 999))
        remain_top_tier[pos] = int(df[mask].shape[0]) if df.shape[0] else 0

    if current_overall and user_slot:
        intervening = utils.picks_until_next_turn(current_overall, int(teams), int(user_slot))
    else:
        intervening = max(0, teams - 1)

    user_bye_weeks = user_bye_weeks or set()
    have_stack = _build_stack_state(user_picked_names, universe_df if universe_df is not None else df)
    pos_rank = _pos_rank_map(df)

    # Strategy bias for THIS pick
    plan_now: List[str] = []
    if selected_strategy_name:
        # find the strategy key by its display name
        strat_key = None
        for k, meta in STRATS.items():
            if meta["name"] == selected_strategy_name or k == selected_strategy_name:
                strat_key = k; break
        if strat_key is None:
            strat_key = "Hero RB + WR Flood"  # safe fallback

        rd = int(round_number)
        plan = STRATS[strat_key]["plan"]
        # use first 3 steps plan for early rounds, then fall back to broad targets
        if rd <= 1:         plan_now = plan[0]
        elif rd == 2:       plan_now = plan[1] if len(plan) > 1 else ["WR","RB","TE"]
        elif rd == 3:       plan_now = plan[2] if len(plan) > 2 else ["WR","RB","TE"]
        else:
            # mid/late default focus by strategy
            if strat_key == "Late QB/TE":
                plan_now = ["RB","WR"]   # keep hammering depth
            elif strat_key == "Zero WR":
                plan_now = ["RB","TE"]
            elif strat_key == "Anchor WR + Early Elite TE":
                plan_now = ["WR","RB","TE"]
            else:  # Hero RB + WR Flood
                plan_now = ["WR","RB","TE"]

    rows = []
    for idx, row in df.iterrows():
        pos = str(row.get("POS","")).upper().strip()
        base = float(_num(row.get("value"), 0.0))
        vbd = float(_num(row.get("vbd"), 0.0))
        rk = _int(row.get("RK"), 999)
        bye = _int(row.get("BYE"), 0)
        ecr_delta = float(_num(row.get("ECR VS. ADP"), 0.0))
        inj = str(row.get("INJURY_RISK","")).strip().lower()
        vol = str(row.get("VOLATILITY","")).strip().lower()
        team = str(row.get("TEAM","")).strip().upper()
        player_name = str(row.get("PLAYER","")).strip()

        why_bits = []
        score = base
        why_bits.append(f"value {base:.1f}")

        # VBD + value vs ADP
        score += w_vbd * vbd
        if vbd != 0: why_bits.append(f"VBD {vbd:+.1f}")
        if ecr_delta:
            score += w_ecr * ecr_delta
            why_bits.append(f"value vs ADP {ecr_delta:+.1f}")

        # Needs & early duplicate guard
        need_gap = max(0, STARTERS.get(pos, 0) - counts.get(pos, 0))
        if need_gap > 0:
            score *= 1.10
            why_bits.append("fills starting need")
        if round_number <= EARLY_ROUNDS and counts.get(pos, 0) >= MAX_AT_POS_EARLY.get(pos, 99):
            score *= 0.90
            why_bits.append("deprioritize duplicate early")

        # Strategy bias (locked)
        if plan_now:
            if pos in plan_now:
                score *= 1.10
                why_bits.append("fits locked strategy")
            else:
                score *= 0.95
                why_bits.append("lower priority for strategy")

        # Tier scarcity bump
        sc_bump = _scarcity_bonus(pos, remain_top_tier.get(pos, 0))
        if sc_bump > 0:
            score *= (1.0 + sc_bump)
            why_bits.append("tier/pos scarcity")

        # Run detection
        if run_pos and pos == run_pos:
            score *= 1.05
            why_bits.append(f"{pos} run")

        # K/DST timing & punting logic
        if pos in ("K","DST"):
            if round_number < total_rounds:
                score *= 0.55 if kd_only_last_round else 0.80
                why_bits.append("delay K/DST")
        if pos in punt_positions and round_number < total_rounds - 1:
            score *= 0.80
            why_bits.append(f"punting {pos} for now")

        # -------- STRICT EARLY-QB DEFERRAL (hard gate) --------
        if pos == "QB" and counts.get("QB",0) == 0 and round_number < qb_forbid_before:
            score *= 0.35
            why_bits.append(f"QB deferred until R{qb_forbid_before}")

        # Additional QB caps to stop spam recommendations when you already have QBs
        # Late QB/TE: absolutely no second QB until final 2 rounds
        if pos == "QB":
            qb_count = counts.get("QB", 0)
            if selected_strategy_name and "Late-Round QB/TE" in selected_strategy_name:
                if qb_count >= 1 and round_number < (total_rounds - 1):
                    score *= 0.20
                    why_bits.append("late-QB plan: hold QB depth")
            else:
                if qb_count >= 2:
                    score *= 0.20
                    why_bits.append("QB depth maxed")
                elif qb_count >= 1 and round_number < max(8, total_rounds - 3):
                    score *= 0.60
                    why_bits.append("already have QB — wait")

        # --- Soft early timing for QB/TE (after hard gate window) ---
        pr = int(pos_rank.get(int(idx), 99))
        if pos == "QB" and counts.get("QB",0) == 0 and round_number < early_qb_round_soft:
            if round_number >= qb_forbid_before:
                elite_qb_window = (pr <= elite_qb_pos_rank)
                stack_here = (team in have_stack.get("WR",set()) or team in have_stack.get("TE",set()))
                if not (elite_qb_window or stack_here):
                    score *= 0.82
                    why_bits.append(f"wait on QB (rank {pr})")
                elif stack_here:
                    score *= (1.0 + stack_bonus)
                    why_bits.append("QB stack leverage")

        if pos == "TE" and counts.get("TE",0) == 0 and round_number < DEFAULT_EARLY_TE_ROUND:
            elite_te_window = (pr <= 2)
            if not elite_te_window:
                score *= 0.88
                why_bits.append(f"wait on TE (rank {pr})")

        # Stacking bonus for WR/TE with your QB
        if pos in ("WR","TE") and team in have_stack.get("QB", set()):
            score *= (1.0 + DEFAULT_STACK_BONUS)
            why_bits.append(f"stack with your QB ({team})")

        # Handcuff priority (late rounds)
        handcuff_to = str(row.get("HANDCUFF_TO","")).strip()
        if round_number >= HANDCUFF_LATE_ROUND and handcuff_to:
            if any(handcuff_to.lower() == n.lower() for n in user_picked_names):
                score *= 1.08
                why_bits.append(f"protect your {handcuff_to} (handcuff)")

        # Risk & late-volatility handling
        if inj == "high": score *= (1.0 - w_inj)
        elif inj == "medium": score *= (1.0 - w_inj/2.0)
        if inj in ("medium","high"): why_bits.append(f"injury risk: {inj}")

        if _rookie_flag(row) and round_number >= HANDCUFF_LATE_ROUND:
            score *= 1.05
            why_bits.append("late rookie ceiling")

        if round_number >= max(8, int(total_rounds * (2/3.0))) and vol in ("medium","high"):
            score *= (1.0 + (w_vol if vol == "high" else w_vol/2.0))
            why_bits.append("late-round volatility upside")

        # Availability to next turn
        gone_p = _likely_gone_prob(rk if rk > 0 else 999, intervening)
        score *= (1.0 + 0.10 * gone_p)
        next_turn_tag = "🔥 likely gone" if gone_p >= 0.65 else ("⏳ might make it" if gone_p <= 0.35 else "")

        # Bye conflict (light)
        if bye and (bye in (user_bye_weeks or set())) and pos in ("RB","WR","TE","QB"):
            score *= 0.98
            why_bits.append(f"bye conflict (wk {bye})")

        rows.append({
            "PLAYER": player_name, "TEAM": team, "POS": pos,
            "score": max(0.0, float(score)), "value": base, "vbd": vbd,
            "why": "; ".join(why_bits) or "best available", "next_turn_tag": next_turn_tag,
        })

    out = pd.DataFrame(rows).sort_values(["score","value","vbd"], ascending=False).reset_index(drop=True)
    return out
