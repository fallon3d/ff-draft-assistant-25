# ... imports and constants unchanged ...

def choose_strategy(
    available_df: pd.DataFrame,
    current_overall: int,
    user_slot: int,
    teams: int,
    total_rounds: int,
    elite_te_value: float = 78.0,
    hero_rb_value: float = 85.0,
    preferred_name: str | None = None,    # <-- NEW
    prefer_margin: float = 0.05,          # 5% nudge to your default
) -> Dict[str, Any]:
    """
    Pick the best starting path based on who is available and likely to remain.
    Uses tunable thresholds for 'Elite TE' and 'Hero RB' detection.
    If `preferred_name` is provided (e.g., 'Hero RB + WR Flood ...'), we give it
    a small boost; if it's within ~5% of the top strategy, we choose it.
    """
    if available_df is None or available_df.empty:
        return {"name":"Anchor WR + Early Elite TE (default)","why":"Empty board fallback."}

    df = available_df.copy()
    if "value" not in df.columns:
        df["value"] = 50.0

    picks_to_next = utils.picks_until_next_turn(current_overall, teams, user_slot)
    next_overall_1 = current_overall
    next_overall_2 = utils.next_pick_overall(next_overall_1 + 1 + picks_to_next, teams, user_slot)
    picks_between_second = max(0, next_overall_2 - (next_overall_1 + 1 + picks_to_next))

    # Detect thresholds now / after
    def _top_value_at_pos(df_, pos):
        sub = df_[df_["POS"] == pos].sort_values("value", ascending=False)
        return float(sub.iloc[0]["value"]) if not sub.empty else 0.0

    top_te_now = _top_value_at_pos(df, "TE")
    elite_te_now = top_te_now >= elite_te_value
    board_after_next = df.sort_values("value", ascending=False).iloc[picks_to_next:].reset_index(drop=True)
    elite_te_wont_last = elite_te_now and (_top_value_at_pos(board_after_next, "TE") < elite_te_value)

    top_rb_now = _top_value_at_pos(df, "RB")
    hero_rb_now = top_rb_now >= hero_rb_value

    # Score each strategy (same as before, with dynamic plan tweaks)
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
        board_at_turn2 = board_after1.iloc[picks_to_next:].reset_index(drop=True)
        # Pick 2
        v_total += _best_for_positions(board_at_turn2, plan[1])
        board_after2 = board_at_turn2.drop(board_at_turn2[board_at_turn2["POS"].isin(plan[1])].head(1).index).reset_index(drop=True)
        # To third pick
        board_at_turn3 = board_after2.iloc[picks_between_second:].reset_index(drop=True)
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
        # find preferred
        pref_tuple = next((t for t in scores if STRATS[t[0]]["name"] == preferred_name), None)
        if pref_tuple:
            _, pref_score, pref_why = pref_tuple
            # Give preferred a 5% nudge
            pref_adj = pref_score * (1.0 + float(prefer_margin))
            if pref_adj >= best_score * 0.995:  # tiny tie-breaker tolerance
                return {"name": preferred_name, "why": pref_why + " (honoring your default)", "score": pref_adj}

    return {"name": STRATS[best_key]["name"], "why": best_why, "score": best_score}
