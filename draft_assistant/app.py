"""
Fantasy Football Draft Assistant (Streamlit)
Tweaks:
- Sidebar controls to tune Elite TE / Hero RB thresholds and K/DST timing
- Positional needs counters above Suggestions (Live + Mock)
"""
import os
import random
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fantasy Football Draft Assistant", layout="wide")
league_info = None  # guard for Export tab

from core import utils, sleeper, roster, evaluation, suggestions, mock_ai, pdf_report

# ---------- CONFIG ----------
config = utils.read_config()

# ---------- SIDEBAR SETTINGS ----------
st.sidebar.title("Settings")

# Identity
sleeper_username = st.sidebar.text_input(
    "Your Sleeper Username",
    value=str(config.get("user_profile", {}).get("sleeper_username", "Fallon3D")),
    key="settings_username",
)
teams_default = int(config.get("draft", {}).get("teams", 12))
user_slot_override = st.sidebar.number_input(
    "Your Draft Slot (1–Teams; 0 = auto)",
    min_value=0, max_value=teams_default,
    value=int(config.get("user_profile", {}).get("user_slot", 0)),
    step=1, key="settings_user_slot",
)

# Live league
league_id = st.sidebar.text_input(
    "Sleeper League ID (Live)",
    value=str(config.get("sleeper", {}).get("league_id", "")),
    key="settings_league_id",
)
poll_seconds = st.sidebar.number_input(
    "Live Poll Interval (seconds)",
    min_value=3, max_value=30,
    value=int(config.get("sleeper", {}).get("poll_seconds", 5)),
    step=1, key="settings_poll_sec",
)

# Draft dims (defaults for both modes)
rounds_default = int(config.get("draft", {}).get("rounds", 15))
teams_setting = st.sidebar.number_input(
    "Default: Number of Teams",
    min_value=2, value=teams_default, key="settings_num_teams",
)
rounds_setting = st.sidebar.number_input(
    "Default: Number of Rounds",
    min_value=1, value=rounds_default, key="settings_num_rounds",
)

st.sidebar.markdown("### Strategy Tuning")
elite_te_value = st.sidebar.number_input(
    "Elite TE threshold (value)",
    min_value=10.0, max_value=140.0,
    value=float(config.get("strategy", {}).get("elite_te_value", 78.0)),
    step=1.0, key="elite_te_value",
    help="Minimum 'value' for a TE to be treated as elite in Round-1 strategy."
)
hero_rb_value = st.sidebar.number_input(
    "Hero RB threshold (value)",
    min_value=10.0, max_value=160.0,
    value=float(config.get("strategy", {}).get("hero_rb_value", 85.0)),
    step=1.0, key="hero_rb_value",
    help="Minimum 'value' for an RB to be considered a Hero RB anchor."
)
kd_only_last_round = st.sidebar.checkbox(
    "Draft K/DST only in the final round",
    value=bool(config.get("strategy", {}).get("kd_only_last_round", True)),
    key="kd_only_last_round",
)

# Uploads (write into draft_assistant/data/)
st.sidebar.markdown("### Data Uploads")
players_file = st.sidebar.file_uploader("Players CSV (base)", type="csv", key="upload_players")
extra1_file = st.sidebar.file_uploader("Extra Players CSV #1", type="csv", key="upload_extra1")
extra2_file = st.sidebar.file_uploader("Extra Players CSV #2", type="csv", key="upload_extra2")
extra3_file = st.sidebar.file_uploader("Extra Players CSV #3", type="csv", key="upload_extra3")
schedule_file = st.sidebar.file_uploader("Schedule CSV (optional)", type="csv", key="upload_schedule")

if st.sidebar.button("Save Settings", key="settings_save_btn"):
    config.setdefault("user_profile", {})["sleeper_username"] = sleeper_username.strip()
    config["user_profile"]["user_slot"] = int(user_slot_override)
    config.setdefault("sleeper", {})["league_id"] = league_id.strip()
    config["sleeper"]["poll_seconds"] = int(poll_seconds)
    config.setdefault("draft", {})["teams"] = int(teams_setting)
    config["draft"]["rounds"] = int(rounds_setting)
    config.setdefault("strategy", {})
    config["strategy"]["elite_te_value"] = float(elite_te_value)
    config["strategy"]["hero_rb_value"] = float(hero_rb_value)
    config["strategy"]["kd_only_last_round"] = bool(kd_only_last_round)
    utils.save_config(config)
    st.sidebar.success("Saved to config.toml")

# Persist uploads next to app.py
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _save_upload(upload, name):
    if upload is not None:
        try:
            df = pd.read_csv(upload)
            df.to_csv(os.path.join(DATA_DIR, name), index=False)
            st.sidebar.success(f"Uploaded: {name}")
        except Exception as e:
            st.sidebar.error(f"{name} error: {e}")

_save_upload(players_file, "sample_players.csv")
_save_upload(extra1_file, "extra_players_1.csv")
_save_upload(extra2_file, "extra_players_2.csv")
_save_upload(extra3_file, "extra_players_3.csv")

if schedule_file is not None:
    try:
        pd.read_csv(schedule_file).to_csv(os.path.join(DATA_DIR, "sample_schedule.csv"), index=False)
        st.sidebar.success("Schedule CSV uploaded.")
    except Exception as e:
        st.sidebar.error(f"Schedule CSV error: {e}")

# ---------- HEADER ----------
st.title("🏈 Fantasy Football Draft Assistant")

# ---------- TABS (Player Board removed) ----------
tab_live, tab_mock, tab_suggest, tab_export = st.tabs(
    ["Live Draft", "Mock Draft", "Suggestions", "Export"]
)

# Helper: build combined player pool
def build_pool():
    df = utils.load_combined_player_pool(
        include_sleeper=True,
        base_path=os.path.join(DATA_DIR, "sample_players.csv"),
        extra_paths=[
            os.path.join(DATA_DIR, "extra_players_1.csv"),
            os.path.join(DATA_DIR, "extra_players_2.csv"),
            os.path.join(DATA_DIR, "extra_players_3.csv"),
        ],
    )
    return utils.normalize_player_headers(df)

# ===================== LIVE DRAFT =====================
with tab_live:
    st.subheader("Live Draft (Sleeper)")
    col1, col2 = st.columns([1, 1])
    auto = col1.toggle("Auto-refresh", value=False, key="live_auto_refresh_toggle")
    col2.button("Refresh now", on_click=st.experimental_rerun, key="live_refresh_btn")

    if auto:
        st.markdown(f"<meta http-equiv='refresh' content='{int(poll_seconds)}'>", unsafe_allow_html=True)
        st.caption(f"Auto-refresh every {int(poll_seconds)}s (polite to Sleeper API).")

    if not league_id:
        st.info("Add your Sleeper League ID in the sidebar, then click **Save Settings**.")
    else:
        league_info = sleeper.get_league_info(league_id)
        if not league_info:
            st.error("Could not fetch league info. Check the League ID and try again.")
        else:
            league_name = league_info.get("name") or league_id
            st.write(f"**League:** {league_name}")
            draft_id = league_info.get("draft_id")

            if not draft_id:
                drafts = sleeper.get_drafts_for_league(league_id)
                if drafts:
                    draft_id = drafts[0].get("draft_id")

            if not draft_id:
                st.info("No active draft found for this league yet.")
            else:
                picks = sleeper.get_picks(draft_id) or []
                users = sleeper.get_users(league_id) or []
                total_picks_made = len(picks)
                total_picks_all = int(teams_setting) * int(rounds_setting)
                next_overall = total_picks_made + 1

                rnd, pick_in_rnd, slot = utils.snake_position(next_overall, int(teams_setting))
                detected_slot = utils.user_roster_id(users, sleeper_username)
                live_user_slot = int(user_slot_override) if int(user_slot_override) > 0 else (detected_slot or 1)
                you_on_clock = (slot == live_user_slot)

                team_display = utils.slot_to_display_name(slot, users)
                if total_picks_made < total_picks_all:
                    st.markdown(
                        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
                        + ("  🎯 _(That’s you)_" if you_on_clock else "")
                    )
                else:
                    st.success("Draft complete.")

                # ---------- SUGGESTIONS (Top 8) ----------
                st.markdown("### Suggestions for This Pick")
                pool = build_pool()
                players_map = sleeper.get_players_nfl() or {}
                picked_names = sleeper.picked_player_names(picks, players_map)
                pool = utils.remove_players_by_name(pool, picked_names)

                your_names = []
                if live_user_slot:
                    your_names = [
                        (p.get("metadata") or {}).get("first_name", "")
                        for i, p in enumerate(picks, start=1)
                        if utils.snake_position(i, int(teams_setting))[2] == live_user_slot
                    ]

                # Needs counters (above list)
                needs_text = suggestions.needs_summary(pool, your_names)
                if needs_text:
                    st.info(needs_text)

                # Strategy choice (Round 1 only) — stored for the session
                if rnd == 1 and ("live_strategy" not in st.session_state):
                    strat = suggestions.choose_strategy(
                        evaluation.evaluate_players(pool, config),
                        current_overall=next_overall, user_slot=live_user_slot,
                        teams=int(teams_setting), total_rounds=int(rounds_setting),
                        elite_te_value=float(elite_te_value), hero_rb_value=float(hero_rb_value),
                    )
                    st.session_state.live_strategy = strat

                if "live_strategy" in st.session_state:
                    st.info(f"**Recommended strategy:** {st.session_state.live_strategy['name']} — {st.session_state.live_strategy['why']}")

                ranked = suggestions.rank_suggestions(
                    evaluation.evaluate_players(pool, config),
                    round_number=rnd, total_rounds=int(rounds_setting),
                    user_picked_names=your_names, pick_log=picks, teams=int(teams_setting),
                    username=sleeper_username, current_overall=next_overall, user_slot=live_user_slot,
                    kd_only_last_round=bool(kd_only_last_round),
                ).head(8)

                if ranked.empty:
                    st.info("No candidates available.")
                else:
                    for _, row in ranked.iterrows():
                        tag = row.get("next_turn_tag", "")
                        st.markdown(f"**{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — Score: {row['score']:.2f} {tag}")
                        st.caption(row['why'])

                # ---------- Team Rosters (collapsed under suggestions) ----------
                with st.expander("Team Rosters (by position)", expanded=False):
                    rosters = roster.build_rosters(picks, users)
                    if not rosters:
                        st.caption("No picks yet.")
                    else:
                        cols = st.columns(3)
                        i = 0
                        for team_name, ros in rosters.items():
                            with cols[i % 3]:
                                highlight = "🟩 " if team_name.lower().startswith(sleeper_username.lower()) else ""
                                st.write(f"{highlight}**{team_name}**")
                                for pos, players in ros.items():
                                    if players:
                                        st.write(f"- {pos}: {', '.join(players)}")
                            i += 1

# ===================== MOCK DRAFT (Practice Mode) =====================
with tab_mock:
    st.subheader("Mock Draft (Practice Mode)")

    mock_url = st.text_input("Sleeper Mock Draft URL", value="", key="mock_sleeper_url")
    colm1, colm2, colm3 = st.columns([1,1,1])
    load_mock_btn = colm1.button("Load / Re-sync Mock", key="mock_load_btn")
    clear_mock_btn = colm2.button("Reset Practice", key="mock_reset_btn")
    auto_sim = colm3.toggle("Auto-sim to your picks", value=False, key="mock_auto_sim")

    if clear_mock_btn:
        st.session_state.pop("mock_state", None)
        st.session_state.pop("mock_strategy", None)
        st.success("Practice state cleared.")

    pool_all = build_pool()

    if load_mock_btn and mock_url.strip():
        draft_id = sleeper.parse_draft_id_from_url(mock_url.strip())
        if not draft_id:
            st.error("Could not parse a draft_id from that URL. It should contain `/draft/<id>`.")
        else:
            picks = sleeper.get_picks(draft_id) or []
            dmeta = sleeper.get_draft(draft_id) or {}
            teams_from_api = int(dmeta.get("settings", {}).get("teams", config.get("draft", {}).get("teams", 12)) or dmeta.get("teams", 12))
            rounds_from_api = int(dmeta.get("settings", {}).get("rounds", config.get("draft", {}).get("rounds", 15)))
            users = dmeta.get("users") or [{"display_name": f"Team {i}", "roster_id": i} for i in range(1, teams_from_api+1)]
            players_map = sleeper.get_players_nfl() or {}

            picked_names = sleeper.picked_player_names(picks, players_map)
            available = utils.remove_players_by_name(pool_all.copy(), picked_names)

            detected_slot = utils.user_roster_id(users, sleeper_username)
            my_slot = int(user_slot_override) if int(user_slot_override) > 0 else (detected_slot or 1)

            st.session_state.mock_state = {
                "draft_id": draft_id,
                "teams": teams_from_api,
                "rounds": rounds_from_api,
                "users": users,
                "your_roster_id": int(my_slot),
                "pick_log": sleeper.picks_to_internal_log(picks, players_map),
                "available": evaluation.evaluate_players(available, config).reset_index(drop=True),
                "current_pick": len(picks) + 1,
            }
            st.success(f"Mock {draft_id} loaded. Synced {len(picks)} picks.")

    if "mock_state" in st.session_state:
        S = st.session_state.mock_state
        teams = int(S["teams"]); rounds = int(S["rounds"])
        pick_log = S["pick_log"]
        next_overall = len(pick_log) + 1
        rnd, pick_in_rnd, slot = utils.snake_position(next_overall, teams)
        you_on_clock = (slot == S.get("your_roster_id"))

        st.write(f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — Slot {slot}" + ("  🎯 _(That’s you)_" if you_on_clock else ""))

        # Auto-sim to your turn
        if auto_sim and not you_on_clock and rnd <= rounds:
            progressed = 0
            while progressed < 50 and not you_on_clock:
                if S["available"].empty: break
                idx = mock_ai.pick_for_team(S["available"], "Balanced", rnd)
                if idx is None or idx not in S["available"].index: break
                pk = S["available"].loc[idx]
                pick_log.append({"round": rnd, "pick_no": pick_in_rnd, "team": f"Slot {slot}",
                                 "metadata": {"first_name": pk["PLAYER"], "last_name": "", "position": pk.get("POS")}})
                S["available"] = S["available"].drop(idx).reset_index(drop=True)
                next_overall += 1
                rnd, pick_in_rnd, slot = utils.snake_position(next_overall, teams)
                you_on_clock = (slot == S.get("your_roster_id"))
                progressed += 1
            S["pick_log"] = pick_log
            S["current_pick"] = next_overall
            st.session_state.mock_state = S
            st.experimental_rerun()

        # ---------- SUGGESTIONS ----------
        st.markdown("### Suggestions for This Pick")
        # Needs counters
        my_names = [p["metadata"]["first_name"] for p in pick_log
                    if utils.slot_for_overall((p["round"]-1)*teams + p["pick_no"], teams) == S.get("your_roster_id")]
        needs_text = suggestions.needs_summary(S["available"], my_names)
        if needs_text:
            st.info(needs_text)

        # Strategy (Round 1)
        if rnd == 1 and ("mock_strategy" not in st.session_state):
            strat = suggestions.choose_strategy(
                S["available"], current_overall=next_overall, user_slot=S.get("your_roster_id"),
                teams=teams, total_rounds=rounds,
                elite_te_value=float(elite_te_value), hero_rb_value=float(hero_rb_value),
            )
            st.session_state.mock_strategy = strat
        if "mock_strategy" in st.session_state:
            st.info(f"**Recommended strategy:** {st.session_state.mock_strategy['name']} — {st.session_state.mock_strategy['why']}")

        ranked = suggestions.rank_suggestions(
            S["available"], round_number=rnd, total_rounds=rounds,
            user_picked_names=my_names, pick_log=pick_log, teams=teams,
            username=sleeper_username, current_overall=next_overall, user_slot=S.get("your_roster_id"),
            kd_only_last_round=bool(kd_only_last_round),
        ).head(8)

        if ranked.empty:
            st.info("No candidates available.")
        else:
            for _, row in ranked.iterrows():
                tag = row.get("next_turn_tag", "")
                st.markdown(f"**{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — Score: {row['score']:.2f} {tag}")
                st.caption(row['why'])

        # ---------- Team Rosters (collapsed) ----------
        with st.expander("Team Rosters (by position)", expanded=False):
            users = S.get("users") or [{"display_name": f"Team {i}", "roster_id": i} for i in range(1, teams+1)]
            simple_picks = []
            for p in pick_log:
                overall = (p["round"] - 1) * teams + p["pick_no"]
                _, _, sl = utils.snake_position(overall, teams)
                simple_picks.append({"roster_id": sl, "metadata": p.get("metadata", {})})
            ros = roster.build_rosters(simple_picks, users)
            cols = st.columns(3)
            i = 0
            for team_name, rmap in ros.items():
                with cols[i % 3]:
                    highlight = "🟩 " if team_name.lower().startswith(sleeper_username.lower()) else ""
                    st.write(f"{highlight}**{team_name}**")
                    for pos, players in rmap.items():
                        if players:
                            st.write(f"- {pos}: {', '.join(players)}")
                i += 1

# ===================== SUGGESTIONS (global glance) =====================
with tab_suggest:
    st.subheader("Global Suggestions Snapshot")
    base_df = build_pool()
    evaluated = evaluation.evaluate_players(base_df, config)
    ranked = suggestions.rank_suggestions(
        evaluated, round_number=1, total_rounds=int(rounds_setting),
        user_picked_names=[], pick_log=[], teams=int(teams_setting),
        username=sleeper_username, current_overall=1, user_slot=int(user_slot_override) or 1,
        kd_only_last_round=bool(kd_only_last_round),
    ).head(8)
    st.info(suggestions.needs_summary(evaluated, []) or "")
    for _, row in ranked.iterrows():
        tag = row.get("next_turn_tag", "")
        st.markdown(f"**{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — Score: {row['score']:.2f} {tag}")
        st.caption(row['why'])

# ===================== EXPORT =====================
with tab_export:
    st.subheader("Export & PDF")
    picks_for_pdf = []
    my_slot = None
    if "mock_state" in st.session_state:
        S = st.session_state.mock_state
        picks_for_pdf = S.get("pick_log", [])
        my_slot = S.get("your_roster_id")
    league_name = (league_info or {}).get("name") if league_info else "My League"
    if st.button("Download Draft PDF", key="export_pdf_btn"):
        pdf_bytes = pdf_report.generate_pdf(league_name, picks_for_pdf, my_slot)
        st.download_button(
            "Download draft_report.pdf",
            data=pdf_bytes,
            file_name="draft_report.pdf",
            mime="application/pdf",
            key="export_pdf_dl",
        )
