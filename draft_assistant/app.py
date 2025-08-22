"""
Fantasy Football Draft Assistant (Streamlit)
Now with:
- Mock Draft that mirrors Live Draft UI, driven by a Sleeper Mock URL
- Unified suggestions engine for Live + Mock (Top 8 + explanations)
- Late-round rookie upside bias
- Recognize user by Sleeper username (default: Fallon3D)
- Player Board removed
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

# Sleeper identification
sleeper_username = st.sidebar.text_input(
    "Your Sleeper Username",
    value=str(config.get("user_profile", {}).get("sleeper_username", "Fallon3D")),
    key="settings_username",
)
league_id = st.sidebar.text_input(
    "Sleeper League ID (Live)",
    value=str(config.get("sleeper", {}).get("league_id", "")),
    key="settings_league_id",
)
poll_seconds = st.sidebar.number_input(
    "Live Poll Interval (seconds)",
    min_value=3,
    max_value=30,
    value=int(config.get("sleeper", {}).get("poll_seconds", 5)),
    step=1,
    key="settings_poll_sec",
)

# Draft dims (defaults for both modes)
teams_default = int(config.get("draft", {}).get("teams", 12))
rounds_default = int(config.get("draft", {}).get("rounds", 15))
teams_setting = st.sidebar.number_input(
    "Default: Number of Teams",
    min_value=2,
    value=teams_default,
    key="settings_num_teams",
)
rounds_setting = st.sidebar.number_input(
    "Default: Number of Rounds",
    min_value=1,
    value=rounds_default,
    key="settings_num_rounds",
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
    config.setdefault("sleeper", {})["league_id"] = league_id.strip()
    config["sleeper"]["poll_seconds"] = int(poll_seconds)
    config.setdefault("draft", {})["teams"] = int(teams_setting)
    config["draft"]["rounds"] = int(rounds_setting)
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
                your_roster_id = utils.user_roster_id(users, sleeper_username)
                you_on_clock = (slot == your_roster_id)

                team_display = utils.slot_to_display_name(slot, users) or f"Slot {slot}"
                if total_picks_made < total_picks_all:
                    st.markdown(
                        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
                        + ("  🎯 _(That’s you)_" if you_on_clock else "")
                    )
                else:
                    st.success("Draft complete.")

                # Team Rosters
                st.markdown("### Team Rosters")
                rosters = roster.build_rosters(picks, users)
                if not rosters:
                    st.info("No picks yet.")
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

                # ---------- SUGGESTIONS (Top 8) ----------
                st.markdown("### Suggestions for This Pick")
                # Build available pool: uploads + (optionally) Sleeper live players
                available_pool = utils.load_combined_player_pool(
                    include_sleeper=True,
                    base_path=os.path.join(DATA_DIR, "sample_players.csv"),
                    extra_paths=[
                        os.path.join(DATA_DIR, "extra_players_1.csv"),
                        os.path.join(DATA_DIR, "extra_players_2.csv"),
                        os.path.join(DATA_DIR, "extra_players_3.csv"),
                    ],
                )
                players_map = sleeper.get_players_nfl() or {}
                picked_names = sleeper.picked_player_names(picks, players_map)
                available_pool = utils.remove_players_by_name(available_pool, picked_names)
                available_pool = utils.normalize_player_headers(available_pool)

                # Your current roster (by names) for needs/bye checks
                your_name = utils.roster_display_name(users, your_roster_id) if your_roster_id else ""
                your_picks = [p for p in picks if str(p.get("roster_id")) == str(your_roster_id)]
                your_names = [ (p.get("metadata") or {}).get("first_name","") for p in your_picks ]

                ranked = suggestions.rank_suggestions(
                    available_pool,
                    round_number=rnd,
                    total_rounds=int(rounds_setting),
                    user_picked_names=your_names,
                    pick_log=picks,
                    teams=int(teams_setting),
                    username=sleeper_username,
                ).head(8)

                if ranked.empty:
                    st.info("No candidates available.")
                else:
                    for _, row in ranked.iterrows():
                        st.markdown(f"**{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — Score: {row['score']:.2f}")
                        st.caption(row['why'])

# ===================== MOCK DRAFT (mirrors Live Draft, from Sleeper Mock URL) =====================
with tab_mock:
    st.subheader("Mock Draft (Practice Mode)")

    mock_url = st.text_input("Sleeper Mock Draft URL", value="", key="mock_sleeper_url")
    colm1, colm2, colm3 = st.columns([1,1,1])
    load_mock_btn = colm1.button("Load / Re-sync Mock", key="mock_load_btn")
    clear_mock_btn = colm2.button("Reset Practice", key="mock_reset_btn")
    auto_sim = colm3.toggle("Auto-sim to your picks", value=False, key="mock_auto_sim")

    if clear_mock_btn:
        st.session_state.pop("mock_state", None)
        st.success("Practice state cleared.")

    # Build full pool up-front
    combined_pool = utils.load_combined_player_pool(
        include_sleeper=True,
        base_path=os.path.join(DATA_DIR, "sample_players.csv"),
        extra_paths=[
            os.path.join(DATA_DIR, "extra_players_1.csv"),
            os.path.join(DATA_DIR, "extra_players_2.csv"),
            os.path.join(DATA_DIR, "extra_players_3.csv"),
        ],
    )
    combined_pool = utils.normalize_player_headers(combined_pool)

    if load_mock_btn and mock_url.strip():
        draft_id = sleeper.parse_draft_id_from_url(mock_url.strip())
        if not draft_id:
            st.error("Could not parse a draft_id from that URL. It should contain `/draft/<id>`.")
        else:
            picks = sleeper.get_picks(draft_id) or []
            dmeta = sleeper.get_draft(draft_id) or {}
            teams_from_api = int(dmeta.get("settings", {}).get("teams", config.get("draft", {}).get("teams", 12)) or dmeta.get("teams", 12))
            rounds_from_api = int(dmeta.get("settings", {}).get("rounds", config.get("draft", {}).get("rounds", 15)))
            users = dmeta.get("users") or []  # some mock endpoints include users; if not, we simulate
            players_map = sleeper.get_players_nfl() or {}

            picked_names = sleeper.picked_player_names(picks, players_map)
            available = utils.remove_players_by_name(combined_pool.copy(), picked_names)
            your_roster_id = utils.user_roster_id(users, sleeper_username) or 1  # fallback to 1 if unknown

            st.session_state.mock_state = {
                "draft_id": draft_id,
                "teams": teams_from_api,
                "rounds": rounds_from_api,
                "users": users,
                "your_roster_id": int(your_roster_id),
                "pick_log": sleeper.picks_to_internal_log(picks, players_map),
                "available": evaluation.evaluate_players(available, config).reset_index(drop=True),
                "current_pick": len(picks) + 1,
            }
            st.success(f"Mock {draft_id} loaded. Synced {len(picks)} picks.")

    # If we have state, show the same UI as Live
    if "mock_state" in st.session_state:
        S = st.session_state.mock_state
        teams = int(S["teams"]); rounds = int(S["rounds"])
        pick_log = S["pick_log"]
        next_overall = len(pick_log) + 1
        rnd, pick_in_rnd, slot = utils.snake_position(next_overall, teams)
        you_on_clock = (slot == S.get("your_roster_id"))

        st.write(f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — Slot {slot}" + ("  🎯 _(That’s you)_" if you_on_clock else ""))

        # Team Rosters (from pick_log)
        st.markdown("### Team Rosters")
        # fabricate users list for display if missing
        users = S.get("users") or [{"display_name": f"Team {i}", "roster_id": i} for i in range(1, teams+1)]
        # Convert internal pick_log to Sleeper-like simple picks for roster builder
        simple_picks = []
        for p in pick_log:
            # Estimate roster_id by snake order slot at that overall
            r, pr, sl = utils.snake_position( (p["round"]-1)*teams + p["pick_no"], teams )
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

        # Auto-sim other teams until it's your turn
        if auto_sim and not you_on_clock and rnd <= rounds:
            progressed = 0
            while progressed < 50 and not you_on_clock:
                # AI pick for this slot
                if S["available"].empty: break
                idx = mock_ai.pick_for_team(S["available"], "Balanced", rnd)
                if idx is None or idx not in S["available"].index: break
                pk = S["available"].loc[idx]
                pick_log.append({"round": rnd, "pick_no": pick_in_rnd, "team": f"Slot {slot}",
                                 "metadata": {"first_name": pk["PLAYER"], "last_name": "", "position": pk.get("POS")}})
                S["available"] = S["available"].drop(idx).reset_index(drop=True)
                # advance
                next_overall += 1
                rnd, pick_in_rnd, slot = utils.snake_position(next_overall, teams)
                you_on_clock = (slot == S.get("your_roster_id"))
                progressed += 1
            S["pick_log"] = pick_log
            S["current_pick"] = next_overall
            st.session_state.mock_state = S
            st.experimental_rerun()

        # Suggestions (Top 8) for practice pick
        st.markdown("### Suggestions for This Pick")
        ranked = suggestions.rank_suggestions(
            S["available"],
            round_number=rnd,
            total_rounds=rounds,
            user_picked_names=[p["metadata"]["first_name"] for p in pick_log if utils.slot_for_overall((p["round"]-1)*teams + p["pick_no"], teams) == S.get("your_roster_id")],
            pick_log=pick_log,
            teams=teams,
            username=sleeper_username,
        ).head(8)

        if ranked.empty:
            st.info("No candidates available.")
        else:
            for _, row in ranked.iterrows():
                st.markdown(f"**{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — Score: {row['score']:.2f}")
                st.caption(row['why'])

        # Practice: make your pick locally
        if you_on_clock and rnd <= rounds:
            choice = st.selectbox("Choose your pick (practice)", options=ranked["PLAYER"].tolist(), key="mock_choice_box")
            if st.button("Draft Selected Player (practice)", key="mock_pick_now"):
                sel_idx = S["available"][S["available"]["PLAYER"] == choice].index
                if not sel_idx.empty:
                    idx = int(sel_idx[0])
                    pk = S["available"].loc[idx]
                    pick_log.append({"round": rnd, "pick_no": pick_in_rnd, "team": "You",
                                     "metadata": {"first_name": pk["PLAYER"], "last_name": "", "position": pk.get("POS")}})
                    S["available"] = S["available"].drop(idx).reset_index(drop=True)
                    S["current_pick"] = next_overall + 1
                    st.session_state.mock_state = S
                    st.experimental_rerun()

# ===================== SUGGESTIONS (global glance) =====================
with tab_suggest:
    st.subheader("Global Suggestions Snapshot")
    base_df = utils.load_combined_player_pool(
        include_sleeper=True,
        base_path=os.path.join(DATA_DIR, "sample_players.csv"),
        extra_paths=[
            os.path.join(DATA_DIR, "extra_players_1.csv"),
            os.path.join(DATA_DIR, "extra_players_2.csv"),
            os.path.join(DATA_DIR, "extra_players_3.csv"),
        ],
    )
    base_df = utils.normalize_player_headers(base_df)
    evaluated = evaluation.evaluate_players(base_df, config)
    # No pick context here; just top 8 values
    ranked = suggestions.rank_suggestions(
        evaluated, round_number=1, total_rounds=int(rounds_setting),
        user_picked_names=[], pick_log=[], teams=int(teams_setting), username=sleeper_username,
    ).head(8)
    for _, row in ranked.iterrows():
        st.markdown(f"**{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — Score: {row['score']:.2f}")
        st.caption(row['why'])

# ===================== EXPORT =====================
with tab_export:
    st.subheader("Export & PDF")
    picks_for_pdf = []
    my_slot = None
    # Prefer mock practice picks if present
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
