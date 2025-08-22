"""
Fantasy Football Draft Assistant (Streamlit)
- Live draft sync (Sleeper public API)
- Mock Draft with AI archetypes
- Player evaluation (VBD-ish) + suggestions
- PDF export (ReportLab)
All zero-cost sources. Robust to missing schedule/coaching files.
"""

import os
import random
import pandas as pd
import streamlit as st

# --- App-wide defaults / state guards ---
st.set_page_config(page_title="Fantasy Football Draft Assistant", layout="wide")
# Ensure this exists even if Live tab never ran
league_info = None  # used by Export tab; avoids NameError

# Local imports
from core import utils, sleeper, roster, evaluation, suggestions, mock_ai, pdf_report

# ---------- CONFIG ----------
config = utils.read_config()

# ---------- SIDEBAR SETTINGS ----------
st.sidebar.title("Settings")

# Sleeper
league_id = st.sidebar.text_input(
    "Sleeper League ID",
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

# Draft dims (global defaults; Mock tab can override)
teams_default = int(config.get("draft", {}).get("teams", 12))
rounds_default = int(config.get("draft", {}).get("rounds", 15))
teams_setting = st.sidebar.number_input(
    "Number of Teams (default)",
    min_value=2,
    value=teams_default,
    key="settings_num_teams",
)
rounds_setting = st.sidebar.number_input(
    "Number of Rounds (default)",
    min_value=1,
    value=rounds_default,
    key="settings_num_rounds",
)

# Uploads to overwrite sample files
st.sidebar.markdown("### Data Uploads")
players_file = st.sidebar.file_uploader("Players CSV", type="csv", key="upload_players")
schedule_file = st.sidebar.file_uploader("Schedule CSV", type="csv", key="upload_schedule")

if st.sidebar.button("Save Settings", key="settings_save_btn"):
    config.setdefault("sleeper", {})["league_id"] = league_id.strip()
    config["sleeper"]["poll_seconds"] = int(poll_seconds)
    config.setdefault("draft", {})["teams"] = int(teams_setting)
    config["draft"]["rounds"] = int(rounds_setting)
    utils.save_config(config)
    st.sidebar.success("Saved to config.toml")

# Persist uploaded files into the data/ folder next to app.py
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

if players_file is not None:
    try:
        df_u = pd.read_csv(players_file)
        df_u.to_csv(os.path.join(DATA_DIR, "sample_players.csv"), index=False)
        st.sidebar.success("Players CSV uploaded.")
    except Exception as e:
        st.sidebar.error(f"Players CSV error: {e}")

if schedule_file is not None:
    try:
        df_s = pd.read_csv(schedule_file)
        df_s.to_csv(os.path.join(DATA_DIR, "sample_schedule.csv"), index=False)
        st.sidebar.success("Schedule CSV uploaded.")
    except Exception as e:
        st.sidebar.error(f"Schedule CSV error: {e}")

# ---------- HEADER ----------
st.title("🏈 Fantasy Football Draft Assistant")

# ---------- TABS ----------
tab_live, tab_mock, tab_board, tab_suggest, tab_export = st.tabs(
    ["Live Draft", "Mock Draft", "Player Board", "Suggestions", "Export"]
)

# ===================== LIVE DRAFT =====================
with tab_live:
    st.subheader("Live Draft (Sleeper)")

    # Auto-refresh controls (no st.autorefresh; use safe alternatives)
    col1, col2 = st.columns([1, 1])
    auto = col1.toggle("Auto-refresh", value=False, key="live_auto_refresh_toggle")
    col2.button("Refresh now", on_click=st.experimental_rerun, key="live_refresh_btn")

    if auto:
        # Lightweight page refresh (reloads the app) every poll_seconds
        # This is simple and Cloud-friendly. Turn off if you prefer manual.
        st.markdown(
            f"<meta http-equiv='refresh' content='{int(poll_seconds)}'>",
            unsafe_allow_html=True,
        )
        st.caption(f"Auto-refresh every {int(poll_seconds)}s (polite to Sleeper API).")

    if not league_id:
        st.info("Add your Sleeper League ID in the sidebar, then click **Save Settings**.")
    else:
        # Pull league & draft
        league_info = sleeper.get_league_info(league_id)
        if not league_info:
            st.error("Could not fetch league info. Check the League ID and try again.")
        else:
            league_name = league_info.get("name") or league_id
            st.write(f"**League:** {league_name}")
            draft_id = league_info.get("draft_id")

            if not draft_id:
                # Fallback: last draft for league
                drafts = sleeper.get_drafts_for_league(league_id)
                if drafts:
                    draft_id = drafts[0].get("draft_id")

            if not draft_id:
                st.info("No active draft found for this league yet.")
            else:
                picks = sleeper.get_picks(draft_id) or []
                users = sleeper.get_users(league_id) or []

                # Current pick / draft complete
                total_picks_made = len(picks)
                total_picks_all = int(teams_setting) * int(rounds_setting)
                next_overall = total_picks_made + 1

                rnd, pick_in_rnd, slot = utils.snake_position(next_overall, int(teams_setting))
                # map slot -> team name if possible
                team_display = utils.slot_to_display_name(slot, users) or f"Slot {slot}"

                if total_picks_made < total_picks_all:
                    st.markdown(
                        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
                    )
                else:
                    st.success("Draft complete.")

                # Team Rosters by Position
                st.markdown("### Team Rosters")
                rosters = roster.build_rosters(picks, users)
                if not rosters:
                    st.info("No picks yet.")
                else:
                    cols = st.columns(3)
                    i = 0
                    for team_name, ros in rosters.items():
                        with cols[i % 3]:
                            st.write(f"**{team_name}**")
                            for pos, players in ros.items():
                                if players:
                                    st.write(f"- {pos}: {', '.join(players)}")
                        i += 1

# ===================== MOCK DRAFT =====================
with tab_mock:
    st.subheader("Mock Draft Simulator")

    # Controls
    user_slot = st.number_input(
        "Your Draft Slot (1 = first pick)",
        min_value=1,
        max_value=int(config.get("draft", {}).get("teams", 12)),
        value=1,
        key="mock_user_slot",
    )
    num_teams = st.number_input(
        "Number of Teams",
        min_value=2,
        value=int(config.get("draft", {}).get("teams", 12)),
        key="mock_num_teams",
    )
    num_rounds = st.number_input(
        "Number of Rounds",
        min_value=1,
        value=int(config.get("draft", {}).get("rounds", 15)),
        key="mock_num_rounds",
    )
    c_start, c_pause, c_reset = st.columns([1, 1, 1])
    start = c_start.button("Start / Resume", key="mock_start")
    pause = c_pause.button("Pause", key="mock_pause")
    reset = c_reset.button("Reset", key="mock_reset")

    # Init / Reset session
    if reset:
        st.session_state.pop("mock_state", None)
        st.success("Mock draft reset.")

    # Load player data
    players_path = os.path.join(DATA_DIR, "sample_players.csv")
    try:
        players_df = pd.read_csv(players_path)
    except Exception:
        players_df = pd.DataFrame()

    players_df = utils.normalize_player_headers(players_df)

    # Start mock
    if ("mock_state" not in st.session_state) and start:
        evaluated = evaluation.evaluate_players(players_df, config)
        # Assign AI strategies
        strategies = list(mock_ai.STRATEGIES)
        random.shuffle(strategies)
        teams_list = []
        for i in range(1, int(num_teams) + 1):
            strategy = strategies[(i - 1) % len(strategies)]
            teams_list.append(
                {"slot": i, "name": "You" if i == int(user_slot) else f"AI {i}", "strategy": strategy, "picks": []}
            )
        st.session_state.mock_state = {
            "running": True,
            "teams": teams_list,
            "available": evaluated.reset_index(drop=True),
            "picks": [],
            "current_pick": 1,
            "user_slot": int(user_slot),
            "num_teams": int(num_teams),
            "num_rounds": int(num_rounds),
        }

    # Pause/Resume
    if "mock_state" in st.session_state:
        if pause:
            st.session_state.mock_state["running"] = False
        if start:
            st.session_state.mock_state["running"] = True

    # Simulation loop (advance AI until user's turn or paused)
    if "mock_state" in st.session_state:
        S = st.session_state.mock_state
        if S["running"]:
            progressed = 0
            # Hard cap steps per rerun to keep UI responsive
            while progressed < 50:
                rnd, pick_in_rnd, slot = utils.snake_position(S["current_pick"], S["num_teams"])
                if rnd > S["num_rounds"]:
                    S["running"] = False
                    break
                if slot == S["user_slot"]:
                    # stop at user's pick
                    break
                # AI picks
                team = next((t for t in S["teams"] if t["slot"] == slot), None)
                if team is None or S["available"].empty:
                    S["running"] = False
                    break
                idx = mock_ai.pick_for_team(S["available"], team["strategy"], rnd)
                if idx is None or idx not in S["available"].index:
                    S["running"] = False
                    break
                pick = S["available"].loc[idx]
                team["picks"].append(pick["PLAYER"])
                S["picks"].append(
                    {"round": rnd, "pick_no": pick_in_rnd, "team": team["name"],
                     "metadata": {"first_name": pick["PLAYER"], "last_name": "", "position": pick.get("POS")}}
                )
                S["available"] = S["available"].drop(idx).reset_index(drop=True)
                S["current_pick"] += 1
                progressed += 1
            st.session_state.mock_state = S

        # Show current state
        S = st.session_state.mock_state
        rnd, pick_in_rnd, slot = utils.snake_position(S["current_pick"], S["num_teams"])
        if rnd <= S["num_rounds"]:
            st.write(f"**Current:** Round {rnd}, Pick {pick_in_rnd} — {'Your turn' if slot == S['user_slot'] else 'AI'}")

        # If it's user's turn, suggest + pick UI
        if slot == S["user_slot"] and rnd <= S["num_rounds"]:
            st.markdown("### Your Pick")
            top5 = suggestions.top_suggestions(S["available"], None, 5)
            if not top5.empty:
                choice = st.selectbox("Top suggestions", options=top5["PLAYER"].tolist(), key="mock_choice_box")
                pick_now = st.button("Draft Selected Player", key="mock_pick_now")
                if pick_now:
                    sel_idx = S["available"][S["available"]["PLAYER"] == choice].index
                    if not sel_idx.empty:
                        idx = int(sel_idx[0])
                        pick = S["available"].loc[idx]
                        S["picks"].append(
                            {"round": rnd, "pick_no": pick_in_rnd, "team": "You",
                             "metadata": {"first_name": pick["PLAYER"], "last_name": "", "position": pick.get("POS")}}
                        )
                        S["available"] = S["available"].drop(idx).reset_index(drop=True)
                        S["current_pick"] += 1
                        st.session_state.mock_state = S
                        st.experimental_rerun()
            else:
                st.info("No available players to pick.")

        # Pick Log
        with st.expander("Pick Log", expanded=True):
            if S["picks"]:
                for p in S["picks"][-25:]:
                    st.write(f"Round {p['round']} • Pick {p['pick_no']} • {p['team']} → {p['metadata']['first_name']}")
            else:
                st.caption("No picks yet.")

        # Your roster
        your_roster = [p["metadata"]["first_name"] for p in S["picks"] if p["team"] == "You"]
        st.markdown("### Your Roster")
        if your_roster:
            st.write(", ".join(your_roster))
        else:
            st.caption("No players yet.")

# ===================== PLAYER BOARD =====================
with tab_board:
    st.subheader("Player Board")
    players_path = os.path.join(DATA_DIR, "sample_players.csv")
    try:
        df_players = pd.read_csv(players_path)
    except Exception:
        df_players = pd.DataFrame()

    df_players = utils.normalize_player_headers(df_players)
    if df_players.empty:
        st.info("No player data. Upload a CSV in the sidebar or keep the sample.")
    else:
        pos_filter = st.multiselect(
            "Position", options=sorted(df_players["POS"].dropna().unique().tolist()),
            default=sorted(df_players["POS"].dropna().unique().tolist()),
            key="pb_pos"
        )
        team_filter = st.multiselect(
            "Team", options=sorted(df_players["TEAM"].dropna().unique().tolist()),
            default=sorted(df_players["TEAM"].dropna().unique().tolist()),
            key="pb_team"
        )
        tier_filter = st.multiselect(
            "Tier", options=sorted(df_players["TIERS"].dropna().unique().astype(str).tolist()),
            default=sorted(df_players["TIERS"].dropna().unique().astype(str).tolist()),
            key="pb_tier"
        )

        filtered = df_players[
            (df_players["POS"].isin(pos_filter)) &
            (df_players["TEAM"].isin(team_filter)) &
            (df_players["TIERS"].astype(str).isin(tier_filter))
        ].copy()

        st.dataframe(filtered[["RK", "PLAYER", "POS", "TEAM", "BYE", "TIERS"]].reset_index(drop=True), use_container_width=True)

        # Draft buttons if it's user's turn in mock mode
        if "mock_state" in st.session_state:
            S = st.session_state.mock_state
            rnd, pick_in_rnd, slot = utils.snake_position(S["current_pick"], S["num_teams"])
            if slot == S.get("user_slot") and rnd <= S.get("num_rounds", 0):
                st.markdown("#### Draft from Board")
                # Display up to 30 rows with draft buttons
                show = filtered.head(30)
                for _, row in show.iterrows():
                    cols = st.columns([5, 2])
                    cols[0].write(f"{int(row['RK'])}. {row['PLAYER']} — {row['POS']} / {row['TEAM']} (Tier {row['TIERS']})")
                    if cols[1].button("Draft", key=f"pb_draft_{row['PLAYER']}"):
                        # perform draft
                        idxs = S["available"][S["available"]["PLAYER"] == row["PLAYER"]].index
                        if not idxs.empty:
                            idx = int(idxs[0])
                            pick = S["available"].loc[idx]
                            S["picks"].append(
                                {"round": rnd, "pick_no": pick_in_rnd, "team": "You",
                                 "metadata": {"first_name": pick["PLAYER"], "last_name": "", "position": pick.get("POS")}}
                            )
                            S["available"] = S["available"].drop(idx).reset_index(drop=True)
                            S["current_pick"] += 1
                            st.session_state.mock_state = S
                            st.experimental_rerun()

# ===================== SUGGESTIONS =====================
with tab_suggest:
    st.subheader("Suggestions")
    players_path = os.path.join(DATA_DIR, "sample_players.csv")
    try:
        base_df = pd.read_csv(players_path)
    except Exception:
        base_df = pd.DataFrame()

    base_df = utils.normalize_player_headers(base_df)
    if base_df.empty:
        st.info("No player data available.")
    else:
        evaluated = evaluation.evaluate_players(base_df, config)
        top20 = evaluated.head(20)
        for _, row in top20.iterrows():
            st.write(
                f"{int(row['RK'])}. **{row['PLAYER']}** ({row['POS']}, {row['TEAM']}) — "
                f"Value: {row['value']:.1f} | VBD: {row['vbd']:.1f} "
                f"| _{row['notes']}_"
            )

# ===================== EXPORT =====================
with tab_export:
    st.subheader("Export & PDF")
    st.write("Generate a simple PDF report of your draft (mock or live).")
    picks_for_pdf = []
    my_slot = None
    # Prefer mock picks (since we track them locally)
    if "mock_state" in st.session_state:
        S = st.session_state.mock_state
        picks_for_pdf = S.get("picks", [])
        my_slot = S.get("user_slot")
    # Fallback league name
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
