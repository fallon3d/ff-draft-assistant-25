"""
Fantasy Football Draft Assistant (Streamlit)
- Live draft sync (Sleeper public API)
- Mock Draft with AI archetypes OR read-only sync from a Sleeper Mock URL
- Player evaluation (VBD-ish) + suggestions
- PDF export (ReportLab)
- Combine player pool from Sleeper live list + up to 3 uploaded CSVs + sample CSV
All zero-cost sources. Robust to missing schedule/coaching files.
"""

import os
import random
import pandas as pd
import streamlit as st

# --- App-wide defaults / state guards ---
st.set_page_config(page_title="Fantasy Football Draft Assistant", layout="wide")
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

# Uploads (write into draft_assistant/data/)
st.sidebar.markdown("### Data Uploads")
players_file = st.sidebar.file_uploader("Players CSV (base)", type="csv", key="upload_players")
extra1_file = st.sidebar.file_uploader("Extra Players CSV #1", type="csv", key="upload_extra1")
extra2_file = st.sidebar.file_uploader("Extra Players CSV #2", type="csv", key="upload_extra2")
extra3_file = st.sidebar.file_uploader("Extra Players CSV #3", type="csv", key="upload_extra3")
schedule_file = st.sidebar.file_uploader("Schedule CSV (optional)", type="csv", key="upload_schedule")

if st.sidebar.button("Save Settings", key="settings_save_btn"):
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

# ---------- TABS ----------
tab_live, tab_mock, tab_board, tab_suggest, tab_export = st.tabs(
    ["Live Draft", "Mock Draft", "Player Board", "Suggestions", "Export"]
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
                team_display = utils.slot_to_display_name(slot, users) or f"Slot {slot}"

                if total_picks_made < total_picks_all:
                    st.markdown(
                        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
                    )
                else:
                    st.success("Draft complete.")

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

# ===================== MOCK DRAFT (supports Sleeper Mock URL) =====================
with tab_mock:
    st.subheader("Mock Draft")

    # Data source controls
    st.markdown("#### Data Source")
    use_sleeper_players = st.checkbox("Include Sleeper live player list", value=True, key="mock_include_sleeper_players")

    # Allow user to provide a Sleeper Mock URL to read current picks
    mock_url = st.text_input("Sleeper Mock Draft URL (optional)", value="", key="mock_sleeper_url")
    load_mock_btn = st.button("Load from Sleeper Mock URL", key="mock_load_btn")

    # Controls for local simulation (still used after loading a Sleeper mock)
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
    c1, c2, c3 = st.columns([1, 1, 1])
    start = c1.button("Start / Resume (AI sim)", key="mock_start")
    pause = c2.button("Pause", key="mock_pause")
    reset = c3.button("Reset", key="mock_reset")

    # Init / Reset
    if reset:
        st.session_state.pop("mock_state", None)
        st.success("Mock draft reset.")

    # Build the combined player pool (Sleeper + CSVs)
    combined_df = utils.load_combined_player_pool(
        include_sleeper=use_sleeper_players,
        base_path=os.path.join(DATA_DIR, "sample_players.csv"),
        extra_paths=[
            os.path.join(DATA_DIR, "extra_players_1.csv"),
            os.path.join(DATA_DIR, "extra_players_2.csv"),
            os.path.join(DATA_DIR, "extra_players_3.csv"),
        ],
    )

    # If the user provided a Sleeper mock URL, load its picks and sync state
    if load_mock_btn and mock_url.strip():
        draft_id = sleeper.parse_draft_id_from_url(mock_url.strip())
        if not draft_id:
            st.error("Could not parse a draft_id from that URL. It should contain `/draft/<id>`.")
        else:
            picks = sleeper.get_picks(draft_id) or []
            players_map = sleeper.get_players_nfl() or {}
            picked_names = sleeper.picked_player_names(picks, players_map)  # list of strings
            # Remove already-picked players from availability
            available = utils.remove_players_by_name(combined_df.copy(), picked_names)
            # Try to read rounds/teams from draft if available
            dmeta = sleeper.get_draft(draft_id) or {}
            rounds_from_api = int(dmeta.get("settings", {}).get("rounds", num_rounds))
            teams_from_api = int(dmeta.get("settings", {}).get("teams", num_teams) or dmeta.get("teams", num_teams))

            # Initialize a read-only synced state (user can continue the draft locally if desired)
            st.session_state.mock_state = {
                "running": False,                   # paused by default (read-only)
                "synced_from_sleeper": True,
                "sleeper_draft_id": draft_id,
                "teams": [{"slot": i, "name": "You" if i == int(user_slot) else f"Team {i}", "strategy": "Balanced", "picks": []}
                          for i in range(1, int(teams_from_api) + 1)],
                "available": evaluation.evaluate_players(available, config).reset_index(drop=True),
                "picks": sleeper.picks_to_internal_log(picks, players_map),
                "current_pick": len(picks) + 1,
                "user_slot": int(user_slot),
                "num_teams": int(teams_from_api),
                "num_rounds": int(rounds_from_api),
            }
            st.success(f"Loaded Sleeper mock {draft_id}. Synced {len(picks)} picks.")

    # Start AI simulation from our local combined pool (if no synced state or user wants to simulate)
    if ("mock_state" not in st.session_state) and start:
        evaluated = evaluation.evaluate_players(combined_df, config)
        strategies = list(mock_ai.STRATEGIES)
        random.shuffle(strategies)
        teams_list = []
        for i in range(1, int(num_teams) + 1):
            strategy = strategies[(i - 1) % len(strategies)]
            teams_list.append({"slot": i, "name": "You" if i == int(user_slot) else f"AI {i}", "strategy": strategy, "picks": []})
        st.session_state.mock_state = {
            "running": True,
            "synced_from_sleeper": False,
            "sleeper_draft_id": None,
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

    # Simulation / UI
    if "mock_state" in st.session_state:
        S = st.session_state.mock_state

        # If this mock was loaded from Sleeper, allow a quick resync button
        if S.get("synced_from_sleeper") and S.get("sleeper_draft_id"):
            if st.button("Re-sync from Sleeper Mock", key="mock_resync_btn"):
                picks = sleeper.get_picks(S["sleeper_draft_id"]) or []
                players_map = sleeper.get_players_nfl() or {}
                picked_names = sleeper.picked_player_names(picks, players_map)
                # Rebuild from current combined pool to capture any new uploads
                combined_df = utils.load_combined_player_pool(
                    include_sleeper=use_sleeper_players,
                    base_path=os.path.join(DATA_DIR, "sample_players.csv"),
                    extra_paths=[
                        os.path.join(DATA_DIR, "extra_players_1.csv"),
                        os.path.join(DATA_DIR, "extra_players_2.csv"),
                        os.path.join(DATA_DIR, "extra_players_3.csv"),
                    ],
                )
                available = utils.remove_players_by_name(combined_df.copy(), picked_names)
                S["available"] = evaluation.evaluate_players(available, config).reset_index(drop=True)
                S["picks"] = sleeper.picks_to_internal_log(picks, players_map)
                S["current_pick"] = len(picks) + 1
                st.session_state.mock_state = S
                st.success(f"Re-synced. Now at {len(picks)} picks.")

        # If running AI, advance until user's turn
        if S["running"]:
            progressed = 0
            while progressed < 50:
                rnd, pick_in_rnd, slot = utils.snake_position(S["current_pick"], S["num_teams"])
                if rnd > S["num_rounds"]:
                    S["running"] = False
                    break
                if slot == S["user_slot"]:
                    break
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

        # Current status
        S = st.session_state.mock_state
        rnd, pick_in_rnd, slot = utils.snake_position(S["current_pick"], S["num_teams"])
        if rnd <= S["num_rounds"]:
            st.write(f"**Current:** Round {rnd}, Pick {pick_in_rnd} — {'Your turn' if slot == S['user_slot'] else 'AI/Other'}")

        # If it's user's turn, offer suggestions from the combined pool (with picked removed)
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
                for p in S["picks"][-50:]:
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
    df_players = utils.load_combined_player_pool(
        include_sleeper=True,   # always show everything here for browsing
        base_path=os.path.join(DATA_DIR, "sample_players.csv"),
        extra_paths=[
            os.path.join(DATA_DIR, "extra_players_1.csv"),
            os.path.join(DATA_DIR, "extra_players_2.csv"),
            os.path.join(DATA_DIR, "extra_players_3.csv"),
        ],
    )
    df_players = utils.normalize_player_headers(df_players)
    if df_players.empty:
        st.info("No player data found. Upload CSVs in the sidebar.")
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
        st.dataframe(
            filtered[["RK", "PLAYER", "POS", "TEAM", "BYE", "TIERS"]].reset_index(drop=True),
            use_container_width=True
        )

# ===================== SUGGESTIONS =====================
with tab_suggest:
    st.subheader("Suggestions")
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
    if base_df.empty:
        st.info("No player data available.")
    else:
        evaluated = evaluation.evaluate_players(base_df, config)
        top20 = evaluated.head(20)
        for _, row in top20.iterrows():
            st.write(
                f"{int(row['RK']) if pd.notna(row['RK']) else 0}. **{row['PLAYER']}** "
                f"({row['POS']}, {row['TEAM']}) — "
                f"Value: {row['value']:.1f} | VBD: {row['vbd']:.1f} "
                f"| _{row['notes']}_"
            )

# ===================== EXPORT =====================
with tab_export:
    st.subheader("Export & PDF")
    st.write("Generate a simple PDF report of your draft (mock or live).")
    picks_for_pdf = []
    my_slot = None
    if "mock_state" in st.session_state:
        S = st.session_state.mock_state
        picks_for_pdf = S.get("picks", [])
        my_slot = S.get("user_slot")
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
