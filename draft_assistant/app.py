"""
Fantasy Football Draft Assistant - Streamlit App
"""
import streamlit as st
import pandas as pd
import random
from datetime import datetime

from core import utils, sleeper, roster, evaluation, suggestions, mock_ai, pdf_report

# Read config
config = utils.read_config()

st.sidebar.title("Settings")
league_id = st.sidebar.text_input("Sleeper League ID", config.get("sleeper", {}).get("league_id", ""))
teams_default = config.get("draft", {}).get("teams", 12)
rounds_default = config.get("draft", {}).get("rounds", 15)
teams = st.sidebar.number_input("Number of Teams", min_value=2, value=teams_default)
rounds = st.sidebar.number_input("Number of Rounds", min_value=1, value=rounds_default)
if st.sidebar.button("Save Settings"):
    config.setdefault("sleeper", {})["league_id"] = league_id
    config.setdefault("draft", {})["teams"] = int(teams)
    config.setdefault("draft", {})["rounds"] = int(rounds)
    utils.save_config(config)
    st.sidebar.success("Settings saved to config.toml")

st.sidebar.title("Data Upload")
players_file = st.sidebar.file_uploader("Upload Players CSV", type="csv")
schedule_file = st.sidebar.file_uploader("Upload Schedule CSV", type="csv")
if schedule_file is not None:
    try:
        pd.read_csv(schedule_file).to_csv("data/sample_schedule.csv", index=False)
        st.sidebar.success("Schedule file uploaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading schedule file: {e}")
if players_file is not None:
    try:
        pd.read_csv(players_file).to_csv("data/sample_players.csv", index=False)
        st.sidebar.success("Players file uploaded.")
    except Exception as e:
        st.sidebar.error(f"Error reading players file: {e}")

st.title("Fantasy Football Draft Assistant")
tabs = st.tabs(["Live Draft", "Mock Draft", "Player Board", "Suggestions", "Export"])

with tabs[0]:
    st.header("Live Draft")
    if not league_id:
        st.info("Enter a Sleeper league ID in the Settings to begin.")
    else:
        league_info = sleeper.get_league_info(league_id)
        if league_info:
            st.write(f"League: {league_info.get('name')} (ID: {league_id})")
            draft_id = league_info.get("draft_id")
            if draft_id:
                picks = sleeper.get_picks(draft_id)
                users = sleeper.get_users(league_id)
                total_picks = len(picks) + 1
                (round_num, pick_in_round, slot) = utils.snake_position(total_picks, teams)
                team_name = None
                for u in users:
                    if str(u.get("roster_id")) == str(slot):
                        team_name = u.get("display_name") or u.get("username")
                if not team_name:
                    team_name = f"Slot {slot}"
                if total_picks <= teams*rounds:
                    st.write(f"**Current Pick:** Round {round_num}, Pick {pick_in_round} (Slot {slot}: {team_name})")
                else:
                    st.success("Draft complete.")
                rosters = roster.build_rosters(picks, users)
                for team, roster_dict in rosters.items():
                    st.write(f"**{team}**")
                    for pos, players in roster_dict.items():
                        if players:
                            st.write(f"- {pos}: {', '.join(players)}")
            else:
                st.info("No active draft found in this league.")
        else:
            st.error("Unable to fetch league info. Check league ID.")

with tabs[1]:
    st.header("Mock Draft Simulator")
    # Mock draft controls
    user_slot = st.number_input("Your Draft Slot (1 = first pick)", min_value=1, max_value=config.get("draft", {}).get("teams", 12), value=1)
    num_teams = st.number_input("Number of Teams", min_value=2, value=config.get("draft", {}).get("teams", 12))
    num_rounds = st.number_input("Number of Rounds", min_value=1, value=config.get("draft", {}).get("rounds", 15))
    start = st.button("Start Mock Draft")
    reset = st.button("Reset Mock Draft")
    # Initialize session state
    if reset:
        st.session_state.pop("mock", None)
    if "mock" not in st.session_state or start:
        # Load player data
        try:
            players_df = pd.read_csv("data/sample_players.csv")
        except Exception:
            players_df = pd.DataFrame()
        players_df = players_df.rename(columns=lambda x: x.strip())
        if "PLAYER NAME" in players_df.columns:
            players_df = players_df.rename(columns={"PLAYER NAME": "PLAYER"})
        elif "PLAYER" not in players_df.columns:
            players_df["PLAYER"] = players_df.get("PLAYER", "")
        # Evaluate players
        available = evaluation.evaluate_players(players_df, config)
        # Initialize teams info
        teams_list = []
        strategies = mock_ai.STRATEGIES.copy()
        random.shuffle(strategies)
        for i in range(1, num_teams+1):
            team = {"slot": i, "name": "You" if i == user_slot else f"AI {i}", 
                    "strategy": strategies[(i-1) % len(strategies)], "picks": []}
            teams_list.append(team)
        # Initialize draft state
        st.session_state.mock = {
            "started": True,
            "teams": teams_list,
            "available": available.reset_index(),
            "picks": [],
            "current_pick": 1,
            "user_slot": user_slot,
            "num_teams": num_teams,
            "num_rounds": num_rounds
        }
    # Simulation loop until user's turn
    if st.session_state.mock.get("started"):
        data = st.session_state.mock
        teams_list = data["teams"]
        available = data["available"]
        current_pick = data["current_pick"]
        while True:
            (rnd, pick_in_round, slot) = utils.snake_position(current_pick, data["num_teams"])
            # Check if draft over
            if rnd > data["num_rounds"]:
                st.success("Mock draft complete.")
                break
            # If it's user's turn, stop simulation
            if slot == data["user_slot"]:
                break
            # AI pick
            team = next((t for t in teams_list if t["slot"] == slot), None)
            if team:
                idx = mock_ai.pick_for_team(available, team["strategy"], rnd)
                if idx is None:
                    break
                pick = available.loc[idx]
                # Record pick
                team["picks"].append(pick["PLAYER"])
                data["picks"].append({
                    "round": rnd,
                    "pick_no": pick_in_round,
                    "team": team["name"],
                    "metadata": {"first_name": pick["PLAYER"], "last_name": ""}
                })
                available = available.drop(idx).reset_index(drop=True)
                data["available"] = available
                data["current_pick"] += 1
                current_pick += 1
            else:
                break
        # Save back
        st.session_state.mock = data
    # After simulation or waiting for user
    if "mock" in st.session_state:
        data = st.session_state.mock
        (rnd, pick_in_round, slot) = utils.snake_position(data["current_pick"], data["num_teams"])
        if slot == data["user_slot"] and rnd <= data["num_rounds"]:
            st.write(f"**Your Turn:** Round {rnd}, Pick {pick_in_round}")
            # Suggest top 5
            top5 = suggestions.top_suggestions(data["available"], None, 5)
            if not top5.empty:
                choice = st.selectbox("Select a player to draft:", top5["PLAYER"].tolist(), key="user_choice")
                if st.button("Draft Selected Player"):
                    sel_name = choice
                    sel_idx = data["available"][data["available"]["PLAYER"] == sel_name].index
                    if not sel_idx.empty:
                        idx = sel_idx[0]
                        pick = data["available"].loc[idx]
                        data["picks"].append({
                            "round": rnd,
                            "pick_no": pick_in_round,
                            "team": "You",
                            "metadata": {"first_name": pick["PLAYER"], "last_name": ""}
                        })
                        data["available"] = data["available"].drop(idx).reset_index(drop=True)
                        data["current_pick"] += 1
                        st.session_state.mock = data
                        st.experimental_rerun()
            else:
                st.info("No available players to pick.")
        # Display draft board and your roster
        st.write("### Draft Picks:")
        for p in data["picks"]:
            st.write(f"Round {p['round']} - {p['team']}: {p['metadata']['first_name']}")
        your_roster = [p["metadata"]["first_name"] for p in data["picks"] if p["team"] == "You"]
        st.write("**Your Roster:**", ", ".join(your_roster))

with tabs[2]:
    st.header("Player Board")
    try:
        players_df = pd.read_csv("data/sample_players.csv")
    except Exception:
        players_df = pd.DataFrame()
    if not players_df.empty:
        players_df = players_df.rename(columns=lambda x: x.strip())
        if "PLAYER NAME" in players_df.columns:
            players_df = players_df.rename(columns={"PLAYER NAME": "PLAYER"})
        if "PLAYER" not in players_df.columns:
            players_df["PLAYER"] = players_df.get("PLAYER", "")
        st.write("Search and filter players:")
        pos_filter = st.multiselect("Position", options=players_df["POS"].unique(), default=list(players_df["POS"].unique()))
        team_filter = st.multiselect("Team", options=players_df["TEAM"].unique(), default=list(players_df["TEAM"].unique()))
        tier_filter = st.multiselect("Tier", options=players_df["TIERS"].unique(), default=list(players_df["TIERS"].unique()))
        filtered = players_df[
            (players_df["POS"].isin(pos_filter)) &
            (players_df["TEAM"].isin(team_filter)) &
            (players_df["TIERS"].isin(tier_filter))
        ]
        st.dataframe(filtered[["RK", "PLAYER", "POS", "TEAM", "BYE", "TIERS"]].reset_index(drop=True))
    else:
        st.info("No player data available. Upload a CSV in Settings.")

with tabs[3]:
    st.header("Suggestions")
    try:
        players_df = pd.read_csv("data/sample_players.csv")
    except Exception:
        players_df = pd.DataFrame()
    if not players_df.empty:
        players_df = players_df.rename(columns=lambda x: x.strip())
        if "PLAYER NAME" in players_df.columns:
            players_df = players_df.rename(columns={"PLAYER NAME": "PLAYER"})
        evaluated = evaluation.evaluate_players(players_df, config)
        top20 = evaluated.head(20)
        for _, row in top20.iterrows():
            st.write(f"{int(row['RK'])}. {row['PLAYER']} ({row['POS']}, {row['TEAM']}) - Value: {row['value']:.1f}, VBD: {row['vbd']:.1f} - Notes: {row['notes']}")
    else:
        st.info("No player data available for suggestions.")

with tabs[4]:
    st.header("Export & PDF")
    st.write("Generate a PDF report of your draft.")
    if st.button("Download PDF"):
        league_name = league_info.get("name") if league_info else "My League"
        picks = st.session_state.mock.get("picks", []) if "mock" in st.session_state else []
        user_slot = st.session_state.mock.get("user_slot") if "mock" in st.session_state else None
        pdf_bytes = pdf_report.generate_pdf(league_name, picks, user_slot)
        st.download_button("Download Draft Report", data=pdf_bytes, file_name="draft_report.pdf", mime="application/pdf")
