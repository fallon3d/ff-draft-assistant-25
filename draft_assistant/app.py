# ... top of file unchanged ...

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

# Draft dims
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

# NEW: Default strategy selector (falls back to Hero RB if not set)
default_strategy_current = str(
    config.get("strategy", {}).get(
        "default_strategy", "Hero RB + WR Flood (optional early TE)"
    )
)
default_strategy = st.sidebar.selectbox(
    "Default Strategy (used as tie-breaker on R1)",
    options=[
        "Anchor WR + Early Elite TE (default)",
        "Modified Zero RB (triple WR + TE, then attack RBs)",
        "Hero RB + WR Flood (optional early TE)",
    ],
    index=[0, 1, 2][
        ["Anchor WR + Early Elite TE (default)",
         "Modified Zero RB (triple WR + TE, then attack RBs)",
         "Hero RB + WR Flood (optional early TE)"].index(default_strategy_current)
        if default_strategy_current in [
            "Anchor WR + Early Elite TE (default)",
            "Modified Zero RB (triple WR + TE, then attack RBs)",
            "Hero RB + WR Flood (optional early TE)",
        ] else 2  # default to Hero RB
    ],
    key="default_strategy_select",
)

elite_te_value = st.sidebar.number_input(
    "Elite TE threshold (value)",
    min_value=10.0, max_value=140.0,
    value=float(config.get("strategy", {}).get("elite_te_value", 78.0)),
    step=1.0, key="elite_te_value",
)
hero_rb_value = st.sidebar.number_input(
    "Hero RB threshold (value)",
    min_value=10.0, max_value=160.0,
    value=float(config.get("strategy", {}).get("hero_rb_value", 85.0)),
    step=1.0, key="hero_rb_value",
)
kd_only_last_round = st.sidebar.checkbox(
    "Draft K/DST only in the final round",
    value=bool(config.get("strategy", {}).get("kd_only_last_round", True)),
    key="kd_only_last_round",
)

# Uploads ... (unchanged)

if st.sidebar.button("Save Settings", key="settings_save_btn"):
    config.setdefault("user_profile", {})["sleeper_username"] = sleeper_username.strip()
    config["user_profile"]["user_slot"] = int(user_slot_override)
    config.setdefault("sleeper", {})["league_id"] = league_id.strip()
    config["sleeper"]["poll_seconds"] = int(poll_seconds)
    config.setdefault("draft", {})["teams"] = int(teams_setting)
    config["draft"]["rounds"] = int(rounds_setting)
    config.setdefault("strategy", {})
    config["strategy"]["default_strategy"] = default_strategy
    config["strategy"]["elite_te_value"] = float(elite_te_value)
    config["strategy"]["hero_rb_value"] = float(hero_rb_value)
    config["strategy"]["kd_only_last_round"] = bool(kd_only_last_round)
    utils.save_config(config)
    st.sidebar.success("Saved to config.toml")

# ... later in Live tab, where we call choose_strategy ...
if rnd == 1 and ("live_strategy" not in st.session_state):
    strat = suggestions.choose_strategy(
        evaluation.evaluate_players(pool, config),
        current_overall=next_overall, user_slot=live_user_slot,
        teams=int(teams_setting), total_rounds=int(rounds_setting),
        elite_te_value=float(elite_te_value), hero_rb_value=float(hero_rb_value),
        preferred_name=default_strategy,  # <-- pass your default
    )
    st.session_state.live_strategy = strat

# ... and in Mock tab ...
if rnd == 1 and ("mock_strategy" not in st.session_state):
    strat = suggestions.choose_strategy(
        S["available"], current_overall=next_overall, user_slot=S.get("your_roster_id"),
        teams=teams, total_rounds=rounds,
        elite_te_value=float(elite_te_value), hero_rb_value=float(hero_rb_value),
        preferred_name=default_strategy,  # <-- pass your default
    )
    st.session_state.mock_strategy = strat

# ... rest of file unchanged ...
