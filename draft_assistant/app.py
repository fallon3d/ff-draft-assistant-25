import os, importlib
import streamlit as st

st.set_page_config(page_title="FF Draft Assistant - Boot Check", page_icon="✅", layout="centered")

st.title("Fantasy Football Draft Assistant")
st.subheader("Milestone A: Boot Check ✅")

# Sanity: required folders
expected = [
    "draft_assistant/",
    "draft_assistant/core/",
    "draft_assistant/data/",
]
ok = True
for p in expected:
    exists = os.path.isdir(p.rstrip("/"))
    st.write(("✅" if exists else "❌"), p)
    ok = ok and exists

# Minimal data check (optional in A)
sample_players = "draft_assistant/data/sample_players.csv"
st.write(("✅" if os.path.exists(sample_players) else "❌"), sample_players)

# Import check for future modules (won't exist yet, that's fine)
try:
    importlib.import_module("core")
    st.write("✅ Python can import 'core' package (once you add it).")
except Exception:
    st.write("ℹ️ 'core' not found yet (expected in Milestone A).")

st.markdown("---")
st.success("If this page loads on Streamlit Cloud, your deploy is working. Next: Milestone B (swap in the full app).")
st.caption("Main file path for Cloud should be: draft_assistant/app.py")
