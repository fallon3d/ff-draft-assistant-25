import sys, streamlit as st
import pandas as pd, requests, pydantic, reportlab, toml, PIL

st.set_page_config(page_title="Health Check", layout="centered")
st.title("Streamlit Cloud Health Check ✅")

st.write("**Python:**", sys.version)
st.write("**streamlit:**", st.__version__)
st.write("**pandas:**", pd.__version__)
st.write("**requests:**", requests.__version__)
st.write("**pydantic:**", pydantic.__version__)
st.write("**reportlab:**", reportlab.Version)
import PIL as pillow
st.write("**Pillow:**", pillow.__version__)

st.success("If you see versions above, the environment is good. Next: restore the real app.py.")
