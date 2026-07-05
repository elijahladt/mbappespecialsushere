"""Read a secret from Streamlit Cloud's st.secrets first (the documented,
reliable mechanism there), falling back to a local .env / os.environ for
running scripts/tests outside the Streamlit app. Deliberately NOT read at
module import time -- capture-once-at-import is fragile if secrets aren't
wired into os.environ before the module first loads.
"""
import os

from dotenv import load_dotenv

load_dotenv()


def get_secret(name: str):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)
