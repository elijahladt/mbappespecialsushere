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


def debug_secret_visibility(name: str) -> str:
    """One-line diagnostic string safe to put INSIDE an exception message --
    no values, just whether/where the key is visible. Exists because the
    interactive debug panel and the error traceback show up in different
    places on Streamlit Cloud (live app page vs. Manage-app logs), and the
    logs view is what's actually easy to copy-paste and share."""
    try:
        import streamlit as st
        secret_keys = list(st.secrets.keys())
        secrets_error = None
    except Exception as e:
        secret_keys = None
        secrets_error = repr(e)
    env_val = os.environ.get(name)
    return (
        f"[debug: st.secrets keys={secret_keys!r} (error={secrets_error}), "
        f"'{name}' in st.secrets={secret_keys is not None and name in secret_keys}, "
        f"'{name}' in os.environ={name in os.environ} (len={len(env_val) if env_val else 0})]"
    )
