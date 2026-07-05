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


# Club football competitions this app knows about. Config-driven so adding a
# league later (Milestone 2/3) is a dict entry, not new code -- ingestion,
# Elo, model, and backtest scripts all take league_id as a parameter and read
# this dict, they don't hardcode "premier_league" anywhere.
#
# football_data_code: the season-CSV code football-data.co.uk uses
#   (https://www.football-data.co.uk/mmz4281/{season}/{code}.csv).
# oddspapi_tournament_id: resolved by direct API lookup (checked categoryName
#   to rule out name collisions with other countries' "Premier League"s) --
#   never guessed from the tournament name alone.
# tier: domestic-league K-factor tier for the club Elo engine (separate tier
#   space from src/features/elo.py's international-competition tiers).
LEAGUES = {
    "premier_league": {
        "label": "Premier League (England)",
        "football_data_code": "E0",
        "oddspapi_tournament_id": 17,
        "tier": 1,
    },
}


# Tennis tours. Unlike club LEAGUES above, live odds can't use one fixed
# OddsPapi tournamentId -- tennis tournaments are individual week-long
# events with a new tournamentId every week, so the live-odds client
# discovers currently-active tournaments at request time (filtered by
# categorySlug + singles-only) instead of reading a hardcoded id from here.
# tennis_data_path_fn builds the historical per-year file URL for each tour
# (confirmed live: ATP at /{year}/{year}.xlsx, WTA at /{year}w/{year}.xlsx).
TENNIS_TOURS = {
    "atp": {
        "label": "ATP (Men)",
        "tennis_data_path_fn": lambda year: f"/{year}/{year}.xlsx",
        "oddspapi_category_slug": "atp",
        "first_year": 2000,
    },
    "wta": {
        "label": "WTA (Women)",
        "tennis_data_path_fn": lambda year: f"/{year}w/{year}.xlsx",
        "oddspapi_category_slug": "wta",
        "first_year": 2000,
    },
}
TENNIS_ODDSPAPI_SPORT_ID = 12


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
