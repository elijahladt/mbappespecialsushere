"""Shared, cached loaders used by both the Kalshi page (streamlit_app.py) and
the BetMGM page (pages/1_BetMGM_Edge_Board.py) -- kept in one place so
Streamlit's cache is shared instead of recomputing the Elo engine twice."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.elo import run_all
from src.features.rest_travel import compute_rest_travel
from src.features.altitude_timezone import compute_altitude_timezone
from src.models.winprob_link import fit_link

HOST_NATIONS = {"United States", "Mexico", "Canada"}

# Checked directly (src/backtest/validate_host_advantage.py): the standard
# +100 Elo home-advantage constant is fit on 150+ years of ALL football, not
# specifically World Cup hosts, and on the last 4 hosts (South Africa 2010,
# Brazil 2014, Russia 2018, Qatar 2022 -- 18 matches) it's actually the
# worst-calibrated value tested; 0 fit best on that small sample. Decoupled
# from the general HOME_ADVANTAGE constant and defaulted lower so it's
# adjustable rather than trusting either extreme on n=18.
DEFAULT_HOST_NATION_BONUS = 50.0

# Split per host country rather than one uniform number: verified via real
# reporting (ESPN et al.) that Mexican fans used organized sleep-disruption
# tactics (loudspeakers, horns, motorcycles outside team hotels) against
# BOTH Ecuador (R32) and England (R16) -- a documented pattern, not a single
# anecdote, and a more aggressive/deliberate form of "home advantage" than
# the 18-match host-nation check above was measuring. No similar reports
# for USA or Canada matches. This CANNOT be statistically fit (n=2, no
# historical base rate for "organized hotel harassment" exists to backtest
# against) -- it's a disclosed judgment call, not a validated number.
DEFAULT_HOST_BONUS_BY_COUNTRY = {
    "United States": DEFAULT_HOST_NATION_BONUS,
    "Canada": DEFAULT_HOST_NATION_BONUS,
    "Mexico": 75.0,
}


@st.cache_resource(show_spinner="Building Elo ratings from ~50k historical matches...")
def load_engine_and_model():
    engine, feature_rows = run_all()
    model, n_train = fit_link(feature_rows)
    tracker, _ = compute_rest_travel()
    alt_tz_tracker, _ = compute_altitude_timezone()
    return engine, model, n_train, tracker, alt_tz_tracker


def effective_diff(engine, team_a, team_b, host_bonus_by_country: dict = None):
    host_bonus_by_country = host_bonus_by_country or DEFAULT_HOST_BONUS_BY_COUNTRY
    adv = host_bonus_by_country.get(team_a, 0.0) if team_a in HOST_NATIONS and team_b not in HOST_NATIONS else 0.0
    return (engine.get(team_a) + adv) - engine.get(team_b)
