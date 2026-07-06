"""Shared, cached loaders used by both the Kalshi page (streamlit_app.py) and
the BetMGM page (pages/1_BetMGM_Edge_Board.py) -- kept in one place so
Streamlit's cache is shared instead of recomputing the Elo engine twice."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.db import get_connection
from src.features.elo import run_all
from src.features.rest_travel import compute_rest_travel
from src.features.altitude_timezone import compute_altitude_timezone
from src.models.winprob_link import fit_link, h2h_diff_live

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


def ensure_db_populated():
    """data/wc.sqlite is gitignored (it's a ~9MB regeneratable cache, not
    source) -- a fresh clone (e.g. Streamlit Cloud) has an empty database
    unless something builds it. Runs the full ingestion pipeline once, only
    if the matches table is actually empty."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()
    if count > 0:
        return

    from src.ingest.historical_results import fetch_and_load as load_historical
    from src.ingest.wc2026_results import fetch_and_load as load_2026
    from src.ingest.enrich_backtest_venues import run as enrich_venues

    load_historical()
    load_2026()
    enrich_venues()


@st.cache_resource(show_spinner="First run: downloading ~50k historical matches + 2026 results (one-time, then cached)...")
def load_engine_and_model():
    ensure_db_populated()
    engine, feature_rows = run_all()
    model, n_train = fit_link(feature_rows)
    tracker, _ = compute_rest_travel()
    alt_tz_tracker, _ = compute_altitude_timezone()
    return engine, model, n_train, tracker, alt_tz_tracker, feature_rows


def effective_diff(engine, team_a, team_b, host_bonus_by_country: dict = None):
    host_bonus_by_country = host_bonus_by_country or DEFAULT_HOST_BONUS_BY_COUNTRY
    adv = host_bonus_by_country.get(team_a, 0.0) if team_a in HOST_NATIONS and team_b not in HOST_NATIONS else 0.0
    return (engine.get(team_a) + adv) - engine.get(team_b)


def full_feature_vector(engine, team_a, team_b, host_bonus_by_country: dict = None):
    """[elo_diff (with host-nation bonus), h2h_diff] -- the model now takes
    both. NOTE on confidence: walk-forward backtest on the tiny WC knockout
    sample (n=49 held-out matches across 2010-2022) found h2h_diff's effect
    is within noise (Brier 0.1648 with it vs. 0.1591 without -- confidence
    intervals almost fully overlap), unlike tennis/club football where much
    larger samples let us confidently say a feature did or didn't help.
    Kept in the model because the underlying rationale (Elo can't represent
    "bogey team" matchups) is sound and the result isn't a clear negative,
    but this is a genuinely unresolved question, not a validated win."""
    return [effective_diff(engine, team_a, team_b, host_bonus_by_country), h2h_diff_live(engine, team_a, team_b)]


def ensure_club_db_populated(league_id: str):
    """Same rationale as ensure_db_populated() above -- club_matches is
    gitignored/regenerable, a fresh clone has it empty. Checked per league_id
    so opening one league's page doesn't block on ingesting every league
    (Milestone 2+ leagues stay untouched until their page is first opened)."""
    from src.db import get_connection
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM club_matches WHERE league_id = ?", (league_id,)
    ).fetchone()[0]
    conn.close()
    if count > 0:
        return

    from src.ingest.football_data_co_uk import fetch_and_load
    fetch_and_load(league_id)


@st.cache_resource(show_spinner="First run for this league: downloading historical results (one-time, then cached)...")
def load_club_engine_and_model(league_id: str):
    from src.features.club_elo import run_all as club_run_all
    from src.models.club_winprob_link import fit_link as club_fit_link

    ensure_club_db_populated(league_id)
    engine, feature_rows = club_run_all(league_id)
    model, n_train = club_fit_link(feature_rows, league_id)
    return engine, model, n_train


def ensure_tennis_db_populated(tour: str):
    """Same rationale as ensure_db_populated()/ensure_club_db_populated()
    above -- tennis_matches is gitignored/regenerable, checked per tour so
    opening the ATP page doesn't block on ingesting WTA history too."""
    from src.db import get_connection
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM tennis_matches WHERE tour = ?", (tour,)
    ).fetchone()[0]
    conn.close()
    if count > 0:
        return

    from src.ingest.tennis_data_co_uk import fetch_and_load
    fetch_and_load(tour)


@st.cache_resource(show_spinner="First run for this tour: downloading historical results (one-time, then cached)...")
def load_tennis_engine_and_model(tour: str):
    from src.features.tennis_elo import run_all as tennis_run_all
    from src.models.tennis_winprob_link import fit_link as tennis_fit_link

    ensure_tennis_db_populated(tour)
    engine, feature_rows = tennis_run_all(tour)
    model, n_train = tennis_fit_link(feature_rows, tour)
    return engine, model, n_train
