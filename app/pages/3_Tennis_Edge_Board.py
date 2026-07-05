import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import TENNIS_TOURS
from src.models.tennis_winprob_link import win_probability
from src.ingest.oddspapi_client import get_betmgm_tennis_matches
from src.ingest.tennis_name_match import normalize_oddspapi_name, resolve_player
from src.trading.devig import implied_prob, devig_proportional
from src.trading.edge_calc import edge, ev_per_dollar
from src.trading.kelly import fractional_kelly_stake
from src.backtest.walk_forward_tennis import walk_forward as tennis_walk_forward, NAIVE_BASELINE_BRIER

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_tennis_engine_and_model

st.set_page_config(page_title="Tennis Edge Board", layout="wide")

st.title("Tennis Edge Board")
st.caption(
    "Separate from the World Cup and club football pages above -- own per-tour Elo pool, "
    "own binary (no draw, no home advantage) win-probability model, own data source "
    "(tennis-data.co.uk historical results + odds). Nothing here touches the other boards "
    "or their data."
)
st.warning(
    "New: single-feature Elo-diff model, walk-forward validated per-year against a naive "
    "50/50 baseline (see below) on ONE tour at a time, no tournament-importance weighting "
    "(Grand Slam vs. ATP250 all use the same K-factor) and no surface-specific rating -- "
    "both real, disclosed simplifications, not bugs. Treat edges with the same skepticism "
    "as the other boards.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Tour")
    tour = st.selectbox("Tour", options=list(TENNIS_TOURS.keys()), format_func=lambda k: TENNIS_TOURS[k]["label"])

    st.header("Position sizing")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    kelly_fraction_pct = st.slider("Kelly fraction", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    max_stake_pct = st.slider("Max stake per match (% of bankroll)", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
    min_edge_filter = st.slider("Only show edges above (probability points)", 0.0, 0.20, 0.0, 0.01)
    if st.button("Refresh odds now"):
        st.cache_data.clear()

tour_cfg = TENNIS_TOURS[tour]
engine, model, n_train = load_tennis_engine_and_model(tour)
st.caption(f"{tour_cfg['label']} Elo engine + win-probability link fit on {n_train // 2} historical matches (2000-present).")


@st.cache_data(ttl=3600, show_spinner="Running walk-forward backtest (year by year)...")
def load_backtest_summary(tour: str):
    from src.features.tennis_elo import run_all as tennis_run_all
    from src.backtest.metrics import brier_score

    _, feature_rows = tennis_run_all(tour)
    probs, outcomes, per_year = tennis_walk_forward(feature_rows)
    if not probs:
        return None
    return {"n": len(probs), "model_brier": brier_score(probs, outcomes), "n_years": len(per_year)}


backtest = load_backtest_summary(tour)
if backtest:
    beats = backtest["model_brier"] < NAIVE_BASELINE_BRIER
    st.caption(
        f"Walk-forward backtest across {backtest['n_years']} years ({backtest['n'] // 2} matches, evaluated from "
        f"both perspectives): model Brier={backtest['model_brier']:.4f} vs. naive 50/50 baseline Brier="
        f"{NAIVE_BASELINE_BRIER:.4f} — "
        + ("model beats the naive baseline." if beats else "model does NOT beat the naive baseline; treat this tour's edges as unvalidated.")
    )

st.caption(
    "Betting market: BetMGM's tennis 'Match Winner' line (via OddsPapi) is a clean 2-way "
    "market -- tennis has no draw, so unlike the club football board this needs no "
    "Draw-No-Bet workaround; edge/Kelly are computed directly against it."
)
st.caption(
    "Name matching: tennis-data.co.uk records players as 'Surname F.' but BetMGM/OddsPapi "
    "returns 'Surname, First' -- converted automatically, but multi-word first names (rare) "
    "may not match and will show as 'not matched' below rather than silently using a "
    "default rating."
)


@st.cache_data(ttl=300, show_spinner="Pulling live BetMGM Match Winner odds via OddsPapi...")
def load_tennis_matches(tour: str):
    return get_betmgm_tennis_matches(tour)


matches = load_tennis_matches(tour)

if not matches:
    st.info(f"No active BetMGM Match Winner lines right now for {tour_cfg['label']}. Check back when a tournament is in progress.")
else:
    rows = []
    for m in matches:
        p1, matched1 = resolve_player(normalize_oddspapi_name(m["player1_raw"]), engine.ratings)
        p2, matched2 = resolve_player(normalize_oddspapi_name(m["player2_raw"]), engine.ratings)

        diff = engine.get(p1) - engine.get(p2)
        model_prob_1 = win_probability(model, diff)
        model_prob_2 = 1 - model_prob_1

        raw1 = implied_prob(m["player1_price"])
        raw2 = implied_prob(m["player2_price"])
        fair1, fair2 = devig_proportional(m["player1_price"], m["player2_price"])

        e1 = edge(model_prob_1, raw1)
        e2 = edge(model_prob_2, raw2)

        stake1 = fractional_kelly_stake(model_prob_1, raw1, bankroll, fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct)
        stake2 = fractional_kelly_stake(model_prob_2, raw2, bankroll, fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct)
        cap_dollars = bankroll * max_stake_pct
        total_stake = stake1 + stake2
        if total_stake > cap_dollars and total_stake > 0:
            scale = cap_dollars / total_stake
            stake1 *= scale
            stake2 *= scale

        for player, matched, model_prob, raw_price, fair_prob, e, stake in (
            (p1, matched1, model_prob_1, raw1, fair1, e1, stake1),
            (p2, matched2, model_prob_2, raw2, fair2, e2, stake2),
        ):
            if abs(e) < min_edge_filter:
                continue
            ev_dollar = stake * ev_per_dollar(model_prob, raw_price)
            profit_if_hit = stake * (1 / raw_price - 1) if raw_price > 0 else 0.0
            rows.append({
                "Match": m["title"],
                "Player": player,
                "Matched to Elo ratings?": "Yes" if matched else "No (using base rating 1500)",
                "Model prob": round(model_prob, 3),
                "BetMGM raw price": round(raw_price, 3),
                "De-vigged fair price": round(fair_prob, 3),
                "Edge (pts)": round(e, 3),
                "EV per $1": round(ev_per_dollar(model_prob, raw_price), 3),
                "Suggested stake ($)": round(stake, 2),
                "EV profit ($)": round(ev_dollar, 2),
                "Profit if hit ($)": round(profit_if_hit, 2),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Match", "Player"])
        total_staked = df["Suggested stake ($)"].sum()
        total_ev = df["EV profit ($)"].sum()
        total_if_hit = df["Profit if hit ($)"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total staked across all bets", f"${total_staked:.2f}")
        m2.metric("Expected value profit", f"${total_ev:+.2f}")
        m3.metric("Profit if every bet hits", f"${total_if_hit:+.2f}")

    st.dataframe(df, hide_index=True, use_container_width=True)

st.divider()
st.subheader(f"Current {tour_cfg['label']} Elo ratings (top 20)")
top20 = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
st.dataframe(pd.DataFrame(top20, columns=["Player", "Elo rating"]), use_container_width=True, hide_index=True)
