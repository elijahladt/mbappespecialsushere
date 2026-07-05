import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import LEAGUES
from src.models.club_winprob_link import win_draw_away_probability, live_feature_vector
from src.ingest.oddspapi_client import get_betmgm_draw_no_bet_matches
from src.trading.devig import implied_prob, devig_proportional
from src.trading.edge_calc import edge, ev_per_dollar
from src.trading.kelly import fractional_kelly_stake
from src.backtest.walk_forward_club import walk_forward as club_walk_forward

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_club_engine_and_model

st.set_page_config(page_title="Club Football Edge Board", layout="wide")

st.title("Club Football Edge Board")
st.caption(
    "Separate from the World Cup pages above -- own Elo pool, own 3-way (home/draw/away) "
    "win-probability model, own data source (football-data.co.uk historical results + odds). "
    "Nothing here touches the World Cup engine or its data."
)
st.warning(
    "New and much less battle-tested than the World Cup board: one league (Premier League), "
    "walk-forward validated per-season against a naive home/draw/away base-rate baseline (see "
    "the backtest summary below). Features are Elo-diff + rolling recent-form -- form was "
    "added specifically to try to close the gap to the real market (see the P&L disclosure "
    "below), and honestly it barely moved the needle: pooled Brier went from 0.5886 to 0.5882 "
    "and the P&L result is essentially unchanged. Treat edges here with extra skepticism.",
    icon="⚠️",
)

with st.sidebar:
    st.header("League")
    league_id = st.selectbox(
        "Competition",
        options=list(LEAGUES.keys()),
        format_func=lambda k: LEAGUES[k]["label"],
    )

    st.header("Position sizing")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    kelly_fraction_pct = st.slider("Kelly fraction", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    max_stake_pct = st.slider("Max stake per match (% of bankroll)", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
    min_edge_filter = st.slider("Only show edges above (probability points)", 0.0, 0.20, 0.0, 0.01)
    if st.button("Refresh odds now"):
        st.cache_data.clear()

league_cfg = LEAGUES[league_id]
engine, model, n_train = load_club_engine_and_model(league_id)
st.caption(f"Club Elo engine + 3-way win-probability link fit on {n_train} historical {league_cfg['label']} matches.")


@st.cache_data(ttl=3600, show_spinner="Running walk-forward backtest (season by season)...")
def load_backtest_summary(league_id: str):
    from src.features.club_elo import run_all as club_run_all
    from src.backtest.metrics import multiclass_brier_score

    _, feature_rows = club_run_all(league_id)
    probs, outcomes, naive_probs, per_season = club_walk_forward(feature_rows)
    if not probs:
        return None
    return {
        "n": len(probs),
        "model_brier": multiclass_brier_score(probs, outcomes),
        "naive_brier": multiclass_brier_score(naive_probs, outcomes),
        "n_seasons": len(per_season),
    }


backtest = load_backtest_summary(league_id)
if backtest:
    beats = backtest["model_brier"] < backtest["naive_brier"]
    st.caption(
        f"Walk-forward backtest across {backtest['n_seasons']} seasons ({backtest['n']} matches): "
        f"model Brier={backtest['model_brier']:.4f} vs. naive home/draw/away base-rate Brier="
        f"{backtest['naive_brier']:.4f} — "
        + ("model beats the naive baseline every season tested." if beats
           else "model does NOT beat the naive baseline; treat this league's edges as unvalidated.")
    )


@st.cache_data(ttl=3600, show_spinner="Running walk-forward P&L simulation against real historical Bet365 odds...")
def load_pnl_summary(league_id: str):
    from src.features.club_elo import run_all as club_run_all
    from src.backtest.pnl_backtest_club import simulate as pnl_simulate

    _, feature_rows = club_run_all(league_id)
    return pnl_simulate(feature_rows, edge_threshold=0.0)


pnl = load_pnl_summary(league_id)
if pnl["n_bets"]:
    st.error(
        f"**Beating a naive baseline is NOT the same as beating the real market.** Walk-forward "
        f"simulation of actually placing fractional-Kelly bets against real historical Bet365 "
        f"closing odds ({pnl['n_bets']} bets, {pnl['n_skipped_no_odds']} matches skipped for missing "
        f"odds): starting from $1,000, this model would have ended with ${pnl['final_bankroll']:.2f} "
        f"({pnl['roi_pct']:+.1f}% ROI). Bet365's closing lines are a much sharper, more relevant "
        f"benchmark than the naive baseline above -- this model does not currently have a real, "
        f"exploitable edge against them. Treat every 'edge' shown below as unproven against a live "
        f"market, not as a validated signal.",
        icon="🚨",
    )

st.caption(
    "Betting market: BetMGM (via OddsPapi) does not offer a straight 3-way 'Full Time Result' "
    "line for club league fixtures -- checked directly. What it does offer is 'Draw No Bet' "
    "(stake refunded if the match is a draw), a clean 2-way market, so the edge/Kelly "
    "comparison below is home-vs-away conditional on no draw (P(home) / (P(home)+P(away))), "
    "matching what's actually bettable. The model's full 3-way probabilities (including the "
    "draw) are still shown for transparency."
)


@st.cache_data(ttl=300, show_spinner="Pulling live BetMGM Draw No Bet odds via OddsPapi...")
def load_draw_no_bet_matches(tournament_id: int):
    return get_betmgm_draw_no_bet_matches(tournament_id)


matches = load_draw_no_bet_matches(league_cfg["oddspapi_tournament_id"])

if not matches:
    st.info(
        "No active BetMGM Draw No Bet lines right now for this league. This is expected "
        "outside the match calendar (e.g. summer off-season) -- bookmakers typically don't "
        "post match-level odds until a few weeks before a season/round kicks off. The "
        "pipeline (ingestion, Elo, model, backtest) is otherwise fully wired up; check back "
        "closer to the next round of fixtures."
    )
else:
    rows = []
    for m in matches:
        home, away = m["home_team"], m["away_team"]
        features = live_feature_vector(engine, home, away)
        p_away, p_draw, p_home = win_draw_away_probability(model, features)

        # Conditional on no draw, to match the Draw No Bet framing.
        denom = p_home + p_away
        cond_home = p_home / denom if denom > 0 else 0.5
        cond_away = 1 - cond_home

        raw_home = implied_prob(m["home_price"])
        raw_away = implied_prob(m["away_price"])
        fair_home, fair_away = devig_proportional(m["home_price"], m["away_price"])

        e_home = edge(cond_home, raw_home)
        e_away = edge(cond_away, raw_away)

        stake_home = fractional_kelly_stake(cond_home, raw_home, bankroll, fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct)
        stake_away = fractional_kelly_stake(cond_away, raw_away, bankroll, fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct)
        # Home/away Draw No Bet are mutually exclusive outcomes of ONE match,
        # not independent bets -- cap the combined stake at the same
        # per-market ceiling rather than letting both legs hit it separately.
        cap_dollars = bankroll * max_stake_pct
        total_stake = stake_home + stake_away
        if total_stake > cap_dollars and total_stake > 0:
            scale = cap_dollars / total_stake
            stake_home *= scale
            stake_away *= scale

        for team, cond_prob, raw_price, fair_prob, e, stake in (
            (home, cond_home, raw_home, fair_home, e_home, stake_home),
            (away, cond_away, raw_away, fair_away, e_away, stake_away),
        ):
            if abs(e) < min_edge_filter:
                continue
            ev_dollar = stake * ev_per_dollar(cond_prob, raw_price)
            profit_if_hit = stake * (1 / raw_price - 1) if raw_price > 0 else 0.0
            rows.append({
                "Match": m["title"],
                "Team": team,
                "P(home) 3-way": round(p_home, 3),
                "P(draw) 3-way": round(p_draw, 3),
                "P(away) 3-way": round(p_away, 3),
                "Model prob (Draw No Bet, cond.)": round(cond_prob, 3),
                "BetMGM raw price": round(raw_price, 3),
                "De-vigged fair price": round(fair_prob, 3),
                "Edge (pts)": round(e, 3),
                "EV per $1": round(ev_per_dollar(cond_prob, raw_price), 3),
                "Suggested stake ($)": round(stake, 2),
                "EV profit ($)": round(ev_dollar, 2),
                "Profit if hit ($)": round(profit_if_hit, 2),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Match", "Team"])
        total_staked = df["Suggested stake ($)"].sum()
        total_ev = df["EV profit ($)"].sum()
        total_if_hit = df["Profit if hit ($)"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total staked across all bets", f"${total_staked:.2f}")
        m2.metric("Expected value profit", f"${total_ev:+.2f}")
        m3.metric("Profit if every bet hits", f"${total_if_hit:+.2f}")

    st.dataframe(df, hide_index=True, use_container_width=True)

st.divider()
st.subheader(f"Current {league_cfg['label']} Elo ratings (top 20)")
top20 = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
st.dataframe(pd.DataFrame(top20, columns=["Team", "Elo rating"]), use_container_width=True, hide_index=True)
