import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.winprob_link import h2h_diff_live
from src.ingest.kalshi_client import get_match_markets
from src.models.goal_simulation import (
    fit_goal_model, simulate_match, simulate_match_efficiency,
    fit_xg_goal_model, simulate_match_xg,
)
from src.features.team_efficiency import build_efficiency_table
from src.models.bootstrap_uncertainty import bootstrap_win_probability, summarize_bootstrap
from src.trading.edge_calc import edge, ev_per_dollar
from src.trading.kelly import fractional_kelly_stake

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_engine_and_model, effective_diff

st.set_page_config(page_title="Match Simulator", layout="wide")

st.title("Match Simulator — 10,000 simulations per match")
st.caption(
    "Every open World Cup match, simulated 10,000 times via a Poisson goal model (not the "
    "Elo-diff logistic regression used on the other boards -- a genuinely different approach: "
    "random scorelines drawn from each side's expected goals, tallied into win/draw/loss and "
    "a most-likely-scoreline table)."
)
st.warning(
    "**Two versions, backtested honestly, very different results.** Walk-forward on "
    "2010-2022 WC knockout matches (src/backtest/validate_goal_simulation.py): the default "
    "Elo-based version scores Brier=0.168 (vs. the main model's 0.165 -- roughly comparable). "
    "An experimental version using each team's own rolling offensive/defensive efficiency "
    "(real goals scored/conceded, not Elo) looked dramatically better on one single real "
    "match (correctly favored Norway over Brazil, which the Elo-based version didn't) -- but "
    "backtested WORSE, consistently, across all 4 tournaments tested (Brier=0.210). That's a "
    "concrete lesson: one compelling anecdote is not the same as real validation. The "
    "efficiency version is available below as an explicitly-labeled experimental toggle, off "
    "by default, not because it's better -- because the underlying question (does recent "
    "scoring form matter beyond Elo) is worth being able to explore.",
    icon="⚠️",
)

engine, model, n_train, tracker, _alt_tz_tracker, feature_rows_full = load_engine_and_model()


@st.cache_resource(show_spinner="Fitting goal-expectancy models...")
def load_goal_models():
    goal_model = fit_goal_model(feature_rows_full)
    efficiency_rows, league_avg_goals = build_efficiency_table()
    latest_efficiency = {}
    for r in efficiency_rows:
        latest_efficiency[r["home_team"]] = (r["home_off_pre"], r["home_def_pre"])
        latest_efficiency[r["away_team"]] = (r["away_off_pre"], r["away_def_pre"])
    return goal_model, league_avg_goals, latest_efficiency


goal_model, league_avg_goals, latest_efficiency = load_goal_models()

with st.sidebar:
    use_efficiency = st.toggle(
        "Use experimental efficiency-based simulation",
        value=False,
        help="Backtested WORSE than the default (Brier 0.210 vs 0.168) -- off by default. "
             "Shown for exploration, not because it's recommended.",
    )
    st.header("Position sizing")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    kelly_fraction_pct = st.slider("Kelly fraction", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    max_stake_pct = st.slider("Max stake per match (% of bankroll)", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
    if st.button("Refresh Kalshi prices now"):
        st.cache_data.clear()


@st.cache_data(ttl=60, show_spinner="Pulling live Kalshi matches...")
def load_kalshi_markets():
    return get_match_markets()


matches = load_kalshi_markets()

if not matches:
    st.info("No open World Cup match markets found on Kalshi right now.")
else:
    st.caption(
        "'P(advances)' converts the simulator's regulation win/draw/loss into a 'who wins the "
        "tie' probability the same way the rest of this app does: P(win) + 0.5 x P(draw) -- "
        "penalties are close to a coin flip, so a drawn simulated match is split 50/50 between "
        "the two sides rather than left out. 'Edge' and 'Suggested stake' compare that number "
        "against Kalshi's live price, using the bankroll/Kelly settings in the sidebar. Check "
        "the boxes on the right for whichever bets you actually want -- the profit totals below "
        "only count checked rows."
    )

    rows = []
    for m in matches:
        team_a, team_b = m["teams"]
        name_a, name_b = team_a["team"], team_b["team"]

        if use_efficiency:
            home_off, home_def = latest_efficiency.get(name_a, (league_avg_goals, league_avg_goals))
            away_off, away_def = latest_efficiency.get(name_b, (league_avg_goals, league_avg_goals))
            sim = simulate_match_efficiency(home_off, home_def, away_off, away_def, league_avg_goals, n_sims=10000)
        else:
            diff = effective_diff(engine, name_a, name_b)
            sim = simulate_match(goal_model, diff, n_sims=10000)

        top_score = sim["top_scorelines"][0]
        p_advance_a = sim["p_home_win"] + 0.5 * sim["p_draw"]
        p_advance_b = 1 - p_advance_a

        for team, p_advance in ((team_a, p_advance_a), (team_b, p_advance_b)):
            if team.get("price") is None:
                continue
            price = team["price"]
            e = edge(p_advance, price)
            stake = fractional_kelly_stake(p_advance, price, bankroll, fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct)
            ev_dollar = stake * ev_per_dollar(p_advance, price)
            profit_if_hit = stake * (1 / price - 1) if price > 0 else 0.0
            rows.append({
                "Match": m["title"],
                "Team": team["team"],
                "P(advances)": round(p_advance, 3),
                "Kalshi price": round(price, 3),
                "Edge (pts)": round(e, 3),
                "Suggested stake ($)": round(stake, 2),
                "EV profit ($)": round(ev_dollar, 2),
                "Profit if hit ($)": round(profit_if_hit, 2),
                "Expected goals": f"{sim['home_expected_goals']:.2f} - {sim['away_expected_goals']:.2f}",
                "Most likely score": top_score["score"],
                "Score prob.": round(top_score["probability"], 3),
                "Bet on this?": False,
            })

    df = pd.DataFrame(rows).sort_values("Edge (pts)", ascending=False).reset_index(drop=True)
    edited_df = st.data_editor(
        df,
        hide_index=True,
        use_container_width=True,
        disabled=[c for c in df.columns if c != "Bet on this?"],
        column_config={
            "Bet on this?": st.column_config.CheckboxColumn(
                "Bet on this?", help="Check to include this bet in the profit totals below.",
            ),
        },
    )

    selected = edited_df[edited_df["Bet on this?"]]
    m1, m2, m3 = st.columns(3)
    m1.metric("Bets selected", f"{len(selected)}")
    m2.metric("Total staked", f"${selected['Suggested stake ($)'].sum():.2f}")
    m3.metric("Predicted profit (EV)", f"${selected['EV profit ($)'].sum():+.2f}",
              help="Sum of stake x (model prob / price - 1) across only the checked rows -- "
                   "the honest, probability-weighted number, not the best-case scenario.")
    if len(selected):
        st.caption(f"Profit if every selected bet hits: ${selected['Profit if hit ($)'].sum():+.2f} (best case, not a forecast).")

    st.divider()
    st.subheader("Deep dive: uncertainty for one match")
    st.caption(
        "10,000 BOOTSTRAP refits (not the same as the goal simulation above) -- resamples the "
        "historical training data itself to show how much the win-probability estimate would "
        "wobble given a small (~142 match) training set. Takes ~60-90 seconds, so it's one "
        "match at a time rather than automatic for all of them."
    )
    match_titles = [m["title"] for m in matches]
    selected_title = st.selectbox("Pick a match", match_titles)
    selected = next(m for m in matches if m["title"] == selected_title)
    sel_a, sel_b = selected["teams"]

    @st.cache_data(show_spinner="Bootstrapping 10,000 model refits (~60-90 seconds, cached after first run)...")
    def cached_bootstrap(sim_diff: float, sim_h2h: float):
        point, boot_samples = bootstrap_win_probability(feature_rows_full, [sim_diff, sim_h2h], n_boot=10000)
        return summarize_bootstrap(point, boot_samples)

    if st.button("Run 10,000-refit uncertainty check"):
        sim_diff = effective_diff(engine, sel_a["team"], sel_b["team"])
        sim_h2h = h2h_diff_live(engine, sel_a["team"], sel_b["team"])
        boot_stats = cached_bootstrap(sim_diff, sim_h2h)
        st.metric(f"{sel_a['team']} advances", f"{boot_stats['point_estimate']:.1%}")
        st.write(
            f"90% confidence interval from 10,000 bootstrap refits: "
            f"**[{boot_stats['ci_low']:.1%}, {boot_stats['ci_high']:.1%}]** (std={boot_stats['std']:.3f})"
        )

st.divider()
st.subheader("Research: xG-based goal simulator (2018 + 2022 World Cups only)")
st.success(
    "**This one actually won its backtest.** Leave-one-tournament-out on 2018/2022 "
    "(src/backtest/validate_xg_goal_sim.py), all three approaches refit on the identical "
    "split for a fair comparison: xG-based goal-sim Brier=0.2188, vs. logistic Brier=0.2312 "
    "and Elo-based goal-sim Brier=0.2366 -- consistently better in BOTH tested years, not "
    "just on average. Uses each team's own actual match xG (StatsBomb shot-quality data) "
    "instead of Elo-implied expected goals. **Cannot score the live matches above** -- there "
    "is no source of real-time xG for 2026 (StatsBomb has no 2026 coverage; confirmed "
    "directly that football-data.org's real 2026 match data has no shots/xG fields at all, "
    "just scores). Shown here as a validated research comparison on historical matches only, "
    "same treatment as the Elo+StatsBomb XGBoost model on the main Kalshi page.",
    icon="✅",
)


@st.cache_resource(show_spinner="Fitting xG-based goal model (StatsBomb 2018+2022)...")
def load_xg_goal_model():
    from src.features.xg_stats import build_xg_training_table
    return fit_xg_goal_model(None), build_xg_training_table()


xg_goal_model, xg_matches_df = load_xg_goal_model()
xg_match_labels = [
    f"{r.date} — {r.home_team} vs {r.away_team}" for r in xg_matches_df.itertuples()
]
selected_xg_label = st.selectbox("Pick a 2018/2022 match", xg_match_labels, key="xg_match_select")
xg_row = xg_matches_df.iloc[xg_match_labels.index(selected_xg_label)]

xg_sim_result = simulate_match_xg(xg_goal_model, xg_row["elo_diff"], n_sims=10000)
elo_sim_result = simulate_match(fit_goal_model(feature_rows_full), xg_row["elo_diff"], n_sims=10000)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**xG-based: {xg_row['home_team']} vs {xg_row['away_team']}**")
    st.write(f"Expected goals: {xg_sim_result['home_expected_goals']:.2f} - {xg_sim_result['away_expected_goals']:.2f}")
    st.write(
        f"P({xg_row['home_team']} win)={xg_sim_result['p_home_win']:.1%}  "
        f"P(draw)={xg_sim_result['p_draw']:.1%}  P({xg_row['away_team']} win)={xg_sim_result['p_away_win']:.1%}"
    )
    st.write(f"Actual result: {'home win' if xg_row['home_win'] == 1 else 'away win'}")
with c2:
    st.markdown(f"**Elo-based (for comparison): {xg_row['home_team']} vs {xg_row['away_team']}**")
    st.write(f"Expected goals: {elo_sim_result['home_expected_goals']:.2f} - {elo_sim_result['away_expected_goals']:.2f}")
    st.write(
        f"P({xg_row['home_team']} win)={elo_sim_result['p_home_win']:.1%}  "
        f"P(draw)={elo_sim_result['p_draw']:.1%}  P({xg_row['away_team']} win)={elo_sim_result['p_away_win']:.1%}"
    )
