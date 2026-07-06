import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.winprob_link import win_probability, h2h_diff_live
from src.ingest.kalshi_client import get_match_markets
from src.ingest.upcoming_fixtures import fetch_upcoming
from src.trading.edge_calc import edge, ev_per_dollar
from src.trading.kelly import fractional_kelly_stake
from src.features.xg_stats import FEATURE_COLUMNS, build_xg_training_table
from src.models.xgb_model import fit_xgb
from src.models.explain import shap_contributions, key_players_statsbomb
from src.backtest.validate_xgb_model import run as run_xgb_validation
from src.features.auto_injury_report import build_live_report, team_auto_flags
from src.backtest.validate_vs_kalshi_r32 import build_comparison as build_r32_comparison
from src.backtest.metrics import brier_score, bootstrap_brier_ci
from src.backtest.bankroll_simulation import simulate as simulate_bankroll
from src.features.player_impact import elo_adjustment_for_team, DEFAULT_IMPACT_SCALE, star_player_boost_for_team, DEFAULT_STAR_BOOST_SCALE
from src.features.altitude_timezone import (
    base_camp_altitude_tz_delta, altitude_tz_elo_adjustment,
    DEFAULT_ALTITUDE_SCALE, DEFAULT_TZ_SCALE,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_engine_and_model, effective_diff, full_feature_vector, HOST_NATIONS, DEFAULT_HOST_BONUS_BY_COUNTRY

INJURY_NOTES_PATH = Path(__file__).resolve().parent.parent / "data" / "injury_notes.json"

st.set_page_config(page_title="World Cup Edge Board", layout="wide")


@st.cache_data(ttl=60, show_spinner="Pulling live Kalshi prices...")
def load_kalshi_markets():
    return get_match_markets()


@st.cache_data(ttl=300, show_spinner="Pulling upcoming fixture venues...")
def load_upcoming_fixtures():
    """Keyed by frozenset({team_a, team_b}) -> {date, venue_city, stage}."""
    return {frozenset({r["home_team"], r["away_team"]}): r for r in fetch_upcoming()}


@st.cache_data(ttl=1800, show_spinner="Checking ESPN match events for cards/injuries...")
def load_auto_injury_report():
    return build_live_report()


@st.cache_resource(show_spinner="Training XGBoost research model (Elo + StatsBomb, 2018+2022)...")
def load_xgb_research_model():
    df = build_xg_training_table()
    model = fit_xgb(df)
    validation = run_xgb_validation()
    return df, model, validation


@st.cache_resource(show_spinner="Backtesting Elo vs real Kalshi opening prices (Round of 32)...")
def load_r32_vs_kalshi():
    rows = build_r32_comparison()
    valid = [r for r in rows if r["kalshi_opening_price"] is not None]
    model_probs = [r["model_prob"] for r in valid]
    kalshi_probs = [r["kalshi_opening_price"] for r in valid]
    outcomes = [r["actual"] for r in valid]
    stats = {
        "model_brier": brier_score(model_probs, outcomes),
        "kalshi_brier": brier_score(kalshi_probs, outcomes),
        "model_ci": bootstrap_brier_ci(model_probs, outcomes),
        "kalshi_ci": bootstrap_brier_ci(kalshi_probs, outcomes),
        "n": len(valid),
    }
    return valid, stats


def load_injury_notes():
    if not INJURY_NOTES_PATH.exists():
        return {}
    notes = json.loads(INJURY_NOTES_PATH.read_text())
    notes.pop("_readme", None)
    return notes


st.title("World Cup 2026 Edge Board — Kalshi")
st.warning(
    "**Small TEST sample -- but a much bigger TRAINING sample now.** The win-probability "
    "link used to train on only ~142 WC knockout matches; it now trains on every decisive, "
    "non-friendly international match in history (~24,500 matches -- qualifiers, continental "
    "championships, Nations Leagues, etc., confirmed via walk-forward to genuinely help: "
    "src/backtest/validate_expanded_training.py, Brier 0.1541 vs. the old 0.1648 on the same "
    "held-out matches). The TEST sample is still just 2010-2022 WC knockout matches (n=49, "
    "Brier 0.1541, 95% CI [0.114, 0.199] vs. naive 50/50's 0.25) -- that part hasn't changed and "
    "still isn't enough to be confident in exact edge sizes, especially the big ones (20+ points) "
    "below, which likely reflect the model missing information the market has (injuries, current "
    "form, squad news), not a real 20-point market mispricing. Use fractional Kelly, trust small "
    "edges more than huge ones. Model includes head-to-head record alongside Elo (added to "
    "address a real gap: Elo assumes team strength is transitive and can't represent 'bogey team' "
    "matchups) -- still a statistical wash even on the bigger training set (0.1541 with it vs. "
    "0.1534 without), but see the live Kalshi comparison further down the page for a more "
    "encouraging real-market result with it included.",
    icon="⚠️",
)

engine, model, n_train, tracker, _alt_tz_tracker, _feature_rows_full = load_engine_and_model()
st.caption(f"Elo engine trained on full match history; win-probability link fit on {n_train} historical competitive (non-friendly) matches with a decisive result -- see the warning above for why this changed from WC-knockout-only.")
st.caption(
    "'Model prob (straight Elo)' does not depend on the live Kalshi price at all and is "
    "identical every reload unless the underlying match history changes. 'Edge' and "
    "'Suggested stake' DO change on every Kalshi refresh, since they use the live price -- "
    "and the table re-sorts with them if 'Sort table by' = Edge, so a match can jump rows "
    "even though its own Model prob hasn't moved."
)
st.caption(
    "Rest days / travel and injury notes below are shown as context only -- a backtest "
    "(src/backtest/validate_extended_features.py) found they do NOT improve the model's "
    "Brier score on available data, so they are NOT folded into the probability."
)
st.caption(
    "'Altitude change' and 'Timezone change' below are computed from each team's REAL "
    "FIFA-published 2026 base camp (not the last city they played in -- teams return to a "
    "fixed base between matches, they don't stay put) vs. the match venue's real elevation "
    "and IANA timezone (properly historical-DST-aware). An earlier version using 'last match "
    "venue' as the reference point was backtested across 2014/2018/2022 and found only a "
    "negligible, noise-level improvement (Brier 0.1600 vs 0.1603, n=35) -- but that version had "
    "a real bug (it ignored teams returning to base camp), so that backtest doesn't apply to "
    "this corrected version. We have no historical base-camp data to re-validate the correction, "
    "so it's offered as a toggle in the sidebar (off by default) rather than folded in by default."
)

with st.sidebar:
    st.header("View")
    sort_by = st.selectbox(
        "Sort table by",
        ["Match (stable)", "Edge (pts)"],
        help="'Edge' re-sorts every time Kalshi prices refresh, so the same match can "
             "jump to a different row -- easy to misread as 'this match's probability "
             "changed' when it's really a different match that moved. 'Match' keeps row "
             "order stable across refreshes; only the values in each row update.",
    )
    show_health = st.toggle(
        "Show health/context data (rest, travel, injuries)",
        value=True,
        help="Off = straight Elo model output only. On = adds rest days, city-change, "
             "and injury-note columns for context. Either way, these are NOT folded into "
             "Model prob -- backtesting found they don't improve it (see caption below).",
    )
    apply_player_impact = st.toggle(
        "Factor suspensions/injuries into win probability (experimental)",
        value=False,
        help="When ON, Edge and Suggested stake are computed from a probability adjusted "
             "for confirmed-missing players, weighted by their real goal+assist share this "
             "tournament (from ESPN match events). This is a disclosed HEURISTIC, not a "
             "statistically fitted adjustment -- we only have 2 historical cases to check "
             "it against, nowhere near enough to validate an effect size. Off by default; "
             "the pure-Elo 'Model prob' column is always shown regardless, for comparison.",
    )
    if apply_player_impact:
        impact_scale = st.slider(
            "Impact scale (Elo points for 100% of a team's output missing)", 0.0, 400.0,
            DEFAULT_IMPACT_SCALE, 10.0,
            help="A guessed constant, not fitted -- tune it and see how sensitive the "
                 "adjustment is. 150 means a team missing a player responsible for ALL its "
                 "goals/assists this tournament gets a 150 Elo-point penalty (capped at 60% "
                 "of that if multiple players are out).",
        )
    apply_star_player = st.toggle(
        "Factor star-player boost into win probability (experimental)",
        value=False,
        help="POSITIVE adjustment for a team's leading individual scorer this tournament "
             "(real ESPN goal+assist counts) -- e.g. Haaland/Messi/Mbappe-caliber output. "
             "Backtested with a StatsBomb proxy on 2018+2022 (leave-one-tournament-out): "
             "INCONCLUSIVE, Brier 0.236 with it vs. 0.233 without on only 97 matches -- not "
             "a validated improvement, shown as an experimental toggle to explore the "
             "'Elo has no concept of individual player quality' gap, not a proven fix for it.",
    )
    if apply_star_player:
        star_boost_scale = st.slider(
            "Star boost scale (Elo points per goal+assist by the leading scorer)", 0.0, 40.0,
            DEFAULT_STAR_BOOST_SCALE, 2.0,
            help="A guessed constant, not fitted -- e.g. 12 means a player with 7 goal+assist "
                 "contributions this tournament (Haaland's tally right now) gives their team "
                 "an +84 Elo boost, capped at 120.",
        )
    apply_altitude_tz = st.toggle(
        "Factor altitude/jet-lag into win probability (experimental)",
        value=False,
        help="Uses each team's REAL FIFA-published 2026 base camp (not the last city they "
             "played in -- teams return to a fixed base between matches) vs. the match "
             "venue's real elevation and timezone. Disclosed heuristic, NOT statistically "
             "validated: we have no historical base-camp data to backtest this exact "
             "(corrected) version against. Only altitude GAINS are penalized (going up hurts "
             "unacclimatized teams; there's no comparable documented benefit to descending).",
    )
    altitude_scale, tz_scale = DEFAULT_ALTITUDE_SCALE, DEFAULT_TZ_SCALE
    if apply_altitude_tz:
        altitude_scale = st.slider("Altitude scale (Elo points per 1000m gained)", 0.0, 100.0, DEFAULT_ALTITUDE_SCALE, 5.0)
        tz_scale = st.slider("Timezone scale (Elo points per hour crossed)", 0.0, 30.0, DEFAULT_TZ_SCALE, 1.0)

    st.caption(
        "Host-nation advantage (Elo points) -- split by country since the general +100 home "
        "constant checked out poorly against the last 4 hosts (18 matches, 0 fit best; see "
        "the Elo-ratings section below), but Mexico specifically has verified, repeated "
        "reports (ESPN et al.) of organized crowd hostility -- fans disrupting sleep with "
        "loudspeakers/horns/motorcycles outside team hotels, done to BOTH Ecuador and England "
        "-- a real, documented pattern distinct from ordinary home support. Can't be "
        "statistically fit (no historical base rate for 'organized hotel harassment' "
        "exists), so Mexico defaults higher than USA/Canada as a disclosed judgment call, "
        "not a validated number."
    )
    usa_host_bonus = st.slider("USA host advantage", 0.0, 200.0, DEFAULT_HOST_BONUS_BY_COUNTRY["United States"], 10.0)
    mexico_host_bonus = st.slider("Mexico host advantage", 0.0, 200.0, DEFAULT_HOST_BONUS_BY_COUNTRY["Mexico"], 10.0)
    canada_host_bonus = st.slider("Canada host advantage", 0.0, 200.0, DEFAULT_HOST_BONUS_BY_COUNTRY["Canada"], 10.0)
    host_bonus_by_country = {"United States": usa_host_bonus, "Mexico": mexico_host_bonus, "Canada": canada_host_bonus}

    st.header("Position sizing")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    kelly_fraction_pct = st.slider("Kelly fraction", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    max_stake_pct = st.slider("Max stake per market (% of bankroll)", min_value=0.01, max_value=0.25, value=0.05, step=0.01)
    min_edge_filter = st.slider("Only show edges above (probability points)", 0.0, 0.20, 0.0, 0.01)
    if st.button("Refresh Kalshi prices now"):
        st.cache_data.clear()

markets = load_kalshi_markets()
need_espn_events = show_health or apply_player_impact or apply_star_player
need_upcoming_fixtures = need_espn_events or apply_altitude_tz
upcoming = load_upcoming_fixtures() if need_upcoming_fixtures else {}
injury_notes = load_injury_notes() if show_health else {}
auto_cards, auto_injuries, auto_goals = load_auto_injury_report() if need_espn_events else (None, None, None)

if not markets:
    st.warning("No open World Cup match markets found on Kalshi right now.")
else:
    rows = []
    for m in markets:
        team_a, team_b = m["teams"]
        diff = effective_diff(engine, team_a["team"], team_b["team"], host_bonus_by_country=host_bonus_by_country)
        h2h_diff = h2h_diff_live(engine, team_a["team"], team_b["team"])
        model_prob_a = win_probability(model, [diff, h2h_diff])
        model_prob_b = 1 - model_prob_a

        rest = {}
        alt_tz = {}
        fixture = None
        if need_upcoming_fixtures:
            fixture = upcoming.get(frozenset({team_a["team"], team_b["team"]}))
            if fixture and show_health:
                year = fixture["date"][:4]
                for team in (team_a["team"], team_b["team"]):
                    rest_days, traveled = tracker.rest_days_and_travel(year, team, fixture["date"], fixture["venue_city"])
                    rest[team] = (rest_days, traveled)

        if fixture:
            for team in (team_a["team"], team_b["team"]):
                alt_delta, tz_delta = base_camp_altitude_tz_delta(team, fixture["date"], fixture["venue_city"])
                alt_tz[team] = (alt_delta, tz_delta)

        impact = {}
        if apply_player_impact and fixture and fixture.get("stage"):
            for team in (team_a["team"], team_b["team"]):
                adj, detail = elo_adjustment_for_team(
                    team, auto_goals, auto_cards, auto_injuries, fixture["stage"], impact_scale=impact_scale,
                )
                impact[team] = (adj, detail)

        star = {}
        if apply_star_player:
            for team in (team_a["team"], team_b["team"]):
                star[team] = star_player_boost_for_team(team, auto_goals, boost_scale=star_boost_scale)

        apply_any_adjustment = apply_player_impact or apply_altitude_tz or apply_star_player

        for team, model_prob in ((team_a, model_prob_a), (team_b, model_prob_b)):
            if team["price"] is None:
                continue

            adjusted_prob = model_prob
            adjustment_detail = []
            if apply_any_adjustment:
                opp_team = team_b["team"] if team is team_a else team_a["team"]
                own_total_adj, opp_total_adj = 0.0, 0.0

                if apply_player_impact:
                    own_adj, own_detail = impact.get(team["team"], (0.0, []))
                    opp_adj, _ = impact.get(opp_team, (0.0, []))
                    own_total_adj += own_adj
                    opp_total_adj += opp_adj
                    adjustment_detail.extend(own_detail)

                if apply_altitude_tz:
                    own_alt, own_tz = alt_tz.get(team["team"], (None, None))
                    opp_alt, opp_tz = alt_tz.get(opp_team, (None, None))
                    own_altitude_adj = altitude_tz_elo_adjustment(own_alt, own_tz, altitude_scale, tz_scale)
                    opp_altitude_adj = altitude_tz_elo_adjustment(opp_alt, opp_tz, altitude_scale, tz_scale)
                    own_total_adj += own_altitude_adj
                    opp_total_adj += opp_altitude_adj
                    if own_altitude_adj != 0:
                        adjustment_detail.append(
                            f"altitude/jet-lag: {own_altitude_adj:+.0f} Elo (altitude change {own_alt}m, tz change {own_tz}h)"
                        )

                if apply_star_player:
                    own_star_adj, own_star_detail = star.get(team["team"], (0.0, []))
                    opp_star_adj, _ = star.get(opp_team, (0.0, []))
                    own_total_adj += own_star_adj
                    opp_total_adj += opp_star_adj
                    adjustment_detail.extend(own_star_detail)

                # This team's Elo effectively shifts by its own penalty minus the
                # opponent's (their disadvantage helps this team's relative odds).
                adjusted_diff = diff + (own_total_adj - opp_total_adj) if team is team_a else -diff + (own_total_adj - opp_total_adj)
                adjusted_h2h = h2h_diff if team is team_a else -h2h_diff
                adjusted_prob = win_probability(model, [adjusted_diff, adjusted_h2h])

            decision_prob = adjusted_prob if apply_any_adjustment else model_prob
            e = edge(decision_prob, team["price"])
            if abs(e) < min_edge_filter:
                continue
            stake = fractional_kelly_stake(
                decision_prob, team["price"], bankroll,
                fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct,
            )
            row = {
                "Match": m["title"],
                "Team (buy YES = advances)": team["team"],
                "Model prob (straight Elo)": round(model_prob, 3),
            }
            if apply_any_adjustment:
                row["Model prob (adjusted)"] = round(adjusted_prob, 3)
                row["Adjustment detail"] = "; ".join(adjustment_detail) if adjustment_detail else "no confirmed impact"
            ev_dollar = stake * ev_per_dollar(decision_prob, team["price"])
            profit_if_hit = stake * (1 / team["price"] - 1)
            row.update({
                "Kalshi price": round(team["price"], 3),
                "Edge (pts)": round(e, 3),
                "EV per $1": round(ev_per_dollar(decision_prob, team["price"]), 3),
                "Suggested stake ($)": round(stake, 2),
                "EV profit ($)": round(ev_dollar, 2),
                "Profit if hit ($)": round(profit_if_hit, 2),
            })
            if show_health:
                rest_days, traveled = rest.get(team["team"], (None, None))
                notes = injury_notes.get(team["team"], [])
                row["Rest days"] = rest_days if rest_days is not None else "?"
                row["Changed city?"] = ("yes" if traveled else "no") if traveled is not None else "?"
                alt_delta, tz_delta = alt_tz.get(team["team"], (None, None))
                row["Altitude change (m)"] = round(alt_delta) if alt_delta is not None else "?"
                row["Timezone change (hrs)"] = round(tz_delta, 1) if tz_delta is not None else "?"
                auto_flags = []
                if fixture and fixture.get("stage"):
                    auto_flags = team_auto_flags(team["team"], auto_cards, auto_injuries, fixture["stage"])
                row["Auto-detected (ESPN cards/injuries)"] = "; ".join(auto_flags) if auto_flags else ""
                row["Manual injury notes"] = "; ".join(notes) if notes else ""
            rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by == "Edge (pts)":
        df = df.sort_values("Edge (pts)", ascending=False)
    else:
        df = df.sort_values(["Match", "Team (buy YES = advances)"])

    if not df.empty:
        total_staked = df["Suggested stake ($)"].sum()
        total_ev = df["EV profit ($)"].sum()
        total_if_hit = df["Profit if hit ($)"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Total staked across all bets", f"${total_staked:.2f}")
        m2.metric("Expected value profit", f"${total_ev:+.2f}",
                   help="Probability-weighted: sum over every recommended bet of stake x (model_prob/price - 1). "
                        "This is what the model's own probabilities say you should expect on average -- the "
                        "honest number, not the highlight-reel one.")
        m3.metric("Profit if every bet hits", f"${total_if_hit:+.2f}",
                   help="Best case, not a forecast: assumes every single recommended bet wins, which won't "
                        "happen (that's not how probability works) -- shown as the upper bound, alongside "
                        "the realistic EV number, not instead of it.")
        st.caption(
            "'Expected value profit' is the number to actually trust -- it already accounts for the bets "
            "that are supposed to lose. 'Profit if every bet hits' is a best-case fantasy included for "
            "context, not a target; treating it as expected is how bettors blow up bankrolls."
        )

    st.dataframe(df, use_container_width=True, hide_index=True)
    if show_health:
        st.caption(
            "'Auto-detected' comes from ESPN's real match events: suspensions are rule-based "
            "(2 yellow cards or a red card, using FIFA's actual 2026 reset windows) and "
            "certain; 'Recent injury concern' just means a player was subbed off or a play "
            "stoppage mentioned an injury in one of their last 3 matches -- a soft signal, "
            "not a guarantee they'll miss the next game. ESPN's dedicated injury-report feed "
            "was checked and is empty for every team in this tournament, so it isn't used."
        )
        st.caption(f"'Manual injury notes' are hand-maintained in data/injury_notes.json ({len(injury_notes)} team(s) currently noted) for anything auto-detection can't catch (pre-tournament news, etc).")

st.divider()
st.info(
    "**10,000-simulation match statistics (win/draw/loss, most likely scoreline) and the "
    "bootstrap confidence-interval deep dive have moved to their own page: 'Match Simulator' "
    "in the sidebar** -- now covers every open match automatically instead of one at a time.",
    icon="🎲",
)

st.divider()
st.subheader("Current Elo ratings (top 20)")
top20 = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
st.dataframe(pd.DataFrame(top20, columns=["Team", "Elo rating"]), use_container_width=True, hide_index=True)

st.divider()
st.subheader("Research model: XGBoost (Elo + StatsBomb shot-quality data)")
st.info(
    "**This model cannot score the live matches above.** It needs StatsBomb shot/xG "
    "data, which only exists for the 2018 and 2022 World Cups (open data) -- there is "
    "no 2026 data since the tournament is still in progress, so it has nothing to work "
    "with for tomorrow's matches. It's shown here as a validated research comparison on "
    "historical matches only. To get a live second-model column next to Elo above, add "
    "an API-Football key (.env locally, or Streamlit Cloud Secrets when deployed) -- that unlocks a deployable variant "
    "using recent form + injuries, which (unlike StatsBomb) has 2026 coverage.",
    icon="🔬",
)

xg_df, xgb_model, validation = load_xgb_research_model()

if validation:
    verdict = "beats" if validation["xgb_brier"] < validation["elo_brier"] else "does not beat"
    st.caption(
        f"Leave-one-tournament-out backtest (train on 2018 predict 2022, and vice versa), "
        f"n={validation['n']} decisive matches: Elo-only Brier={validation['elo_brier']:.4f} vs. "
        f"XGBoost Brier={validation['xgb_brier']:.4f} -- XGBoost {verdict} this Elo baseline, but "
        f"the confidence intervals overlap substantially at this sample size, so treat this as "
        f"inconclusive, not a proven win. (Note: this Elo baseline is refit on the same small "
        f"98-row sample for a fair comparison -- it is NOT the same as the production Elo model "
        f"above, which trains on ~24,500 competitive matches across all history and backtests "
        f"at Brier 0.1541.)"
    )

match_labels = [f"{r.date} — {r.home_team} vs {r.away_team}" for r in xg_df.itertuples()]
selected = st.selectbox("Explore a historical match", match_labels)
sel_row = xg_df.iloc[match_labels.index(selected)]

feature_row = {c: sel_row[c] for c in FEATURE_COLUMNS}
xgb_prob = xgb_model.predict_proba([[feature_row[c] for c in FEATURE_COLUMNS]])[0, 1]
actual = "Home win" if sel_row["home_win"] == 1 else "Away/draw"
st.write(f"**{sel_row['home_team']} vs {sel_row['away_team']}** ({sel_row['date']}) — "
         f"XGBoost P(home wins) = {xgb_prob:.3f} — actual result: {actual}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Why this prediction (SHAP feature contributions)**")
    contributions = shap_contributions(xgb_model, feature_row, FEATURE_COLUMNS)
    st.dataframe(pd.DataFrame(contributions), hide_index=True, use_container_width=True)
with col2:
    st.markdown("**Key players (top xG contributors, from StatsBomb)**")
    home_players = key_players_statsbomb(sel_row["match_id"], sel_row["home_team"])
    away_players = key_players_statsbomb(sel_row["match_id"], sel_row["away_team"])
    st.write(f"{sel_row['home_team']}:", pd.DataFrame(home_players) if home_players else "no shots recorded")
    st.write(f"{sel_row['away_team']}:", pd.DataFrame(away_players) if away_players else "no shots recorded")

st.divider()
st.subheader("Elo model vs. real Kalshi opening prices (2026 Round of 32)")
st.caption(
    "A genuine live-market backtest, not simulated: the 16 Round of 32 matches already "
    "played this tournament, walk-forward Elo (each match uses only data available before "
    "it -- no peeking at its own or later results) vs. Kalshi's own opening price on that "
    "match's market. 'Opening price' here is each market's first-hour CLOSING read, not its "
    "literal first tick -- checked directly against Kalshi's data and found the very first "
    "tick is a listing-seed artifact (both sides of a market showed the identical price, "
    "which is impossible for a real two-outcome market), so it's excluded rather than used."
)

r32_rows, r32_stats = load_r32_vs_kalshi()
st.write(
    f"n={r32_stats['n']} team-market outcomes (each of the 16 matches counted once per side): "
    f"Elo Brier={r32_stats['model_brier']:.4f} (95% CI [{r32_stats['model_ci'][0]:.4f}, {r32_stats['model_ci'][1]:.4f}]), "
    f"Kalshi opening Brier={r32_stats['kalshi_brier']:.4f} (95% CI [{r32_stats['kalshi_ci'][0]:.4f}, {r32_stats['kalshi_ci'][1]:.4f}])."
)
if r32_stats["model_brier"] < r32_stats["kalshi_brier"]:
    st.write("Elo beat Kalshi's opening price on this sample -- but n=32 is small, the CIs "
             "overlap, and opening prices are inherently less efficient than closing lines "
             "(days of trading to react to news happen after a market opens), so this is a "
             "real but modest result, not proof of a durable, exploitable edge.")

r32_df = pd.DataFrame(r32_rows)[["date", "match", "team", "model_prob", "kalshi_opening_price", "actual"]]
r32_df.columns = ["Date", "Match", "Team", "Elo model prob", "Kalshi opening price", "Actual"]
r32_df["Actual"] = r32_df["Actual"].map({1: "won", 0: "lost"})
st.dataframe(r32_df.sort_values("Date"), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Bankroll simulation: fractional Kelly staked against those Round of 32 opening prices")
st.caption(
    "Only bets with positive edge get a stake (Kelly clips the rest to $0). Same-day bets "
    "are all sized off the bankroll as of the START of that day, not updated bet-by-bet "
    "within the day -- a real bettor placing same-day bets doesn't know an earlier same-day "
    "result yet. This uses opening prices, which are easier to beat than closing lines, and "
    "only 16 matches -- a good run here is not proof of a durable edge."
)
bsim_col1, bsim_col2, bsim_col3 = st.columns(3)
with bsim_col1:
    sim_bankroll = st.number_input("Starting bankroll ($)", min_value=1.0, value=100.0, step=10.0, key="sim_bankroll")
with bsim_col2:
    sim_fraction = st.slider("Kelly fraction", 0.05, 1.0, 0.25, 0.05, key="sim_fraction")
with bsim_col3:
    sim_use_cap = st.toggle("Apply 5% max-stake cap", value=True, key="sim_use_cap")

sim_history, sim_final = simulate_bankroll(
    bankroll=sim_bankroll, fraction=sim_fraction, max_stake_pct=0.05 if sim_use_cap else None,
    rows=r32_rows,
)
sim_bet_rows = []
for day in sim_history:
    for b in day["bets"]:
        sim_bet_rows.append({
            "Date": day["date"], "Team": b["team"], "Model prob": round(b["model_prob"], 3),
            "Kalshi opening": round(b["kalshi_opening_price"], 3), "Stake ($)": round(b["stake"], 2),
            "Result": "won" if b["actual"] == 1 else "lost", "P&L ($)": round(b["pnl"], 2),
        })
st.write(
    f"**${sim_bankroll:.2f} -> ${sim_final:.2f}** ({(sim_final / sim_bankroll - 1) * 100:+.1f}%), "
    f"{len(sim_bet_rows)} bets placed of 32 team-market opportunities."
)
st.dataframe(pd.DataFrame(sim_bet_rows), hide_index=True, use_container_width=True)
