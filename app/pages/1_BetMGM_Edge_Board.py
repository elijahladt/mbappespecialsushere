import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.winprob_link import win_probability, h2h_diff_live
from src.ingest.oddspapi_client import get_betmgm_moneyline_matches
from src.ingest.upcoming_fixtures import fetch_upcoming
from src.features.auto_injury_report import build_live_report, team_auto_flags
from src.features.player_impact import elo_adjustment_for_team, DEFAULT_IMPACT_SCALE, star_player_boost_for_team, DEFAULT_STAR_BOOST_SCALE
from src.features.altitude_timezone import (
    base_camp_altitude_tz_delta, altitude_tz_elo_adjustment,
    DEFAULT_ALTITUDE_SCALE, DEFAULT_TZ_SCALE,
)
from src.trading.devig import implied_prob, devig_proportional
from src.trading.edge_calc import edge, ev_per_dollar
from src.trading.kelly import fractional_kelly_stake

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_engine_and_model, effective_diff, DEFAULT_HOST_BONUS_BY_COUNTRY

st.set_page_config(page_title="BetMGM Edge Board", layout="wide")

with st.expander("🔧 Debug: secret detection (safe -- shows key names/lengths only, never values)"):
    try:
        secret_names = list(st.secrets.keys())
    except Exception as e:
        secret_names = None
        st.write(f"st.secrets raised: {e!r}")
    st.write(f"Keys visible in st.secrets: {secret_names}")
    import os as _os
    for _name in ("ODDSPAPI_KEY", "API_FOOTBALL_KEY"):
        in_secrets = secret_names is not None and _name in secret_names
        env_val = _os.environ.get(_name)
        st.write(
            f"{_name}: in st.secrets={in_secrets}, "
            f"in os.environ={_name in _os.environ} (len={len(env_val) if env_val else 0})"
        )


@st.cache_data(ttl=300, show_spinner="Pulling live BetMGM odds via OddsPapi...")
def load_betmgm_matches():
    return get_betmgm_moneyline_matches()


@st.cache_data(ttl=300, show_spinner="Pulling upcoming fixture stages...")
def load_upcoming_fixtures():
    return {frozenset({r["home_team"], r["away_team"]}): r for r in fetch_upcoming()}


@st.cache_data(ttl=1800, show_spinner="Checking ESPN match events for cards/injuries...")
def load_auto_injury_report():
    return build_live_report()


st.title("World Cup 2026 Edge Board — BetMGM")
st.warning(
    "Same model, same caveats as the Kalshi board: small-sample validated (see the main "
    "page for the historical Brier-score backtests), so trust small edges more than huge "
    "ones and size with fractional Kelly, not full Kelly.",
    icon="⚠️",
)
st.caption(
    "BetMGM has no public API of its own -- this uses OddsPapi (oddspapi.io) as a "
    "third-party aggregator, the only legitimate way to get real BetMGM prices (BetMGM's "
    "internal API is undocumented and scraping it would violate their ToS). Market is "
    "BetMGM's 'Winner (incl. overtime)' line -- the same 2-way, no-draw framing as Kalshi's "
    "advance markets, matching what the Elo model here predicts (WC knockout ties)."
)
st.caption(
    "Sportsbooks bake in a margin (both sides' implied probabilities sum to over 100%), "
    "unlike Kalshi. 'BetMGM raw price' is what you actually pay -- Edge and Kelly stake are "
    "computed against THAT, since that's your real cost. 'De-vigged fair price' removes the "
    "margin proportionally, shown only as a read on BetMGM's true belief for comparison."
)

engine, model, n_train, tracker, _alt_tz_tracker, _feature_rows_full = load_engine_and_model()
st.caption(f"Elo engine trained on full match history; win-probability link fit on {n_train} historical competitive (non-friendly) matches with a decisive result -- see the main Kalshi page for why this expanded beyond WC-knockout-only.")
st.caption(
    "'Altitude change' / 'Timezone change' use each team's REAL FIFA-published 2026 base "
    "camp (not the last city they played -- teams return to a fixed base between matches) "
    "vs. the venue's real elevation/timezone. We have no historical base-camp data to "
    "backtest this corrected version, so it's off by default -- toggle in the sidebar."
)

with st.sidebar:
    st.header("View")
    sort_by = st.selectbox("Sort table by", ["Match (stable)", "Edge (pts)"])
    apply_player_impact = st.toggle(
        "Factor suspensions/injuries into win probability (experimental)",
        value=False,
        help="Same heuristic as the Kalshi board: shifts Elo diff by a confirmed-missing "
             "player's real goal+assist share this tournament. Disclosed guess, not "
             "statistically validated -- off by default.",
    )
    impact_scale = DEFAULT_IMPACT_SCALE
    if apply_player_impact:
        impact_scale = st.slider("Impact scale (Elo points for 100% of output missing)", 0.0, 400.0, DEFAULT_IMPACT_SCALE, 10.0)

    apply_star_player = st.toggle(
        "Factor star-player boost into win probability (experimental)",
        value=False,
        help="Same heuristic as the Kalshi board: POSITIVE adjustment for a team's leading "
             "scorer's real goal+assist count this tournament. Backtested with a StatsBomb "
             "proxy on 2018+2022: INCONCLUSIVE (Brier 0.236 with it vs 0.233 without, n=97) -- "
             "not a validated improvement, off by default.",
    )
    star_boost_scale = DEFAULT_STAR_BOOST_SCALE
    if apply_star_player:
        star_boost_scale = st.slider("Star boost scale (Elo points per goal+assist)", 0.0, 40.0, DEFAULT_STAR_BOOST_SCALE, 2.0)

    apply_altitude_tz = st.toggle(
        "Factor altitude/jet-lag into win probability (experimental)",
        value=False,
        help="Same as the Kalshi board: real FIFA-published 2026 base camp vs. venue "
             "elevation/timezone. Disclosed heuristic, not statistically validated -- off "
             "by default.",
    )
    altitude_scale, tz_scale = DEFAULT_ALTITUDE_SCALE, DEFAULT_TZ_SCALE
    if apply_altitude_tz:
        altitude_scale = st.slider("Altitude scale (Elo points per 1000m gained)", 0.0, 100.0, DEFAULT_ALTITUDE_SCALE, 5.0)
        tz_scale = st.slider("Timezone scale (Elo points per hour crossed)", 0.0, 30.0, DEFAULT_TZ_SCALE, 1.0)

    st.caption(
        "Host-nation advantage split by country -- Mexico defaults higher given verified, "
        "repeated reports (ESPN et al.) of organized crowd hostility (hotel harassment, "
        "sleep disruption) against BOTH Ecuador and England, distinct from ordinary home "
        "support and not reported for USA/Canada. Can't be statistically fit (no historical "
        "base rate exists) -- disclosed judgment call, not a validated number."
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
    if st.button("Refresh BetMGM odds now"):
        st.cache_data.clear()

matches = load_betmgm_matches()
upcoming = load_upcoming_fixtures()
auto_cards, auto_injuries, auto_goals = load_auto_injury_report() if (apply_player_impact or apply_star_player) else (None, None, None)

if not matches:
    st.warning("No BetMGM World Cup knockout moneyline matches found right now.")
else:
    rows = []
    for m in matches:
        team_a, team_b = m["teams"]
        diff = effective_diff(engine, team_a["team"], team_b["team"], host_bonus_by_country=host_bonus_by_country)
        h2h_diff = h2h_diff_live(engine, team_a["team"], team_b["team"])
        model_prob_a = win_probability(model, [diff, h2h_diff])
        model_prob_b = 1 - model_prob_a

        fair_a, fair_b = devig_proportional(team_a["decimal_odds"], team_b["decimal_odds"])

        fixture = upcoming.get(frozenset({team_a["team"], team_b["team"]}))
        alt_tz = {}
        if fixture:
            for team in (team_a["team"], team_b["team"]):
                alt_tz[team] = base_camp_altitude_tz_delta(team, fixture["date"], fixture["venue_city"])

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

        for team, model_prob, fair_prob in ((team_a, model_prob_a, fair_a), (team_b, model_prob_b, fair_b)):
            raw_price = implied_prob(team["decimal_odds"])

            decision_prob = model_prob
            adj_prob, adjustment_detail = None, []
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

                adjusted_diff = (diff if team is team_a else -diff) + (own_total_adj - opp_total_adj)
                adjusted_h2h = h2h_diff if team is team_a else -h2h_diff
                adj_prob = win_probability(model, [adjusted_diff, adjusted_h2h])
                decision_prob = adj_prob

            e = edge(decision_prob, raw_price)
            if abs(e) < min_edge_filter:
                continue
            stake = fractional_kelly_stake(decision_prob, raw_price, bankroll, fraction=kelly_fraction_pct, max_stake_pct=max_stake_pct)

            row = {
                "Match": m["title"],
                "Team": team["team"],
                "Model prob (straight Elo)": round(model_prob, 3),
            }
            if apply_any_adjustment:
                row["Model prob (adjusted)"] = round(adj_prob, 3)
                row["Adjustment detail"] = "; ".join(adjustment_detail) if adjustment_detail else "no confirmed impact"
            alt_delta, tz_delta = alt_tz.get(team["team"], (None, None))
            row["Altitude change (m)"] = round(alt_delta) if alt_delta is not None else "?"
            row["Timezone change (hrs)"] = round(tz_delta, 1) if tz_delta is not None else "?"
            ev_dollar = stake * ev_per_dollar(decision_prob, raw_price)
            profit_if_hit = stake * (1 / raw_price - 1)
            row.update({
                "Decimal odds": team["decimal_odds"],
                "BetMGM raw price": round(raw_price, 3),
                "De-vigged fair price": round(fair_prob, 3),
                "Edge (pts)": round(e, 3),
                "EV per $1": round(ev_per_dollar(decision_prob, raw_price), 3),
                "Suggested stake ($)": round(stake, 2),
                "EV profit ($)": round(ev_dollar, 2),
                "Profit if hit ($)": round(profit_if_hit, 2),
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by == "Edge (pts)":
        df = df.sort_values("Edge (pts)", ascending=False)
    else:
        df = df.sort_values(["Match", "Team"])

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

    st.dataframe(df, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Current Elo ratings (top 20)")
top20 = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
st.dataframe(pd.DataFrame(top20, columns=["Team", "Elo rating"]), use_container_width=True, hide_index=True)
