"""Backtest the Elo win-probability model against real Kalshi OPENING prices
for the 2026 World Cup Round of 32 -- the 16 matches that have already been
played and settled this tournament. This is a genuine live-market comparison
(distinct from the 2010-2022 historical backtest), using actual money-backed
prices rather than a simulated market.

Walk-forward: for each R32 match, Elo is recomputed from scratch using ONLY
matches strictly before that match's date -- the live dashboard's cached Elo
engine already includes these R32 results in its ratings, which would leak
the outcome back into "what the model would have said beforehand."
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import HOME_ADVANTAGE, run_all
from src.models.winprob_link import win_probability, h2h_diff_live
from src.ingest.kalshi_client import BASE_URL, get_markets_for_event
from src.ingest.wc2026_results import normalize_team
from src.backtest.metrics import brier_score, bootstrap_brier_ci

HOST_NATIONS = {"United States", "Mexico", "Canada"}


def get_settled_events(series_ticker="KXWCADVANCE"):
    events, cursor = [], ""
    while True:
        params = {"series_ticker": series_ticker, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/events", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events.extend(data.get("events", []))
        cursor = data.get("cursor", "")
        if not cursor:
            break
    return events


def get_opening_price(ticker: str, open_time_iso: str):
    """The very first candle's `open_dollars` is a listing-seed artifact, not
    a real trade: verified directly against Kalshi's own data -- it shows the
    exact same value (e.g. 0.8100) on BOTH sides of a two-outcome market and
    across unrelated matchups, which is mathematically impossible for a real
    price (two complementary outcomes can't both open near the same price).
    The first candle's CLOSE is the earliest price after that artifact
    settles, and lines up sensibly with the opposing side's own close
    (checked: e.g. Belgium 0.65 / Senegal 0.36 on the same match, sums to
    ~1 as a real market should)."""
    open_dt = datetime.fromisoformat(open_time_iso.replace("Z", "+00:00"))
    start_ts = int(open_dt.timestamp())
    end_ts = start_ts + 24 * 3600  # first day of trading is plenty to find one candle
    resp = requests.get(
        f"{BASE_URL}/series/KXWCADVANCE/markets/{ticker}/candlesticks",
        params={"period_interval": 60, "start_ts": start_ts, "end_ts": end_ts},
        timeout=30,
    )
    resp.raise_for_status()
    candles = resp.json().get("candlesticks", [])
    candles = [c for c in candles if c.get("price", {}).get("close_dollars") not in (None, "0.0000")]
    if not candles:
        return None
    return float(candles[0]["price"]["close_dollars"])


def elo_asof(cutoff_date: str):
    """Rebuild Elo using only matches strictly before cutoff_date -- avoids
    leaking the very match being predicted (or same-day matches) into its
    own pre-match rating. Delegates to src/features/elo.py's run_all() so
    this backtest can never silently drift out of sync with the live
    dashboard's methodology (year-boundary reversion, h2h tracking, etc.) --
    a duplicated reimplementation here already caused exactly that risk once."""
    engine, _ = run_all(cutoff_date=cutoff_date)
    return engine


def build_comparison():
    events = get_settled_events()
    rows = []
    for event in events:
        markets = get_markets_for_event(event["event_ticker"])
        if len(markets) != 2:
            continue
        match_date = markets[0]["occurrence_datetime"][:10]
        engine = elo_asof(match_date)

        # Fit the win-probability link using only knockout history strictly
        # before this match too, for the same no-leakage reason.
        model, n_train = fit_link_asof(match_date)

        for m in markets:
            team = normalize_team(m["yes_sub_title"].replace(" advances", "").strip())
            opponent_market = [mm for mm in markets if mm["ticker"] != m["ticker"]][0]
            opponent = normalize_team(opponent_market["yes_sub_title"].replace(" advances", "").strip())

            adv = HOME_ADVANTAGE if team in HOST_NATIONS and opponent not in HOST_NATIONS else 0.0
            diff = (engine.get(team) + adv) - engine.get(opponent)
            model_prob = win_probability(model, [diff, h2h_diff_live(engine, team, opponent)])

            opening_price = get_opening_price(m["ticker"], m["open_time"])
            actual = 1 if m["result"] == "yes" else 0

            rows.append({
                "match": event["title"], "date": match_date, "team": team,
                "model_prob": model_prob, "kalshi_opening_price": opening_price,
                "actual": actual, "n_train": n_train,
            })
    return rows


def fit_link_asof(cutoff_date: str):
    """Same delegation rationale as elo_asof() -- reuses run_all() rather
    than a separate reimplementation of the chronological loop."""
    from src.models.winprob_link import build_training_set
    from sklearn.linear_model import LogisticRegression

    _, feature_rows = run_all(cutoff_date=cutoff_date)
    X, y = build_training_set(feature_rows)
    model = LogisticRegression()
    model.fit(X, y)
    return model, len(y)


if __name__ == "__main__":
    rows = build_comparison()
    valid = [r for r in rows if r["kalshi_opening_price"] is not None]
    print(f"{len(valid)} of {len(rows)} team-markets had a usable opening price\n")

    for r in sorted(valid, key=lambda r: r["date"]):
        print(f"{r['date']} {r['match']:28s} {r['team']:>15s}  "
              f"model={r['model_prob']:.3f}  kalshi_open={r['kalshi_opening_price']:.3f}  "
              f"actual={'won' if r['actual'] else 'lost'}")

    model_probs = [r["model_prob"] for r in valid]
    kalshi_probs = [r["kalshi_opening_price"] for r in valid]
    outcomes = [r["actual"] for r in valid]

    model_brier = brier_score(model_probs, outcomes)
    kalshi_brier = brier_score(kalshi_probs, outcomes)
    m_lo, m_hi = bootstrap_brier_ci(model_probs, outcomes)
    k_lo, k_hi = bootstrap_brier_ci(kalshi_probs, outcomes)

    print(f"\nn={len(valid)} team-market outcomes (each match counted twice, once per side)")
    print(f"Elo model:      Brier={model_brier:.4f} (95% CI [{m_lo:.4f}, {m_hi:.4f}])")
    print(f"Kalshi opening: Brier={kalshi_brier:.4f} (95% CI [{k_lo:.4f}, {k_hi:.4f}])")
