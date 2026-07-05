"""Thin client for Kalshi's public (no-auth) market-data endpoints, scoped to
FIFA World Cup match markets. Read-only: only pulls prices, never places orders.
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.wc2026_results import normalize_team

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
WC_MATCH_SERIES = "KXWCADVANCE"


def _get(path, params=None):
    resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_open_events(series_ticker=WC_MATCH_SERIES):
    events, cursor = [], ""
    while True:
        params = {"series_ticker": series_ticker, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        data = _get("/events", params)
        events.extend(data.get("events", []))
        cursor = data.get("cursor", "")
        if not cursor:
            break
    return events


def get_markets_for_event(event_ticker):
    return _get("/markets", {"event_ticker": event_ticker}).get("markets", [])


def _mid_price(market) -> float | None:
    bid = float(market.get("yes_bid_dollars") or 0)
    ask = float(market.get("yes_ask_dollars") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    last = float(market.get("last_price_dollars") or 0)
    return last if last > 0 else None


def get_match_markets(series_ticker=WC_MATCH_SERIES):
    """Returns one row per open World Cup match: two teams, each with a
    Kalshi-implied win (advance) probability from the yes bid/ask midpoint."""
    rows = []
    for event in get_open_events(series_ticker):
        markets = get_markets_for_event(event["event_ticker"])
        teams = []
        for m in markets:
            team = normalize_team(m["yes_sub_title"].replace(" advances", "").strip())
            price = _mid_price(m)
            teams.append({"team": team, "price": price, "ticker": m["ticker"]})
        if len(teams) == 2:
            rows.append({
                "event_ticker": event["event_ticker"],
                "title": event["title"],
                "sub_title": event.get("sub_title", ""),
                "teams": teams,
            })
    return rows


if __name__ == "__main__":
    for row in get_match_markets():
        a, b = row["teams"]
        print(f"{row['title']:35s} | {a['team']:>15s} {a['price']:.3f}  vs  {b['team']:<15s} {b['price']:.3f}")
