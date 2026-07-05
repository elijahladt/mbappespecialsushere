"""Thin client for OddsPapi (oddspapi.io), used to pull real BetMGM odds for
World Cup knockout matches. BetMGM itself has no public API -- confirmed via
research, no developer portal, scraping their site would violate ToS -- so
this aggregator is the legitimate path to real BetMGM prices.

Requires ODDSPAPI_KEY in a local .env (see .env.example).

Market 10728/10729 ("Winner (incl. overtime)", moneyline, 2-way with no draw
option) is the target -- confirmed via direct API calls to be what BetMGM
actually offers for World Cup knockout matches, matching our Elo model's
"who wins the tie" framing (same semantics as Kalshi's advance market).
"""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_secret, debug_secret_visibility
from src.ingest.wc2026_results import normalize_team

BASE_URL = "https://api.oddspapi.io"
WC_TOURNAMENT_ID = 16
MONEYLINE_MARKET_ID = "10728"  # outcomes: 10728 = participant 1, 10729 = participant 2


def _get(path: str, params: dict):
    api_key = get_secret("ODDSPAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "ODDSPAPI_KEY not set. Locally: copy .env.example to .env and add your key from "
            "https://oddspapi.io/. On Streamlit Cloud: add it under App settings -> Secrets. "
            + debug_secret_visibility("ODDSPAPI_KEY")
        )
    resp = requests.get(f"{BASE_URL}{path}", params={**params, "apiKey": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _participant_names(ids: list):
    data = _get("/v4/participants", {"sportId": 10})
    return {int(k): v for k, v in data.items() if int(k) in ids}


def get_betmgm_moneyline_matches():
    """Returns one row per World Cup knockout match with a BetMGM 'Winner
    (incl. overtime)' line: team names + decimal odds for each side."""
    fixtures = _get("/v4/odds-by-tournaments", {
        "bookmaker": "betmgm", "tournamentIds": WC_TOURNAMENT_ID,
    })

    candidates = []
    ids_needed = set()
    for fx in fixtures:
        markets = fx.get("bookmakerOdds", {}).get("betmgm", {}).get("markets", {})
        ml = markets.get(MONEYLINE_MARKET_ID)
        if not ml:
            continue
        outcomes = ml.get("outcomes", {})
        p1 = outcomes.get("10728", {}).get("players", {}).get("0", {})
        p2 = outcomes.get("10729", {}).get("players", {}).get("0", {})
        if not (p1.get("price") and p2.get("price")):
            continue
        candidates.append({
            "fixture_id": fx["fixtureId"],
            "participant1_id": fx["participant1Id"],
            "participant2_id": fx["participant2Id"],
            "start_time": fx["startTime"],
            "price1": float(p1["price"]),
            "price2": float(p2["price"]),
        })
        ids_needed.update([fx["participant1Id"], fx["participant2Id"]])

    names = _participant_names(list(ids_needed))
    rows = []
    for c in candidates:
        team1 = normalize_team(names.get(c["participant1_id"], f"id{c['participant1_id']}"))
        team2 = normalize_team(names.get(c["participant2_id"], f"id{c['participant2_id']}"))
        rows.append({
            "title": f"{team1} vs {team2}",
            "start_time": c["start_time"],
            "teams": [
                {"team": team1, "decimal_odds": c["price1"]},
                {"team": team2, "decimal_odds": c["price2"]},
            ],
        })
    return rows


if __name__ == "__main__":
    for row in get_betmgm_moneyline_matches():
        a, b = row["teams"]
        print(f"{row['title']:30s} | {a['team']:>15s} {a['decimal_odds']:.2f}  vs  {b['team']:<15s} {b['decimal_odds']:.2f}")
