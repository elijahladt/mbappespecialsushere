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


def _participant_names(ids: list, sport_id: int = 10):
    data = _get("/v4/participants", {"sportId": sport_id})
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


# Club league fixtures: BetMGM does NOT offer a straight 3-way "Full Time
# Result" (1X2) market for club league matches -- checked directly against
# Premier League tournament ID 17 (categoryName "England"), 31 markets
# returned, no market ID 101. What IS offered includes Draw No Bet (market
# ID 10214/10215: outcome 10214 = home team excl. draw, 10215 = away team
# excl. draw, stake refunded on a draw) -- a clean 2-way market that reuses
# the existing binary Kelly/edge machinery with zero new trading code.
DRAW_NO_BET_MARKET_ID = "10214"


def get_betmgm_draw_no_bet_matches(tournament_id: int):
    """Returns one row per club league fixture with a BetMGM 'Draw No Bet'
    line: team names + decimal odds for each side (draw excluded/refunded).
    Rows where the market isn't active yet (e.g. off-season, odds not
    posted) are skipped rather than returned with placeholder prices."""
    fixtures = _get("/v4/odds-by-tournaments", {
        "bookmaker": "betmgm", "tournamentIds": tournament_id,
    })

    candidates = []
    ids_needed = set()
    for fx in fixtures:
        markets = fx.get("bookmakerOdds", {}).get("betmgm", {}).get("markets", {})
        dnb = markets.get(DRAW_NO_BET_MARKET_ID)
        if not dnb or not dnb.get("marketActive", False):
            continue
        outcomes = dnb.get("outcomes", {})
        p1 = outcomes.get("10214", {}).get("players", {}).get("0", {})
        p2 = outcomes.get("10215", {}).get("players", {}).get("0", {})
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
        team1 = names.get(c["participant1_id"], f"id{c['participant1_id']}")
        team2 = names.get(c["participant2_id"], f"id{c['participant2_id']}")
        rows.append({
            "title": f"{team1} vs {team2}",
            "start_time": c["start_time"],
            "home_team": team1,
            "away_team": team2,
            "home_price": c["price1"],
            "away_price": c["price2"],
        })
    return rows


# Tennis: unlike a club league, a tennis "tournament" on OddsPapi is a
# single week-long event with its own tournamentId that only exists for
# that event -- there's no fixed id to hardcode the way there is for the
# Premier League. Live fixtures are discovered at request time instead.
TENNIS_MATCH_WINNER_MARKET_ID = "121"  # confirmed directly: outcome 121 = player1, 122 = player2

# Slugs to exclude when discovering "singles" tournaments -- OddsPapi lists
# doubles/mixed draws as separate tournaments under the same ATP/WTA
# categorySlug, and this project only models singles matches.
_NON_SINGLES_SLUG_MARKERS = ("doubles", "mixed")


def discover_active_tennis_tournaments(category_slug: str):
    """Returns [(tournamentId, tournamentName), ...] for ATP or WTA (pass
    category_slug="atp" or "wta") singles tournaments that currently have
    live or upcoming fixtures -- checked directly against the live API
    (Wimbledon singles draws included, e.g. tournamentId 2555/2559)."""
    from src.config import TENNIS_ODDSPAPI_SPORT_ID
    tournaments = _get("/v4/tournaments", {"sportId": TENNIS_ODDSPAPI_SPORT_ID})
    matches = []
    for t in tournaments:
        if t.get("categorySlug") != category_slug:
            continue
        slug = t.get("tournamentSlug", "")
        if any(marker in slug for marker in _NON_SINGLES_SLUG_MARKERS):
            continue
        if t.get("liveFixtures", 0) > 0 or t.get("upcomingFixtures", 0) > 0:
            matches.append((t["tournamentId"], t.get("tournamentName", "")))
    return matches


def get_betmgm_tennis_matches(category_slug: str):
    """Returns one row per live/upcoming ATP or WTA singles fixture with a
    BetMGM 'Match Winner' line: player names (raw OddsPapi format, i.e.
    "Surname, First" -- normalize before joining to Elo ratings) + decimal
    odds for each side."""
    from src.config import TENNIS_ODDSPAPI_SPORT_ID

    tournaments = discover_active_tennis_tournaments(category_slug)
    if not tournaments:
        return []
    tournament_ids = ",".join(str(tid) for tid, _ in tournaments)

    fixtures = _get("/v4/odds-by-tournaments", {
        "bookmaker": "betmgm", "tournamentIds": tournament_ids,
    })

    candidates = []
    ids_needed = set()
    for fx in fixtures:
        markets = fx.get("bookmakerOdds", {}).get("betmgm", {}).get("markets", {})
        mw = markets.get(TENNIS_MATCH_WINNER_MARKET_ID)
        if not mw or not mw.get("marketActive", False):
            continue
        outcomes = mw.get("outcomes", {})
        p1 = outcomes.get("121", {}).get("players", {}).get("0", {})
        p2 = outcomes.get("122", {}).get("players", {}).get("0", {})
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

    names = _participant_names(list(ids_needed), sport_id=TENNIS_ODDSPAPI_SPORT_ID)
    rows = []
    for c in candidates:
        p1_raw = names.get(c["participant1_id"], f"id{c['participant1_id']}")
        p2_raw = names.get(c["participant2_id"], f"id{c['participant2_id']}")
        rows.append({
            "title": f"{p1_raw} vs {p2_raw}",
            "start_time": c["start_time"],
            "player1_raw": p1_raw,
            "player2_raw": p2_raw,
            "player1_price": c["price1"],
            "player2_price": c["price2"],
        })
    return rows


if __name__ == "__main__":
    for row in get_betmgm_moneyline_matches():
        a, b = row["teams"]
        print(f"{row['title']:30s} | {a['team']:>15s} {a['decimal_odds']:.2f}  vs  {b['team']:<15s} {b['decimal_odds']:.2f}")

    print("\nClub football (Premier League) Draw No Bet lines:")
    club_rows = get_betmgm_draw_no_bet_matches(17)
    if not club_rows:
        print("  (none active right now -- expected in the off-season; check again closer to kickoff)")
    for row in club_rows:
        print(f"  {row['title']:40s} | {row['home_team']:>15s} {row['home_price']:.2f}  vs  {row['away_team']:<15s} {row['away_price']:.2f}")

    from src.ingest.tennis_name_match import normalize_oddspapi_name
    for tour in ("atp", "wta"):
        print(f"\n{tour.upper()} Match Winner lines:")
        tennis_rows = get_betmgm_tennis_matches(tour)
        if not tennis_rows:
            print("  (none active right now)")
        for row in tennis_rows:
            p1 = normalize_oddspapi_name(row["player1_raw"])
            p2 = normalize_oddspapi_name(row["player2_raw"])
            print(f"  {row['title']:45s} | {p1:>18s} {row['player1_price']:.2f}  vs  {p2:<18s} {row['player2_price']:.2f}")
