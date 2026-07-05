"""Pull completed 2026 FIFA World Cup match results from ESPN's public scoreboard
API (undocumented but widely used, read-only, no auth) and load them into wc.sqlite
so Elo ratings can be brought current through today.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection
from src.ingest.historical_results import TIER_WORLD_CUP

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
TOURNAMENT_START = date(2026, 6, 11)

# ESPN team names occasionally differ from the historical dataset's naming convention.
# Extend this as mismatches surface (a name that never accrues history is a good sign
# a row belongs here).
TEAM_ALIASES = {
    "Türkiye": "Turkey",
    "USA": "United States",
    "South Korea": "South Korea",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
}


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def fetch_day(day: date):
    resp = requests.get(SCOREBOARD_URL, params={"dates": day.strftime("%Y%m%d")}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("events", [])


def parse_event(event):
    comp = event["competitions"][0]
    if not comp["status"]["type"]["completed"]:
        return None
    home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
    away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
    stage = event.get("season", {}).get("slug", "")
    match_date = comp["date"][:10]
    address = comp.get("venue", {}).get("address", {})
    return {
        "date": match_date,
        "home_team": normalize_team(home["team"]["displayName"]),
        "away_team": normalize_team(away["team"]["displayName"]),
        "home_score": int(home["score"]),
        "away_score": int(away["score"]),
        "stage": stage,
        "venue_city": address.get("city"),
        "venue_country": address.get("country"),
    }


def fetch_and_load(through: date | None = None):
    through = through or date.today()
    rows = []
    day = TOURNAMENT_START
    while day <= through:
        for event in fetch_day(day):
            parsed = parse_event(event)
            if parsed:
                host_playing = parsed["home_team"] in ("United States", "Mexico", "Canada")
                rows.append((
                    parsed["date"], parsed["home_team"], parsed["away_team"],
                    parsed["home_score"], parsed["away_score"],
                    "FIFA World Cup", TIER_WORLD_CUP,
                    0 if host_playing else 1,
                    "espn_2026_live", parsed["stage"],
                    parsed["venue_city"], parsed["venue_country"],
                ))
        day += timedelta(days=1)

    conn = get_connection()
    conn.executemany(
        """INSERT OR REPLACE INTO matches
           (date, home_team, away_team, home_score, away_score, tournament, tier, neutral, source, stage, venue_city, venue_country)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    # Diagnostic: flag any 2026 team never seen in historical data -- likely a naming
    # mismatch that will silently give that team a fresh, uninformative Elo rating.
    known_teams = {r[0] for r in conn.execute(
        "SELECT DISTINCT home_team FROM matches WHERE source = 'martj42_historical'"
    ).fetchall()}
    teams_2026 = {r[1] for r in rows} | {r[2] for r in rows}
    unmatched = sorted(teams_2026 - known_teams)
    conn.close()

    print(f"Loaded {len(rows)} completed 2026 World Cup matches through {through}.")
    if unmatched:
        print(f"WARNING: teams with no historical match (check TEAM_ALIASES): {unmatched}")


if __name__ == "__main__":
    fetch_and_load()
