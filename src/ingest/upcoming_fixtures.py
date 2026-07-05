"""Fetch scheduled (not yet played) World Cup fixtures with venue info from
ESPN, so the dashboard can show rest-days/travel context for upcoming matches
too, not just completed ones."""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.wc2026_results import fetch_day, normalize_team


def fetch_upcoming(days_ahead: int = 20):
    rows = []
    day = date.today()
    end = day + timedelta(days=days_ahead)
    while day <= end:
        for event in fetch_day(day):
            comp = event["competitions"][0]
            if comp["status"]["type"]["completed"]:
                continue
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            address = comp.get("venue", {}).get("address", {})
            rows.append({
                "date": comp["date"][:10],
                "home_team": normalize_team(home["team"]["displayName"]),
                "away_team": normalize_team(away["team"]["displayName"]),
                "venue_city": address.get("city"),
                "stage": event.get("season", {}).get("slug", ""),
            })
        day += timedelta(days=1)
    return rows


if __name__ == "__main__":
    for r in fetch_upcoming():
        print(r)
