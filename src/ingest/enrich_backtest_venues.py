"""Backfill venue city/country + round stage onto existing historical World Cup
rows (already loaded from the martj42 dataset) for the tournaments used in the
walk-forward backtest, by cross-referencing ESPN's scoreboard API. Needed for
the rest-days/travel-distance features, since the base historical dataset has
no venue or round information at all.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection
from src.ingest.wc2026_results import fetch_day, parse_event

# (start, end) inclusive date ranges covering each tournament, used only to
# enrich rows that already exist in the DB -- never inserts new match rows.
BACKTEST_TOURNAMENTS = {
    2010: (date(2010, 6, 11), date(2010, 7, 11)),
    2014: (date(2014, 6, 12), date(2014, 7, 13)),
    2018: (date(2018, 6, 14), date(2018, 7, 15)),
    2022: (date(2022, 11, 20), date(2022, 12, 18)),
}


def enrich_year(conn, start: date, end: date):
    matched, unmatched = 0, 0
    day = start
    while day <= end:
        for event in fetch_day(day):
            parsed = parse_event(event)
            if not parsed:
                continue
            cur = conn.execute(
                """UPDATE matches SET venue_city = ?, venue_country = ?, stage = ?
                   WHERE date = ? AND tournament = 'FIFA World Cup'
                   AND ((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))""",
                (parsed["venue_city"], parsed["venue_country"], parsed["stage"],
                 parsed["date"], parsed["home_team"], parsed["away_team"],
                 parsed["away_team"], parsed["home_team"]),
            )
            if cur.rowcount > 0:
                matched += 1
            else:
                unmatched += 1
                print(f"  no DB match for {parsed['date']} {parsed['home_team']} vs {parsed['away_team']}")
        day += timedelta(days=1)
    return matched, unmatched


def run():
    conn = get_connection()
    for year, (start, end) in BACKTEST_TOURNAMENTS.items():
        matched, unmatched = enrich_year(conn, start, end)
        conn.commit()
        print(f"{year}: enriched {matched} rows, {unmatched} unmatched")
    conn.close()


if __name__ == "__main__":
    run()
