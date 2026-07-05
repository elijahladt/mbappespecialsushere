"""Rest-days and travel-change tracking, computed as legitimate public proxies
for team fatigue/condition since individual biometric data (e.g. wearable
recovery scores) isn't available for opposing national team players -- that
data is private and no legitimate source publishes it. These are the honest,
publicly-derivable stand-ins: how many days since the team's last match in
this tournament, and whether they had to relocate to a new host city.

NOTE: a walk-forward comparison (src/backtest/validate_extended_features.py)
found these do NOT improve the win-probability model's Brier score over plain
Elo on the available data (n=23, essentially a wash) -- so they are NOT fed
into win_probability(). They're surfaced on the dashboard as context only.
"""
import sys
from datetime import date as _date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection


def _parse_date(s: str) -> _date:
    return _date(int(s[:4]), int(s[5:7]), int(s[8:10]))


class RestTravelTracker:
    def __init__(self):
        self.last_seen = {}  # (year, team) -> (date_str, venue_city)

    def state(self, year: str, team: str):
        return self.last_seen.get((year, team))

    def rest_days_and_travel(self, year: str, team: str, current_date: str, current_city):
        prev = self.state(year, team)
        if prev is None:
            return None, None
        prev_date, prev_city = prev
        rest_days = (_parse_date(current_date) - _parse_date(prev_date)).days
        traveled = int(current_city != prev_city) if current_city and prev_city else None
        return rest_days, traveled

    def process_match(self, year: str, home: str, away: str, date_str: str, venue_city):
        row = {}
        for side, team in (("home", home), ("away", away)):
            rest, traveled = self.rest_days_and_travel(year, team, date_str, venue_city)
            row[f"{side}_rest_days"] = rest
            row[f"{side}_traveled"] = traveled
        for team in (home, away):
            self.last_seen[(year, team)] = (date_str, venue_city)
        return row


def compute_rest_travel(conn=None):
    """Returns (tracker, features) where features is keyed by
    (date, home_team, away_team) -> {home_rest_days, away_rest_days,
    home_traveled, away_traveled} for every completed FIFA World Cup match.
    `tracker` retains final per-team state so callers can also compute rest/
    travel for a hypothetical upcoming match."""
    own_conn = conn is None
    conn = conn or get_connection()
    matches = conn.execute(
        """SELECT date, home_team, away_team, venue_city
           FROM matches WHERE tournament = 'FIFA World Cup'
           ORDER BY date ASC, rowid ASC"""
    ).fetchall()
    if own_conn:
        conn.close()

    tracker = RestTravelTracker()
    features = {}
    for date_str, home, away, venue_city in matches:
        year = date_str[:4]
        features[(date_str, home, away)] = tracker.process_match(year, home, away, date_str, venue_city)

    return tracker, features


if __name__ == "__main__":
    tracker, feats = compute_rest_travel()
    sample = [(k, v) for k, v in feats.items() if v["home_rest_days"] is not None][:10]
    print(f"Computed rest/travel features for {len(feats)} World Cup matches.")
    for k, v in sample:
        print(k, v)
