import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "wc.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    tournament TEXT NOT NULL,
    tier INTEGER NOT NULL,
    neutral INTEGER NOT NULL,
    source TEXT NOT NULL,
    stage TEXT,
    venue_city TEXT,
    venue_country TEXT,
    UNIQUE(date, home_team, away_team, tournament)
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team);

CREATE TABLE IF NOT EXISTS elo_ratings (
    team TEXT NOT NULL,
    date TEXT NOT NULL,
    rating REAL NOT NULL,
    PRIMARY KEY (team, date)
);

CREATE TABLE IF NOT EXISTS club_matches (
    date TEXT NOT NULL,
    league_id TEXT NOT NULL,
    season TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_score INTEGER NOT NULL,
    away_score INTEGER NOT NULL,
    competition_tier INTEGER NOT NULL,
    source TEXT NOT NULL,
    b365_home REAL, b365_draw REAL, b365_away REAL,
    pinnacle_home REAL, pinnacle_draw REAL, pinnacle_away REAL,
    UNIQUE(date, league_id, home_team, away_team)
);
CREATE INDEX IF NOT EXISTS idx_club_matches_date ON club_matches(date);
CREATE INDEX IF NOT EXISTS idx_club_matches_league_teams ON club_matches(league_id, home_team, away_team);

CREATE TABLE IF NOT EXISTS club_elo_ratings (
    league_id TEXT NOT NULL,
    team TEXT NOT NULL,
    date TEXT NOT NULL,
    rating REAL NOT NULL,
    PRIMARY KEY (league_id, team, date)
);

-- Tennis has no home/away (neutral court) and no draws, so it's stored as
-- winner/loser rather than home/away -- a genuinely different shape from
-- club_matches, not just a relabeling.
CREATE TABLE IF NOT EXISTS tennis_matches (
    date TEXT NOT NULL,
    tour TEXT NOT NULL,
    tournament TEXT NOT NULL,
    surface TEXT,
    round TEXT,
    winner TEXT NOT NULL,
    loser TEXT NOT NULL,
    comment TEXT,
    source TEXT NOT NULL,
    b365_winner REAL, b365_loser REAL,
    pinnacle_winner REAL, pinnacle_loser REAL,
    UNIQUE(date, tour, tournament, winner, loser)
);
CREATE INDEX IF NOT EXISTS idx_tennis_matches_date ON tennis_matches(date);
CREATE INDEX IF NOT EXISTS idx_tennis_matches_tour_players ON tennis_matches(tour, winner, loser);

CREATE TABLE IF NOT EXISTS tennis_elo_ratings (
    tour TEXT NOT NULL,
    player TEXT NOT NULL,
    date TEXT NOT NULL,
    rating REAL NOT NULL,
    PRIMARY KEY (tour, player, date)
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn
