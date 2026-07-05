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
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn
