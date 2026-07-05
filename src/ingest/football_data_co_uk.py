"""Load domestic league results + bookmaker closing odds from football-data.co.uk
into club_matches. Free, no auth, season-partitioned CSVs
(https://www.football-data.co.uk/mmz4281/{season}/{code}.csv), confirmed live
and covering English Premier League (code E0) back to 1993/94.

Column layout has drifted over ~30 years of files: FTHG/FTAG/FTR (score/result)
and Date are present in every season, but odds columns vary by era and
provider ("PSCH/PSCD/PSCA" Pinnacle-closing in recent files, older "PH/PD/PA",
none at all before ~2000) -- read defensively with .get(), never assume a
column exists.
"""
import csv
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection
from src.config import LEAGUES

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# football-data.co.uk season codes, e.g. "2425" = 2024/25. Earliest season
# confirmed to return real E0 data is 1993/94; stop once a fetch 404s/empty
# rather than hardcoding an end year, so this keeps working next season too.
FIRST_SEASON_START_YEAR = 1993


def _season_codes(first_year: int = FIRST_SEASON_START_YEAR):
    year = first_year
    codes = []
    now_year = datetime.now(timezone.utc).year
    while year <= now_year + 1:
        codes.append(f"{year % 100:02d}{(year + 1) % 100:02d}")
        year += 1
    return codes


def _parse_date(raw: str) -> str:
    """DD/MM/YY or DD/MM/YYYY -> YYYY-MM-DD. Two-digit years use the standard
    football-data.co.uk convention: 70-99 -> 1900s, 00-69 -> 2000s."""
    day, month, year = raw.split("/")
    if len(year) == 2:
        yy = int(year)
        year = str(1900 + yy) if yy >= 70 else str(2000 + yy)
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _first_present(row: dict, *keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "NA"):
            try:
                return float(v)
            except ValueError:
                return None
    return None


def fetch_season(league_id: str, code: str, season: str) -> list:
    resp = requests.get(f"{BASE_URL}/{season}/{code}.csv", timeout=30)
    if resp.status_code != 200 or not resp.text.strip():
        return []
    resp.encoding = "utf-8-sig"
    reader = csv.DictReader(io.StringIO(resp.text))

    league_cfg = LEAGUES[league_id]
    tier = league_cfg["tier"]
    rows = []
    for row in reader:
        if not row.get("HomeTeam") or not row.get("FTHG") or row["FTHG"] in ("", "NA"):
            continue
        try:
            date = _parse_date(row["Date"])
        except (KeyError, ValueError):
            continue
        rows.append((
            date,
            league_id,
            season,
            row["HomeTeam"].strip(),
            row["AwayTeam"].strip(),
            int(row["FTHG"]),
            int(row["FTAG"]),
            tier,
            "football_data_co_uk",
            _first_present(row, "B365H"),
            _first_present(row, "B365D"),
            _first_present(row, "B365A"),
            _first_present(row, "PSCH", "PSH", "PH"),
            _first_present(row, "PSCD", "PSD", "PD"),
            _first_present(row, "PSCA", "PSA", "PA"),
        ))
    return rows


def fetch_and_load(league_id: str = "premier_league"):
    league_cfg = LEAGUES[league_id]
    code = league_cfg["football_data_code"]

    all_rows = []
    for season in _season_codes():
        season_rows = fetch_season(league_id, code, season)
        if season_rows:
            all_rows.extend(season_rows)

    conn = get_connection()
    conn.executemany(
        """INSERT OR IGNORE INTO club_matches
           (date, league_id, season, home_team, away_team, home_score, away_score,
            competition_tier, source, b365_home, b365_draw, b365_away,
            pinnacle_home, pinnacle_draw, pinnacle_away)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        all_rows,
    )
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM club_matches WHERE league_id = ?", (league_id,)
    ).fetchone()[0]
    conn.close()
    print(f"Loaded {len(all_rows)} {league_id} rows across {len(_season_codes())} candidate seasons, {count} rows now in DB for this league.")


if __name__ == "__main__":
    fetch_and_load("premier_league")
