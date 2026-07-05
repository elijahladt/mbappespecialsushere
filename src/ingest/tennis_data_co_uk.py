"""Load ATP/WTA historical match results + bookmaker odds from
tennis-data.co.uk into tennis_matches. Free, no auth, one .xlsx per season
(confirmed live: ATP at /{year}/{year}.xlsx from 2000, WTA at
/{year}w/{year}.xlsx). Their HTTPS is misconfigured (TLS handshake fails);
plain HTTP works fine and is what's used here.

Column layout drifts slightly by tour and era (WTA best-of-3 has W1-W3/L1-L3,
ATP best-of-5 has W1-W5/L1-L5; older years may be missing odds columns
entirely) -- read defensively with .get(), never assume a column exists.
"""
import io
import sys
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection
from src.config import TENNIS_TOURS

BASE_URL = "http://www.tennis-data.co.uk"


def _first_present(row: dict, *keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, "", "NA"):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None
    return None


def fetch_season(tour: str, year: int) -> list:
    tour_cfg = TENNIS_TOURS[tour]
    url = BASE_URL + tour_cfg["tennis_data_path_fn"](year)
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or not resp.content:
        return []

    try:
        wb = openpyxl.load_workbook(io.BytesIO(resp.content), read_only=True, data_only=True)
    except Exception:
        return []
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        wb.close()
        return []

    rows = []
    for values in rows_iter:
        row = dict(zip(header, values))
        winner, loser = row.get("Winner"), row.get("Loser")
        date_val = row.get("Date")
        if not winner or not loser or not date_val:
            continue
        comment = (row.get("Comment") or "").strip()
        if comment == "Walkover":
            continue  # no match actually played
        if hasattr(date_val, "strftime"):
            date = date_val.strftime("%Y-%m-%d")
        else:
            continue

        rows.append((
            date,
            tour,
            str(row.get("Tournament") or ""),
            row.get("Surface"),
            row.get("Round"),
            str(winner).strip(),
            str(loser).strip(),
            comment or None,
            "tennis_data_co_uk",
            _first_present(row, "B365W"),
            _first_present(row, "B365L"),
            _first_present(row, "PSW"),
            _first_present(row, "PSL"),
        ))
    wb.close()
    return rows


def fetch_and_load(tour: str = "atp"):
    tour_cfg = TENNIS_TOURS[tour]
    first_year = tour_cfg["first_year"]
    now_year = datetime.now(timezone.utc).year

    all_rows = []
    for year in range(first_year, now_year + 1):
        season_rows = fetch_season(tour, year)
        if season_rows:
            all_rows.extend(season_rows)

    conn = get_connection()
    conn.executemany(
        """INSERT OR IGNORE INTO tennis_matches
           (date, tour, tournament, surface, round, winner, loser, comment, source,
            b365_winner, b365_loser, pinnacle_winner, pinnacle_loser)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        all_rows,
    )
    conn.commit()
    count = conn.execute(
        "SELECT COUNT(*) FROM tennis_matches WHERE tour = ?", (tour,)
    ).fetchone()[0]
    conn.close()
    print(f"Loaded {len(all_rows)} {tour} rows ({first_year}-{now_year}), {count} rows now in DB for this tour.")


if __name__ == "__main__":
    fetch_and_load("atp")
    fetch_and_load("wta")
