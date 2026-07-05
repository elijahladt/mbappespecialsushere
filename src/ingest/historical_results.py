"""Load historical international football results (1872-present) into wc.sqlite.

Source: martj42/international_results (public GitHub CSV mirror of the
well-known Jurisoo "international football results" dataset).
"""
import csv
import io
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

# Competition tiers drive Elo K-factor: higher tier = higher-stakes match = bigger rating swings.
TIER_WORLD_CUP = 4
TIER_CONTINENTAL_FINALS = 3
TIER_QUALIFIERS = 2
TIER_FRIENDLY = 1

CONTINENTAL_FINALS_KEYWORDS = [
    "copa américa", "copa america", "uefa euro", "african cup of nations",
    "afc asian cup", "gold cup", "concacaf championship", "confederations cup",
    "oceania nations cup",
]
QUALIFIER_KEYWORDS = ["qualification", "qualifier"]


def classify_tier(tournament: str) -> int:
    t = tournament.lower()
    if t == "fifa world cup":
        return TIER_WORLD_CUP
    if any(k in t for k in QUALIFIER_KEYWORDS):
        return TIER_QUALIFIERS
    if any(k in t for k in CONTINENTAL_FINALS_KEYWORDS):
        return TIER_CONTINENTAL_FINALS
    if t == "friendly":
        return TIER_FRIENDLY
    # Minor/regional cups, unofficial tournaments etc. -> treat like friendlies.
    return TIER_FRIENDLY


def fetch_and_load():
    resp = requests.get(RESULTS_URL, timeout=60)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    rows = []
    skipped = 0
    duplicate_2026_wc = 0
    for row in reader:
        if row["home_score"] in ("", "NA") or row["away_score"] in ("", "NA"):
            skipped += 1
            continue
        tournament = row["tournament"]
        # This CSV is community-maintained and turns out to already include
        # 2026 results (through early July) -- including World Cup matches
        # that src.ingest.wc2026_results also fetches from ESPN separately.
        # Without this filter, any match appearing in both (often off by a
        # day between sources) gets its Elo update counted TWICE. ESPN is the
        # sole source of truth for 2026 World Cup matches; skip this source's
        # copies of them.
        if tournament == "FIFA World Cup" and row["date"] >= "2026-01-01":
            duplicate_2026_wc += 1
            continue
        rows.append((
            row["date"],
            row["home_team"],
            row["away_team"],
            int(row["home_score"]),
            int(row["away_score"]),
            tournament,
            classify_tier(tournament),
            1 if row["neutral"].strip().upper() == "TRUE" else 0,
            "martj42_historical",
        ))

    conn = get_connection()
    conn.executemany(
        """INSERT OR IGNORE INTO matches
           (date, home_team, away_team, home_score, away_score, tournament, tier, neutral, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM matches WHERE source = 'martj42_historical'").fetchone()[0]
    conn.close()
    print(f"Loaded {len(rows)} historical rows (skipped {skipped} with no score, {duplicate_2026_wc} duplicate 2026 World Cup rows already covered by ESPN), {count} rows now in DB from this source.")


if __name__ == "__main__":
    fetch_and_load()
