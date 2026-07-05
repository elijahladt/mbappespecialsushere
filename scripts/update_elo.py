"""Refresh historical + 2026 results, then print current Elo ratings.
Run this daily during the tournament to keep ratings current."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest.historical_results import fetch_and_load as load_historical
from src.ingest.wc2026_results import fetch_and_load as load_2026
from src.features.elo import run_all

if __name__ == "__main__":
    load_historical()
    load_2026()
    engine, rows = run_all()
    print(f"\nElo ratings refreshed from {len(rows)} matches.")
