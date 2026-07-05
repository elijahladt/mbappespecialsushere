"""Print current open Kalshi World Cup match markets and prices."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingest.kalshi_client import get_match_markets

if __name__ == "__main__":
    for row in get_match_markets():
        a, b = row["teams"]
        print(f"{row['title']:35s} | {a['team']:>15s} {a['price']:.3f}  vs  {b['team']:<15s} {b['price']:.3f}")
