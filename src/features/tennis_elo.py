"""Chronological Elo rating engine for tennis, parameterized by tour ("atp"
or "wta") so each has its own isolated pool -- same rationale as
src/features/club_elo.py's per-league pools, but tennis itself is
structurally simpler than club football: matches are winner/loser (no
home/away, no neutral-site distinction to make -- tennis is essentially
always neutral court barring Davis Cup/Billie Jean King Cup ties, which
aren't ingested here), and there's no draw.

v1 uses a single constant K-factor rather than stratifying by tournament
level (Grand Slam vs. ATP250 etc.) -- a disclosed simplification, not an
oversight: tennis-data.co.uk's Series/Tier column would support that split
later, but resisting it until a walk-forward backtest shows the extra
complexity is actually earning its keep (same discipline as every other
feature in this project).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

BASE_RATING = 1500.0
K_FACTOR = 32.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


class TennisEloEngine:
    """One isolated rating pool for a single tour ("atp" or "wta")."""

    def __init__(self, tour: str):
        self.tour = tour
        self.ratings = {}

    def get(self, player: str) -> float:
        return self.ratings.get(player, BASE_RATING)

    def process_match(self, winner: str, loser: str):
        winner_pre = self.get(winner)
        loser_pre = self.get(loser)
        exp_winner = expected_score(winner_pre, loser_pre)

        delta = K_FACTOR * (1.0 - exp_winner)
        self.ratings[winner] = winner_pre + delta
        self.ratings[loser] = loser_pre - delta

        return {"winner_elo_pre": winner_pre, "loser_elo_pre": loser_pre, "exp_winner_pre": exp_winner}


def run_all(tour: str, conn=None):
    """Process every match for one tour chronologically. Returns
    (engine, feature_rows) with PRE-match ratings only -- same leakage-free
    contract as the club/WC Elo engines."""
    own_conn = conn is None
    conn = conn or get_connection()
    matches = conn.execute(
        """SELECT date, tournament, surface, round, winner, loser, comment
           FROM tennis_matches WHERE tour = ? ORDER BY date ASC, rowid ASC""",
        (tour,),
    ).fetchall()

    engine = TennisEloEngine(tour)
    feature_rows = []
    for date, tournament, surface, round_, winner, loser, comment in matches:
        result = engine.process_match(winner, loser)
        feature_rows.append({
            "date": date,
            "tournament": tournament,
            "surface": surface,
            "round": round_,
            "winner": winner,
            "loser": loser,
            "comment": comment,
            **result,
        })

    if own_conn:
        conn.close()
    return engine, feature_rows


if __name__ == "__main__":
    for tour in ("atp", "wta"):
        engine, rows = run_all(tour)
        top = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:15]
        print(f"\n{tour.upper()}: processed {len(rows)} matches, {len(engine.ratings)} players rated.")
        for player, rating in top:
            print(f"  {player:25s} {rating:.1f}")
