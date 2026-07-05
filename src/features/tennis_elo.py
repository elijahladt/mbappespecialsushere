"""Chronological Elo rating engine for tennis, parameterized by tour ("atp"
or "wta") so each has its own isolated pool -- same rationale as
src/features/club_elo.py's per-league pools, but tennis itself is
structurally simpler than club football: matches are winner/loser (no
home/away, no neutral-site distinction to make -- tennis is essentially
always neutral court barring Davis Cup/Billie Jean King Cup ties, which
aren't ingested here), and there's no draw.

Two additions beyond the v1 single-feature model, both driven by a
walk-forward-P&L finding that Elo-diff alone falls well short of the
market's own calibration (see src/backtest/pnl_backtest_tennis.py):

1. Tournament-level K-factor, via Series (ATP)/Tier (WTA) -- a Grand Slam
   result should move ratings more than a regional 250-level one. WTA's
   Tier column has drifted over the years (clean "WTA1000/500/250" labels
   in recent seasons, "Premier"/"International" pre-2021, and a long tail
   of odd numeric codes like "WTA263" in older files that don't cleanly
   fit either scheme) -- classify_series_tier() uses keyword matching with
   a safe low-tier fallback for anything unrecognized, rather than an
   exact-match dict that would silently misclassify the long tail.
2. A SEPARATE per-(player, surface) Elo pool run alongside the overall
   pool -- surface matters enormously in tennis (a player's hard-court and
   clay-court ability can differ a lot) and this was the single biggest
   gap identified. Both overall_elo_diff and surface_elo_diff are exposed
   in feature_rows; the model decides how to weight them (surface ratings
   are noisier early in a player's career on that surface, so keeping
   overall as a separate feature lets the model fall back on it).
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

BASE_RATING = 1500.0
K_FACTOR = 32.0
FORM_WINDOW = 10  # matches, for the rolling recent-form feature


def _form_rate(results_deque) -> float:
    """Win rate over the trailing window, or a neutral 0.5 prior if the
    player has no tracked history yet (new to the dataset)."""
    if not results_deque:
        return 0.5
    return sum(results_deque) / len(results_deque)

# Tournament-level K-factor multiplier tiers. Applied to BOTH the overall
# and surface-specific engines.
K_BY_TIER = {1: 24.0, 2: 32.0, 3: 40.0, 4: 48.0}  # regional/250 -> 500/Premier -> Masters/1000 -> Grand Slam


def classify_series_tier(series) -> int:
    if not series:
        return 1
    s = series.lower()
    if "grand slam" in s:
        return 4
    if "1000" in s or "masters cup" in s or "tour championships" in s:
        return 3
    if "500" in s or "premier" in s:
        return 2
    return 1  # ATP250/International/WTA250/unrecognized legacy codes


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


class TennisEloEngine:
    """One isolated rating pool -- used both for the overall per-tour pool
    and, separately, for each per-surface pool."""

    def __init__(self, label: str):
        self.label = label
        self.ratings = {}

    def get(self, player: str) -> float:
        return self.ratings.get(player, BASE_RATING)

    def process_match(self, winner: str, loser: str, tier: int = 1):
        winner_pre = self.get(winner)
        loser_pre = self.get(loser)
        exp_winner = expected_score(winner_pre, loser_pre)

        delta = K_BY_TIER.get(tier, K_FACTOR) * (1.0 - exp_winner)
        self.ratings[winner] = winner_pre + delta
        self.ratings[loser] = loser_pre - delta

        return {"winner_elo_pre": winner_pre, "loser_elo_pre": loser_pre, "exp_winner_pre": exp_winner}


def run_all(tour: str, conn=None):
    """Process every match for one tour chronologically, updating BOTH the
    overall Elo pool and a separate per-surface pool. Returns
    (engine, feature_rows) where `engine` is the OVERALL pool (kept as the
    primary return value for backward compatibility with existing callers
    that only want overall ratings, e.g. the live dashboard's "current
    ratings" table); per-surface engines are attached as
    engine.surface_engines but not returned separately.
    feature_rows carries PRE-match ratings only from both pools -- same
    leakage-free contract as the club/WC Elo engines."""
    own_conn = conn is None
    conn = conn or get_connection()
    matches = conn.execute(
        """SELECT date, tournament, surface, round, series, best_of, winner, loser, comment,
                  b365_winner, b365_loser, pinnacle_winner, pinnacle_loser
           FROM tennis_matches WHERE tour = ? ORDER BY date ASC, rowid ASC""",
        (tour,),
    ).fetchall()

    engine = TennisEloEngine(tour)
    surface_engines = {}  # surface -> TennisEloEngine
    recent_results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    h2h = defaultdict(lambda: defaultdict(int))  # frozenset({a,b}) -> {player: wins}
    feature_rows = []
    for (date, tournament, surface, round_, series, best_of, winner, loser, comment,
         b365_winner, b365_loser, pinnacle_winner, pinnacle_loser) in matches:
        tier = classify_series_tier(series)

        # Read form/h2h BEFORE this match updates them -- leak-free, same
        # discipline as reading Elo's pre-match ratings before process_match.
        winner_form_pre = _form_rate(recent_results[winner])
        loser_form_pre = _form_rate(recent_results[loser])
        pair_key = frozenset({winner, loser})
        winner_h2h_pre = h2h[pair_key][winner]
        loser_h2h_pre = h2h[pair_key][loser]

        overall_result = engine.process_match(winner, loser, tier)

        surface_key = surface or "Unknown"
        if surface_key not in surface_engines:
            surface_engines[surface_key] = TennisEloEngine(f"{tour}:{surface_key}")
        surface_result = surface_engines[surface_key].process_match(winner, loser, tier)

        feature_rows.append({
            "date": date,
            "tournament": tournament,
            "surface": surface,
            "round": round_,
            "series": series,
            "tier": tier,
            "best_of": best_of,
            "winner": winner,
            "loser": loser,
            "comment": comment,
            "b365_winner": b365_winner, "b365_loser": b365_loser,
            "pinnacle_winner": pinnacle_winner, "pinnacle_loser": pinnacle_loser,
            "winner_elo_pre": overall_result["winner_elo_pre"],
            "loser_elo_pre": overall_result["loser_elo_pre"],
            "winner_surface_elo_pre": surface_result["winner_elo_pre"],
            "loser_surface_elo_pre": surface_result["loser_elo_pre"],
            "winner_form_pre": winner_form_pre,
            "loser_form_pre": loser_form_pre,
            "winner_h2h_pre": winner_h2h_pre,
            "loser_h2h_pre": loser_h2h_pre,
        })

        recent_results[winner].append(1)
        recent_results[loser].append(0)
        h2h[pair_key][winner] += 1

    # Attached so live inference (app pages) can read CURRENT form/h2h state
    # for any two named players the same way it reads engine.ratings/
    # engine.surface_engines -- not just usable inside this backtest loop.
    engine.surface_engines = surface_engines
    engine.recent_results = recent_results
    engine.h2h = h2h
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
        print(f"  Surfaces tracked: {list(engine.surface_engines.keys())}")
