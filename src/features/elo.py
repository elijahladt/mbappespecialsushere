"""Chronological Elo rating engine for national teams.

Methodology follows the publicly documented World Football Elo approach:
K-factor scaled by competition importance, a goal-difference multiplier so
blowouts move ratings more than 1-goal wins, and a fixed home-advantage bonus
that's zeroed out on neutral ground.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0
K_BY_TIER = {1: 20, 2: 40, 3: 50, 4: 60}  # friendly, qualifiers, continental finals, World Cup

# Season-boundary mean reversion (FiveThirtyEight's NFL/NBA convention:
# shrink ratings partway back toward BASE_RATING at each year boundary, so
# old dominance can't coast indefinitely) -- TESTED and REJECTED. Every
# fraction tried (0.05 to 0.33) made both the 2010-2022 historical walk-
# forward backtest AND the live 2026 Kalshi Round-of-32 backtest WORSE,
# monotonically -- more reversion, worse Brier, no exceptions. Likely
# cause: NFL/NBA teams play 16+ games/season with real roster turnover
# between seasons; international teams play far fewer matches/year (often
# 8-12) and the existing K-factor already reacts fast to real results
# (up to 60 for WC-tier matches) -- there's no seasonal roster reset this
# is compensating for, so it was just injecting noise. Kept at 0
# (disabled) rather than removed entirely so this negative result and the
# reasoning behind it stays visible, not silently forgotten.
REVERSION_FRACTION = 0.0


def goal_diff_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


class EloEngine:
    def __init__(self):
        self.ratings = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, BASE_RATING)

    def process_match(self, home, away, home_score, away_score, tier, neutral):
        """Update ratings for one match; returns pre-match ratings/expectation
        so callers can build leakage-free features from the same pass."""
        home_pre = self.get(home)
        away_pre = self.get(away)
        adv = 0.0 if neutral else HOME_ADVANTAGE
        exp_home = expected_score(home_pre + adv, away_pre)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = K_BY_TIER.get(tier, 20) * goal_diff_multiplier(home_score - away_score)
        delta = k * (actual_home - exp_home)

        self.ratings[home] = home_pre + delta
        self.ratings[away] = away_pre - delta

        return {
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "exp_home_pre": exp_home,
        }


def run_all(conn=None, cutoff_date: str = None):
    """Process every match chronologically (optionally only matches strictly
    before cutoff_date, for walk-forward/as-of backtests that need to avoid
    leaking a match's own or later results into its pre-match rating).
    Returns (engine, feature_rows) where feature_rows carries the PRE-match
    ratings for every match -- safe to use for training since no match ever
    sees information from its own or later outcomes."""
    own_conn = conn is None
    conn = conn or get_connection()
    if cutoff_date:
        matches = conn.execute(
            """SELECT date, home_team, away_team, home_score, away_score, tournament, tier, neutral, stage
               FROM matches WHERE date < ? ORDER BY date ASC, rowid ASC""",
            (cutoff_date,),
        ).fetchall()
    else:
        matches = conn.execute(
            """SELECT date, home_team, away_team, home_score, away_score, tournament, tier, neutral, stage
               FROM matches ORDER BY date ASC, rowid ASC"""
        ).fetchall()

    engine = EloEngine()
    # Head-to-head record between two specific national teams -- added
    # after a discussion about tactical/matchup effects Elo can't represent
    # (Elo treats strength as one transitive scalar; some teams genuinely
    # have a "bogey" opponent regardless of overall rating). Tracked across
    # ALL match history (not just WC knockout matches, which is all the
    # win-probability link itself trains on) since more history means a
    # more stable signal -- same rationale as tennis's h2h feature.
    h2h = defaultdict(lambda: defaultdict(int))  # frozenset({a,b}) -> {team: wins}
    feature_rows = []
    current_year = None
    for date, home, away, home_score, away_score, tournament, tier, neutral, stage in matches:
        year = date[:4]
        if current_year is not None and year != current_year:
            for team in engine.ratings:
                engine.ratings[team] = BASE_RATING + (1 - REVERSION_FRACTION) * (engine.ratings[team] - BASE_RATING)
        current_year = year

        pair_key = frozenset({home, away})
        home_h2h_pre = h2h[pair_key][home]
        away_h2h_pre = h2h[pair_key][away]

        result = engine.process_match(home, away, home_score, away_score, tier, bool(neutral))
        feature_rows.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": tournament,
            "tier": tier,
            "neutral": bool(neutral),
            "stage": stage,
            "home_h2h_pre": home_h2h_pre,
            "away_h2h_pre": away_h2h_pre,
            **result,
        })

        if home_score > away_score:
            h2h[pair_key][home] += 1
        elif away_score > home_score:
            h2h[pair_key][away] += 1
        # draws add nothing to either side's h2h win count

    # Attached so live inference (app pages) can read CURRENT h2h state for
    # any two named teams, same as engine.ratings.
    engine.h2h = h2h
    if own_conn:
        conn.close()
    return engine, feature_rows


if __name__ == "__main__":
    engine, rows = run_all()
    top = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
    print(f"Processed {len(rows)} matches, {len(engine.ratings)} teams rated.")
    print("Current top 20 Elo ratings:")
    for team, rating in top:
        print(f"  {team:30s} {rating:.1f}")
