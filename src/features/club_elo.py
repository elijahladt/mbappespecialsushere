"""Chronological Elo rating engine for club football, parameterized by
league_id so each domestic league gets its own isolated rating pool (same
team name playing in two different competitions never collides) --
mirrors src/features/elo.py's methodology exactly (goal-difference
multiplier, home-advantage constant) but is a separate module: the WC
engine is proven and live, and club football's tier axis (domestic league
vs continental competition) is a different taxonomy from the WC engine's
tier axis (friendly -> qualifier -> continental -> World Cup), so the two
should never share one K_BY_TIER dict.

Cross-league bridging (e.g. Champions League) is intentionally NOT special
Elo math -- it just means two matches with league_id="CL" update ratings
that live in each team's own domestic pool (see run_all: ratings are keyed
by (league_id, team) for domestic matches, but a club's "current rating" for
display purposes is read from its own domestic pool regardless of which
competition last updated it).
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

BASE_RATING = 1500.0
HOME_ADVANTAGE = 100.0
FORM_WINDOW = 10  # matches, for the rolling recent-form feature -- added
                   # after a walk-forward P&L backtest showed pure Elo-diff
                   # falls well short of the market's own calibration; form
                   # catches momentum/injury swings faster than Elo drifts.


def _form_rate(results_deque) -> float:
    """Average normalized result (1.0 win / 0.5 draw / 0.0 loss) over the
    trailing window, or a neutral 0.5 if the team has no tracked history
    yet -- same 0-1 scale and same neutral-prior convention as tennis's
    _form_rate() in tennis_elo.py, so both features mean the same thing."""
    if not results_deque:
        return 0.5
    return sum(results_deque) / len(results_deque)

# competition_tier -> K-factor. 1 = domestic league play. Higher tiers
# (continental group/knockout) get added here once Milestone 3 (Champions
# League) is built -- not guessed ahead of time.
K_BY_TIER = {1: 30}


def goal_diff_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(rating_a - rating_b) / 400.0))


class ClubEloEngine:
    """One isolated rating pool for a single league_id. Domestic-league-only
    for now (see module docstring re: cross-league bridging) -- ratings for
    a team are scoped to this engine's own league_id."""

    def __init__(self, league_id: str):
        self.league_id = league_id
        self.ratings = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, BASE_RATING)

    def process_match(self, home, away, home_score, away_score, competition_tier):
        home_pre = self.get(home)
        away_pre = self.get(away)
        exp_home = expected_score(home_pre + HOME_ADVANTAGE, away_pre)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = K_BY_TIER.get(competition_tier, 30) * goal_diff_multiplier(home_score - away_score)
        delta = k * (actual_home - exp_home)

        self.ratings[home] = home_pre + delta
        self.ratings[away] = away_pre - delta

        return {
            "home_elo_pre": home_pre,
            "away_elo_pre": away_pre,
            "exp_home_pre": exp_home,
        }


def run_all(league_id: str, conn=None):
    """Process every match for one league chronologically. Returns
    (engine, feature_rows) with PRE-match ratings only -- safe for training,
    same leakage-free contract as src/features/elo.py's run_all()."""
    own_conn = conn is None
    conn = conn or get_connection()
    matches = conn.execute(
        """SELECT date, season, home_team, away_team, home_score, away_score, competition_tier,
                  b365_home, b365_draw, b365_away, pinnacle_home, pinnacle_draw, pinnacle_away
           FROM club_matches WHERE league_id = ? ORDER BY date ASC, rowid ASC""",
        (league_id,),
    ).fetchall()

    engine = ClubEloEngine(league_id)
    recent_results = defaultdict(lambda: deque(maxlen=FORM_WINDOW))
    feature_rows = []
    for (date, season, home, away, home_score, away_score, competition_tier,
         b365_home, b365_draw, b365_away, pinnacle_home, pinnacle_draw, pinnacle_away) in matches:
        # Read form BEFORE this match updates it -- leak-free, same
        # discipline as reading Elo's pre-match ratings before process_match.
        home_form_pre = _form_rate(recent_results[home])
        away_form_pre = _form_rate(recent_results[away])

        result = engine.process_match(home, away, home_score, away_score, competition_tier)
        feature_rows.append({
            "date": date,
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "competition_tier": competition_tier,
            "home_form_pre": home_form_pre,
            "away_form_pre": away_form_pre,
            # Real historical bookmaker odds -- NOT used as model features (that
            # would be a different, odds-implied-probability model), only kept
            # here so backtests can simulate what actually betting against them
            # would have earned, alongside the calibration-only Brier checks.
            "b365_home": b365_home, "b365_draw": b365_draw, "b365_away": b365_away,
            "pinnacle_home": pinnacle_home, "pinnacle_draw": pinnacle_draw, "pinnacle_away": pinnacle_away,
            **result,
        })

        if home_score > away_score:
            home_result, away_result = 1.0, 0.0
        elif home_score == away_score:
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0.0, 1.0
        recent_results[home].append(home_result)
        recent_results[away].append(away_result)

    # Attached so live inference (app pages) can read CURRENT form state for
    # any two named teams, same as engine.ratings.
    engine.recent_results = recent_results
    if own_conn:
        conn.close()
    return engine, feature_rows


if __name__ == "__main__":
    engine, rows = run_all("premier_league")
    top = sorted(engine.ratings.items(), key=lambda kv: -kv[1])[:20]
    print(f"Processed {len(rows)} premier_league matches, {len(engine.ratings)} teams rated.")
    print("Current top 20 Elo ratings:")
    for team, rating in top:
        print(f"  {team:30s} {rating:.1f}")
