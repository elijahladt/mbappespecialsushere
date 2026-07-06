"""Chronological, leak-free offensive/defensive efficiency tracker for
national teams -- rolling average goals scored ("offensive efficiency")
and goals conceded ("defensive efficiency") over a team's trailing
EFFICIENCY_WINDOW matches, tracked pre-match (no peeking at a match's own
or later goals).

Built from real match results already ingested (src/features/elo.py's same
`matches` table) rather than API-Football: confirmed directly that
API-Football's free tier blocks team-statistics for the 2026 season
(same wall as fixtures/injuries -- "Free plans do not have access to this
season, try from 2022 to 2024"), even though the /leagues endpoint lists
2026 as an available season for the World Cup. Real match scores give the
same underlying signal (how many goals a team tends to score/concede) and
aren't blocked at all -- and cover every historical tournament too, not
just seasons a paid API tier would unlock.
"""
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

EFFICIENCY_WINDOW = 20  # matches


def _avg_or_default(values_deque, default: float):
    if not values_deque:
        return default
    return sum(values_deque) / len(values_deque)


def build_efficiency_table(league_avg_goals: float = None):
    """Returns (rows, league_avg_goals) where rows is a list of dicts, one
    per match, with PRE-match offensive/defensive efficiency for both
    teams. league_avg_goals is computed from the data if not given --
    needed by the Dixon-Coles-lite expected-goals formula in
    goal_simulation.py."""
    conn = get_connection()
    matches = conn.execute(
        """SELECT date, home_team, away_team, home_score, away_score
           FROM matches ORDER BY date ASC, rowid ASC"""
    ).fetchall()
    conn.close()

    if league_avg_goals is None:
        totals = [m[3] + m[4] for m in matches]
        league_avg_goals = (sum(totals) / len(totals)) / 2 if totals else 1.35  # per TEAM per match

    scored = defaultdict(lambda: deque(maxlen=EFFICIENCY_WINDOW))
    conceded = defaultdict(lambda: deque(maxlen=EFFICIENCY_WINDOW))

    rows = []
    for date, home, away, home_score, away_score in matches:
        home_off_pre = _avg_or_default(scored[home], league_avg_goals)
        home_def_pre = _avg_or_default(conceded[home], league_avg_goals)
        away_off_pre = _avg_or_default(scored[away], league_avg_goals)
        away_def_pre = _avg_or_default(conceded[away], league_avg_goals)

        rows.append({
            "date": date, "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "home_off_pre": home_off_pre, "home_def_pre": home_def_pre,
            "away_off_pre": away_off_pre, "away_def_pre": away_def_pre,
        })

        scored[home].append(home_score)
        conceded[home].append(away_score)
        scored[away].append(away_score)
        conceded[away].append(home_score)

    return rows, league_avg_goals


if __name__ == "__main__":
    rows, league_avg = build_efficiency_table()
    print(f"League average goals per team per match: {league_avg:.3f}")
    print(f"{len(rows)} matches processed.\n")
    last = rows[-1]
    print(f"Most recent match: {last['home_team']} vs {last['away_team']} ({last['date']})")
    print(f"  {last['home_team']}: offense={last['home_off_pre']:.2f}, defense (goals conceded)={last['home_def_pre']:.2f}")
    print(f"  {last['away_team']}: offense={last['away_off_pre']:.2f}, defense (goals conceded)={last['away_def_pre']:.2f}")
