"""Poisson goal-based match simulator -- a genuinely different model from
the Elo-diff logistic regression used elsewhere in this project: instead of
directly classifying "who wins," this simulates a random SCORELINE for each
team 10,000 times and derives win/draw/loss (and bonus outputs: correct
score, over/under) from the resulting distribution.

Two ways to get expected goals, both feeding the same Poisson simulation
core (see simulate_from_expected_goals):
  1. fit_goal_model() / expected_goals() -- simple linear regression of
     actual goal_diff on effective Elo diff (the original v1 approach).
  2. expected_goals_from_efficiency() -- a proper two-sided, Dixon-Coles-
     lite model: each team's own rolling offensive/defensive efficiency
     (src/features/team_efficiency.py), combined multiplicatively. This is
     the richer model -- it can tell a high-scoring/leaky team apart from a
     low-scoring/solid one even at the same Elo diff, which #1 can't.
Both are backtested honestly against each other in
src/backtest/validate_goal_simulation.py before either is trusted.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import HOME_ADVANTAGE


def fit_goal_model(feature_rows):
    """Regresses actual goal difference on effective Elo diff across ALL
    historical matches (not just knockout matches -- more data, and goal-
    scoring behavior isn't knockout-specific the way "who advances" is).
    Returns (slope, intercept, avg_total_goals)."""
    elo_diffs, goal_diffs, totals = [], [], []
    for r in feature_rows:
        adv = 0.0 if r["neutral"] else HOME_ADVANTAGE
        elo_diffs.append((r["home_elo_pre"] + adv) - r["away_elo_pre"])
        goal_diffs.append(r["home_score"] - r["away_score"])
        totals.append(r["home_score"] + r["away_score"])

    elo_diffs = np.array(elo_diffs)
    goal_diffs = np.array(goal_diffs)
    A = np.vstack([elo_diffs, np.ones_like(elo_diffs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, goal_diffs, rcond=None)
    avg_total_goals = float(np.mean(totals))
    return {"slope": float(slope), "intercept": float(intercept), "avg_total_goals": avg_total_goals}


def expected_goals(goal_model: dict, effective_elo_diff: float):
    expected_diff = goal_model["slope"] * effective_elo_diff + goal_model["intercept"]
    total = goal_model["avg_total_goals"]
    home = max((total + expected_diff) / 2, 0.05)
    away = max((total - expected_diff) / 2, 0.05)
    return home, away


def fit_xg_goal_model(feature_rows):
    """Research-only expected-goals model: regresses each team's OWN actual
    match xG (StatsBomb shot-quality data, 2018+2022 only) on their Elo-diff
    perspective (their Elo - opponent's Elo + home advantage). Every match
    contributes two rows (each team's own perspective), which both widens
    the tiny StatsBomb sample and keeps it symmetric the way the WC
    knockout logistic link's symmetrization does elsewhere in this project.

    IMPORTANT: this can NEVER be used for live 2026 inference -- there is no
    source of real-time xG for the current tournament (StatsBomb has no
    2026 coverage; checked directly that football-data.org's real, working
    2026 match data has no shots/xG/statistics fields at all, just scores).
    This is a research-only comparison, same treatment as the existing
    Elo+StatsBomb XGBoost model -- see validate_goal_simulation.py for the
    honest leave-one-tournament-out backtest before trusting it for anything,
    and even then it only ever gets shown against 2018/2022 matches.

    `feature_rows` is unused directly (kept for a consistent call signature
    with the other fit_*_model functions) -- build_xg_training_table()
    below already does its own Elo run internally.
    """
    from src.ingest.statsbomb_data import load_team_match_stats
    from src.features.xg_stats import build_xg_training_table

    # build_xg_training_table() already does the match_id <-> (date, home,
    # away) join correctly (including the StatsBomb team-name normalization)
    # -- reuse it to get match_id -> elo_diff, then bring in BOTH teams' own
    # actual xg_for from team_match_stats for the symmetrized regression.
    merged = build_xg_training_table()
    elo_diff_by_match = dict(zip(merged["match_id"], merged["elo_diff"]))

    team_stats = load_team_match_stats()
    diffs, xgs = [], []
    for _, row in team_stats.iterrows():
        match_id = row["match_id"]
        if match_id not in elo_diff_by_match:
            continue
        home_perspective_diff = elo_diff_by_match[match_id]
        team_diff = home_perspective_diff if row["is_home"] else -home_perspective_diff
        diffs.append(team_diff)
        xgs.append(row["xg_for"])

    diffs = np.array(diffs)
    xgs = np.array(xgs)
    A = np.vstack([diffs, np.ones_like(diffs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, xgs, rcond=None)
    return {"slope": float(slope), "intercept": float(intercept), "n": len(xgs)}


def expected_goals_from_xg(xg_model: dict, effective_elo_diff: float):
    """home/away expected goals using the xG-fitted model -- each side's
    own perspective diff plugged into the same fitted line."""
    home = max(xg_model["slope"] * effective_elo_diff + xg_model["intercept"], 0.05)
    away = max(xg_model["slope"] * (-effective_elo_diff) + xg_model["intercept"], 0.05)
    return home, away


def simulate_match_xg(xg_model: dict, effective_elo_diff: float, n_sims: int = 10000, seed: int = 42):
    home_lambda, away_lambda = expected_goals_from_xg(xg_model, effective_elo_diff)
    return simulate_from_expected_goals(home_lambda, away_lambda, n_sims=n_sims, seed=seed)


def expected_goals_from_efficiency(home_off: float, home_def: float, away_off: float, away_def: float,
                                     league_avg_goals: float):
    """Dixon-Coles-lite multiplicative expected goals: each side's expected
    goals = league average * (own attacking strength) * (opponent's
    defensive weakness), where strength/weakness are each team's own
    rolling goals-for/against relative to the league average. Two teams at
    the SAME Elo diff can get different expected scorelines here if one is
    high-scoring/leaky and the other low-scoring/solid -- fit_goal_model()
    above can't tell them apart, since it only ever sees the Elo diff."""
    home_attack = home_off / league_avg_goals
    away_defense_weakness = away_def / league_avg_goals
    away_attack = away_off / league_avg_goals
    home_defense_weakness = home_def / league_avg_goals

    home_lambda = max(league_avg_goals * home_attack * away_defense_weakness, 0.05)
    away_lambda = max(league_avg_goals * away_attack * home_defense_weakness, 0.05)
    return home_lambda, away_lambda


def simulate_from_expected_goals(home_lambda: float, away_lambda: float, n_sims: int = 10000, seed: int = 42):
    """Poisson simulation core shared by both expected-goals methods above."""
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(home_lambda, size=n_sims)
    away_goals = rng.poisson(away_lambda, size=n_sims)

    home_win = float(np.mean(home_goals > away_goals))
    draw = float(np.mean(home_goals == away_goals))
    away_win = float(np.mean(home_goals < away_goals))
    over_2_5 = float(np.mean((home_goals + away_goals) > 2.5))

    scorelines, counts = np.unique(np.stack([home_goals, away_goals], axis=1), axis=0, return_counts=True)
    order = np.argsort(-counts)
    top_scores = [
        {"score": f"{int(scorelines[i][0])}-{int(scorelines[i][1])}", "probability": float(counts[i] / n_sims)}
        for i in order[:5]
    ]

    return {
        "home_expected_goals": home_lambda,
        "away_expected_goals": away_lambda,
        "p_home_win": home_win,
        "p_draw": draw,
        "p_away_win": away_win,
        "p_over_2_5": over_2_5,
        "top_scorelines": top_scores,
        "n_sims": n_sims,
    }


def simulate_match(goal_model: dict, effective_elo_diff: float, n_sims: int = 10000, seed: int = 42):
    """Backward-compatible entry point using the original Elo-diff-only
    expected-goals method (fit_goal_model/expected_goals)."""
    home_lambda, away_lambda = expected_goals(goal_model, effective_elo_diff)
    return simulate_from_expected_goals(home_lambda, away_lambda, n_sims=n_sims, seed=seed)


def simulate_match_efficiency(home_off: float, home_def: float, away_off: float, away_def: float,
                                league_avg_goals: float, n_sims: int = 10000, seed: int = 42):
    """Entry point using the efficiency-based (Dixon-Coles-lite) method."""
    home_lambda, away_lambda = expected_goals_from_efficiency(home_off, home_def, away_off, away_def, league_avg_goals)
    return simulate_from_expected_goals(home_lambda, away_lambda, n_sims=n_sims, seed=seed)


if __name__ == "__main__":
    from src.features.elo import run_all
    from src.features.team_efficiency import build_efficiency_table

    _, feature_rows = run_all()
    goal_model = fit_goal_model(feature_rows)
    print(f"Elo-based goal model: slope={goal_model['slope']:.5f}, intercept={goal_model['intercept']:.4f}, "
          f"avg_total_goals={goal_model['avg_total_goals']:.3f}\n")

    for label, diff in [("Even matchup", 0.0), ("Moderate favorite (+150)", 150.0), ("Heavy favorite (+400)", 400.0)]:
        result = simulate_match(goal_model, diff, n_sims=10000)
        print(f"[Elo-based] {label} (effective_elo_diff={diff:+.0f}):")
        print(f"  Expected goals: home={result['home_expected_goals']:.2f}, away={result['away_expected_goals']:.2f}")
        print(f"  P(home win)={result['p_home_win']:.3f}  P(draw)={result['p_draw']:.3f}  P(away win)={result['p_away_win']:.3f}\n")

    _, league_avg = build_efficiency_table()
    print(f"[Efficiency-based] League avg goals/team/match: {league_avg:.3f}")
    result = simulate_match_efficiency(home_off=3.15, home_def=0.95, away_off=2.00, away_def=1.00,
                                       league_avg_goals=league_avg, n_sims=10000)
    print(f"  Norway (off=3.15,def=0.95) vs Brazil (off=2.00,def=1.00):")
    print(f"  Expected goals: {result['home_expected_goals']:.2f} - {result['away_expected_goals']:.2f}")
    print(f"  P(Norway win)={result['p_home_win']:.3f}  P(draw)={result['p_draw']:.3f}  P(Brazil win)={result['p_away_win']:.3f}")
