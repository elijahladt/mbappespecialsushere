"""Elo-diff -> 3-way (home/draw/away) win-probability model for club
football. Unlike the World Cup knockout link (src/models/winprob_link.py,
a binary "who wins the tie" classifier fit only on decisive matches), club
league matches genuinely end in a draw ~20-25% of the time and that outcome
must be modeled directly -- this is a multinomial classifier, not binary.

Single feature (elo_diff_with_home_advantage) for v1, matching the same
"prove the simple thing calibrates first" discipline as winprob_link.py --
resist adding more features until src/backtest/walk_forward_club.py shows
this baseline is honestly calibrated.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.club_elo import HOME_ADVANTAGE, run_all


def effective_elo_diff(row) -> float:
    return (row["home_elo_pre"] + HOME_ADVANTAGE) - row["away_elo_pre"]


def outcome_label(row) -> int:
    """0 = away win, 1 = draw, 2 = home win -- ordered so class index lines
    up with sklearn's sorted-classes convention for predict_proba columns."""
    if row["home_score"] > row["away_score"]:
        return 2
    if row["home_score"] == row["away_score"]:
        return 1
    return 0


def build_training_set(feature_rows):
    X = [[effective_elo_diff(row)] for row in feature_rows]
    y = [outcome_label(row) for row in feature_rows]
    return np.array(X), np.array(y)


def fit_link(feature_rows=None, league_id: str = "premier_league"):
    if feature_rows is None:
        _, feature_rows = run_all(league_id)
    X, y = build_training_set(feature_rows)
    # lbfgs (the default solver) natively fits a multinomial model whenever
    # the target has >2 classes -- no multi_class kwarg needed/supported on
    # newer scikit-learn.
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, len(y)


def win_draw_away_probability(model, elo_diff_home_minus_away: float):
    """Returns (p_away, p_draw, p_home) -- summing to 1 by construction,
    ordered to match outcome_label's 0/1/2 convention."""
    probs = model.predict_proba([[elo_diff_home_minus_away]])[0]
    classes = list(model.classes_)
    return tuple(probs[classes.index(c)] for c in (0, 1, 2))


if __name__ == "__main__":
    model, n = fit_link()
    print(f"Fit on {n} Premier League matches.")
    print(f"Classes: {model.classes_}")
    for diff in (-300, -150, -50, 0, 50, 150, 300):
        p_away, p_draw, p_home = win_draw_away_probability(model, diff)
        print(f"  eff_elo_diff={diff:+5d} -> P(home)={p_home:.3f} P(draw)={p_draw:.3f} P(away)={p_away:.3f}  sum={p_home+p_draw+p_away:.3f}")
