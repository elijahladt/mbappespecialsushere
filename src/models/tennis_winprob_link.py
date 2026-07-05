"""Elo-diff -> binary win-probability model for tennis. Simpler than both
existing links: no draws (winprob is P(player_a beats player_b) directly,
no 3-way split like club football) and no home advantage (tennis is
neutral-court, unlike club football's fixed home team).

Training-set symmetrization: tennis-data.co.uk records each match as
winner/loser, which is a POST-match label -- using it directly as
(elo_diff = winner_pre - loser_pre, label=1) for every single row would be
degenerate (100% of labels would be 1, since "winner" is true by
definition). The fix is to add BOTH perspectives of every match to the
training set: (winner_pre - loser_pre, 1) AND (loser_pre - winner_pre, 0).
This is not double-counting in a harmful sense -- it's exactly what makes
the dataset symmetric around zero, which is the correct constraint for a
1v1 rating system with no home-side bias (P(A beats B) should equal
1 - P(B beats A) exactly, and a logistic fit on asymmetric data wouldn't
guarantee that; fitting on the symmetrized set does, up to noise).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.tennis_elo import run_all


def build_training_set(feature_rows):
    X, y = [], []
    for row in feature_rows:
        diff = row["winner_elo_pre"] - row["loser_elo_pre"]
        X.append([diff])
        y.append(1)
        X.append([-diff])
        y.append(0)
    return np.array(X), np.array(y)


def fit_link(feature_rows=None, tour: str = "atp"):
    if feature_rows is None:
        _, feature_rows = run_all(tour)
    X, y = build_training_set(feature_rows)
    model = LogisticRegression()
    model.fit(X, y)
    return model, len(y)


def win_probability(model, elo_diff_a_minus_b: float) -> float:
    """Probability player A (elo_diff = A's rating - B's rating) wins."""
    return model.predict_proba([[elo_diff_a_minus_b]])[0, 1]


if __name__ == "__main__":
    for tour in ("atp", "wta"):
        model, n = fit_link(tour=tour)
        print(f"\n{tour.upper()}: fit on {n} symmetrized rows ({n // 2} real matches).")
        print(f"Coefficient (per Elo point): {model.coef_[0][0]:.5f}, intercept: {model.intercept_[0]:.4f} (should be ~0)")
        for diff in (-400, -200, -100, 0, 100, 200, 400):
            print(f"  elo_diff={diff:+5d} -> P(A wins) = {win_probability(model, diff):.3f}")
