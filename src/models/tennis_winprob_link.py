"""Multi-feature win-probability model for tennis. Started as a single
Elo-diff feature; extended after a walk-forward P&L backtest showed that
single feature falls well short of the market's own calibration
(src/backtest/pnl_backtest_tennis.py). Still no draws, still no home
advantage (tennis is neutral-court) -- those structural facts didn't
change, only the feature set did.

Features (all pre-match, leak-free):
  0. elo_diff          -- overall Elo rating difference
  1. surface_elo_diff  -- per-surface Elo difference (surface matters a
                           lot in tennis; a player's hard/clay/grass
                           ability can differ substantially)
  2. form_diff         -- rolling win-rate difference over the last
                           FORM_WINDOW matches (catches current form/
                           injury-comeback swings faster than Elo drifts)
  3. h2h_diff           -- head-to-head win-count difference between
                           these two specific players
  4. is_bo5             -- 1 if this is a best-of-5 match (ATP Grand
                           Slams), 0 otherwise (Bo5 reduces upset
                           frequency relative to Bo3)
  5. bo5_x_elo          -- interaction term (is_bo5 * elo_diff), so the
                           model can learn that favorites convert their
                           edge MORE reliably over 5 sets than 3

Training-set symmetrization (unchanged from v1): tennis-data.co.uk labels
matches winner/loser, a POST-match label -- using it directly as
(features_from_winner_perspective, 1) for every row would be degenerate.
Both perspectives of every match are added: winner-perspective labeled 1,
loser-perspective labeled 0. This is what keeps the model's implied
P(A beats B) == 1 - P(B beats A) exactly, the correct symmetry for a
no-home-advantage 1v1 rating system.

Retirement handling: matches decided by retirement or walkover-adjacent
"Awarded" results are EXCLUDED from this training set (though they still
fed the Elo/form/h2h updates in tennis_elo.py) -- a player retiring hurt
isn't a skill-based result and including it as a normal win/loss teaches
the model something false about relative ability that day.
"""
import sys
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.tennis_elo import run_all, _form_rate

FEATURE_NAMES = ["elo_diff", "surface_elo_diff", "form_diff", "h2h_diff", "is_bo5", "bo5_x_elo"]
EXCLUDED_COMMENTS = ("Retired", "Awarded")


def features_for_perspective(row, a_is_winner: bool):
    """Build the feature vector from either the winner's or the loser's
    perspective as 'player A' -- used both to build the symmetrized
    training set (both perspectives, from historical rows) and, with a
    hand-built pseudo-row, for live inference."""
    if a_is_winner:
        elo_diff = row["winner_elo_pre"] - row["loser_elo_pre"]
        surface_diff = row["winner_surface_elo_pre"] - row["loser_surface_elo_pre"]
        form_diff = row["winner_form_pre"] - row["loser_form_pre"]
        h2h_diff = row["winner_h2h_pre"] - row["loser_h2h_pre"]
    else:
        elo_diff = row["loser_elo_pre"] - row["winner_elo_pre"]
        surface_diff = row["loser_surface_elo_pre"] - row["winner_surface_elo_pre"]
        form_diff = row["loser_form_pre"] - row["winner_form_pre"]
        h2h_diff = row["loser_h2h_pre"] - row["winner_h2h_pre"]
    is_bo5 = 1.0 if row.get("best_of") == 5 else 0.0
    return [elo_diff, surface_diff, form_diff, h2h_diff, is_bo5, is_bo5 * elo_diff]


def build_training_set(feature_rows):
    X, y = [], []
    for row in feature_rows:
        if row.get("comment") in EXCLUDED_COMMENTS:
            continue
        X.append(features_for_perspective(row, True)); y.append(1)
        X.append(features_for_perspective(row, False)); y.append(0)
    return np.array(X), np.array(y)


def fit_link(feature_rows=None, tour: str = "atp"):
    if feature_rows is None:
        _, feature_rows = run_all(tour)
    X, y = build_training_set(feature_rows)
    model = LogisticRegression()
    model.fit(X, y)
    return model, len(y)


def win_probability(model, features) -> float:
    """`features` is the 6-element vector from features_for_perspective()
    or live_feature_vector() -- P(the 'A' side of that vector wins)."""
    return model.predict_proba([features])[0, 1]


def live_feature_vector(engine, player_a: str, player_b: str, surface: str = None, is_bo5: bool = False):
    """Build the same 6-feature vector for a LIVE, not-yet-played match,
    reading current state off the trained engine (engine.ratings,
    engine.surface_engines, engine.recent_results, engine.h2h -- all
    attached by tennis_elo.run_all). If the surface isn't known/resolved,
    surface_diff falls back to the overall elo_diff (a neutral choice: the
    surface feature contributes nothing extra rather than guessing wrong)."""
    elo_diff = engine.get(player_a) - engine.get(player_b)

    surface_engine = engine.surface_engines.get(surface) if surface else None
    surface_diff = (surface_engine.get(player_a) - surface_engine.get(player_b)) if surface_engine else elo_diff

    form_a = _form_rate(engine.recent_results.get(player_a, deque()))
    form_b = _form_rate(engine.recent_results.get(player_b, deque()))

    pair_key = frozenset({player_a, player_b})
    h2h_a = engine.h2h.get(pair_key, {}).get(player_a, 0)
    h2h_b = engine.h2h.get(pair_key, {}).get(player_b, 0)

    is_bo5_val = 1.0 if is_bo5 else 0.0
    return [elo_diff, surface_diff, form_a - form_b, h2h_a - h2h_b, is_bo5_val, is_bo5_val * elo_diff]


if __name__ == "__main__":
    for tour in ("atp", "wta"):
        model, n = fit_link(tour=tour)
        print(f"\n{tour.upper()}: fit on {n} symmetrized rows ({n // 2} real matches, retirements/awarded excluded).")
        print("Coefficients:")
        for name, coef in zip(FEATURE_NAMES, model.coef_[0]):
            print(f"  {name:16s} {coef:+.5f}")
        print(f"  intercept        {model.intercept_[0]:+.4f} (should be ~0)")
