"""Elo-diff + recent-form -> 3-way (home/draw/away) win-probability model
for club football. Unlike the World Cup knockout link (src/models/
winprob_link.py, a binary "who wins the tie" classifier fit only on
decisive matches), club league matches genuinely end in a draw ~20-25% of
the time and that outcome must be modeled directly -- this is a
multinomial classifier, not binary.

Started as a single elo_diff feature ("prove the simple thing calibrates
first"); extended after a walk-forward P&L backtest showed that single
feature falls well short of the market's own calibration (see
src/backtest/pnl_backtest_club.py) -- home_form_diff (rolling
win/draw/loss rate over the last FORM_WINDOW matches, from
src/features/club_elo.py) was added to catch momentum/injury swings that
a slow-moving Elo rating doesn't react to as quickly.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.club_elo import HOME_ADVANTAGE, run_all, _form_rate

FEATURE_NAMES = ["elo_diff", "form_diff"]


def effective_elo_diff(row) -> float:
    return (row["home_elo_pre"] + HOME_ADVANTAGE) - row["away_elo_pre"]


def build_features(row):
    return [effective_elo_diff(row), row["home_form_pre"] - row["away_form_pre"]]


def outcome_label(row) -> int:
    """0 = away win, 1 = draw, 2 = home win -- ordered so class index lines
    up with sklearn's sorted-classes convention for predict_proba columns."""
    if row["home_score"] > row["away_score"]:
        return 2
    if row["home_score"] == row["away_score"]:
        return 1
    return 0


def build_training_set(feature_rows):
    X = [build_features(row) for row in feature_rows]
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


def win_draw_away_probability(model, features):
    """`features` is the [elo_diff, form_diff] vector from build_features()
    or live_feature_vector(). Returns (p_away, p_draw, p_home) -- summing
    to 1 by construction, ordered to match outcome_label's 0/1/2 convention."""
    probs = model.predict_proba([features])[0]
    classes = list(model.classes_)
    return tuple(probs[classes.index(c)] for c in (0, 1, 2))


def live_feature_vector(engine, home_team: str, away_team: str, host_bonus_by_country: dict = None,
                         home_country: str = None):
    """Build the same [elo_diff, form_diff] vector for a LIVE, not-yet-played
    match, reading current state off the trained engine (engine.ratings,
    engine.recent_results -- both attached by club_elo.run_all)."""
    adv = HOME_ADVANTAGE
    if host_bonus_by_country and home_country:
        adv += host_bonus_by_country.get(home_country, 0.0)
    elo_diff = (engine.get(home_team) + adv) - engine.get(away_team)
    home_form = _form_rate(engine.recent_results.get(home_team, []))
    away_form = _form_rate(engine.recent_results.get(away_team, []))
    return [elo_diff, home_form - away_form]


if __name__ == "__main__":
    model, n = fit_link()
    print(f"Fit on {n} Premier League matches.")
    print(f"Classes: {model.classes_}")
    print("Coefficients (per class, ordered elo_diff/form_diff):")
    for cls, coefs in zip(model.classes_, model.coef_):
        print(f"  class {cls}: {dict(zip(FEATURE_NAMES, coefs))}")
    for diff in (-300, -150, -50, 0, 50, 150, 300):
        p_away, p_draw, p_home = win_draw_away_probability(model, [diff, 0.0])
        print(f"  eff_elo_diff={diff:+5d}, form_diff=0 -> P(home)={p_home:.3f} P(draw)={p_draw:.3f} P(away)={p_away:.3f}  sum={p_home+p_draw+p_away:.3f}")
