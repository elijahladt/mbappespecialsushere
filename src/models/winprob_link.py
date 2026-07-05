"""Elo-diff -> match-winner probability, fit on historical World Cup KNOCKOUT
matches specifically (not group stage), since Kalshi's advance/winner markets
settle to a definite winner after regulation/extra-time/penalties -- a
different target than a 3-way 1X2 group-stage market.

Known data limitation: the historical results dataset records only the final
score, not whether a draw was decided by a penalty shootout, so drawn knockout
matches (shootout-decided) are excluded from the training set below -- we
simply don't know who advanced in those rows. See the shootout_adjustment()
heuristic for how that gap is handled at prediction time instead.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import HOME_ADVANTAGE, run_all

# Round-of-32/Round-of-16 (or equivalent first true single-elimination round)
# start date for each modern-format World Cup. Matches on/after this date within
# that tournament are knockout matches; a hardcoded lookup because the historical
# results dataset carries no per-match round/stage label of its own.
WC_KNOCKOUT_START = {
    1986: "1986-06-15", 1990: "1990-06-23", 1994: "1994-07-02",
    1998: "1998-06-27", 2002: "2002-06-15", 2006: "2006-06-24",
    2010: "2010-06-26", 2014: "2014-06-28", 2018: "2018-06-30",
    2022: "2022-12-03", 2026: "2026-06-28",
}


def is_knockout(date: str, tournament: str) -> bool:
    if tournament != "FIFA World Cup":
        return False
    year = int(date[:4])
    start = WC_KNOCKOUT_START.get(year)
    return start is not None and date >= start


def effective_elo_diff(row) -> float:
    adv = 0.0 if row["neutral"] else HOME_ADVANTAGE
    return (row["home_elo_pre"] + adv) - row["away_elo_pre"]


def build_training_set(feature_rows):
    X, y = [], []
    for row in feature_rows:
        if not is_knockout(row["date"], row["tournament"]):
            continue
        if row["home_score"] == row["away_score"]:
            continue  # shootout-decided; excluded, see module docstring
        X.append([effective_elo_diff(row)])
        y.append(1 if row["home_score"] > row["away_score"] else 0)
    return np.array(X), np.array(y)


def fit_link(feature_rows=None):
    if feature_rows is None:
        _, feature_rows = run_all()
    X, y = build_training_set(feature_rows)
    model = LogisticRegression()
    model.fit(X, y)
    return model, len(y)


def win_probability(model, elo_diff_home_minus_away: float) -> float:
    """Probability the 'home' side (first team, elo_diff = home - away
    including any home-advantage/neutral adjustment already applied by the
    caller) wins the tie outright."""
    p = model.predict_proba([[elo_diff_home_minus_away]])[0, 1]
    return shootout_adjustment(p)


def shootout_adjustment(p: float, band: float = 0.10, shrink: float = 0.15) -> float:
    """Toss-up knockout matches are disproportionately likely to be decided by
    penalties, which are close to a coin flip regardless of team strength.
    Nudge probabilities inside the toss-up band slightly toward 0.5. This is a
    documented placeholder heuristic -- Milestone B should replace it with an
    empirical shootout-frequency/outcome model fit on round-labeled data."""
    if abs(p - 0.5) < band:
        return 0.5 + (p - 0.5) * (1 - shrink)
    return p


if __name__ == "__main__":
    model, n = fit_link()
    print(f"Fit on {n} historical WC knockout matches with a decisive result.")
    print(f"Coefficient (per Elo point): {model.coef_[0][0]:.5f}, intercept: {model.intercept_[0]:.4f}")
    for diff in (-300, -150, -50, 0, 50, 150, 300):
        print(f"  eff_elo_diff={diff:+5d} -> P(home wins tie) = {win_probability(model, diff):.3f}")
