"""Honest check: does adding rest-days/travel-change to the win-probability
link actually improve out-of-sample Brier score, or is it just added
complexity on a tiny dataset? Only fit where genuinely helpful.

Rest/travel data only exists from 2010 onward (venue enrichment), so this can
only be walk-forward tested on 2018 and 2022 (each needs >=2 prior tournaments
of feature-complete training data). Compares directly against the base
Elo-only model on the SAME held-out matches.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import run_all
from src.features.rest_travel import compute_rest_travel
from src.models.winprob_link import (
    WC_KNOCKOUT_START, effective_elo_diff, is_knockout, shootout_adjustment,
)
from src.backtest.metrics import brier_score

EXTENDED_TEST_YEARS = [2018, 2022]


def merge_features(feature_rows, rest_travel):
    merged = []
    for r in feature_rows:
        rt = rest_travel.get((r["date"], r["home_team"], r["away_team"]))
        if rt is None:
            continue
        merged.append({**r, **rt})
    return merged


def build_extended_xy(rows):
    X, y = [], []
    for r in rows:
        if not is_knockout(r["date"], r["tournament"]):
            continue
        if r["home_score"] == r["away_score"]:
            continue
        if r["home_rest_days"] is None or r["away_rest_days"] is None:
            continue
        if r["home_traveled"] is None or r["away_traveled"] is None:
            continue
        rest_diff = r["home_rest_days"] - r["away_rest_days"]
        travel_diff = r["away_traveled"] - r["home_traveled"]  # positive = away side traveled, home side didn't
        X.append([effective_elo_diff(r), rest_diff, travel_diff])
        y.append(1 if r["home_score"] > r["away_score"] else 0)
    return np.array(X), np.array(y)


def run():
    _, feature_rows = run_all()
    _, rest_travel = compute_rest_travel()
    merged = merge_features(feature_rows, rest_travel)

    base_probs, base_outcomes = [], []
    ext_probs, ext_outcomes = [], []

    for year in EXTENDED_TEST_YEARS:
        cutoff = WC_KNOCKOUT_START[year]
        train_rows = [r for r in merged if r["date"] < cutoff]
        X_train, y_train = build_extended_xy(train_rows)
        if len(y_train) < 10:
            print(f"{year}: skipped, only {len(y_train)} feature-complete training rows")
            continue

        ext_model = LogisticRegression()
        ext_model.fit(X_train, y_train)

        base_X_train = X_train[:, [0]]
        base_model = LogisticRegression()
        base_model.fit(base_X_train, y_train)

        test_rows = [r for r in merged if is_knockout(r["date"], r["tournament"])
                     and r["date"][:4] == str(year) and r["home_score"] != r["away_score"]
                     and r["home_rest_days"] is not None and r["away_rest_days"] is not None
                     and r["home_traveled"] is not None and r["away_traveled"] is not None]

        for r in test_rows:
            rest_diff = r["home_rest_days"] - r["away_rest_days"]
            travel_diff = r["away_traveled"] - r["home_traveled"]
            diff = effective_elo_diff(r)
            outcome = 1 if r["home_score"] > r["away_score"] else 0

            p_ext = shootout_adjustment(ext_model.predict_proba([[diff, rest_diff, travel_diff]])[0, 1])
            p_base = shootout_adjustment(base_model.predict_proba([[diff]])[0, 1])

            ext_probs.append(p_ext); ext_outcomes.append(outcome)
            base_probs.append(p_base); base_outcomes.append(outcome)

        print(f"{year}: trained on {len(y_train)} rows, tested on {len(test_rows)} matches")

    if ext_probs:
        print(f"\nBase (Elo-only)         Brier = {brier_score(base_probs, base_outcomes):.4f}")
        print(f"Extended (+rest/travel) Brier = {brier_score(ext_probs, ext_outcomes):.4f}")
        print(f"n = {len(ext_probs)} held-out matches (2018+2022 only -- all the data with rest/travel history available)")
    else:
        print("No feature-complete held-out matches available for comparison.")


if __name__ == "__main__":
    run()
