"""Honest check: does adding altitude-change and timezone-crossing ('jet
lag') to the win-probability link improve out-of-sample Brier score, or is
it just added complexity? Same discipline as validate_extended_features.py
(rest/travel) -- only trust it if it actually helps on held-out matches.

Venue data (and so these features) exist from 2010 onward, same as
rest/travel, so walk-forward tested on 2014/2018/2022 (each needs at least
one prior tournament of feature-complete training history).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import run_all
from src.features.altitude_timezone import compute_altitude_timezone
from src.models.winprob_link import (
    WC_KNOCKOUT_START, effective_elo_diff, is_knockout, shootout_adjustment,
)
from src.backtest.metrics import brier_score

EXTENDED_TEST_YEARS = [2014, 2018, 2022]


def merge_features(feature_rows, alt_tz):
    merged = []
    for r in feature_rows:
        f = alt_tz.get((r["date"], r["home_team"], r["away_team"]))
        if f is None:
            continue
        merged.append({**r, **f})
    return merged


def build_extended_xy(rows):
    X, y = [], []
    for r in rows:
        if not is_knockout(r["date"], r["tournament"]):
            continue
        if r["home_score"] == r["away_score"]:
            continue
        if r["home_altitude_delta"] is None or r["away_altitude_delta"] is None:
            continue
        if r["home_tz_delta_hours"] is None or r["away_tz_delta_hours"] is None:
            continue
        altitude_diff = r["home_altitude_delta"] - r["away_altitude_delta"]
        tz_diff = r["home_tz_delta_hours"] - r["away_tz_delta_hours"]
        X.append([effective_elo_diff(r), altitude_diff, tz_diff])
        y.append(1 if r["home_score"] > r["away_score"] else 0)
    return np.array(X), np.array(y)


def run():
    _, feature_rows = run_all()
    _, alt_tz = compute_altitude_timezone()
    merged = merge_features(feature_rows, alt_tz)

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

        base_model = LogisticRegression()
        base_model.fit(X_train[:, [0]], y_train)

        test_rows = [r for r in merged if is_knockout(r["date"], r["tournament"])
                     and r["date"][:4] == str(year) and r["home_score"] != r["away_score"]
                     and r["home_altitude_delta"] is not None and r["away_altitude_delta"] is not None
                     and r["home_tz_delta_hours"] is not None and r["away_tz_delta_hours"] is not None]

        for r in test_rows:
            altitude_diff = r["home_altitude_delta"] - r["away_altitude_delta"]
            tz_diff = r["home_tz_delta_hours"] - r["away_tz_delta_hours"]
            diff = effective_elo_diff(r)
            outcome = 1 if r["home_score"] > r["away_score"] else 0

            p_ext = shootout_adjustment(ext_model.predict_proba([[diff, altitude_diff, tz_diff]])[0, 1])
            p_base = shootout_adjustment(base_model.predict_proba([[diff]])[0, 1])

            ext_probs.append(p_ext); ext_outcomes.append(outcome)
            base_probs.append(p_base); base_outcomes.append(outcome)

        print(f"{year}: trained on {len(y_train)} rows, tested on {len(test_rows)} matches")

    if ext_probs:
        base_brier = brier_score(base_probs, base_outcomes)
        ext_brier = brier_score(ext_probs, ext_outcomes)
        print(f"\nBase (Elo-only)                Brier = {base_brier:.4f}")
        print(f"Extended (+altitude/timezone)  Brier = {ext_brier:.4f}")
        print(f"n = {len(ext_probs)} held-out matches ({'+'.join(str(y) for y in EXTENDED_TEST_YEARS)})")
        verdict = "beats" if ext_brier < base_brier else "does not beat"
        print(f"Verdict: altitude/timezone features {verdict} the Elo-only baseline on this sample.")
    else:
        print("No feature-complete held-out matches available for comparison.")


if __name__ == "__main__":
    run()
