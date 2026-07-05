"""Walk-forward validation of the Elo-diff win-probability link across past
World Cups: for each tournament year, fit the link using ONLY knockout matches
from earlier tournaments, then score its predictions on that year's matches.
No tournament's outcome ever informs its own prediction.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import run_all
from src.models.winprob_link import (
    WC_KNOCKOUT_START, build_training_set, effective_elo_diff, is_knockout,
    shootout_adjustment,
)
from src.backtest.metrics import brier_score, bootstrap_brier_ci, reliability_buckets

TEST_YEARS = [2010, 2014, 2018, 2022]  # per plan; earlier tournaments (1986-2006) only ever serve as training history


def walk_forward(feature_rows):
    years = sorted(WC_KNOCKOUT_START.keys())
    all_probs, all_outcomes, per_year = [], [], {}

    for year in years:
        if year not in TEST_YEARS:
            continue
        cutoff = WC_KNOCKOUT_START[year]

        train_rows = [r for r in feature_rows if r["date"] < cutoff]
        X_train, y_train = build_training_set(train_rows)
        if len(y_train) < 10:
            continue  # not enough prior knockout history to fit a meaningful link yet

        model = LogisticRegression()
        model.fit(X_train, y_train)

        test_rows = [r for r in feature_rows if is_knockout(r["date"], r["tournament"])
                     and r["date"][:4] == str(year) and r["home_score"] != r["away_score"]]
        if not test_rows:
            continue

        year_probs, year_outcomes = [], []
        for r in test_rows:
            diff = effective_elo_diff(r)
            p = shootout_adjustment(model.predict_proba([[diff]])[0, 1])
            outcome = 1 if r["home_score"] > r["away_score"] else 0
            year_probs.append(p)
            year_outcomes.append(outcome)

        per_year[year] = {
            "n": len(year_outcomes),
            "n_train": len(y_train),
            "brier": brier_score(year_probs, year_outcomes),
        }
        all_probs.extend(year_probs)
        all_outcomes.extend(year_outcomes)

    return all_probs, all_outcomes, per_year


if __name__ == "__main__":
    _, feature_rows = run_all()
    probs, outcomes, per_year = walk_forward(feature_rows)

    print("Walk-forward results (train = only knockout matches strictly before the test tournament):\n")
    for year, stats in per_year.items():
        print(f"  {year}: n={stats['n']:2d} test matches (trained on {stats['n_train']} prior knockout matches), Brier={stats['brier']:.4f}")

    if probs:
        pooled_brier = brier_score(probs, outcomes)
        lo, hi = bootstrap_brier_ci(probs, outcomes)
        print(f"\nPooled 2010-2022: n={len(probs)}, Brier={pooled_brier:.4f} (95% CI [{lo:.4f}, {hi:.4f}])")
        print(f"Naive 50/50 baseline Brier = 0.2500 for comparison.")

        print("\nCalibration by bucket (mean predicted vs realized win rate):")
        for row in reliability_buckets(probs, outcomes):
            print(f"  {row['bucket']:12s} n={row['n']:2d}  predicted={row['mean_predicted']:.3f}  realized={row['realized_rate']:.3f}")
    else:
        print("\nNo held-out predictions were generated -- insufficient training history.")
