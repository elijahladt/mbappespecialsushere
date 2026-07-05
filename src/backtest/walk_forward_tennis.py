"""Walk-forward validation of the tennis win-probability link: for each
year, fit on ONLY matches from strictly earlier years, then score
predictions on that year -- no year's outcome ever informs its own
prediction. Same discipline as the club/WC walk-forward backtests.

Evaluation uses the same symmetrization as build_training_set (every held-out
match scored from both perspectives) so the naive baseline is exactly the
WC's familiar 50/50 -- symmetrized outcomes are 50/50 by construction, so a
constant-0.5 prediction always scores Brier=0.25, an honest apples-to-apples
comparison point.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.tennis_elo import run_all
from src.models.tennis_winprob_link import build_training_set, features_for_perspective, EXCLUDED_COMMENTS
from src.backtest.metrics import brier_score, bootstrap_brier_ci, reliability_buckets

NAIVE_BASELINE_BRIER = 0.25  # constant p=0.5 on a symmetrized 50/50 label set


def walk_forward(feature_rows, min_train: int = 500):
    years = sorted({int(r["date"][:4]) for r in feature_rows})
    per_year = {}
    all_probs, all_outcomes = [], []

    for year in years:
        cutoff = f"{year}-01-01"
        train_rows = [r for r in feature_rows if r["date"] < cutoff]
        test_rows = [r for r in feature_rows if r["date"][:4] == str(year) and r.get("comment") not in EXCLUDED_COMMENTS]
        if len(train_rows) < min_train or not test_rows:
            continue

        X_train, y_train = build_training_set(train_rows)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        year_probs, year_outcomes = [], []
        for r in test_rows:
            year_probs.append(model.predict_proba([features_for_perspective(r, True)])[0, 1]); year_outcomes.append(1)
            year_probs.append(model.predict_proba([features_for_perspective(r, False)])[0, 1]); year_outcomes.append(0)

        per_year[year] = {
            "n_matches": len(test_rows),
            "n_train_matches": len(train_rows),
            "brier": brier_score(year_probs, year_outcomes),
        }
        all_probs.extend(year_probs)
        all_outcomes.extend(year_outcomes)

    return all_probs, all_outcomes, per_year


if __name__ == "__main__":
    for tour in ("atp", "wta"):
        _, feature_rows = run_all(tour)
        probs, outcomes, per_year = walk_forward(feature_rows)

        print(f"\n{tour.upper()} walk-forward (train = only years strictly before the test year):")
        for year, stats in per_year.items():
            print(f"  {year}: {stats['n_matches']:4d} matches (trained on {stats['n_train_matches']:6d} prior matches), Brier={stats['brier']:.4f}")

        if probs:
            pooled_brier = brier_score(probs, outcomes)
            lo, hi = bootstrap_brier_ci(probs, outcomes)
            print(f"Pooled: n={len(probs)}, Brier={pooled_brier:.4f} (95% CI [{lo:.4f}, {hi:.4f}]), naive 50/50 baseline Brier={NAIVE_BASELINE_BRIER:.4f}")
            print("Model beats the naive baseline." if pooled_brier < NAIVE_BASELINE_BRIER else "Model does NOT beat the naive baseline.")
            print("Calibration by bucket:")
            for row in reliability_buckets(probs, outcomes):
                print(f"  {row['bucket']:12s} n={row['n']:5d}  predicted={row['mean_predicted']:.3f}  realized={row['realized_rate']:.3f}")
