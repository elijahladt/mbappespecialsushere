"""Does training the win-probability link on ALL competitive (non-friendly)
international matches -- not just the ~142 WC knockout matches used today --
actually help? User's ask, verbatim: expand the training set to every
competitive international game, excluding friendlies, to fix the model's
tiny-sample problem (this project has repeatedly found that the ~142-match
WC-knockout-only training set is the real bottleneck behind most failed
feature experiments).

"Competitive" here means literally "tournament != Friendly" -- confirmed
directly against the database that "Friendly" is the only friendly label
(no other spellings, no NULLs), so this is an unambiguous filter, not a
judgment call. This pulls in World Cup/continental-championship qualifiers,
continental championships (Euro, Copa America, AFCON, Asian Cup, etc.),
Nations Leagues, and everything else that isn't literally a friendly --
about 24,500 decisive non-friendly matches across history vs. the current
~142-match WC-knockout-only training set (roughly 170x more data).

Same walk-forward discipline as src/backtest/walk_forward.py, same held-out
test matches (2010/2014/2018/2022 WC knockout stages) -- the ONLY thing that
changes is what the model is allowed to train on. Reported honestly either
way, per the project's standing rule: don't keep an added feature/change
unless it actually wins the backtest.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import run_all
from src.models.winprob_link import (
    WC_KNOCKOUT_START, build_features, is_knockout, shootout_adjustment,
    build_training_set as build_competitive_training_set,
    build_knockout_only_training_set,
)
from src.backtest.walk_forward import TEST_YEARS
from src.backtest.metrics import brier_score, bootstrap_brier_ci, reliability_buckets


def walk_forward_comparison(feature_rows):
    years = sorted(WC_KNOCKOUT_START.keys())
    baseline_probs, expanded_probs, outcomes = [], [], []
    per_year = {}

    for year in years:
        if year not in TEST_YEARS:
            continue
        cutoff = WC_KNOCKOUT_START[year]

        train_rows = [r for r in feature_rows if r["date"] < cutoff]

        X_base, y_base = build_knockout_only_training_set(train_rows)
        X_exp, y_exp = build_competitive_training_set(train_rows)
        if len(y_base) < 10 or len(y_exp) < 10:
            continue

        baseline_model = LogisticRegression()
        baseline_model.fit(X_base, y_base)
        expanded_model = LogisticRegression()
        expanded_model.fit(X_exp, y_exp)

        test_rows = [r for r in feature_rows if is_knockout(r["date"], r["tournament"])
                     and r["date"][:4] == str(year) and r["home_score"] != r["away_score"]]
        if not test_rows:
            continue

        year_base, year_exp, year_outcomes = [], [], []
        for r in test_rows:
            feats = build_features(r)
            p_base = shootout_adjustment(baseline_model.predict_proba([feats])[0, 1])
            p_exp = shootout_adjustment(expanded_model.predict_proba([feats])[0, 1])
            outcome = 1 if r["home_score"] > r["away_score"] else 0
            year_base.append(p_base)
            year_exp.append(p_exp)
            year_outcomes.append(outcome)

        per_year[year] = {
            "n": len(year_outcomes),
            "n_train_baseline": len(y_base),
            "n_train_expanded": len(y_exp),
            "baseline_brier": brier_score(year_base, year_outcomes),
            "expanded_brier": brier_score(year_exp, year_outcomes),
        }
        baseline_probs.extend(year_base)
        expanded_probs.extend(year_exp)
        outcomes.extend(year_outcomes)

    return baseline_probs, expanded_probs, outcomes, per_year


if __name__ == "__main__":
    _, feature_rows = run_all()
    baseline_probs, expanded_probs, outcomes, per_year = walk_forward_comparison(feature_rows)

    print("Walk-forward: WC-knockout-only training (current) vs. all-competitive-matches training (expanded)\n")
    for year, stats in per_year.items():
        print(f"  {year}: n={stats['n']:2d} test matches -- "
              f"baseline (trained on {stats['n_train_baseline']:4d} WC knockout matches): Brier={stats['baseline_brier']:.4f}  "
              f"expanded (trained on {stats['n_train_expanded']:6d} competitive matches): Brier={stats['expanded_brier']:.4f}")

    if outcomes:
        base_brier = brier_score(baseline_probs, outcomes)
        exp_brier = brier_score(expanded_probs, outcomes)
        b_lo, b_hi = bootstrap_brier_ci(baseline_probs, outcomes)
        e_lo, e_hi = bootstrap_brier_ci(expanded_probs, outcomes)
        print(f"\nPooled 2010-2022 (n={len(outcomes)}):")
        print(f"  Baseline (WC knockout only):      Brier={base_brier:.4f} (95% CI [{b_lo:.4f}, {b_hi:.4f}])")
        print(f"  Expanded (all competitive matches): Brier={exp_brier:.4f} (95% CI [{e_lo:.4f}, {e_hi:.4f}])")
        verdict = "beats" if exp_brier < base_brier else "does NOT beat"
        print(f"\n  Expanded training set {verdict} the current WC-knockout-only baseline on this sample.")

        print("\nCalibration by bucket -- baseline:")
        for row in reliability_buckets(baseline_probs, outcomes):
            print(f"  {row['bucket']:12s} n={row['n']:2d}  predicted={row['mean_predicted']:.3f}  realized={row['realized_rate']:.3f}")
        print("\nCalibration by bucket -- expanded:")
        for row in reliability_buckets(expanded_probs, outcomes):
            print(f"  {row['bucket']:12s} n={row['n']:2d}  predicted={row['mean_predicted']:.3f}  realized={row['realized_rate']:.3f}")
    else:
        print("\nNo held-out predictions were generated -- insufficient training history.")
