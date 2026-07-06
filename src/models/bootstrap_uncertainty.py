"""Bootstrap uncertainty for the WC win-probability model: instead of a
single point-estimate probability, refit the model many times on resampled
historical training data and report the spread. This exists because a
single number (e.g. "73%") hides how much that number would have wobbled if
the historical dataset had come out even slightly differently.

n_boot defaults to 2,000, not 10,000: the training set now used by
build_training_set() is ~24,500 competitive (non-friendly) matches, not the
old ~142 WC-knockout-only set, so each refit is far more expensive (a
LogisticRegression fit on ~24k rows, timed at ~35ms/refit vs. a few ms at
the old size) -- 10,000 refits would take ~6.5 minutes instead of the
~60-90 seconds this feature originally targeted. 2,000 replicates is still
a standard, well-justified number for a bootstrap CI (Efron's own guidance
is 1,000-2,000 for standard-error/CI estimates; 10,000 was already more
precision than needed, just cheap enough to afford at the old sample size).

Not the same question as "simulate the match 10,000 times" (which would
just resample outcomes at a FIXED probability and converge back to that
same number by the law of large numbers, adding no information) -- this
resamples the TRAINING DATA, which is what actually captures model
uncertainty given a finite sample.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.winprob_link import build_training_set, shootout_adjustment, fit_link


def bootstrap_win_probability(feature_rows, query_features, n_boot: int = 2000, seed: int = 42):
    """Returns (point_estimate, samples). point_estimate is fit on the full,
    unresampled training set -- the same model used everywhere else in the
    app -- so this function only adds an uncertainty band AROUND the real
    prediction, it doesn't replace it with something else."""
    X, y = build_training_set(feature_rows)
    n = len(y)

    full_model = LogisticRegression()
    full_model.fit(X, y)
    point_estimate = shootout_adjustment(full_model.predict_proba([query_features])[0, 1])

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_boot, y_boot = X[idx], y[idx]
        if len(np.unique(y_boot)) < 2:
            # Degenerate resample (all one class) -- essentially impossible
            # at n~24,500 but kept as a guard; reuse the previous valid
            # sample rather than crash
            # or silently bias the distribution with a fabricated 0/1.
            samples[i] = samples[i - 1] if i > 0 else point_estimate
            continue
        model = LogisticRegression()
        model.fit(X_boot, y_boot)
        samples[i] = shootout_adjustment(model.predict_proba([query_features])[0, 1])

    return point_estimate, samples


def summarize_bootstrap(point_estimate: float, samples, ci: float = 0.90):
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = 100 - lo_pct
    lo, hi = np.percentile(samples, [lo_pct, hi_pct])
    return {
        "point_estimate": point_estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "std": float(np.std(samples)),
        "n_boot": len(samples),
    }


if __name__ == "__main__":
    import time
    from src.features.elo import run_all

    _, feature_rows = run_all()

    for label, query in [
        ("Even matchup (elo_diff=0, h2h_diff=0)", [0.0, 0.0]),
        ("Moderate favorite (elo_diff=+150, h2h_diff=+1)", [150.0, 1.0]),
        ("Heavy favorite (elo_diff=+400, h2h_diff=+2)", [400.0, 2.0]),
    ]:
        t0 = time.time()
        point, samples = bootstrap_win_probability(feature_rows, query, n_boot=2000)
        stats = summarize_bootstrap(point, samples)
        elapsed = time.time() - t0
        print(f"{label}:")
        print(f"  Point estimate: {point:.3f}")
        print(f"  90% CI ({stats['n_boot']} bootstrap refits): [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}] (std={stats['std']:.3f})")
        print(f"  ({elapsed:.1f}s for {stats['n_boot']} refits)\n")
