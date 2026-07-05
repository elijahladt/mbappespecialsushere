import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backtest.walk_forward import walk_forward
from src.features.elo import run_all
from src.backtest.metrics import brier_score, bootstrap_brier_ci, reliability_buckets

if __name__ == "__main__":
    _, feature_rows = run_all()
    probs, outcomes, per_year = walk_forward(feature_rows)

    for year, stats in per_year.items():
        print(f"{year}: n={stats['n']}, trained on {stats['n_train']} prior matches, Brier={stats['brier']:.4f}")
    if probs:
        lo, hi = bootstrap_brier_ci(probs, outcomes)
        print(f"\nPooled: n={len(probs)}, Brier={brier_score(probs, outcomes):.4f} (95% CI [{lo:.4f}, {hi:.4f}]) vs naive 0.25")
        for row in reliability_buckets(probs, outcomes):
            print(row)
