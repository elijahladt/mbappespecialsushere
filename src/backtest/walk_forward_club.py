"""Walk-forward validation of the club 3-way win-probability link: for each
season, fit on ONLY matches from strictly earlier seasons, then score
predictions on that season -- no season's outcome ever informs its own
prediction. Same discipline as src/backtest/walk_forward.py, adapted for
club football's per-season structure and 3-way (not binary) outcome.

Gate before shipping a league: this season-average Brier score must beat a
naive baseline that predicts each season's OWN empirical home/draw/away base
rates (also fit walk-forward, from prior seasons only) -- a much stronger
baseline than the WC's near-50/50 knockout framing, since club home/draw/away
base rates are genuinely informative on their own.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.club_elo import run_all
from src.models.club_winprob_link import build_training_set, build_features, outcome_label
from src.backtest.metrics import multiclass_brier_score, bootstrap_brier_ci


def _season_order(feature_rows):
    """Seasons ordered by their earliest match date -- not by the season
    code string, since football-data.co.uk codes like '9900' (1999/00) sort
    incorrectly against '0001' (2000/01) under plain string comparison."""
    first_date = {}
    for row in feature_rows:
        season = row.get("season")
        if season not in first_date or row["date"] < first_date[season]:
            first_date[season] = row["date"]
    return [s for s, _ in sorted(first_date.items(), key=lambda kv: kv[1])]


def walk_forward(feature_rows, min_train: int = 380):
    per_season = {}
    all_probs, all_outcomes = [], []
    all_naive_probs = []

    seasons = _season_order(feature_rows)
    for i, season in enumerate(seasons):
        train_rows = [r for r in feature_rows if r["season"] in seasons[:i]]
        test_rows = [r for r in feature_rows if r["season"] == season]
        if len(train_rows) < min_train or not test_rows:
            continue

        X_train, y_train = build_training_set(train_rows)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        classes = list(model.classes_)

        # Naive baseline: prior seasons' empirical home/draw/away base rates.
        naive = np.array([np.mean(y_train == c) for c in (0, 1, 2)])

        season_probs, season_outcomes, season_naive = [], [], []
        for r in test_rows:
            raw = model.predict_proba([build_features(r)])[0]
            p = tuple(raw[classes.index(c)] for c in (0, 1, 2))
            label = outcome_label(r)
            season_probs.append(p)
            season_outcomes.append(label)
            season_naive.append(tuple(naive))

        per_season[season] = {
            "n": len(season_outcomes),
            "n_train": len(y_train),
            "brier": multiclass_brier_score(season_probs, season_outcomes),
            "naive_brier": multiclass_brier_score(season_naive, season_outcomes),
        }
        all_probs.extend(season_probs)
        all_outcomes.extend(season_outcomes)
        all_naive_probs.extend(season_naive)

    return all_probs, all_outcomes, all_naive_probs, per_season


if __name__ == "__main__":
    _, feature_rows = run_all("premier_league")
    probs, outcomes, naive_probs, per_season = walk_forward(feature_rows)

    print("Walk-forward results (train = only seasons strictly before the test season):\n")
    for season, stats in per_season.items():
        print(f"  {season}: n={stats['n']:3d} matches (trained on {stats['n_train']:5d} prior matches), "
              f"Brier={stats['brier']:.4f}  naive_baseline_Brier={stats['naive_brier']:.4f}")

    if probs:
        pooled_brier = multiclass_brier_score(probs, outcomes)
        pooled_naive = multiclass_brier_score(naive_probs, outcomes)
        home_probs = [p[2] for p in probs]
        home_outcomes = [1 if o == 2 else 0 for o in outcomes]
        lo, hi = bootstrap_brier_ci(home_probs, home_outcomes)
        print(f"\nPooled: n={len(probs)}, model Brier={pooled_brier:.4f}, naive-baseline Brier={pooled_naive:.4f}")
        print(f"(bootstrap CI below is for the home-win margin only, as a spot check: [{lo:.4f}, {hi:.4f}])")
        if pooled_brier < pooled_naive:
            print("Model beats the naive empirical base-rate baseline.")
        else:
            print("Model does NOT beat the naive baseline -- do not ship this league board without investigating.")
    else:
        print("\nNo held-out predictions were generated -- insufficient training history.")
