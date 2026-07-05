"""XGBoost classifier on Elo + StatsBomb shot-quality features. Given the
tiny sample (98 decisive 2018+2022 matches, see src/features/xg_stats.py),
this is heavily regularized: shallow trees, high min_child_weight/gamma, few
rounds, early stopping on a validation split. Even so, treat it as a research
comparison, not a trusted standalone signal, until src/backtest/validate_xgb_model.py
shows it actually beats Elo out-of-sample.
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.xg_stats import FEATURE_COLUMNS, build_xg_training_table

XGB_PARAMS = dict(
    max_depth=2,
    min_child_weight=5,
    gamma=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    learning_rate=0.05,
    n_estimators=200,
    eval_metric="logloss",
)


def fit_xgb(df, feature_columns=FEATURE_COLUMNS, val_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = max(1, int(len(df) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X = df[feature_columns].values
    y = df["home_win"].values

    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=20)
    model.fit(
        X[train_idx], y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False,
    )
    return model


if __name__ == "__main__":
    df = build_xg_training_table()
    model = fit_xgb(df)
    print(f"Trained on {len(df)} rows, best iteration: {model.best_iteration}")
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {feat:18s} {imp:.3f}")
