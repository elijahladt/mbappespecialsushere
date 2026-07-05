"""Leave-one-tournament-out validation: train on 2018 -> test 2022, and vice
versa (the only split possible with 2 StatsBomb-covered tournaments). Compares
the XGBoost (Elo+StatsBomb) model against an Elo-only baseline fit on the
IDENTICAL rows/split, so any difference is attributable to the added features,
not to a different training set. Reports honestly either way.
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.xg_stats import FEATURE_COLUMNS, build_xg_training_table
from src.models.xgb_model import XGB_PARAMS
from src.backtest.metrics import brier_score, bootstrap_brier_ci

YEARS = [2018, 2022]


def run():
    df = build_xg_training_table()
    elo_probs, elo_outcomes = [], []
    xgb_probs, xgb_outcomes = [], []

    for year in YEARS:
        train = df[df["year"] != year]
        test = df[df["year"] == year]
        if len(train) < 10 or len(test) < 1:
            continue

        elo_model = LogisticRegression()
        elo_model.fit(train[["elo_diff"]].values, train["home_win"].values)

        xgb_model = xgb.XGBClassifier(**{k: v for k, v in XGB_PARAMS.items()})
        xgb_model.fit(train[FEATURE_COLUMNS].values, train["home_win"].values)

        elo_pred = elo_model.predict_proba(test[["elo_diff"]].values)[:, 1]
        xgb_pred = xgb_model.predict_proba(test[FEATURE_COLUMNS].values)[:, 1]
        outcomes = test["home_win"].values

        elo_probs.extend(elo_pred); elo_outcomes.extend(outcomes)
        xgb_probs.extend(xgb_pred); xgb_outcomes.extend(outcomes)

        print(f"{year}: train={len(train)}, test={len(test)}, "
              f"Elo Brier={brier_score(elo_pred, outcomes):.4f}, "
              f"XGBoost Brier={brier_score(xgb_pred, outcomes):.4f}")

    if elo_probs:
        elo_brier = brier_score(elo_probs, elo_outcomes)
        xgb_brier = brier_score(xgb_probs, xgb_outcomes)
        elo_lo, elo_hi = bootstrap_brier_ci(elo_probs, elo_outcomes)
        xgb_lo, xgb_hi = bootstrap_brier_ci(xgb_probs, xgb_outcomes)
        print(f"\nPooled (n={len(elo_probs)}):")
        print(f"  Elo-only baseline:        Brier={elo_brier:.4f} (95% CI [{elo_lo:.4f}, {elo_hi:.4f}])")
        print(f"  XGBoost (Elo+StatsBomb):  Brier={xgb_brier:.4f} (95% CI [{xgb_lo:.4f}, {xgb_hi:.4f}])")
        verdict = "beats" if xgb_brier < elo_brier else "does NOT beat"
        print(f"\n  Verdict: XGBoost {verdict} the Elo-only baseline on this held-out sample.")
        return {"elo_brier": elo_brier, "xgb_brier": xgb_brier, "n": len(elo_probs)}
    return None


if __name__ == "__main__":
    run()
