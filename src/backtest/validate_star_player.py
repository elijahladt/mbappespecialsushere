"""Leave-one-tournament-out validation of the star-player (leading
individual attacker's cumulative xG) feature -- same methodology as
validate_xgb_model.py: train on 2018 predict 2022, and vice versa, compare
against an Elo-only baseline fit on the IDENTICAL split so any difference
is attributable to the added feature, not a different training set.

Motivation: does a team having ONE standout individual attacker (like
Norway's Haaland) carry real signal beyond team-level Elo? Elo has no
concept of individual player quality at all -- this tests whether that's
actually a gap worth closing, or just a plausible-sounding story.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.xg_stats import build_xg_training_table
from src.features.star_player_tracker import build_star_player_table
from src.backtest.metrics import brier_score, bootstrap_brier_ci

YEARS = [2018, 2022]


def build_combined_table():
    elo_df = build_xg_training_table()
    star_df = build_star_player_table().set_index(["match_id", "team"])["leading_scorer_xg_pre"]

    rows = []
    for _, r in elo_df.iterrows():
        try:
            home_star = star_df.loc[(r["match_id"], r["home_team"])]
            away_star = star_df.loc[(r["match_id"], r["away_team"])]
        except KeyError:
            continue  # team-name join miss -- skip rather than guess
        rows.append({**r.to_dict(), "star_diff": home_star - away_star})

    import pandas as pd
    return pd.DataFrame(rows)


def run():
    df = build_combined_table()
    elo_probs, star_probs, outcomes_all = [], [], []

    for year in YEARS:
        train = df[df["year"] != year]
        test = df[df["year"] == year]
        if len(train) < 10 or len(test) < 1:
            continue

        elo_model = LogisticRegression()
        elo_model.fit(train[["elo_diff"]].values, train["home_win"].values)

        star_model = LogisticRegression()
        star_model.fit(train[["elo_diff", "star_diff"]].values, train["home_win"].values)

        elo_pred = elo_model.predict_proba(test[["elo_diff"]].values)[:, 1]
        star_pred = star_model.predict_proba(test[["elo_diff", "star_diff"]].values)[:, 1]
        outcomes = test["home_win"].values

        elo_probs.extend(elo_pred); star_probs.extend(star_pred); outcomes_all.extend(outcomes)
        print(f"{year}: train={len(train)}, test={len(test)}, "
              f"Elo-only Brier={brier_score(elo_pred, outcomes):.4f}, "
              f"Elo+star Brier={brier_score(star_pred, outcomes):.4f}")

    if elo_probs:
        elo_brier = brier_score(elo_probs, outcomes_all)
        star_brier = brier_score(star_probs, outcomes_all)
        elo_lo, elo_hi = bootstrap_brier_ci(elo_probs, outcomes_all)
        star_lo, star_hi = bootstrap_brier_ci(star_probs, outcomes_all)
        print(f"\nPooled (n={len(elo_probs)}):")
        print(f"  Elo-only:   Brier={elo_brier:.4f} (95% CI [{elo_lo:.4f}, {elo_hi:.4f}])")
        print(f"  Elo+star:   Brier={star_brier:.4f} (95% CI [{star_lo:.4f}, {star_hi:.4f}])")
        verdict = "beats" if star_brier < elo_brier else "does NOT beat"
        print(f"\n  Verdict: adding the star-player feature {verdict} the Elo-only baseline on this held-out sample.")
        return {"elo_brier": elo_brier, "star_brier": star_brier, "n": len(elo_probs)}
    return None


if __name__ == "__main__":
    run()
