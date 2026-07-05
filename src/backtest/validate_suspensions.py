"""Does knowing a team's top recent xG-contributor is suspended (2 yellows /
red card, per src/features/suspensions.py) improve the win-probability model?
Tested the same honest way as rest/travel and StatsBomb features: added as an
extra feature to the leave-one-tournament-out backtest, only trusted if it
actually lowers the Brier score on held-out matches.
"""
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.xg_stats import build_xg_training_table, FEATURE_COLUMNS
from src.features.suspensions import suspended_players
from src.ingest.statsbomb_data import load_player_match_stats
from src.models.xgb_model import XGB_PARAMS
from src.backtest.metrics import brier_score

YEARS = [2018, 2022]


def team_top_scorer_as_of(player_stats, team, year, before_date):
    prior = player_stats[(player_stats["team"] == team) & (player_stats["match_date"] < before_date)
                          & (player_stats["year"] == year)]
    if prior.empty:
        return None
    top = prior.groupby("player")["xg"].sum().sort_values(ascending=False)
    return top.index[0] if len(top) else None


def build_suspension_feature(xg_df):
    cards = pd.read_parquet(Path(__file__).resolve().parent.parent.parent / "data" / "espn_events" / "historical_cards.parquet")
    player_stats = load_player_match_stats()
    # player_match_stats has no date/year -- merge from team_match_stats for that.
    from src.ingest.statsbomb_data import load_team_match_stats
    team_stats = load_team_match_stats()[["match_id", "match_date", "year"]].drop_duplicates()
    player_stats = player_stats.merge(team_stats, on="match_id")

    suspension_diffs = []
    for _, r in xg_df.iterrows():
        suspended = suspended_players(cards[cards["year"] == r["year"]], upcoming_stage=_stage_for_row(r))
        home_top = team_top_scorer_as_of(player_stats, r["home_team"], r["year"], r["date"])
        away_top = team_top_scorer_as_of(player_stats, r["away_team"], r["year"], r["date"])
        home_out = int(home_top is not None and any(home_top in p for p in suspended.get(r["home_team"], [])))
        away_out = int(away_top is not None and any(away_top in p for p in suspended.get(r["away_team"], [])))
        suspension_diffs.append(away_out - home_out)  # positive = away's key player is out, favors home
    return suspension_diffs


def _stage_for_row(r):
    return r.get("stage")


if __name__ == "__main__":
    df = build_xg_training_table()

    from src.features.elo import run_all
    _, feature_rows = run_all()
    stage_by_match = {(fr["date"], fr["home_team"], fr["away_team"]): fr["stage"] for fr in feature_rows}
    df["stage"] = [stage_by_match.get((r.date, r.home_team, r.away_team)) for r in df.itertuples()]

    df["suspension_diff"] = build_suspension_feature(df)
    print("Non-zero suspension_diff rows:", (df["suspension_diff"] != 0).sum(), "of", len(df))

    extended_cols = FEATURE_COLUMNS + ["suspension_diff"]
    elo_probs, elo_outcomes, ext_probs, ext_outcomes = [], [], [], []
    for year in YEARS:
        train = df[df["year"] != year]
        test = df[df["year"] == year]

        xgb_base = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_base.fit(train[FEATURE_COLUMNS].values, train["home_win"].values)
        ext_model = xgb.XGBClassifier(**XGB_PARAMS)
        ext_model.fit(train[extended_cols].values, train["home_win"].values)

        base_pred = xgb_base.predict_proba(test[FEATURE_COLUMNS].values)[:, 1]
        ext_pred = ext_model.predict_proba(test[extended_cols].values)[:, 1]
        outcomes = test["home_win"].values

        elo_probs.extend(base_pred); elo_outcomes.extend(outcomes)
        ext_probs.extend(ext_pred); ext_outcomes.extend(outcomes)
        print(f"{year}: XGBoost(no suspension) Brier={brier_score(base_pred, outcomes):.4f}, "
              f"XGBoost(+suspension) Brier={brier_score(ext_pred, outcomes):.4f}")

    print(f"\nPooled: without suspension feature Brier={brier_score(elo_probs, elo_outcomes):.4f}, "
          f"with suspension feature Brier={brier_score(ext_probs, ext_outcomes):.4f}")
