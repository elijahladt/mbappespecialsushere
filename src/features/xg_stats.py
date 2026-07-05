"""Merge Elo ratings with StatsBomb team-match shot-quality stats into one
feature table for the XGBoost research model.

Trained on ALL decisive (non-draw) 2018+2022 matches (group + knockout), not
just knockout ties like the Elo link -- StatsBomb only covers 2 tournaments,
so restricting to knockout-only would leave under 30 rows, too little for a
tree ensemble to have any chance. This is a deliberate mismatch with the live
Kalshi "who advances" framing (a group-stage win and a knockout-tie win are
correlated but not identical targets) -- documented here and surfaced in the
dashboard caveat rather than glossed over.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import run_all
from src.ingest.statsbomb_data import load_team_match_stats

FEATURE_COLUMNS = ["elo_diff", "xg_diff", "shots_diff", "possession_diff", "setpiece_diff"]


def build_xg_training_table():
    engine, feature_rows = run_all()
    elo_by_match = {(r["date"], r["home_team"], r["away_team"]): r for r in feature_rows}

    team_stats = load_team_match_stats()
    home = team_stats[team_stats["is_home"]]
    away = team_stats[~team_stats["is_home"]]
    merged = home.merge(away, on="match_id", suffixes=("_home", "_away"))

    rows = []
    for _, r in merged.iterrows():
        key = (r["match_date_home"], r["team_home"], r["team_away"])
        elo_row = elo_by_match.get(key)
        if elo_row is None:
            continue  # StatsBomb/team-name join miss -- skip rather than guess
        home_score, away_score = elo_row["home_score"], elo_row["away_score"]
        if home_score == away_score:
            continue  # decisive matches only, see module docstring

        from src.models.winprob_link import effective_elo_diff
        rows.append({
            "match_id": r["match_id"],
            "date": key[0], "home_team": key[1], "away_team": key[2],
            "elo_diff": effective_elo_diff(elo_row),
            "xg_diff": r["xg_for_home"] - r["xg_for_away"],
            "shots_diff": r["shots_for_home"] - r["shots_for_away"],
            "possession_diff": r["possession_pct_home"] - r["possession_pct_away"],
            "setpiece_diff": r["setpiece_xg_share_home"] - r["setpiece_xg_share_away"],
            "home_win": int(home_score > away_score),
            "year": r["year_home"],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_xg_training_table()
    print(f"Built {len(df)} decisive-match rows with full Elo+StatsBomb features.")
    print(df[["year"]].value_counts())
    print(df[FEATURE_COLUMNS + ["home_win"]].describe())
