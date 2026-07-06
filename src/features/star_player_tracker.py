"""Chronological, leak-free tracker of each team's leading individual
attacker's cumulative output within a tournament -- built to test whether
"this team has one standout individual player" (e.g. Norway with Haaland)
carries real predictive signal beyond team-level Elo, which has no concept
of individual player quality at all.

Uses StatsBomb's per-shot xG (already cached from the XGBoost research
model work) as the individual-output metric for the 2018/2022 backtest --
not exact goals, but a reasonable proxy for "how dangerous is this
player's attacking output," and it's already ingested with zero extra
work. The LIVE (2026) version uses real ESPN goal+assist counts instead
(see src/features/player_impact.py) since StatsBomb has no 2026 coverage --
two different metrics for the same underlying concept, same as every other
historical-vs-live data gap already disclosed in this project.
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.statsbomb_data import load_team_match_stats, load_player_match_stats


def build_star_player_table():
    """Returns a DataFrame with one row per (match_id, team) giving that
    team's leading scorer's cumulative xG STRICTLY BEFORE this match (i.e.
    not counting this match's own events) -- safe to use as a pre-match
    feature. Tournament-scoped: tallies reset at the start of each year's
    World Cup, matching how "this player has been on fire THIS tournament"
    is actually the claim being tested, not their career total."""
    team_stats = load_team_match_stats()[["match_id", "match_date", "year", "team", "is_home"]]
    player_stats = load_player_match_stats()
    player_stats = player_stats.merge(team_stats[["match_id", "match_date", "year"]].drop_duplicates("match_id"),
                                       on="match_id", how="left")

    player_xg_by_match = player_stats.groupby(["match_id", "team", "player"])["xg"].sum().reset_index()
    match_dates = team_stats[["match_id", "match_date", "year"]].drop_duplicates("match_id")
    player_xg_by_match = player_xg_by_match.merge(match_dates, on="match_id")
    player_xg_by_match = player_xg_by_match.sort_values(["year", "match_date", "match_id"])

    cumulative = defaultdict(lambda: defaultdict(float))  # year -> {(team, player): cumulative xg}
    rows = []
    for match_id, group in player_xg_by_match.groupby("match_id", sort=False):
        year = group["year"].iloc[0]
        teams_in_match = group["team"].unique()
        for team in teams_in_match:
            player_cum = cumulative[year]
            team_players = {p: xg for (t, p), xg in player_cum.items() if t == team}
            leading_scorer_xg_pre = max(team_players.values()) if team_players else 0.0
            rows.append({"match_id": match_id, "team": team, "leading_scorer_xg_pre": leading_scorer_xg_pre})

        # Update AFTER recording pre-match state for every team in this match.
        for _, r in group.iterrows():
            cumulative[year][(r["team"], r["player"])] += r["xg"]

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_star_player_table()
    print(f"{len(df)} (match, team) rows.")
    print(df.sort_values("leading_scorer_xg_pre", ascending=False).head(10))
