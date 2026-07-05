"""Pull 2018 + 2022 World Cup event data from StatsBomb's free open data
(via statsbombpy, no auth needed) and aggregate to per-match team-level shot
quality/possession features and per-match per-player xG contribution.

Open data only has full coverage for 2018 and 2022 -- other World Cups in the
open-data release (1958-1990) only have a handful of "famous match" events
digitized, not full tournaments, so they're not usable for systematic
team-level rolling stats.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.wc2026_results import normalize_team

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "statsbomb"
WC_SEASONS = {2018: 3, 2022: 106}
WC_COMPETITION_ID = 43

SET_PIECE_TYPES = {"Free Kick", "Corner", "Penalty"}


def _events_cached(match_id: int) -> pd.DataFrame:
    from statsbombpy import sb
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"events_{match_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    ev = sb.events(match_id=match_id)
    ev.to_parquet(path)
    return ev


def aggregate_match(match_id: int, home_team: str, away_team: str):
    """Returns (team_rows, player_rows) for one match."""
    ev = _events_cached(match_id)
    # Period 5 = penalty shootout -- exclude it. A converted shootout penalty
    # carries a high individual xG (~0.78) but isn't a run-of-play chance, and
    # including it badly inflates match xG totals vs. any publicly reported figure.
    ev = ev[ev["period"] != 5]
    shots = ev[ev["type"] == "Shot"].copy()
    shots["shot_statsbomb_xg"] = shots["shot_statsbomb_xg"].fillna(0.0)
    total_events = len(ev)

    team_rows = []
    for team, opponent, is_home in ((home_team, away_team, True), (away_team, home_team, False)):
        team_shots = shots[shots["team"] == team]
        opp_shots = shots[shots["team"] == opponent]
        setpiece_xg = team_shots[team_shots["shot_type"].isin(SET_PIECE_TYPES)]["shot_statsbomb_xg"].sum()
        team_rows.append({
            "match_id": match_id,
            "is_home": is_home,
            "team": normalize_team(team),
            "opponent": normalize_team(opponent),
            "xg_for": float(team_shots["shot_statsbomb_xg"].sum()),
            "xg_against": float(opp_shots["shot_statsbomb_xg"].sum()),
            "shots_for": int(len(team_shots)),
            "shots_against": int(len(opp_shots)),
            "possession_pct": float((ev["possession_team"] == team).sum() / total_events) if total_events else None,
            "setpiece_xg_share": float(setpiece_xg / team_shots["shot_statsbomb_xg"].sum()) if team_shots["shot_statsbomb_xg"].sum() > 0 else 0.0,
        })

    player_rows = []
    for _, row in shots.iterrows():
        player_rows.append({
            "match_id": match_id,
            "team": normalize_team(row["team"]),
            "player": row["player"],
            "xg": float(row["shot_statsbomb_xg"]),
        })

    return team_rows, player_rows


def build_all(seasons=WC_SEASONS):
    from statsbombpy import sb
    all_team_rows, all_player_rows = [], []
    for year, season_id in seasons.items():
        matches = sb.matches(competition_id=WC_COMPETITION_ID, season_id=season_id)
        for _, m in matches.iterrows():
            team_rows, player_rows = aggregate_match(m["match_id"], m["home_team"], m["away_team"])
            for r in team_rows:
                r["year"] = year
                r["match_date"] = m["match_date"]
            all_team_rows.extend(team_rows)
            all_player_rows.extend(player_rows)
        print(f"{year}: aggregated {len(matches)} matches")

    team_df = pd.DataFrame(all_team_rows)
    player_df = pd.DataFrame(all_player_rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    team_df.to_parquet(CACHE_DIR / "team_match_stats.parquet")
    player_df.to_parquet(CACHE_DIR / "player_match_stats.parquet")
    return team_df, player_df


def load_team_match_stats() -> pd.DataFrame:
    path = CACHE_DIR / "team_match_stats.parquet"
    if not path.exists():
        team_df, _ = build_all()
        return team_df
    return pd.read_parquet(path)


def load_player_match_stats() -> pd.DataFrame:
    path = CACHE_DIR / "player_match_stats.parquet"
    if not path.exists():
        _, player_df = build_all()
        return player_df
    return pd.read_parquet(path)


if __name__ == "__main__":
    team_df, player_df = build_all()
    print(f"\nTotal: {len(team_df)} team-match rows, {len(player_df)} player-shot rows")
    print(team_df.head(10))

    known_teams = set()
    import sqlite3
    from src.db import get_connection
    conn = get_connection()
    known_teams = {r[0] for r in conn.execute(
        "SELECT DISTINCT home_team FROM matches WHERE source = 'martj42_historical'"
    ).fetchall()}
    conn.close()
    unmatched = sorted(set(team_df["team"]) - known_teams)
    if unmatched:
        print(f"WARNING: StatsBomb teams with no historical match (check TEAM_ALIASES): {unmatched}")
