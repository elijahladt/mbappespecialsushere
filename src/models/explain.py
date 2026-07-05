"""Per-prediction explainability: SHAP feature contributions for the XGBoost
model, plus a "key players" breakdown -- not literal model inputs (the model
only sees team-aggregated stats), but the players whose output most drove
those team-level numbers, shown transparently as that.
"""
import sys
from pathlib import Path

import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.statsbomb_data import load_player_match_stats


def shap_contributions(model, feature_row: dict, feature_columns):
    explainer = shap.TreeExplainer(model)
    x = [[feature_row[c] for c in feature_columns]]
    shap_values = explainer.shap_values(x)
    row = shap_values[0] if shap_values.ndim == 2 else shap_values
    contributions = [
        {"feature": feature_columns[i], "value": feature_row[feature_columns[i]], "contribution": float(row[i])}
        for i in range(len(feature_columns))
    ]
    return sorted(contributions, key=lambda c: -abs(c["contribution"]))


def key_players_statsbomb(match_id, team: str, top_n: int = 3):
    """Top xG contributors for `team` in a StatsBomb-covered match (2018/2022
    backtest matches only -- live 2026 matches have no StatsBomb data, see
    key_players_live below)."""
    players = load_player_match_stats()
    match_players = players[(players["match_id"] == match_id) & (players["team"] == team)]
    if match_players.empty:
        return []
    agg = match_players.groupby("player")["xg"].sum().sort_values(ascending=False)
    return [{"player": p, "xg": float(v)} for p, v in agg.head(top_n).items() if v > 0]


def key_players_live(team: str, top_n: int = 3):
    """For live 2026 matches (no StatsBomb data): fall back to the
    API-Football squad list. Requires API_FOOTBALL_KEY -- returns an
    explicit unavailable marker rather than silently empty if not configured,
    so the dashboard can say why instead of just showing nothing."""
    try:
        from src.ingest.api_football_client import resolve_team_id, get_squad
        team_id = resolve_team_id(team)
        if team_id is None:
            return {"available": False, "reason": f"Team '{team}' not found via API-Football"}
        squad = get_squad(team_id)
        # No universal "form" ranking without per-player statistics endpoint
        # (paid tier); list by position as a reasonable, honest default.
        notable = [p["name"] for p in squad[:top_n]]
        return {"available": True, "players": notable}
    except RuntimeError as e:
        return {"available": False, "reason": str(e)}
