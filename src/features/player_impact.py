"""Adjust Model prob for confirmed-missing key players, weighted by their
real contribution (goals+assists) to the team this tournament -- using
ESPN's free match-event data (goal/assist events), since StatsBomb has no
2026 coverage and API-Football's free tier is season-gated away from 2026.

This is a disclosed HEURISTIC, not a statistically fitted adjustment: we only
have 2 historical cases where a team's top scorer was suspended (see
src/backtest/validate_suspensions.py), nowhere near enough to fit or
validate an effect size. IMPACT_SCALE is a guess, exposed as a tunable
dashboard parameter rather than hidden -- this is why the whole feature is
toggleable and off by default.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.suspensions import suspended_players

DEFAULT_IMPACT_SCALE = 150.0  # Elo points for a team missing 100% of its goal+assist output
HARD_INJURY_WEIGHT = 0.6      # weight for a player confirmed subbed off injured (uncertain they'll actually sit out)
SOFT_INJURY_WEIGHT = 0.25     # weight for a softer "stoppage mentioned injury" signal
MAX_TEAM_IMPACT = 0.6         # cap: never treat a team as losing more than 60% of its output to missing players


def team_goal_contribution_shares(goals_df, team: str):
    """{player: share_of_team_goal_plus_assist_events} for one team, using
    all goal/assist events recorded so far (whatever window `goals_df` covers)."""
    team_goals = goals_df[goals_df["team"] == team]
    if team_goals.empty:
        return {}
    counts = team_goals.groupby("player").size()
    total = counts.sum()
    return (counts / total).to_dict()


def missing_players_for_team(team: str, cards_df, injuries_df, upcoming_stage: str):
    """Returns {player: weight} where weight in (0, 1] reflects how confident
    we are this player misses the upcoming match: 1.0 for a confirmed
    suspension, HARD/SOFT_INJURY_WEIGHT for the two injury-signal tiers."""
    out = {}
    suspended = suspended_players(cards_df, upcoming_stage)
    for entry in suspended.get(team, []):
        player = entry.split(" (")[0]
        out[player] = 1.0

    if injuries_df is not None and not injuries_df.empty:
        team_injuries = injuries_df[injuries_df["team"] == team]
        for _, row in team_injuries.sort_values("date").groupby("player").tail(1).iterrows():
            weight = HARD_INJURY_WEIGHT if row.get("confidence") == "hard" else SOFT_INJURY_WEIGHT
            out[row["player"]] = max(out.get(row["player"], 0), weight)
    return out


def elo_adjustment_for_team(team: str, goals_df, cards_df, injuries_df, upcoming_stage: str,
                              impact_scale: float = DEFAULT_IMPACT_SCALE):
    """Negative Elo-point adjustment for `team` given its confirmed/likely
    missing players, weighted by each player's real goal+assist share."""
    missing = missing_players_for_team(team, cards_df, injuries_df, upcoming_stage)
    if not missing:
        return 0.0, []
    shares = team_goal_contribution_shares(goals_df, team)

    total_impact = 0.0
    detail = []
    for player, confidence_weight in missing.items():
        share = shares.get(player, 0.0)
        impact = share * confidence_weight
        total_impact += impact
        if share > 0:
            detail.append(f"{player}: {share:.0%} of team's goal contributions, weight={confidence_weight:.2f}")

    total_impact = min(total_impact, MAX_TEAM_IMPACT)
    return -total_impact * impact_scale, detail


if __name__ == "__main__":
    from datetime import date
    from src.features.auto_injury_report import build_live_report

    cards, injuries, goals = build_live_report()
    for team in ["Argentina", "Egypt", "Portugal", "Spain", "United States", "Belgium"]:
        adj, detail = elo_adjustment_for_team(team, goals, cards, injuries, upcoming_stage="round-of-16")
        print(f"{team}: Elo adjustment = {adj:+.1f}  {detail}")
