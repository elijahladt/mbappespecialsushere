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

# Star-player boost: does a team with ONE standout individual attacker (e.g.
# Norway's Haaland) deserve a bump beyond what team-level Elo captures? Elo
# has no concept of individual player quality at all. Backtested with a
# StatsBomb-based proxy (src/backtest/validate_star_player.py, leave-one-
# tournament-out on 2018/2022): result was INCONCLUSIVE -- Brier 0.236 with
# it vs. 0.233 without, on only 97 matches across 2 tournaments, not a clean
# win. Shipped anyway as a disclosed, OFF-by-default heuristic (same
# treatment as the injury adjustment above) since the underlying question
# ("should individual star quality matter") is real even though we couldn't
# prove an effect size on this small a sample -- not statistically
# validated, don't treat it as more than a toggle to experiment with.
DEFAULT_STAR_BOOST_SCALE = 12.0  # Elo points per goal+assist by the team's leading scorer this tournament
MAX_STAR_BOOST = 120.0            # cap: even a huge individual tally shouldn't swing more than this many Elo points


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


def leading_scorer_count(goals_df, team: str):
    """(player, count) for the team's single biggest goal+assist contributor
    this tournament, or (None, 0) if the team has no recorded events yet.
    Uses the raw COUNT, not the share used by team_goal_contribution_shares()
    above -- share alone can't distinguish "one player got both of this
    team's 2 total contributions" from "one player has 8 of this team's 10",
    and it's the absolute output that reflects real individual quality."""
    team_goals = goals_df[goals_df["team"] == team]
    if team_goals.empty:
        return None, 0
    counts = team_goals.groupby("player").size()
    top_player = counts.idxmax()
    return top_player, int(counts.max())


def star_player_boost_for_team(team: str, goals_df, boost_scale: float = DEFAULT_STAR_BOOST_SCALE):
    """POSITIVE Elo-point adjustment for `team` based on its leading
    scorer's real goal+assist count this tournament -- the mirror image of
    elo_adjustment_for_team() above (which only ever subtracts, for
    confirmed-missing players). See module-level note on DEFAULT_STAR_BOOST_SCALE
    for the (inconclusive) backtest this is based on."""
    player, count = leading_scorer_count(goals_df, team)
    if not player or count == 0:
        return 0.0, []
    boost = min(count * boost_scale, MAX_STAR_BOOST)
    return boost, [f"{player}: {count} goal+assist contribution(s) this tournament (leading scorer)"]


if __name__ == "__main__":
    from datetime import date
    from src.features.auto_injury_report import build_live_report

    cards, injuries, goals = build_live_report()
    for team in ["Argentina", "Egypt", "Portugal", "Spain", "United States", "Belgium"]:
        adj, detail = elo_adjustment_for_team(team, goals, cards, injuries, upcoming_stage="round-of-16")
        print(f"{team}: Injury/suspension Elo adjustment = {adj:+.1f}  {detail}")

    print()
    for team in ["Norway", "Brazil", "France", "Morocco", "Argentina"]:
        boost, detail = star_player_boost_for_team(team, goals)
        print(f"{team}: Star-player Elo boost = {boost:+.1f}  {detail}")
