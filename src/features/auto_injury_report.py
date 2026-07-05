"""Live auto-detected player-availability report for 2026: combines the
suspension tracker (certain, rule-based) with recent in-match injury
mentions (softer signal -- "this player was subbed off hurt in their last
match", not a guarantee they'll miss the next one). This is what replaces
having to manually type injury notes for anything ESPN's match events
already reveal; data/injury_notes.json remains for anything this doesn't
catch (e.g. news that surfaces outside of a match, pre-tournament injuries).
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.espn_match_events import build_events_for_date_range
from src.ingest.wc2026_results import TOURNAMENT_START
from src.features.suspensions import suspended_players


def build_live_report(through: date = None):
    through = through or date.today()
    cards, injuries, goals = build_events_for_date_range(TOURNAMENT_START, through)
    return cards, injuries, goals


def team_auto_flags(team: str, cards, injuries, upcoming_stage: str):
    """Returns a list of human-readable auto-detected flags for one team
    ahead of a specific upcoming match."""
    flags = []
    suspended = suspended_players(cards, upcoming_stage)
    for entry in suspended.get(team, []):
        flags.append(f"OUT (suspended): {entry}")

    if not injuries.empty:
        team_injuries = injuries[injuries["team"] == team]
        if not team_injuries.empty:
            recent = team_injuries.sort_values("date").tail(3)
            for _, row in recent.iterrows():
                flags.append(f"Recent injury concern ({row['date']}): {row['player']}")
    return flags


if __name__ == "__main__":
    cards, injuries, goals = build_live_report()
    for team in ["Argentina", "Egypt", "Portugal", "Spain"]:
        flags = team_auto_flags(team, cards, injuries, upcoming_stage="round-of-16")
        print(team, flags)
