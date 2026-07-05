"""FIFA World Cup 2026 yellow-card accumulation suspension rules, verified via
web search (not assumed): a player is suspended for the next match after 2
yellow cards, but the accumulation slate is wiped clean twice -- once after
the group stage, and again after the quarterfinals -- so nobody enters the
semifinal carrying a card, and knockout accumulation doesn't inherit group-
stage cards. A straight red card is always at least a 1-match ban.

A ban applies ONLY to the single next match after the triggering card, then
clears (yellow count resets to 0 once served). This walks each team's card
history match-by-match within a window to track that correctly, rather than
naively flagging anyone who has ever reached 2 yellows/a red in the window.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.espn_match_events import build_events_for_date_range

SUSPENSION_WINDOWS = {
    # ESPN uses slightly different stage slugs across tournaments
    # (e.g. "quarterfinal" for 2026 vs "quarterfinals" for 2018/2022) --
    # cover both rather than assume one convention.
    "group-stage": "group",
    "round-of-32": "knockout1",
    "round-of-16": "knockout1",
    "quarterfinal": "knockout1",
    "quarterfinals": "knockout1",
    "semifinal": "knockout2",
    "semifinals": "knockout2",
    "third-place": "knockout2",
    "3rd-place-match": "knockout2",
    "final": "knockout2",
}


def _window(stage: str):
    return SUSPENSION_WINDOWS.get(stage)


def _currently_suspended(team_cards_in_window):
    """team_cards_in_window: rows for one team/player, one window, each row
    one card event. Walks match-by-match (grouping same-date cards as one
    match) tracking yellow count and pending suspension, and returns
    (is_suspended_for_next_match, reason)."""
    by_match = team_cards_in_window.sort_values("date").groupby("date")
    yellow_count = 0
    suspended_next, reason = False, None
    for _, match_cards in by_match:
        if suspended_next:
            suspended_next, reason = False, None  # this match serves the ban
        if (match_cards["card_type"] == "red").any():
            suspended_next, reason = True, "red card"
        yellow_count += (match_cards["card_type"] == "yellow").sum()
        if yellow_count >= 2:
            suspended_next, reason = True, "2 yellow cards"
            yellow_count = 0  # ban served next match, slate clears
    return suspended_next, reason


def suspended_players(cards_df, upcoming_stage: str):
    """{team: [reasons]} for players who will sit out the next match given
    the card history so far. `upcoming_stage` picks the accumulation window
    (e.g. 'round-of-16' -> group-stage cards don't count, round-of-32 cards do)."""
    window = _window(upcoming_stage)
    if window is None or cards_df.empty:
        return {}

    same_window = cards_df[cards_df["stage"].map(_window) == window]
    result = {}
    for (team, player), rows in same_window.groupby(["team", "player"]):
        is_suspended, reason = _currently_suspended(rows)
        if is_suspended:
            result.setdefault(team, []).append(f"{player} ({reason})")
    return result


if __name__ == "__main__":
    from datetime import date
    from src.ingest.wc2026_results import TOURNAMENT_START
    cards, _, _ = build_events_for_date_range(TOURNAMENT_START, date.today())
    suspended = suspended_players(cards, upcoming_stage="round-of-16")
    print("Suspended for Round of 16:")
    for team, players in suspended.items():
        print(f"  {team}: {players}")
