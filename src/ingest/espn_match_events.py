"""Pull card (yellow/red) and in-match injury-substitution events from ESPN's
match-summary `keyEvents` feed. Free, no auth, works for both live 2026
matches and historical 2018/2022/etc matches (same endpoint, just a different
date). This is what powers automatic suspension detection (task: replace
manually-typed injury notes with something that reads real match events) --
ESPN's dedicated "injuries" endpoint exists but is verified EMPTY for this
World Cup across every team, so it's not usable; keyEvents is the real signal.
"""
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.ingest.wc2026_results import fetch_day, normalize_team

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary"
CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "espn_events"

INJURY_TEXT_RE = re.compile(r"because of an injury(?: (?P<player>[^.(]+))?\s*\(?(?P<team>[^)]*)\)?", re.IGNORECASE)
GOAL_SCORER_RE = re.compile(r"^(?:Goal!|Own Goal by)\s*[^.]*\.\s*(?P<scorer>.+?)\s*\((?P<team>[^)]+)\)", re.IGNORECASE)
GOAL_ASSIST_RE = re.compile(r"Assisted by\s+(?P<assister>.+?)(?:\s+with|\s+following|\.|$)", re.IGNORECASE)


def _summary_cached(event_id: str) -> dict:
    import json
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"summary_{event_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    resp = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    path.write_text(json.dumps(data))
    return data


def parse_card_and_injury_events(key_events: list, home_team: str, away_team: str):
    cards, injuries, goals = [], [], []
    for e in key_events:
        etype = e.get("type", {}).get("type", "")
        text = e.get("text", "") or e.get("shortText", "") or ""
        minute = e.get("clock", {}).get("displayValue", "")

        if etype in ("yellow-card", "red-card"):
            team = e.get("team", {}).get("displayName")
            m = re.match(r"^([^(]+)\(", text)
            player = m.group(1).strip() if m else e.get("shortText", "").rsplit(" ", 2)[0]
            if team:
                cards.append({
                    "team": normalize_team(team), "player": player,
                    "card_type": "red" if etype == "red-card" else "yellow", "minute": minute,
                })
        elif etype == "substitution" and "injury" in text.lower():
            # Hard signal: the player actually left the pitch.
            team = e.get("team", {}).get("displayName")
            m = re.search(r"replaces\s+(?P<player>.+?)\s+because of an injury", text, re.IGNORECASE)
            player_off = m.group("player").strip() if m else None
            if team and player_off:
                injuries.append({"team": normalize_team(team), "player": player_off, "minute": minute,
                                  "detail": text, "confidence": "hard"})
        elif etype == "start-delay" and "injury" in text.lower():
            # Soft signal: a stoppage mentioned an injury, but the player may
            # have played on -- doesn't necessarily mean they came off.
            m = INJURY_TEXT_RE.search(text)
            if m and m.group("player"):
                injuries.append({
                    "team": normalize_team(m.group("team").strip()) if m.group("team") else None,
                    "player": m.group("player").strip(), "minute": minute, "detail": text,
                    "confidence": "soft",
                })
        elif etype in ("goal", "goal---header", "goal---free-kick", "goal---penalty"):
            # own-goal and var---goal-not-awarded are deliberately excluded: an
            # own goal isn't a measure of the scorer's attacking quality, and a
            # disallowed goal never happened.
            m = GOAL_SCORER_RE.match(text)
            if not m:
                continue
            team = normalize_team(m.group("team").strip())
            scorer = m.group("scorer").strip()
            goals.append({"team": team, "player": scorer, "minute": minute, "role": "goal"})
            am = GOAL_ASSIST_RE.search(text)
            if am:
                goals.append({"team": team, "player": am.group("assister").strip(), "minute": minute, "role": "assist"})
    return cards, injuries, goals


def fetch_match_events(event_id: str, home_team: str, away_team: str):
    data = _summary_cached(event_id)
    key_events = data.get("keyEvents", [])
    return parse_card_and_injury_events(key_events, home_team, away_team)


def build_events_for_date_range(start: date, end: date):
    """Returns (cards_df, injuries_df, goals_df) for every completed match in [start, end]."""
    all_cards, all_injuries, all_goals = [], [], []
    day = start
    while day <= end:
        for event in fetch_day(day):
            comp = event["competitions"][0]
            if not comp["status"]["type"]["completed"]:
                continue
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")["team"]["displayName"]
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")["team"]["displayName"]
            match_date = comp["date"][:10]
            stage = event.get("season", {}).get("slug", "")
            cards, injuries, goals = fetch_match_events(event["id"], home, away)
            for c in cards:
                c.update(date=match_date, home_team=normalize_team(home), away_team=normalize_team(away), stage=stage)
            for i in injuries:
                i.update(date=match_date, home_team=normalize_team(home), away_team=normalize_team(away), stage=stage)
            for g in goals:
                g.update(date=match_date, home_team=normalize_team(home), away_team=normalize_team(away), stage=stage)
            all_cards.extend(cards)
            all_injuries.extend(injuries)
            all_goals.extend(goals)
        day += timedelta(days=1)
    return pd.DataFrame(all_cards), pd.DataFrame(all_injuries), pd.DataFrame(all_goals)


if __name__ == "__main__":
    from src.ingest.wc2026_results import TOURNAMENT_START
    cards, injuries, goals = build_events_for_date_range(TOURNAMENT_START, date.today())
    print(f"{len(cards)} card events, {len(injuries)} in-match injury events, {len(goals)} goal/assist events (2026 so far)")
    if not cards.empty:
        print(cards.head(10))
    if not injuries.empty:
        print(injuries.head(10))
    if not goals.empty:
        print(goals.head(10))
