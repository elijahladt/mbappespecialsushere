"""Thin client for API-Football (api-football.com / api-sports.io), used to
supplement StatsBomb's WC-only event data with current squad/injury/recent-form
info for ALL 2026 teams -- including nations StatsBomb has no coverage for.

Requires API_FOOTBALL_KEY in a local .env (see .env.example). Free tier is
100 requests/day, so every call here is cached to sqlite with a TTL -- this
should be run once a day at most, not on every dashboard refresh.
"""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import get_secret, debug_secret_visibility
from src.db import get_connection

BASE_URL = "https://v3.football.api-sports.io"

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_football_cache (
    cache_key TEXT PRIMARY KEY,
    fetched_at INTEGER NOT NULL,
    response_json TEXT NOT NULL
);
"""


def _cache_conn():
    conn = get_connection()
    conn.executescript(CACHE_SCHEMA)
    return conn


def _cached_get(path: str, params: dict, ttl_seconds: int = 86400):
    import json
    cache_key = f"{path}?{sorted(params.items())}"
    conn = _cache_conn()
    row = conn.execute(
        "SELECT fetched_at, response_json FROM api_football_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    now = int(time.time())
    if row and now - row[0] < ttl_seconds:
        conn.close()
        return json.loads(row[1])

    api_key = get_secret("API_FOOTBALL_KEY")
    if not api_key:
        conn.close()
        raise RuntimeError(
            "API_FOOTBALL_KEY not set. Locally: copy .env.example to .env and add your free "
            "key from https://www.api-football.com/. On Streamlit Cloud: add it under App "
            "settings -> Secrets. " + debug_secret_visibility("API_FOOTBALL_KEY")
        )

    resp = requests.get(
        f"{BASE_URL}{path}",
        headers={"x-apisports-key": api_key},
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    conn.execute(
        "INSERT OR REPLACE INTO api_football_cache (cache_key, fetched_at, response_json) VALUES (?, ?, ?)",
        (cache_key, now, json.dumps(data)),
    )
    conn.commit()
    conn.close()
    return data


def resolve_team_id(team_name: str):
    data = _cached_get("/teams", {"search": team_name}, ttl_seconds=30 * 86400)
    results = data.get("response", [])
    if not results:
        return None
    return results[0]["team"]["id"]


class PlanRestricted(Exception):
    """Raised when the API's own response says the current plan can't serve
    this request (e.g. free tier is season-gated to 2022-2024, so 2026 data
    -- exactly what we need live -- comes back as an explicit plan error, not
    just an empty result). Distinguishing this from "genuinely no data" matters:
    silently treating it as zero would misreport "no injuries" as a real finding."""
    pass


def _check_plan_error(data: dict):
    errors = data.get("errors")
    if isinstance(errors, dict) and "plan" in errors:
        raise PlanRestricted(errors["plan"])


def get_recent_form(team_id: int, last: int = 5):
    """Recent fixtures (any competition) -- used for a recent-form feature
    that isn't limited to StatsBomb's WC-only coverage. NOTE: the free tier's
    season cutoff (2022-2024) blocks this for 2026 -- verified directly,
    raises PlanRestricted rather than returning a misleading empty list."""
    data = _cached_get("/fixtures", {"team": team_id, "last": last})
    _check_plan_error(data)
    return data.get("response", [])


def get_squad(team_id: int):
    """Not season-gated -- works on the free tier, returns the current
    registered squad."""
    data = _cached_get("/players/squads", {"team": team_id}, ttl_seconds=7 * 86400)
    _check_plan_error(data)
    results = data.get("response", [])
    return results[0]["players"] if results else []


def get_injuries(team_id: int, season: int = 2026):
    """NOTE: blocked on the free tier for 2026 (season cutoff is 2022-2024,
    verified directly against the API) -- raises PlanRestricted."""
    data = _cached_get("/injuries", {"team": team_id, "season": season}, ttl_seconds=3600)
    _check_plan_error(data)
    return data.get("response", [])


def team_recent_form_summary(team_name: str, last: int = 5):
    """Points per game over the last N matches across all competitions, plus
    current injury count. On the free tier both underlying calls are
    season-gated to 2022-2024 and will raise PlanRestricted for 2026 -- this
    returns that fact explicitly rather than pretending the data is just empty."""
    team_id = resolve_team_id(team_name)
    if team_id is None:
        return None
    try:
        fixtures = get_recent_form(team_id, last=last)
    except PlanRestricted as e:
        return {"team_id": team_id, "available": False, "reason": str(e)}

    points, counted = 0, 0
    for f in fixtures:
        goals = f.get("goals", {})
        home_id = f["teams"]["home"]["id"]
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            continue
        is_home = home_id == team_id
        gf, ga_ = (gh, ga) if is_home else (ga, gh)
        if gf > ga_:
            points += 3
        elif gf == ga_:
            points += 1
        counted += 1
    ppg = points / counted if counted else None

    try:
        injuries = get_injuries(team_id)
    except PlanRestricted as e:
        return {"team_id": team_id, "available": False, "reason": str(e),
                "recent_ppg": ppg, "n_recent_matches": counted}

    return {
        "team_id": team_id, "available": True,
        "recent_ppg": ppg,
        "n_recent_matches": counted,
        "n_injuries": len(injuries),
        "injured_players": [i["player"]["name"] for i in injuries],
    }


if __name__ == "__main__":
    import json
    summary = team_recent_form_summary("Argentina")
    print(json.dumps(summary, indent=2))
