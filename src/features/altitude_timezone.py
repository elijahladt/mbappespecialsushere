"""Altitude and timezone-crossing ('jet lag') features -- proper versions of
what the crude 'Changed city?' flag in rest_travel.py was trying to capture.

Altitude matters specifically because Mexico City (2026 host, ~2240m) and
several 2010 South Africa host cities (highveld plateau, ~1200-1750m) are
genuinely high-altitude; reduced oxygen measurably hurts aerobic performance
for teams not acclimatized -- a well-documented effect in football, distinct
from generic travel fatigue.

Timezone-delta is computed with the real IANA timezone at the actual match
date (via zoneinfo, so historical DST rules are handled correctly) rather
than a fixed offset -- 2018 Russia spanned multiple real timezones (Moscow to
Ekaterinburg is +2h), which a same-city/different-city flag can't see at all.
"""
import sys
from datetime import date as _date
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection

# (altitude_meters, IANA timezone) -- venue_city strings exactly as stored
# in matches.venue_city (see src/ingest/wc2026_results.py / enrich_backtest_venues.py).
VENUE_REFERENCE = {
    # 2026 host cities (US / Mexico / Canada)
    "Arlington, Texas": (180, "America/Chicago"),
    "Atlanta, Georgia": (320, "America/New_York"),
    "East Rutherford, New Jersey": (5, "America/New_York"),
    "Foxborough, Massachusetts": (50, "America/New_York"),
    "Houston, Texas": (15, "America/Chicago"),
    "Inglewood, California": (30, "America/Los_Angeles"),
    "Kansas City, Missouri": (330, "America/Chicago"),
    "Miami Gardens, Florida": (3, "America/New_York"),
    "Philadelphia, Pennsylvania": (12, "America/New_York"),
    "Santa Clara, California": (24, "America/Los_Angeles"),
    "Seattle, Washington": (10, "America/Los_Angeles"),
    "Toronto": (76, "America/Toronto"),
    "Vancouver": (1, "America/Vancouver"),
    "Guadalajara": (1566, "America/Mexico_City"),
    "Guadalupe": (538, "America/Monterrey"),
    "Mexico City": (2240, "America/Mexico_City"),
    # 2022 Qatar -- single small cluster, near sea level, one timezone (control case)
    "Al Khor": (10, "Asia/Qatar"), "Al Wakrah": (5, "Asia/Qatar"),
    "Ar-Rayyan": (12, "Asia/Qatar"), "Doha": (10, "Asia/Qatar"), "Lusail": (5, "Asia/Qatar"),
    # 2018 Russia -- real timezone spread
    "Ekaterinburg": (237, "Asia/Yekaterinburg"), "Kaliningrad": (15, "Europe/Kaliningrad"),
    "Kazan": (57, "Europe/Moscow"), "Moscow": (156, "Europe/Moscow"),
    "Nizhny Novgorod": (82, "Europe/Moscow"), "Rostov-on-Don": (70, "Europe/Moscow"),
    "Saint Petersburg": (3, "Europe/Moscow"), "Samara": (155, "Europe/Samara"),
    "Saransk": (170, "Europe/Moscow"), "Sochi": (50, "Europe/Moscow"),
    "Volgograd": (142, "Europe/Volgograd"),
    # 2014 Brazil -- moderate altitude spread, real timezone spread (Manaus/Cuiaba)
    "Belo Horizonte": (852, "America/Sao_Paulo"), "Brasília": (1172, "America/Sao_Paulo"),
    "Cuiabá": (165, "America/Cuiaba"), "Curitiba": (934, "America/Sao_Paulo"),
    "Fortaleza": (16, "America/Fortaleza"), "Manaus": (92, "America/Manaus"),
    "Natal": (30, "America/Fortaleza"), "Porto Alegre": (10, "America/Sao_Paulo"),
    "Recife": (10, "America/Fortaleza"), "Rio de Janeiro": (2, "America/Sao_Paulo"),
    "Salvador": (8, "America/Bahia"), "Sao Paulo": (760, "America/Sao_Paulo"),
    # 2010 South Africa -- single timezone, HUGE altitude spread (highveld vs coast)
    "Bloemfontein": (1400, "Africa/Johannesburg"), "Cape Town": (20, "Africa/Johannesburg"),
    "Durban": (10, "Africa/Johannesburg"), "Gqeberha": (60, "Africa/Johannesburg"),
    "Johannesburg": (1750, "Africa/Johannesburg"), "Nelspruit": (660, "Africa/Johannesburg"),
    "Polokwane": (1230, "Africa/Johannesburg"), "Port Elizabeth": (60, "Africa/Johannesburg"),
    "Pretoria": (1330, "Africa/Johannesburg"), "Rustenburg": (1150, "Africa/Johannesburg"),
}


# Real, FIFA-published 2026 Team Base Camp Training Sites (fwc26teambasecamps.fifa.com) --
# where each team actually stays and trains for the WHOLE tournament, traveling out to
# match venues and back rather than relocating match-to-match. The chronological
# "last match venue" tracker below is the wrong reference point for 2026: a team based
# in, say, Irvine, California that plays a match in Mexico City returns to Irvine
# afterward, not to Mexico City -- so their NEXT match's altitude/timezone change
# should be measured from Irvine, not from Mexico City.
TEAM_BASE_CAMP = {
    "United States": ("Irvine, California", 40, "America/Los_Angeles"),
    "Algeria": ("Lawrence, Kansas", 280, "America/Chicago"),
    "Argentina": ("Kansas City, Kansas", 270, "America/Chicago"),
    "Australia": ("Alameda, California", 10, "America/Los_Angeles"),
    "Austria": ("Santa Barbara, California", 10, "America/Los_Angeles"),
    "Belgium": ("Renton, Washington", 15, "America/Los_Angeles"),
    "Bosnia and Herzegovina": ("Sandy, Utah", 1320, "America/Denver"),
    "Brazil": ("Morristown, New Jersey", 110, "America/New_York"),
    "Cape Verde": ("Tampa, Florida", 15, "America/New_York"),
    "Canada": ("Vancouver, British Columbia", 1, "America/Vancouver"),
    "Colombia": ("Zapopan, Jalisco, Mexico", 1570, "America/Mexico_City"),
    "DR Congo": ("Houston, Texas", 15, "America/Chicago"),
    "Ivory Coast": ("Chester, Pennsylvania", 10, "America/New_York"),
    "Croatia": ("Alexandria, Virginia", 30, "America/New_York"),
    "Curacao": ("Boca Raton, Florida", 5, "America/New_York"),
    "Czech Republic": ("Mansfield, Texas", 200, "America/Chicago"),
    "Ecuador": ("Columbus, Ohio", 230, "America/New_York"),
    "England": ("Kansas City, Missouri", 270, "America/Chicago"),
    "Egypt": ("Spokane, Washington", 575, "America/Los_Angeles"),
    "France": ("Waltham, Massachusetts", 30, "America/New_York"),
    "Germany": ("Winston-Salem, North Carolina", 280, "America/New_York"),
    "Ghana": ("Smithfield, Rhode Island", 30, "America/New_York"),
    "Haiti": ("Galloway Township, New Jersey", 10, "America/New_York"),
    "Iran": ("Tijuana, Baja California, Mexico", 20, "America/Tijuana"),
    "Iraq": ("White Sulphur Springs, West Virginia", 580, "America/New_York"),
    "Japan": ("Nashville, Tennessee", 150, "America/Chicago"),
    "Jordan": ("Portland, Oregon", 15, "America/Los_Angeles"),
    "South Korea": ("Zapopan, Jalisco, Mexico", 1570, "America/Mexico_City"),
    "Mexico": ("Mexico City, Mexico", 2240, "America/Mexico_City"),
    "Morocco": ("Basking Ridge, New Jersey", 90, "America/New_York"),
    "Netherlands": ("Riverside, Missouri", 230, "America/Chicago"),
    "New Zealand": ("San Diego, California", 20, "America/Los_Angeles"),
    "Norway": ("Greensboro, North Carolina", 270, "America/New_York"),
    "Panama": ("New Tecumseth, Ontario, Canada", 245, "America/Toronto"),
    "Paraguay": ("San Jose, California", 25, "America/Los_Angeles"),
    "Portugal": ("Palm Beach Gardens, Florida", 5, "America/New_York"),
    "Qatar": ("Santa Barbara, California", 10, "America/Los_Angeles"),
    "Saudi Arabia": ("Austin, Texas", 150, "America/Chicago"),
    "Scotland": ("Charlotte, North Carolina", 230, "America/New_York"),
    "Senegal": ("Piscataway, New Jersey", 30, "America/New_York"),
    "South Africa": ("San Agustín Tlaxiaca, Hidalgo, Mexico", 2100, "America/Mexico_City"),
    "Spain": ("Chattanooga, Tennessee", 210, "America/New_York"),
    "Sweden": ("Frisco, Texas", 200, "America/Chicago"),
    "Switzerland": ("San Diego, California", 20, "America/Los_Angeles"),
    "Tunisia": ("Santiago, Nuevo León, Mexico", 600, "America/Monterrey"),
    "Turkey": ("Mesa, Arizona", 370, "America/Phoenix"),
    "Uruguay": ("Playa del Carmen, Quintana Roo, Mexico", 10, "America/Cancun"),
    "Uzbekistan": ("Marietta, Georgia", 330, "America/New_York"),
}


def base_camp_altitude_tz_delta(team: str, match_date_str: str, venue_city: str):
    """The CORRECTED 2026 feature: altitude/timezone delta from the team's
    fixed tournament base camp to the match venue -- not from whatever city
    they last played in. Returns (altitude_delta_m, tz_delta_hours) or
    (None, None) if the team or venue isn't in our reference data."""
    base = TEAM_BASE_CAMP.get(team)
    venue = VENUE_REFERENCE.get(venue_city)
    if base is None or venue is None:
        return None, None
    _, base_altitude, base_tz = base
    venue_altitude, venue_tz = venue
    y, m, d = (int(x) for x in match_date_str.split("-"))
    match_date = _date(y, m, d)
    base_offset = _utc_offset_hours(base_tz, match_date)
    venue_offset = _utc_offset_hours(venue_tz, match_date)
    return venue_altitude - base_altitude, abs(venue_offset - base_offset)


def _utc_offset_hours(tz_name: str, on_date: _date) -> float:
    zone = ZoneInfo(tz_name)
    dt = _date(on_date.year, on_date.month, on_date.day)
    import datetime
    offset = datetime.datetime(dt.year, dt.month, dt.day, 12, tzinfo=zone).utcoffset()
    return offset.total_seconds() / 3600.0


DEFAULT_ALTITUDE_SCALE = 30.0  # Elo points per 1000m gained in altitude (going UP only -- no bonus for going down)
DEFAULT_TZ_SCALE = 8.0         # Elo points per hour of timezone crossed (symmetric -- either direction disrupts)


def altitude_tz_elo_adjustment(altitude_delta_m, tz_delta_hours,
                                 altitude_scale: float = DEFAULT_ALTITUDE_SCALE,
                                 tz_scale: float = DEFAULT_TZ_SCALE) -> float:
    """Disclosed heuristic, NOT statistically fitted: unlike the old
    last-match-venue version (which WAS backtested across 2014/2018/2022 and
    found negligible, see validate_altitude_timezone.py), this base-camp-
    corrected version has no matching historical data to validate against --
    we don't have reliable 2010-2022 team base camp locations. Only altitude
    GAINS are penalized (physiologically, going up hurts an unacclimatized
    team; there's no comparably documented advantage to going down)."""
    if altitude_delta_m is None or tz_delta_hours is None:
        return 0.0
    altitude_penalty = -max(altitude_delta_m, 0) / 1000.0 * altitude_scale
    tz_penalty = -tz_delta_hours * tz_scale
    return altitude_penalty + tz_penalty


class AltitudeTimezoneTracker:
    def __init__(self):
        self.last_seen = {}  # (year, team) -> (altitude_m, utc_offset_hours)

    def query(self, year: str, team: str, date_str: str, venue_city: str):
        """Read-only: altitude/timezone delta for `team` playing at `venue_city`
        on `date_str`, without mutating tracked state -- for upcoming matches
        that haven't been played yet."""
        ref = VENUE_REFERENCE.get(venue_city)
        prev = self.last_seen.get((year, team))
        if ref is None or prev is None:
            return None, None
        altitude, tz_name = ref
        y, m, d = (int(x) for x in date_str.split("-"))
        offset = _utc_offset_hours(tz_name, _date(y, m, d))
        prev_altitude, prev_offset = prev
        return altitude - prev_altitude, abs(offset - prev_offset)

    def process_match(self, year: str, home: str, away: str, date_str: str, venue_city: str):
        ref = VENUE_REFERENCE.get(venue_city)
        row = {}
        if ref is None:
            for side in ("home", "away"):
                row[f"{side}_altitude_delta"] = None
                row[f"{side}_tz_delta_hours"] = None
            return row

        altitude, tz_name = ref
        y, m, d = (int(x) for x in date_str.split("-"))
        offset = _utc_offset_hours(tz_name, _date(y, m, d))

        for side, team in (("home", home), ("away", away)):
            prev = self.last_seen.get((year, team))
            if prev is None:
                row[f"{side}_altitude_delta"] = None
                row[f"{side}_tz_delta_hours"] = None
            else:
                prev_altitude, prev_offset = prev
                row[f"{side}_altitude_delta"] = altitude - prev_altitude
                row[f"{side}_tz_delta_hours"] = abs(offset - prev_offset)

        for team in (home, away):
            self.last_seen[(year, team)] = (altitude, offset)
        return row


def compute_altitude_timezone(conn=None):
    """(tracker, features) where features is keyed by (date, home, away) ->
    {home_altitude_delta, home_tz_delta_hours, away_altitude_delta, away_tz_delta_hours}."""
    own_conn = conn is None
    conn = conn or get_connection()
    matches = conn.execute(
        """SELECT date, home_team, away_team, venue_city
           FROM matches WHERE tournament = 'FIFA World Cup' AND venue_city IS NOT NULL
           ORDER BY date ASC, rowid ASC"""
    ).fetchall()
    if own_conn:
        conn.close()

    tracker = AltitudeTimezoneTracker()
    features = {}
    for date_str, home, away, venue_city in matches:
        year = date_str[:4]
        features[(date_str, home, away)] = tracker.process_match(year, home, away, date_str, venue_city)
    return tracker, features


if __name__ == "__main__":
    missing_cities = set()
    conn = get_connection()
    all_cities = {r[0] for r in conn.execute("SELECT DISTINCT venue_city FROM matches WHERE venue_city IS NOT NULL").fetchall()}
    conn.close()
    for c in all_cities:
        if c not in VENUE_REFERENCE:
            missing_cities.add(c)
    if missing_cities:
        print(f"WARNING: no altitude/timezone reference for: {sorted(missing_cities)}")

    tracker, feats = compute_altitude_timezone()
    sample = [(k, v) for k, v in feats.items() if v.get("home_altitude_delta") is not None][:10]
    print(f"Computed altitude/timezone features for {len(feats)} World Cup matches.")
    for k, v in sample:
        print(k, v)
