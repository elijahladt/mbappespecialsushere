"""Is the standard +100 Elo home-advantage constant well-calibrated for World
Cup HOST NATIONS specifically, or is hosting a bigger/smaller effect than a
routine home qualifier? Tests several candidate values against the actual
group+knockout results of the last 4 hosts (South Africa 2010, Brazil 2014,
Russia 2018, Qatar 2022) -- confirmed via direct query that all of these are
recorded as home_team with neutral=0, i.e. this is exactly the mechanism
already in use, not a hypothetical.

Uses each host's pre-match Elo (walk-forward, computed fresh from only
matches strictly before that date) and scores against the actual match
outcome (1/0.5/0 for win/draw/loss), the same convention the Elo engine
itself uses internally -- not the binary knockout-only win_probability link,
since hosts' group-stage matches (which can draw) are included too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db import get_connection
from src.features.elo import EloEngine, expected_score

HOSTS = {
    "South Africa": ("2010-06-01", "2010-07-15"),
    "Brazil": ("2014-06-01", "2014-07-15"),
    "Russia": ("2018-06-01", "2018-07-20"),
    "Qatar": ("2022-11-01", "2022-12-20"),
}

CANDIDATE_HOME_ADVANTAGES = [0, 50, 100, 150, 200, 250, 300]


def get_host_matches(host, start, end):
    conn = get_connection()
    rows = conn.execute(
        """SELECT date, home_team, away_team, home_score, away_score
           FROM matches WHERE tournament='FIFA World Cup' AND date BETWEEN ? AND ?
           AND home_team=? ORDER BY date""",
        (start, end, host),
    ).fetchall()
    conn.close()
    return rows


def elo_asof(cutoff_date):
    conn = get_connection()
    matches = conn.execute(
        """SELECT date, home_team, away_team, home_score, away_score, tier, neutral
           FROM matches WHERE date < ? ORDER BY date ASC, rowid ASC""",
        (cutoff_date,),
    ).fetchall()
    conn.close()
    engine = EloEngine()
    for date, home, away, hs, as_, tier, neutral in matches:
        engine.process_match(home, away, hs, as_, tier, bool(neutral))
    return engine


def run():
    all_matches = []
    for host, (start, end) in HOSTS.items():
        for date, home, away, hs, as_ in get_host_matches(host, start, end):
            engine = elo_asof(date)
            host_elo = engine.get(home)
            opp_elo = engine.get(away)
            actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            all_matches.append({
                "host": host, "date": date, "opponent": away,
                "host_elo": host_elo, "opp_elo": opp_elo, "actual": actual,
            })
            print(f"{host:15s} vs {away:15s} ({date})  host_elo={host_elo:.0f}  opp_elo={opp_elo:.0f}  "
                  f"result={'W' if actual==1 else ('D' if actual==0.5 else 'L')}")

    print(f"\n{len(all_matches)} host matches total (South Africa 3, Brazil 7, Russia 5, Qatar 3)\n")
    print(f"{'home_advantage':>15s}  {'Brier score':>12s}  {'mean predicted':>15s}  {'actual host win rate':>20s}")
    for adv in CANDIDATE_HOME_ADVANTAGES:
        errors = []
        preds = []
        for m in all_matches:
            pred = expected_score(m["host_elo"] + adv, m["opp_elo"])
            errors.append((pred - m["actual"]) ** 2)
            preds.append(pred)
        brier = sum(errors) / len(errors)
        mean_pred = sum(preds) / len(preds)
        actual_rate = sum(m["actual"] for m in all_matches) / len(all_matches)
        marker = "  <-- current default" if adv == 100 else ""
        print(f"{adv:>15d}  {brier:>12.4f}  {mean_pred:>15.3f}  {actual_rate:>20.3f}{marker}")

    best = min(CANDIDATE_HOME_ADVANTAGES, key=lambda adv: sum(
        (expected_score(m["host_elo"] + adv, m["opp_elo"]) - m["actual"]) ** 2 for m in all_matches
    ) / len(all_matches))
    print(f"\nBest-fitting home_advantage on this sample: {best} (n=18 matches -- treat this as a rough "
          f"read, not a precisely fitted constant, given the small sample)")


if __name__ == "__main__":
    run()
