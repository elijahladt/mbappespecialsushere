"""Simulate what a $X bankroll, staked at fractional Kelly against Kalshi's
opening prices, would have actually returned across the settled 2026 Round
of 32 matches -- using the real walk-forward Elo probabilities and real
Kalshi opening prices from validate_vs_kalshi_r32.py.

Processed in chronological order, grouped by date: bets placed on the same
day are all sized off the bankroll as of the START of that day (not updated
mid-day), since a real bettor placing same-day bets doesn't know the outcome
of an earlier same-day match yet. Bankroll updates once per day, after all
that day's bets settle.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.backtest.validate_vs_kalshi_r32 import build_comparison
from src.trading.kelly import kelly_fraction

MATCH_RESULT_PAYOUT = 1.0  # $1 payout per contract on a correct YES


def simulate(bankroll: float = 100.0, fraction: float = 0.25, max_stake_pct: float | None = 0.05, rows=None):
    """`rows` lets callers (e.g. the dashboard) pass in an already-fetched
    comparison instead of hitting the Kalshi API again on every parameter
    tweak -- build_comparison() makes ~100 live requests, too slow to redo
    on every slider move."""
    rows = rows if rows is not None else build_comparison()
    rows = [r for r in rows if r["kalshi_opening_price"] is not None]
    by_date = defaultdict(list)
    for r in rows:
        by_date[r["date"]].append(r)

    history = []
    for date in sorted(by_date):
        day_bankroll = bankroll
        day_pnl = 0.0
        day_bets = []
        for r in by_date[date]:
            p, c = r["model_prob"], r["kalshi_opening_price"]
            f = kelly_fraction(p, c) * fraction
            if max_stake_pct is not None:
                f = min(f, max_stake_pct)
            stake = day_bankroll * f
            if stake <= 0:
                continue
            contracts = stake / c
            payout = contracts * MATCH_RESULT_PAYOUT if r["actual"] == 1 else 0.0
            pnl = payout - stake
            day_pnl += pnl
            day_bets.append({**r, "stake": stake, "pnl": pnl})
        bankroll += day_pnl
        history.append({"date": date, "bets": day_bets, "day_pnl": day_pnl, "bankroll_after": bankroll})

    return history, bankroll


if __name__ == "__main__":
    start = 100.0
    history, final = simulate(bankroll=start, fraction=0.25, max_stake_pct=0.05)

    total_staked, total_bets, total_wins = 0.0, 0, 0
    for day in history:
        if not day["bets"]:
            continue
        print(f"{day['date']}  (bankroll going in: ${day['bankroll_after'] - day['day_pnl']:.2f})")
        for b in day["bets"]:
            outcome = "WON" if b["actual"] == 1 else "lost"
            print(f"    {b['team']:>20s}  model={b['model_prob']:.3f}  kalshi_open={b['kalshi_opening_price']:.3f}  "
                  f"stake=${b['stake']:.2f}  {outcome}  pnl={b['pnl']:+.2f}")
            total_staked += b["stake"]
            total_bets += 1
            total_wins += 1 if b["actual"] == 1 else 0
        print(f"  -> day P&L: {day['day_pnl']:+.2f}, bankroll after: ${day['bankroll_after']:.2f}\n")

    print(f"Start: ${start:.2f}  ->  Final: ${final:.2f}  ({(final / start - 1) * 100:+.1f}%)")
    print(f"{total_bets} bets placed (of 32 team-market opportunities), {total_wins} won, ${total_staked:.2f} total staked across the tournament")
