"""Walk-forward PROFIT simulation for tennis -- same purpose as
pnl_backtest_club.py: walk_forward_tennis.py only checks calibration
(Brier score), this simulates actually placing fractional-Kelly bets
against the REAL historical Bet365 Match Winner odds stored in
tennis_matches (b365_winner/b365_loser) and tracks a compounding bankroll.

Not a leak: b365_winner/b365_loser are pre-match prices FOR A NAMED PLAYER,
just stored under a column name that reflects who happened to win --
using "the real price quoted for the player named winner in this row" is
exactly the same as using "the real price quoted for player X" where X is
determined before the match. The label only determines which of the two
bets (if placed) wins the simulation, resolved from the actual, known
outcome, same as pnl_backtest_club.py.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.tennis_elo import run_all
from src.models.tennis_winprob_link import build_training_set, features_for_perspective, EXCLUDED_COMMENTS
from src.trading.devig import implied_prob
from src.trading.kelly import fractional_kelly_stake


def simulate(feature_rows, edge_threshold: float = 0.0, starting_bankroll: float = 1000.0,
             kelly_fraction: float = 0.25, max_stake_pct: float = 0.05, min_train: int = 500):
    bankroll = starting_bankroll
    bankroll_history = [bankroll]
    n_bets, n_wins, n_skipped_no_odds = 0, 0, 0

    years = sorted({int(r["date"][:4]) for r in feature_rows})
    for year in years:
        cutoff = f"{year}-01-01"
        train_rows = [r for r in feature_rows if r["date"] < cutoff]
        test_rows = [r for r in feature_rows if r["date"][:4] == str(year)]
        if len(train_rows) < min_train or not test_rows:
            continue

        X_train, y_train = build_training_set(train_rows)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        for r in test_rows:
            b365_winner, b365_loser = r["b365_winner"], r["b365_loser"]
            if not b365_winner or not b365_loser or b365_winner <= 1 or b365_loser <= 1:
                n_skipped_no_odds += 1
                continue
            if r.get("comment") in EXCLUDED_COMMENTS:
                n_skipped_no_odds += 1
                continue

            p_winner_side = model.predict_proba([features_for_perspective(r, True)])[0, 1]
            p_loser_side = 1 - p_winner_side

            legs = []  # (won_the_match, model_prob, decimal_odds)
            for won, model_prob, price in ((True, p_winner_side, b365_winner), (False, p_loser_side, b365_loser)):
                raw_price = implied_prob(price)
                if model_prob - raw_price > edge_threshold:
                    legs.append((won, model_prob, price))
            if not legs:
                continue

            stakes = [fractional_kelly_stake(p, implied_prob(price), bankroll, kelly_fraction, max_stake_pct)
                      for _, p, price in legs]
            cap = bankroll * max_stake_pct
            total = sum(stakes)
            if total > cap and total > 0:
                stakes = [s * cap / total for s in stakes]

            match_pnl = 0.0
            for (won, _, price), stake in zip(legs, stakes):
                n_bets += 1
                if won:
                    n_wins += 1
                    match_pnl += stake * (price - 1)
                else:
                    match_pnl -= stake
            bankroll += match_pnl
            bankroll_history.append(bankroll)

    return {
        "final_bankroll": bankroll,
        "roi_pct": (bankroll - starting_bankroll) / starting_bankroll * 100,
        "n_bets": n_bets,
        "n_wins": n_wins,
        "win_rate": n_wins / n_bets if n_bets else None,
        "n_skipped_no_odds": n_skipped_no_odds,
        "peak_bankroll": max(bankroll_history),
        "trough_bankroll": min(bankroll_history),
    }


if __name__ == "__main__":
    for tour in ("atp", "wta"):
        _, feature_rows = run_all(tour)
        print(f"\n{tour.upper()} walk-forward P&L simulation (fractional Kelly vs. real historical Bet365 odds, $1000 start):\n")
        for threshold in (0.0, 0.02, 0.05):
            result = simulate(feature_rows, edge_threshold=threshold)
            print(f"Edge threshold >{threshold:.0%}:")
            print(f"  Final bankroll: ${result['final_bankroll']:.2f} (ROI {result['roi_pct']:+.1f}%)")
            print(f"  Bets placed: {result['n_bets']} (win rate {result['win_rate']:.1%})" if result['n_bets'] else "  No bets placed at this threshold.")
            print(f"  Bankroll range: ${result['trough_bankroll']:.2f} - ${result['peak_bankroll']:.2f}")
            print(f"  Matches skipped (no stored Bet365 odds): {result['n_skipped_no_odds']}\n")
