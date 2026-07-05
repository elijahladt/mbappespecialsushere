"""Walk-forward PROFIT simulation for club football -- a different question
from walk_forward_club.py's Brier/calibration check. That backtest answers
"was the model's probability honest?"; this one answers "if you'd actually
placed fractional-Kelly bets against the REAL historical Bet365 closing
odds stored in club_matches, how much would you have made?"

Same walk-forward discipline (train on seasons strictly before the test
season, never peek forward) and the exact same per-match betting logic
already used in the live BetMGM/Club Football Edge Board pages (edge vs.
RAW bookmaker price, fractional Kelly, per-match aggregate stake cap) --
just replayed against history with a real, compounding bankroll instead of
a snapshot of today's odds.

Rows with no stored Bet365 price are skipped (not simulated as a loss or a
free win) -- disclosed via n_skipped_no_odds, not silently dropped.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.club_elo import run_all
from src.models.club_winprob_link import build_training_set, build_features, outcome_label
from src.backtest.walk_forward_club import _season_order
from src.trading.devig import implied_prob
from src.trading.kelly import fractional_kelly_stake


def simulate(feature_rows, edge_threshold: float = 0.0, starting_bankroll: float = 1000.0,
             kelly_fraction: float = 0.25, max_stake_pct: float = 0.05, min_train: int = 380):
    bankroll = starting_bankroll
    bankroll_history = [bankroll]
    n_bets, n_wins, n_skipped_no_odds = 0, 0, 0

    seasons = _season_order(feature_rows)
    for i, season in enumerate(seasons):
        train_rows = [r for r in feature_rows if r["season"] in seasons[:i]]
        test_rows = [r for r in feature_rows if r["season"] == season]
        if len(train_rows) < min_train or not test_rows:
            continue

        X_train, y_train = build_training_set(train_rows)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        classes = list(model.classes_)

        for r in test_rows:
            odds = (r["b365_home"], r["b365_draw"], r["b365_away"])
            if any(not o or o <= 1 for o in odds):
                n_skipped_no_odds += 1
                continue

            raw_probs = model.predict_proba([build_features(r)])[0]
            p_away, p_draw, p_home = (raw_probs[classes.index(c)] for c in (0, 1, 2))
            actual = outcome_label(r)  # 0=away, 1=draw, 2=home

            legs = []  # (label, model_prob, raw_price)
            for label, model_prob, price in ((2, p_home, odds[0]), (1, p_draw, odds[1]), (0, p_away, odds[2])):
                raw_price = implied_prob(price)
                if model_prob - raw_price > edge_threshold:
                    legs.append((label, model_prob, price))
            if not legs:
                continue

            stakes = [fractional_kelly_stake(p, implied_prob(price), bankroll, kelly_fraction, max_stake_pct)
                      for _, p, price in legs]
            cap = bankroll * max_stake_pct
            total = sum(stakes)
            if total > cap and total > 0:
                stakes = [s * cap / total for s in stakes]

            match_pnl = 0.0
            for (label, _, price), stake in zip(legs, stakes):
                n_bets += 1
                if label == actual:
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
    _, feature_rows = run_all("premier_league")
    print("Premier League walk-forward P&L simulation (fractional Kelly vs. real historical Bet365 odds, $1000 start):\n")
    for threshold in (0.0, 0.02, 0.05):
        result = simulate(feature_rows, edge_threshold=threshold)
        print(f"Edge threshold >{threshold:.0%}:")
        print(f"  Final bankroll: ${result['final_bankroll']:.2f} (ROI {result['roi_pct']:+.1f}%)")
        print(f"  Bets placed: {result['n_bets']} (win rate {result['win_rate']:.1%})" if result['n_bets'] else "  No bets placed at this threshold.")
        print(f"  Bankroll range: ${result['trough_bankroll']:.2f} - ${result['peak_bankroll']:.2f}")
        print(f"  Matches skipped (no stored Bet365 odds): {result['n_skipped_no_odds']}\n")
