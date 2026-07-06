"""Walk-forward validation of the Poisson goal-based match simulator(s)
against the existing, already-validated Elo-diff logistic model, on the
SAME 2010-2022 WC knockout test matches used everywhere else in this
project -- same discipline as every other feature/model added here: don't
trust a new approach just because it sounds reasonable, check it against a
real held-out comparison first.

Two goal-sim variants compared: the original Elo-diff-only expected-goals
regression, and the newer efficiency-based (Dixon-Coles-lite) version using
each team's own rolling offensive/defensive efficiency
(src/features/team_efficiency.py) -- both leak-free by construction (each
match's efficiency numbers only reflect STRICTLY EARLIER matches), so no
separate train/test refit is needed for the efficiency tracker itself, only
for the logistic and elo-goal models which actually fit parameters.

The goal models produce P(draw) directly (unlike the logistic model, which
only ever answers "who advances"); converted to a comparable "advances"
probability by splitting the draw mass 50/50 between the two sides -- the
same "penalties are close to a coin flip" assumption already used elsewhere
(src/models/winprob_link.py's shootout_adjustment), just applied as a direct
split here instead of a band-based nudge.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.elo import run_all, HOME_ADVANTAGE
from src.features.team_efficiency import build_efficiency_table
from src.models.winprob_link import WC_KNOCKOUT_START, build_training_set, build_features, is_knockout, shootout_adjustment
from src.models.goal_simulation import fit_goal_model, simulate_match, simulate_match_efficiency
from src.backtest.metrics import brier_score, bootstrap_brier_ci

TEST_YEARS = [2010, 2014, 2018, 2022]


def walk_forward_comparison(feature_rows, n_sims: int = 2000):
    """n_sims defaults lower than the live 10,000 here since this reruns the
    simulation once per held-out match across 4 tournaments -- 2000 is
    still plenty stable for a Brier-score comparison, and keeps the
    backtest fast; the live dashboard uses the full 10,000."""
    efficiency_rows, league_avg_goals = build_efficiency_table()
    efficiency_by_key = {(r["date"], r["home_team"], r["away_team"]): r for r in efficiency_rows}

    years = sorted(WC_KNOCKOUT_START.keys())
    logit_probs, goal_probs, eff_probs, outcomes = [], [], [], []
    per_year = {}

    for year in years:
        if year not in TEST_YEARS:
            continue
        cutoff = WC_KNOCKOUT_START[year]

        train_rows = [r for r in feature_rows if r["date"] < cutoff]
        X_train, y_train = build_training_set(train_rows)
        if len(y_train) < 10:
            continue
        logit_model = LogisticRegression()
        logit_model.fit(X_train, y_train)

        goal_model = fit_goal_model(train_rows)

        test_rows = [r for r in feature_rows if is_knockout(r["date"], r["tournament"])
                     and r["date"][:4] == str(year) and r["home_score"] != r["away_score"]]
        if not test_rows:
            continue

        year_logit, year_goal, year_eff, year_outcomes = [], [], [], []
        for r in test_rows:
            p_logit = shootout_adjustment(logit_model.predict_proba([build_features(r)])[0, 1])

            adv = 0.0 if r["neutral"] else HOME_ADVANTAGE
            effective_diff = (r["home_elo_pre"] + adv) - r["away_elo_pre"]
            sim = simulate_match(goal_model, effective_diff, n_sims=n_sims)
            p_goal = sim["p_home_win"] + 0.5 * sim["p_draw"]

            eff_row = efficiency_by_key.get((r["date"], r["home_team"], r["away_team"]))
            if eff_row is not None:
                eff_sim = simulate_match_efficiency(
                    eff_row["home_off_pre"], eff_row["home_def_pre"],
                    eff_row["away_off_pre"], eff_row["away_def_pre"],
                    league_avg_goals, n_sims=n_sims,
                )
                p_eff = eff_sim["p_home_win"] + 0.5 * eff_sim["p_draw"]
            else:
                p_eff = p_goal  # join miss fallback -- shouldn't happen, same source table

            outcome = 1 if r["home_score"] > r["away_score"] else 0
            year_logit.append(p_logit)
            year_goal.append(p_goal)
            year_eff.append(p_eff)
            year_outcomes.append(outcome)

        per_year[year] = {
            "n": len(year_outcomes),
            "logit_brier": brier_score(year_logit, year_outcomes),
            "goal_brier": brier_score(year_goal, year_outcomes),
            "eff_brier": brier_score(year_eff, year_outcomes),
        }
        logit_probs.extend(year_logit)
        goal_probs.extend(year_goal)
        eff_probs.extend(year_eff)
        outcomes.extend(year_outcomes)

    return logit_probs, goal_probs, eff_probs, outcomes, per_year


if __name__ == "__main__":
    _, feature_rows = run_all()
    logit_probs, goal_probs, eff_probs, outcomes, per_year = walk_forward_comparison(feature_rows)

    print("Walk-forward comparison: logistic vs. Elo-based goal-sim vs. efficiency-based goal-sim\n")
    for year, stats in per_year.items():
        print(f"  {year}: n={stats['n']:2d} -- logistic={stats['logit_brier']:.4f}  "
              f"elo-goal-sim={stats['goal_brier']:.4f}  efficiency-goal-sim={stats['eff_brier']:.4f}")

    if logit_probs:
        logit_brier = brier_score(logit_probs, outcomes)
        goal_brier = brier_score(goal_probs, outcomes)
        eff_brier = brier_score(eff_probs, outcomes)
        l_lo, l_hi = bootstrap_brier_ci(logit_probs, outcomes)
        g_lo, g_hi = bootstrap_brier_ci(goal_probs, outcomes)
        e_lo, e_hi = bootstrap_brier_ci(eff_probs, outcomes)
        print(f"\nPooled (n={len(outcomes)}):")
        print(f"  Logistic model:            Brier={logit_brier:.4f} (95% CI [{l_lo:.4f}, {l_hi:.4f}])")
        print(f"  Elo-based goal-sim:        Brier={goal_brier:.4f} (95% CI [{g_lo:.4f}, {g_hi:.4f}])")
        print(f"  Efficiency-based goal-sim: Brier={eff_brier:.4f} (95% CI [{e_lo:.4f}, {e_hi:.4f}])")
        best = min([("logistic", logit_brier), ("elo-goal-sim", goal_brier), ("efficiency-goal-sim", eff_brier)], key=lambda x: x[1])
        print(f"\n  Best on this sample: {best[0]} (Brier={best[1]:.4f}) -- but check the CIs above before calling this a real difference.")
