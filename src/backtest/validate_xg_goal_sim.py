"""Leave-one-tournament-out validation of the xG-based goal simulator
(src/models/goal_simulation.py's fit_xg_goal_model/simulate_match_xg)
against the Elo-only goal simulator and the plain logistic model, ALL
refit on the exact same small StatsBomb-covered 2018+2022 subset for a
fair comparison -- same discipline, same methodology as
validate_xgb_model.py and validate_goal_simulation.py.

xG is StatsBomb-only (no 2026 coverage, confirmed no alternative source
has it either), so this is a research-only comparison: even if it wins
here, it can never be used for live inference. Reported honestly either
way, per the user's own instruction: keep it only if it actually helps.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.features.xg_stats import build_xg_training_table
from src.ingest.statsbomb_data import load_team_match_stats
from src.models.goal_simulation import fit_goal_model, simulate_match, simulate_match_xg
from src.backtest.metrics import brier_score, bootstrap_brier_ci

YEARS = [2018, 2022]


def _fit_xg_model_on_subset(merged_train, team_stats):
    """Same regression as fit_xg_goal_model(), restricted to the match_ids
    in merged_train -- needed so the xG model is trained on strictly the
    same held-out split as the logistic/elo-goal-sim comparisons."""
    elo_diff_by_match = dict(zip(merged_train["match_id"], merged_train["elo_diff"]))
    diffs, xgs = [], []
    for _, row in team_stats.iterrows():
        match_id = row["match_id"]
        if match_id not in elo_diff_by_match:
            continue
        home_perspective_diff = elo_diff_by_match[match_id]
        team_diff = home_perspective_diff if row["is_home"] else -home_perspective_diff
        diffs.append(team_diff)
        xgs.append(row["xg_for"])
    diffs, xgs = np.array(diffs), np.array(xgs)
    A = np.vstack([diffs, np.ones_like(diffs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, xgs, rcond=None)
    return {"slope": float(slope), "intercept": float(intercept), "n": len(xgs)}


def _fit_elo_goal_model_on_subset(df_train):
    """Same regression as fit_goal_model(), but working directly off the
    xg_stats merged dataframe's elo_diff/home_win columns (this table only
    has decisive matches with a known home_win, not raw scores -- fit
    against the win/loss OUTCOME's implied goal_diff sign isn't available,
    so approximate using +1/-1 as a stand-in goal_diff and the table's
    known average total, same spirit as fit_goal_model but on the smaller
    decisive-only StatsBomb subset for a fair like-for-like comparison)."""
    elo_diffs = df_train["elo_diff"].values
    goal_diff_sign = np.where(df_train["home_win"].values == 1, 1.0, -1.0)
    A = np.vstack([elo_diffs, np.ones_like(elo_diffs)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, goal_diff_sign, rcond=None)
    avg_total_goals = 2.7  # standard football average; this subset has no raw score column
    return {"slope": float(slope), "intercept": float(intercept), "avg_total_goals": avg_total_goals}


def run(n_sims: int = 2000):
    merged = build_xg_training_table()
    team_stats = load_team_match_stats()

    logit_probs, elo_sim_probs, xg_sim_probs, outcomes = [], [], [], []
    per_year = {}

    for year in YEARS:
        train = merged[merged["year"] != year]
        test = merged[merged["year"] == year]
        if len(train) < 10 or len(test) < 1:
            continue

        logit_model = LogisticRegression()
        logit_model.fit(train[["elo_diff"]].values, train["home_win"].values)

        elo_goal_model = _fit_elo_goal_model_on_subset(train)
        xg_goal_model = _fit_xg_model_on_subset(train, team_stats)

        year_logit, year_elo_sim, year_xg_sim, year_outcomes = [], [], [], []
        for _, row in test.iterrows():
            p_logit = logit_model.predict_proba([[row["elo_diff"]]])[0, 1]

            elo_sim = simulate_match(elo_goal_model, row["elo_diff"], n_sims=n_sims)
            p_elo_sim = elo_sim["p_home_win"] + 0.5 * elo_sim["p_draw"]

            xg_sim = simulate_match_xg(xg_goal_model, row["elo_diff"], n_sims=n_sims)
            p_xg_sim = xg_sim["p_home_win"] + 0.5 * xg_sim["p_draw"]

            year_logit.append(p_logit)
            year_elo_sim.append(p_elo_sim)
            year_xg_sim.append(p_xg_sim)
            year_outcomes.append(row["home_win"])

        per_year[year] = {
            "n": len(year_outcomes),
            "logit_brier": brier_score(year_logit, year_outcomes),
            "elo_sim_brier": brier_score(year_elo_sim, year_outcomes),
            "xg_sim_brier": brier_score(year_xg_sim, year_outcomes),
        }
        logit_probs.extend(year_logit)
        elo_sim_probs.extend(year_elo_sim)
        xg_sim_probs.extend(year_xg_sim)
        outcomes.extend(year_outcomes)

    return logit_probs, elo_sim_probs, xg_sim_probs, outcomes, per_year


if __name__ == "__main__":
    logit_probs, elo_sim_probs, xg_sim_probs, outcomes, per_year = run()

    print("Leave-one-tournament-out: logistic vs. Elo-based goal-sim vs. xG-based goal-sim (2018/2022 only)\n")
    for year, stats in per_year.items():
        print(f"  {year}: n={stats['n']:2d} -- logistic={stats['logit_brier']:.4f}  "
              f"elo-goal-sim={stats['elo_sim_brier']:.4f}  xg-goal-sim={stats['xg_sim_brier']:.4f}")

    if logit_probs:
        logit_brier = brier_score(logit_probs, outcomes)
        elo_sim_brier = brier_score(elo_sim_probs, outcomes)
        xg_sim_brier = brier_score(xg_sim_probs, outcomes)
        l_lo, l_hi = bootstrap_brier_ci(logit_probs, outcomes)
        e_lo, e_hi = bootstrap_brier_ci(elo_sim_probs, outcomes)
        x_lo, x_hi = bootstrap_brier_ci(xg_sim_probs, outcomes)
        print(f"\nPooled (n={len(outcomes)}):")
        print(f"  Logistic:       Brier={logit_brier:.4f} (95% CI [{l_lo:.4f}, {l_hi:.4f}])")
        print(f"  Elo-goal-sim:   Brier={elo_sim_brier:.4f} (95% CI [{e_lo:.4f}, {e_hi:.4f}])")
        print(f"  xG-goal-sim:    Brier={xg_sim_brier:.4f} (95% CI [{x_lo:.4f}, {x_hi:.4f}])")
        best = min([("logistic", logit_brier), ("elo-goal-sim", elo_sim_brier), ("xg-goal-sim", xg_sim_brier)], key=lambda x: x[1])
        verdict = "beats" if xg_sim_brier < min(logit_brier, elo_sim_brier) else "does NOT beat"
        print(f"\n  xG-goal-sim {verdict} both other approaches on this sample (best overall: {best[0]}, Brier={best[1]:.4f}).")
