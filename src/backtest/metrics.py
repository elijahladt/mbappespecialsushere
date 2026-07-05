import numpy as np


def brier_score(probs, outcomes) -> float:
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def bootstrap_brier_ci(probs, outcomes, n_boot: int = 2000, seed: int = 42):
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = brier_score(probs[idx], outcomes[idx])
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def multiclass_brier_score(prob_triples, outcome_labels) -> float:
    """Multi-class (3-way) Brier score: mean squared error between the
    predicted probability vector and the one-hot outcome vector, summed
    across classes per sample. prob_triples entries and outcome_labels use
    the same (away=0, draw=1, home=2) class ordering as club_winprob_link.
    Reduces to the standard binary Brier score when there are 2 classes."""
    probs = np.asarray(prob_triples, dtype=float)
    n_classes = probs.shape[1]
    one_hot = np.zeros_like(probs)
    for i, label in enumerate(outcome_labels):
        one_hot[i, label] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_buckets(probs, outcomes, n_buckets: int = 5):
    """Bucket predictions and compare mean predicted prob to realized win rate --
    a calibration sanity check (not just a single scalar score)."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, n_buckets + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bucket": f"[{lo:.1f}, {hi:.1f})",
            "n": n,
            "mean_predicted": float(probs[mask].mean()),
            "realized_rate": float(outcomes[mask].mean()),
        })
    return rows
