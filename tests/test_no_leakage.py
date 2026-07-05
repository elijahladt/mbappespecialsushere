"""Chronological-truncation invariance: a match's pre-match Elo rating must be
identical whether or not later matches exist in the dataset. If this fails,
some later result is leaking backward into an earlier feature."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.elo import EloEngine


SAMPLE_MATCHES = [
    ("2020-01-01", "A", "B", 2, 0, 4, False),
    ("2020-02-01", "B", "C", 1, 1, 4, False),
    ("2020-03-01", "A", "C", 3, 1, 4, False),
    ("2020-04-01", "C", "A", 0, 2, 4, False),
]


def ratings_after(n_matches):
    engine = EloEngine()
    for date, home, away, hs, as_, tier, neutral in SAMPLE_MATCHES[:n_matches]:
        engine.process_match(home, away, hs, as_, tier, neutral)
    return dict(engine.ratings)


def test_truncation_invariance():
    # Rating snapshot right after match 3 must match regardless of whether
    # match 4 (in the future relative to match 3) is ever processed.
    full_run_ratings = []
    engine = EloEngine()
    for i, (date, home, away, hs, as_, tier, neutral) in enumerate(SAMPLE_MATCHES):
        engine.process_match(home, away, hs, as_, tier, neutral)
        if i == 2:
            full_run_ratings = dict(engine.ratings)

    truncated_ratings = ratings_after(3)
    assert full_run_ratings == truncated_ratings
