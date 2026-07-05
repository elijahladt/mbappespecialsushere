import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trading.devig import implied_prob, devig_proportional


def test_implied_prob():
    assert abs(implied_prob(2.0) - 0.5) < 1e-9
    assert abs(implied_prob(4.0) - 0.25) < 1e-9


def test_devig_removes_overround():
    fair_a, fair_b = devig_proportional(1.90, 1.90)
    assert abs(fair_a - 0.5) < 1e-9
    assert abs(fair_b - 0.5) < 1e-9
    assert abs((fair_a + fair_b) - 1.0) < 1e-9


def test_devig_preserves_relative_odds():
    fair_a, fair_b = devig_proportional(1.44, 2.75)  # Brazil vs Norway
    assert fair_a > fair_b
    assert abs((fair_a + fair_b) - 1.0) < 1e-9
