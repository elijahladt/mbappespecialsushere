import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trading.kelly import kelly_fraction, fractional_kelly_stake
from src.trading.edge_calc import edge, ev_per_dollar


def test_kelly_hand_computed():
    assert abs(kelly_fraction(0.6, 0.5) - 0.2) < 1e-9


def test_kelly_no_edge_clipped_to_zero():
    assert kelly_fraction(0.4, 0.5) == 0.0


def test_kelly_full_edge_case():
    # p=1, any price<1 -> should bet everything (f*=1)
    assert abs(kelly_fraction(1.0, 0.5) - 1.0) < 1e-9


def test_fractional_kelly_respects_max_stake_cap():
    stake = fractional_kelly_stake(p=0.99, price=0.5, bankroll=1000, fraction=1.0, max_stake_pct=0.05)
    assert stake == 50.0  # capped at 5% of bankroll despite huge edge


def test_edge_and_ev():
    assert abs(edge(0.6, 0.5) - 0.1) < 1e-9
    assert abs(ev_per_dollar(0.6, 0.5) - 0.2) < 1e-9
