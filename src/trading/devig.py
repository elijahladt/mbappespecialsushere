"""De-vig bookmaker odds. Unlike Kalshi (where the quoted price already is
close to a fair probability), a sportsbook price bakes in a margin/overround:
both sides' implied probabilities sum to MORE than 100%. The de-vigged
number is the model's best estimate of the sportsbook's true belief, useful
for comparison; the RAW implied probability is what you actually pay, so
edge/Kelly should be computed against the raw price, not the de-vigged one.
"""


def implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def devig_proportional(decimal_odds_a: float, decimal_odds_b: float):
    """Proportional (multiplicative) de-vig: scale each side's raw implied
    probability down by the total overround. Returns (fair_prob_a, fair_prob_b)."""
    raw_a, raw_b = implied_prob(decimal_odds_a), implied_prob(decimal_odds_b)
    overround = raw_a + raw_b
    return raw_a / overround, raw_b / overround
