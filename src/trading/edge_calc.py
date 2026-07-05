"""Edge = model's fair probability minus the market's quoted probability."""


def edge(model_prob: float, market_price: float) -> float:
    return model_prob - market_price


def ev_per_dollar(model_prob: float, market_price: float) -> float:
    """Expected value per $1 staked buying a YES contract at market_price."""
    if market_price <= 0:
        return 0.0
    return (model_prob / market_price) - 1
