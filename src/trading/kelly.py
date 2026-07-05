"""Binary-contract Kelly staking.

For a Kalshi YES contract priced at c (a $1 payout on success), buying one
contract is mathematically the same bet as decimal odds d = 1/c. Kelly's
formula for decimal odds d and true probability p is f* = (p*d - 1) / (d - 1),
which simplifies to the form below when expressed directly in terms of the
contract price c.
"""


def kelly_fraction(p: float, price: float) -> float:
    """Optimal fraction of bankroll to stake on YES at price `price` given
    model probability `p`. Negative values (no edge) are clipped to 0 --
    a negative fraction here just means "buy NO instead", which the caller
    should evaluate as its own separate (1-p) vs (1-price) case."""
    if price <= 0 or price >= 1:
        return 0.0
    f = (p - price) / (1 - price)
    return max(f, 0.0)


def fractional_kelly_stake(p: float, price: float, bankroll: float,
                            fraction: float = 0.25, max_stake_pct: float = 0.05) -> float:
    """Dollar stake using a fraction of full Kelly, hard-capped as a percent of
    bankroll. Fractional Kelly (not full Kelly) is the default because Kelly
    sizing is extremely sensitive to probability error, and this model has not
    yet been validated against historical World Cups (see Milestone B)."""
    f_full = kelly_fraction(p, price)
    f_applied = min(f_full * fraction, max_stake_pct)
    return bankroll * f_applied
