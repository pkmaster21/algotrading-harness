"""Transaction cost model: flat slippage in basis points per side, plus a flat commission.

SimBroker uses `apply_costs` to price fills and construction uses it to estimate the
cost of an entry when sizing, so both apply the same formula.
"""

from __future__ import annotations

from pydantic import Field

from harness.config import CostConfig
from harness.models import FrozenModel, Side


class CostBreakdown(FrozenModel):
    fill_price: float = Field(gt=0)
    slippage_cost: float = Field(ge=0)  # |fill_price - reference_price| * quantity * multiplier
    commission: float = Field(ge=0)


def apply_costs(
    side: Side,
    reference_price: float,
    quantity: int,
    cfg: CostConfig,
    multiplier: int = 1,
) -> CostBreakdown:
    """Price a trade at `reference_price` (the fill bar's open, or a sizing estimate).

    A buy fills at `ref * (1 + bps / 10_000)` and a sell at `ref * (1 - bps / 10_000)`,
    so slippage always works against the trader. Commission is charged once per order
    with nonzero quantity. Pure function: no state, no I/O.
    """
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")
    if quantity < 0:
        raise ValueError(f"quantity must be non-negative, got {quantity}")
    rate = cfg.slippage_bps / 10_000
    fill_price = reference_price * (1 + rate) if side == "buy" else reference_price * (1 - rate)
    return CostBreakdown(
        fill_price=fill_price,
        slippage_cost=abs(fill_price - reference_price) * quantity * multiplier,
        commission=cfg.commission_per_order if quantity > 0 else 0.0,
    )
