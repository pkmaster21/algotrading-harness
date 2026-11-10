import pytest

from harness.config import CostConfig
from harness.costs import apply_costs

FIVE_BPS = CostConfig(slippage_bps=5.0, commission_per_order=0.0)


def test_buy_pays_slippage_up() -> None:
    c = apply_costs("buy", 100.0, 10, FIVE_BPS)
    assert c.fill_price == pytest.approx(100.05)
    assert c.slippage_cost == pytest.approx(0.5)
    assert c.commission == 0.0


def test_sell_pays_slippage_down() -> None:
    c = apply_costs("sell", 200.0, 3, FIVE_BPS)
    assert c.fill_price == pytest.approx(199.9)
    assert c.slippage_cost == pytest.approx(0.3)


def test_commission_once_per_nonzero_order() -> None:
    cfg = CostConfig(slippage_bps=0.0, commission_per_order=1.25)
    assert apply_costs("buy", 50.0, 7, cfg).commission == 1.25
    assert apply_costs("buy", 50.0, 0, cfg).commission == 0.0


def test_multiplier_scales_slippage_cost() -> None:
    c = apply_costs("buy", 2.0, 1, FIVE_BPS, multiplier=100)
    assert c.slippage_cost == pytest.approx(0.1)


@pytest.mark.parametrize("price,qty", [(0.0, 1), (-1.0, 1), (10.0, -1)])
def test_rejects_bad_inputs(price: float, qty: int) -> None:
    with pytest.raises(ValueError):
        apply_costs("buy", price, qty, FIVE_BPS)
