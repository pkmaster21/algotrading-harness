"""Acceptance: buy SPY on the first bar and hold, with max_positions 1.

The engine uses whole shares. Final equity must equal
    shares * last_close + leftover_cash
where shares and leftover_cash come from the first fill (next-day open plus slippage).
"""

import numpy as np
import pytest

from harness.broker.sim import SimBroker
from harness.engine import simulate
from tests.fixture_strategies import BuyAndHold
from tests.synthetic import COSTS, construction, make_panel

STARTING_CASH = 10_000.0


def _wavy_panel(n: int = 250, gap_up: float = 0.0):
    t = np.arange(n, dtype=float)
    closes = 400 + 30 * np.sin(t / 15) + 0.2 * t
    opens = np.concatenate([[closes[0]], closes[:-1] + 0.7 * np.cos(t[1:])])
    opens[1] += gap_up  # first fill session
    return make_panel(["SPY"], n, opens={"SPY": opens}, closes={"SPY": closes})


@pytest.mark.parametrize("gap_up", [0.0, 25.0], ids=["normal open", "gap up past cash"])
def test_final_equity_is_shares_times_last_close_plus_leftover_cash(gap_up: float) -> None:
    panel = _wavy_panel(gap_up=gap_up)
    result = simulate(
        BuyAndHold(),
        panel,
        SimBroker(STARTING_CASH, COSTS),
        ["SPY"],
        construction(max_positions=1),
        COSTS,
    )
    (fill,) = result.fills
    assert fill.signal_date == panel.dates[0]
    assert fill.fill_date == panel.dates[1]
    assert fill.reference_price == panel.field("open")[1, 0]
    assert fill.price == pytest.approx(fill.reference_price * 1.0005)
    assert isinstance(fill.quantity, int)  # whole shares

    shares = fill.quantity
    leftover_cash = STARTING_CASH - shares * fill.price - fill.commission
    last_close = float(panel.field("close")[-1, 0])
    assert 0 <= leftover_cash < fill.price + 0.5  # the whole account is invested (< 1 share left)
    assert result.equity[-1].equity == pytest.approx(shares * last_close + leftover_cash, abs=0.01)
    assert result.final_positions["SPY"].quantity == shares


def test_gap_up_shrinks_the_order_instead_of_borrowing() -> None:
    panel = _wavy_panel(gap_up=25.0)
    result = simulate(
        BuyAndHold(),
        panel,
        SimBroker(STARTING_CASH, COSTS),
        ["SPY"],
        construction(max_positions=1),
        COSTS,
    )
    (update,) = [u for u in result.updates if u.fills]
    assert update.reason == "reduced_for_cash"
    assert result.orders[update.order_id].quantity > update.fills[0].quantity
    assert all(p.cash >= 0 for p in result.equity)
