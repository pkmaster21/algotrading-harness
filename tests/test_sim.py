"""Acceptance: no fill ever uses a price from the bar that generated the signal.

Plus SimBroker's own rules: next-open fills, sells before buys, cash caps, cancellations,
and forced closes at a stale (never future) close.
"""

import datetime as dt

import pytest

from harness.broker.base import LookAheadError
from harness.broker.sim import SimBroker
from harness.config import CostConfig
from harness.engine import simulate
from harness.models import Leg, Order, Stock
from tests.fixture_strategies import Churn
from tests.synthetic import COSTS, construction, make_panel, price_lookup

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
D0, D1, D2 = dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)


def _order(symbol: str, side: str, qty: int, created_on: dt.date = D0) -> Order:
    action = "enter" if side == "buy" else "exit"
    return Order(
        created_on=created_on,
        action=action,
        legs=(Leg(instrument=Stock(symbol=symbol), side=side),),
        quantity=qty,
        strength=1.0 if action == "enter" else None,
    )


# --------------------------------------------------------------------------- acceptance


def test_no_fill_uses_a_price_from_its_signal_bar() -> None:
    panel = make_panel(SYMBOLS, 150, missing={"NVDA": range(60, 63), "META": range(100, 150)})
    result = simulate(Churn(), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS)
    price = price_lookup(panel)
    next_open = [f for f in result.fills if f.kind == "next_open"]
    assert len(next_open) > 200  # the test is only meaningful with many fills

    for fill in next_open:
        sym = fill.instrument.key
        signal_bar = {price(sym, fill.signal_date, f) for f in ("open", "high", "low", "close")}
        assert fill.fill_date > fill.signal_date
        assert fill.price_date == fill.fill_date
        assert fill.reference_price == price(sym, fill.fill_date, "open")
        assert fill.reference_price not in signal_bar
        assert result.orders[fill.order_id].created_on == fill.signal_date
        slip = 1 + 0.0005 if fill.side == "buy" else 1 - 0.0005
        assert fill.price == pytest.approx(fill.reference_price * slip)


def test_forced_close_uses_a_close_strictly_before_the_trigger() -> None:
    # AAPL stops trading after session 10; it is held from the open of session 1.
    panel = make_panel(["AAPL"], 30, missing={"AAPL": range(11, 30)})
    from tests.fixture_strategies import BuyAndHold

    result = simulate(
        BuyAndHold(BuyAndHold.Params(symbol="AAPL")),
        panel,
        SimBroker(10_000, COSTS),
        ["AAPL"],
        construction(max_positions=1, max_missing_sessions=5),
        COSTS,
    )
    (forced,) = [f for f in result.fills if f.kind == "forced_close"]
    price = price_lookup(panel)
    assert forced.price_date == panel.dates[10]  # last bar
    assert forced.fill_date == forced.signal_date == panel.dates[15]  # 5th missing session
    assert forced.reference_price == price("AAPL", panel.dates[10], "close")
    assert forced.price == pytest.approx(forced.reference_price * (1 - 0.0005))
    assert forced.price_date < forced.signal_date
    (event,) = [e for e in result.events if e.kind == "position_closed_no_data"]
    assert event.symbol == "AAPL" and event.date == panel.dates[15]
    assert result.final_positions == {}
    assert result.equity[-1].cash == pytest.approx(result.equity[-1].equity)


def test_missing_counter_resets_when_bars_return() -> None:
    panel = make_panel(["AAPL"], 30, missing={"AAPL": [5, 6, 7, 8, 10, 11, 12, 13]})
    from tests.fixture_strategies import BuyAndHold

    result = simulate(
        BuyAndHold(BuyAndHold.Params(symbol="AAPL")),
        panel,
        SimBroker(10_000, COSTS),
        ["AAPL"],
        construction(max_positions=1, max_missing_sessions=5),
        COSTS,
    )
    assert not [f for f in result.fills if f.kind == "forced_close"]
    assert "AAPL" in result.final_positions


# --------------------------------------------------------------------------- SimBroker unit


def test_fill_in_signal_session_raises() -> None:
    broker = SimBroker(10_000, COSTS)
    broker.submit([_order("AAPL", "buy", 1, created_on=D1)])
    with pytest.raises(LookAheadError):
        broker.on_session_open(D1, {"AAPL": 100.0})


def test_fill_before_signal_session_raises() -> None:
    broker = SimBroker(10_000, COSTS)
    broker.submit([_order("AAPL", "buy", 1, created_on=D2)])
    with pytest.raises(LookAheadError):
        broker.on_session_open(D1, {"AAPL": 100.0})


def test_sessions_must_advance() -> None:
    broker = SimBroker(10_000, COSTS)
    broker.on_session_open(D1, {})
    with pytest.raises(ValueError):
        broker.on_session_open(D1, {})


def test_buy_fills_at_open_plus_slippage() -> None:
    broker = SimBroker(10_000, COSTS)
    (oid,) = broker.submit([_order("AAPL", "buy", 10)])
    broker.on_session_open(D1, {"AAPL": 200.0})
    (update,) = broker.poll_updates()
    (fill,) = update.fills
    assert (update.order_id, update.status, update.reason) == (oid, "filled", "")
    assert fill.price == pytest.approx(200.1)
    assert broker.account().cash == pytest.approx(10_000 - 2001.0)
    assert broker.account().positions["AAPL"].quantity == 10
    assert broker.poll_updates() == []


def test_no_open_cancels_order() -> None:
    broker = SimBroker(10_000, COSTS)
    broker.submit([_order("AAPL", "buy", 10)])
    broker.on_session_open(D1, {"MSFT": 50.0})
    (update,) = broker.poll_updates()
    assert (update.status, update.reason) == ("canceled", "no_bar")
    broker.on_session_open(D2, {"AAPL": 50.0})  # day order: not retried
    assert broker.poll_updates() == []


def test_sells_execute_before_buys_and_fund_them() -> None:
    broker = SimBroker(1_000, CostConfig(slippage_bps=0.0, commission_per_order=0.0))
    broker.submit([_order("AAPL", "buy", 10)])
    broker.on_session_open(D1, {"AAPL": 100.0})  # all cash into AAPL
    broker.poll_updates()
    broker.submit([_order("MSFT", "buy", 5, created_on=D1), _order("AAPL", "sell", 10, D1)])
    broker.on_session_open(D2, {"AAPL": 110.0, "MSFT": 200.0})
    sell, buy = broker.poll_updates()
    assert sell.fills[0].side == "sell" and buy.fills[0].side == "buy"
    assert buy.reason == "" and buy.fills[0].quantity == 5
    assert broker.account().cash == pytest.approx(1_100 - 1_000)


def test_gap_up_reduces_buy_to_affordable_whole_shares() -> None:
    broker = SimBroker(1_000, COSTS)
    broker.submit([_order("AAPL", "buy", 10)])  # sized for ~100, opens at 120
    broker.on_session_open(D1, {"AAPL": 120.0})
    (update,) = broker.poll_updates()
    assert update.reason == "reduced_for_cash"
    assert update.fills[0].quantity == 8  # 8 * 120.06 = 960.48
    assert broker.account().cash >= 0


def test_cannot_afford_one_share_cancels() -> None:
    broker = SimBroker(100, COSTS)
    broker.submit([_order("AAPL", "buy", 1)])
    broker.on_session_open(D1, {"AAPL": 150.0})
    (update,) = broker.poll_updates()
    assert (update.status, update.reason) == ("canceled", "insufficient_cash")
    assert broker.account().cash == 100


def test_commission_counts_against_cash() -> None:
    broker = SimBroker(1_000, CostConfig(slippage_bps=0.0, commission_per_order=5.0))
    broker.submit([_order("AAPL", "buy", 10)])
    broker.on_session_open(D1, {"AAPL": 100.0})
    (update,) = broker.poll_updates()
    assert update.fills[0].quantity == 9  # 10 * 100 + 5 > 1000
    assert broker.account().cash == pytest.approx(1_000 - 905)


def test_sell_not_held_is_canceled() -> None:
    broker = SimBroker(1_000, COSTS)
    broker.submit([_order("AAPL", "sell", 3)])
    broker.on_session_open(D1, {"AAPL": 100.0})
    (update,) = broker.poll_updates()
    assert (update.status, update.reason) == ("canceled", "not_held")


def test_buy_for_held_symbol_raises() -> None:
    broker = SimBroker(10_000, COSTS)
    broker.submit([_order("AAPL", "buy", 1)])
    broker.on_session_open(D1, {"AAPL": 100.0})
    broker.submit([_order("AAPL", "buy", 1, created_on=D1)])
    with pytest.raises(ValueError, match="pyramiding"):
        broker.on_session_open(D2, {"AAPL": 100.0})


def test_force_close_requires_a_held_symbol_and_a_stale_price() -> None:
    broker = SimBroker(10_000, COSTS)
    with pytest.raises(ValueError):
        broker.force_close("AAPL", 100.0, D0, D2)
    broker.submit([_order("AAPL", "buy", 1)])
    broker.on_session_open(D1, {"AAPL": 100.0})
    with pytest.raises(ValueError):  # price from the trigger session itself: rejected by Fill
        broker.force_close("AAPL", 100.0, D2, D2)


def test_option_orders_not_supported() -> None:
    from harness.models import OptionContract

    call = OptionContract(underlying="AAPL", expiry=D2, strike=100, right="call")
    order = Order(created_on=D0, action="enter", legs=(Leg(instrument=call, side="buy"),),
                  quantity=1, strength=1.0)  # fmt: skip
    with pytest.raises(NotImplementedError):
        SimBroker(10_000, COSTS).submit([order])


def test_cancel_all_reports_reason() -> None:
    broker = SimBroker(10_000, COSTS)
    ids = broker.submit([_order("AAPL", "buy", 1), _order("MSFT", "buy", 1)])
    assert broker.cancel_all("end_of_data") == ids
    assert [(u.order_id, u.reason) for u in broker.poll_updates()] == [
        (ids[0], "end_of_data"), (ids[1], "end_of_data"),
    ]  # fmt: skip
