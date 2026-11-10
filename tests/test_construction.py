import datetime as dt
import math

import pytest

from harness.config import ConstructionConfig, CostConfig
from harness.construction import build_orders
from harness.models import PortfolioSnapshot, Position, Signal, Stock

TODAY = dt.date(2024, 3, 1)
COSTS = CostConfig(slippage_bps=5.0, commission_per_order=0.0)
FIVE = ConstructionConfig(max_positions=5, max_missing_sessions=5)


def _pos(symbol: str, qty: int, price: float = 100.0) -> Position:
    return Position(
        instrument=Stock(symbol=symbol),
        quantity=qty,
        avg_price=price,
        entry_date=dt.date(2024, 2, 1),
        entry_order_id=f"sim-{symbol}",
    )


def _snap(cash: float, positions: dict[str, Position] | None = None, equity: float | None = None):
    positions = positions or {}
    if equity is None:
        equity = cash + sum(p.quantity * p.avg_price for p in positions.values())
    return PortfolioSnapshot(as_of=TODAY, cash=cash, equity=equity, positions=positions)


def _kinds(result) -> list[tuple[str, str]]:
    return [(e.kind, e.symbol) for e in result.events]


def test_equal_weight_whole_shares_sized_at_close_plus_slippage() -> None:
    result = build_orders([Signal.enter("AAPL", 1.0)], _snap(10_000), {"AAPL": 150.0}, FIVE, COSTS)
    (order,) = result.orders
    assert order.action == "enter"
    assert order.symbol == "AAPL"
    assert order.created_on == TODAY
    assert order.quantity == math.floor(2_000 / (150.0 * 1.0005))  # 13
    assert order.legs[0].side == "buy"


def test_single_slot_invests_whole_account() -> None:
    one = ConstructionConfig(max_positions=1, max_missing_sessions=5)
    result = build_orders([Signal.enter("SPY", 1.0)], _snap(10_000), {"SPY": 400.0}, one, COSTS)
    assert result.orders[0].quantity == math.floor(10_000 / (400.0 * 1.0005))  # 24


def test_ranked_by_strength_then_symbol_and_capped_at_slots() -> None:
    signals = [
        Signal.enter("XOM", 0.5),
        Signal.enter("MSFT", 2.0),
        Signal.enter("JPM", 2.0),
        Signal.enter("AAPL", 3.0),
    ]
    closes = {s: 100.0 for s in ("XOM", "MSFT", "JPM", "AAPL")}
    held = {f"H{i}": _pos(f"H{i}", 1) for i in range(2)}  # 2 held, so 3 slots
    result = build_orders(signals, _snap(10_000, held), closes, FIVE, COSTS)
    assert [o.symbol for o in result.orders] == ["AAPL", "JPM", "MSFT"]
    assert ("entry_skipped_no_slot", "XOM") in _kinds(result)


def test_exit_held_sells_full_quantity_and_frees_slot() -> None:
    held = {f"H{i}": _pos(f"H{i}", 4) for i in range(5)}  # full
    signals = [Signal.exit("H0"), Signal.enter("AAPL", 1.0)]
    result = build_orders(signals, _snap(0.0, held), {"H0": 100.0, "AAPL": 100.0}, FIVE, COSTS)
    exit_order, entry_order = result.orders
    assert (exit_order.action, exit_order.symbol, exit_order.quantity) == ("exit", "H0", 4)
    assert exit_order.legs[0].side == "sell"
    # Entry is funded only by estimated exit proceeds: 4 * 100 * 0.9995 = 399.8.
    assert entry_order.symbol == "AAPL"
    assert entry_order.quantity == math.floor(399.8 / 100.05)  # 3


def test_exit_not_held_is_ignored_and_logged() -> None:
    result = build_orders([Signal.exit("AAPL", reason="stop")], _snap(1_000), {}, FIVE, COSTS)
    assert result.orders == ()
    assert result.events[0].kind == "exit_ignored_not_held"
    assert result.events[0].symbol == "AAPL"
    assert result.events[0].detail == "stop"
    assert result.events[0].date == TODAY


def test_entry_for_held_symbol_ignored_even_when_exiting_today() -> None:
    held = {"AAPL": _pos("AAPL", 5)}
    signals = [Signal.exit("AAPL"), Signal.enter("AAPL", 9.0)]
    result = build_orders(signals, _snap(1_000, held), {"AAPL": 100.0}, FIVE, COSTS)
    assert [(o.action, o.symbol) for o in result.orders] == [("exit", "AAPL")]
    assert result.events[0].kind == "entry_ignored_held"
    assert result.events[0].detail == "exiting today"


def test_duplicate_entries_keep_strongest() -> None:
    signals = [Signal.enter("AAPL", 1.0, "weak"), Signal.enter("AAPL", 2.0, "strong")]
    result = build_orders(signals, _snap(10_000), {"AAPL": 100.0}, FIVE, COSTS)
    (order,) = result.orders
    assert order.reason == "strong"
    assert order.strength == 2.0
    assert ("signal_duplicate", "AAPL") in _kinds(result)


def test_duplicate_exit_logged_once_sold_once() -> None:
    held = {"AAPL": _pos("AAPL", 5)}
    result = build_orders(
        [Signal.exit("AAPL"), Signal.exit("AAPL")], _snap(0, held), {"AAPL": 1.0}, FIVE, COSTS
    )
    assert len(result.orders) == 1
    assert _kinds(result) == [("signal_duplicate", "AAPL")]


def test_entry_without_close_skipped() -> None:
    result = build_orders([Signal.enter("META", 1.0)], _snap(10_000), {}, FIVE, COSTS)
    assert result.orders == ()
    assert _kinds(result) == [("entry_skipped_no_price", "META")]


def test_insufficient_cash_skips_without_using_slot() -> None:
    one = ConstructionConfig(max_positions=2, max_missing_sessions=5)
    signals = [Signal.enter("BIG", 2.0), Signal.enter("SMALL", 1.0)]
    # target = 1000 each; BIG at 5000 cannot buy one share, SMALL can.
    result = build_orders(signals, _snap(2_000), {"BIG": 5_000.0, "SMALL": 10.0}, one, COSTS)
    assert [o.symbol for o in result.orders] == ["SMALL"]
    assert ("entry_skipped_no_cash", "BIG") in _kinds(result)


def test_budget_caps_later_entries() -> None:
    # equity 10k but only 2.5k cash: first entry takes 2k, second gets the remaining ~0.5k.
    held = {"H0": _pos("H0", 75)}
    signals = [Signal.enter("A", 2.0), Signal.enter("B", 1.0)]
    result = build_orders(signals, _snap(2_500, held), {"A": 10.0, "B": 10.0}, FIVE, COSTS)
    a, b = result.orders
    assert a.quantity == math.floor(2_000 / 10.005)
    spent = a.quantity * 10.005
    assert b.quantity == math.floor((2_500 - spent) / 10.005)


def test_commission_included_in_allocation() -> None:
    costs = CostConfig(slippage_bps=0.0, commission_per_order=10.0)
    one = ConstructionConfig(max_positions=1, max_missing_sessions=5)
    result = build_orders([Signal.enter("SPY", 1.0)], _snap(1_000), {"SPY": 99.0}, one, costs)
    assert result.orders[0].quantity == 10  # 10 * 99 + 10 = 1000


def test_deterministic_regardless_of_signal_order() -> None:
    sigs = [Signal.enter(s, 1.0) for s in ("D", "B", "A", "C", "F", "E")]
    closes = {s.symbol: 50.0 for s in sigs}
    a = build_orders(sigs, _snap(10_000), closes, FIVE, COSTS)
    b = build_orders(list(reversed(sigs)), _snap(10_000), closes, FIVE, COSTS)
    assert [o.symbol for o in a.orders] == [o.symbol for o in b.orders] == list("ABCDE")


@pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
def test_nonpositive_or_nan_close_treated_as_no_price(bad: float) -> None:
    result = build_orders([Signal.enter("AAPL", 1.0)], _snap(10_000), {"AAPL": bad}, FIVE, COSTS)
    assert _kinds(result) == [("entry_skipped_no_price", "AAPL")]


def test_duplicate_exit_for_unheld_symbol_logged_as_duplicate() -> None:
    result = build_orders([Signal.exit("AAPL"), Signal.exit("AAPL")], _snap(1_000), {}, FIVE, COSTS)
    assert result.orders == ()
    assert _kinds(result) == [("exit_ignored_not_held", "AAPL"), ("signal_duplicate", "AAPL")]
