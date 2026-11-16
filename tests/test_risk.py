"""Risk manager: each limit, the session mark, and the two rules that must never break
(a risk-reducing sell always passes; an approved buy never breaches a limit)."""

from __future__ import annotations

import datetime as dt
import random
from collections.abc import Mapping

import pytest

from harness.broker.sim import SimBroker
from harness.config import OrderLimits, RiskConfig
from harness.engine import simulate
from harness.models import (
    AccountState,
    Action,
    Leg,
    OptionContract,
    Order,
    PortfolioSnapshot,
    Position,
    Stock,
)
from harness.risk import RiskDecision, RiskManager, RiskState, RiskStateError, _whole_units
from tests.fixture_strategies import BuyAndHold, Churn
from tests.synthetic import COSTS, construction, make_panel

D0, D1, D2, D3 = (dt.date(2024, 1, d) for d in (2, 3, 4, 5))
LOOSE = OrderLimits(max_order_notional=1e9, max_orders_per_session=1000)


def cfg(
    position: float = 1.0, exposure: float = 1.0, daily: float = 0.03, drawdown: float = 0.15
) -> RiskConfig:
    return RiskConfig(
        name="test",
        max_position_pct=position,
        max_exposure_pct=exposure,
        daily_loss_cap_pct=daily,
        max_drawdown_halt_pct=drawdown,
    )


def manager(
    config: RiskConfig | None = None,
    limits: OrderLimits = LOOSE,
    markup: float = 0.0,
    equity: float = 10_000.0,
    marked: dt.date | None = D0,
) -> RiskManager:
    """A manager already marked for D0 at `equity` (unless `marked` is None)."""
    m = RiskManager(config or cfg(), limits, markup, RiskState.initial(equity))
    if marked is not None:
        m.on_mark(marked, equity)
    return m


def portfolio(
    cash: float, holdings: Mapping[str, tuple[int, float]] | None = None, as_of: dt.date = D0
) -> tuple[PortfolioSnapshot, dict[str, float]]:
    """A snapshot and its marks from {symbol: (quantity, last close)}."""
    holdings = holdings or {}
    positions = {
        s: Position(
            instrument=Stock(symbol=s),
            quantity=q,
            avg_price=mark,
            entry_date=D0 - dt.timedelta(days=30),
            entry_order_id="e",
        )
        for s, (q, mark) in holdings.items()
    }
    equity = cash + sum(q * mark for q, mark in holdings.values())
    snapshot = PortfolioSnapshot(as_of=as_of, cash=cash, equity=equity, positions=positions)
    return snapshot, {s: mark for s, (_, mark) in holdings.items()}


def buy(symbol: str, quantity: int, created_on: dt.date = D0) -> Order:
    return _order(symbol, "enter", quantity, created_on)


def sell(symbol: str, quantity: int, created_on: dt.date = D0) -> Order:
    return _order(symbol, "exit", quantity, created_on)


def _order(symbol: str, action: Action, quantity: int, created_on: dt.date) -> Order:
    return Order(
        created_on=created_on,
        action=action,
        legs=(Leg(instrument=Stock(symbol=symbol), side="buy" if action == "enter" else "sell"),),
        quantity=quantity,
        strength=1.0 if action == "enter" else None,
    )


def approved(review) -> list[tuple[str, int]]:  # type: ignore[no-untyped-def]
    return [(o.symbol, o.quantity) for o in review.orders]


def rules(review) -> list[str | None]:  # type: ignore[no-untyped-def]
    return [d.rule for d in review.decisions]


# --------------------------------------------------------------------------- buys: each limit


def test_order_within_every_limit_passes_untouched() -> None:
    snapshot, marks = portfolio(10_000)
    order = buy("AAPL", 20)
    review = manager(cfg(position=0.25)).review([order], snapshot, {**marks, "AAPL": 100.0})
    assert review.orders == (order,)  # the same object: nothing rewritten
    assert review.events == ()
    assert rules(review) == [None]


def test_position_limit_shrinks_to_the_largest_quantity_that_fits() -> None:
    snapshot, _ = portfolio(10_000)
    review = manager(cfg(position=0.25)).review([buy("AAPL", 30)], snapshot, {"AAPL": 100.0})
    assert approved(review) == [("AAPL", 25)]  # 25% of 10,000 at $100
    assert rules(review) == ["max_position"]
    (event,) = review.events
    assert (event.kind, event.symbol, event.date) == ("risk_shrunk", "AAPL", D0)
    assert "max_position" in event.detail


def test_limits_are_measured_at_the_worst_case_entry_price() -> None:
    snapshot, _ = portfolio(10_000)
    m = manager(cfg(position=0.25), markup=0.02)  # judged at $102, not $100
    review = m.review([buy("AAPL", 30)], snapshot, {"AAPL": 100.0})
    assert approved(review) == [("AAPL", 24)]  # 24 * 102 = 2448 <= 2500 < 25 * 102


def test_exposure_limit_counts_what_is_already_held() -> None:
    snapshot, marks = portfolio(4_000, {"MSFT": (60, 100.0)})  # 60% invested
    review = manager(cfg(position=0.80, exposure=0.80)).review(
        [buy("AAPL", 30)], snapshot, {**marks, "AAPL": 100.0}
    )
    assert approved(review) == [("AAPL", 20)]
    assert rules(review) == ["max_exposure"]


def test_earlier_entries_use_up_exposure_for_later_ones() -> None:
    snapshot, _ = portfolio(10_000)
    review = manager(cfg(position=0.40, exposure=0.50)).review(
        [buy("AAPL", 40), buy("MSFT", 40), buy("NVDA", 40)],
        snapshot,
        {"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0},
    )
    assert approved(review) == [("AAPL", 40), ("MSFT", 10)]
    assert rules(review) == [None, "max_exposure", "max_exposure"]
    assert [e.kind for e in review.events] == ["risk_shrunk", "risk_rejected"]


@pytest.mark.parametrize("orders_reversed", [False, True])
def test_exits_are_credited_before_entries_wherever_they_appear(orders_reversed: bool) -> None:
    # 80% in MSFT. Selling it frees the room the AAPL entry needs.
    snapshot, marks = portfolio(2_000, {"MSFT": (80, 100.0)})
    orders = [sell("MSFT", 80), buy("AAPL", 50)]
    if orders_reversed:
        orders.reverse()
    review = manager(cfg(position=0.50)).review(orders, snapshot, {**marks, "AAPL": 100.0})
    assert sorted(approved(review)) == [("AAPL", 50), ("MSFT", 80)]
    assert [o.symbol for o in review.orders] == [o.symbol for o in orders]  # input order kept
    assert review.events == ()


def test_exit_without_a_mark_is_approved_but_not_credited() -> None:
    snapshot, _ = portfolio(2_000, {"MSFT": (80, 100.0)})
    review = manager().review([sell("MSFT", 80), buy("AAPL", 50)], snapshot, {"AAPL": 100.0})
    assert approved(review) == [("MSFT", 80), ("AAPL", 20)]  # only the 2,000 of room there was


def test_order_notional_limit() -> None:
    snapshot, _ = portfolio(10_000)
    limits = OrderLimits(max_order_notional=1_000.0, max_orders_per_session=100)
    review = manager(limits=limits).review([buy("AAPL", 30)], snapshot, {"AAPL": 100.0})
    assert approved(review) == [("AAPL", 10)]
    assert rules(review) == ["max_order_notional"]


def test_order_count_limit_rejects_entries_but_never_exits() -> None:
    snapshot, marks = portfolio(
        4_000, {"MSFT": (20, 100.0), "NVDA": (20, 100.0), "AMZN": (20, 100.0)}
    )
    limits = OrderLimits(max_order_notional=1e9, max_orders_per_session=2)
    review = manager(limits=limits).review(
        [buy("AAPL", 5), sell("MSFT", 20), sell("NVDA", 20), sell("AMZN", 20)],
        snapshot,
        {**marks, "AAPL": 100.0},
    )
    assert approved(review) == [("MSFT", 20), ("NVDA", 20), ("AMZN", 20)]
    assert rules(review) == ["max_orders", None, None, None]

    review = manager(limits=limits).review(
        [sell("MSFT", 20), buy("AAPL", 5), buy("JPM", 5)],
        snapshot,
        {**marks, "AAPL": 100.0, "JPM": 100.0},
    )
    assert approved(review) == [("MSFT", 20), ("AAPL", 5)]
    assert rules(review) == [None, None, "max_orders"]


def test_buy_without_a_price_is_rejected() -> None:
    snapshot, _ = portfolio(10_000)
    for marks in ({}, {"AAPL": float("nan")}, {"AAPL": 0.0}):
        review = manager().review([buy("AAPL", 5)], snapshot, marks)
        assert review.orders == () and rules(review) == ["no_price"]
        assert [e.kind for e in review.events] == ["risk_rejected"]


def test_buy_is_rejected_when_equity_is_not_positive() -> None:
    snapshot = PortfolioSnapshot(as_of=D0, cash=-5.0, equity=-5.0)
    m = manager(equity=10_000.0, marked=None)
    m.on_mark(D0, -5.0)
    review = m.review([buy("AAPL", 1)], snapshot, {"AAPL": 100.0})
    assert review.orders == ()


def test_adding_to_a_held_position_counts_the_shares_already_held() -> None:
    snapshot, marks = portfolio(8_000, {"AAPL": (20, 100.0)})
    review = manager(cfg(position=0.25)).review([buy("AAPL", 20)], snapshot, marks)
    assert approved(review) == [("AAPL", 5)]  # 20 held + 5 = 25% of 10,000


# --------------------------------------------------------------------------- sells always pass


def test_sell_of_the_held_quantity_passes_in_every_state() -> None:
    snapshot, marks = portfolio(0.0, {"AAPL": (100, 100.0)}, as_of=D1)
    limits = OrderLimits(max_order_notional=1.0, max_orders_per_session=1)
    m = RiskManager(cfg(position=0.01, exposure=0.01), limits, 0.0, RiskState.initial(100_000.0))
    m.on_mark(D0, 100_000.0)
    events = m.on_mark(D1, 10_000.0)  # a 90% loss: daily cap and drawdown halt both trip
    assert [e.kind for e in events] == ["entries_halted", "entries_halted"]
    assert m.state.halted and m.state.entries_blocked_for == D1

    order = sell("AAPL", 100, created_on=D1)
    review = m.review([order, buy("MSFT", 1, created_on=D1)], snapshot, {**marks, "MSFT": 1.0})
    assert review.orders == (order,)
    assert rules(review) == [None, "drawdown_halt"]


def test_sell_larger_than_the_position_is_cut_and_cannot_open_a_short() -> None:
    snapshot, marks = portfolio(0.0, {"AAPL": (10, 100.0)})
    review = manager().review([sell("AAPL", 15)], snapshot, marks)
    assert approved(review) == [("AAPL", 10)]
    assert rules(review) == ["not_held"]
    assert [e.kind for e in review.events] == ["risk_shrunk"]


def test_sell_of_a_symbol_not_held_is_rejected() -> None:
    snapshot, marks = portfolio(10_000)
    review = manager().review([sell("AAPL", 5)], snapshot, marks)
    assert review.orders == () and rules(review) == ["not_held"]


def test_two_sells_of_one_symbol_cannot_sell_it_twice() -> None:
    snapshot, marks = portfolio(0.0, {"AAPL": (10, 100.0)})
    review = manager().review([sell("AAPL", 6), sell("AAPL", 6), sell("AAPL", 6)], snapshot, marks)
    assert approved(review) == [("AAPL", 6), ("AAPL", 4)]
    assert rules(review) == [None, "not_held", "not_held"]


# --------------------------------------------------------------------------- daily cap and halt


def test_daily_loss_cap_blocks_that_sessions_entries_only() -> None:
    m = manager(cfg(daily=0.03, drawdown=0.50))
    events = m.on_mark(D1, 9_700.0)  # exactly 3% down
    assert [e.kind for e in events] == ["entries_halted"]
    assert "daily_loss_cap" in events[0].detail
    snapshot, _ = portfolio(9_700.0, as_of=D1)
    review = m.review([buy("AAPL", 1, created_on=D1)], snapshot, {"AAPL": 100.0})
    assert rules(review) == ["daily_loss_cap"]

    assert m.on_mark(D2, 9_650.0) == ()  # a small loss the next day: entries are back
    assert m.state.entries_blocked_for is None
    snapshot, _ = portfolio(9_650.0, as_of=D2)
    review = m.review([buy("AAPL", 1, created_on=D2)], snapshot, {"AAPL": 100.0})
    assert approved(review) == [("AAPL", 1)]


def test_a_loss_or_drawdown_exactly_at_the_limit_trips_it() -> None:
    # Numbers chosen to be exact in binary, so "exactly at the limit" is really exact.
    m = manager(cfg(daily=0.03125, drawdown=0.5), equity=1024.0)
    events = m.on_mark(D1, 992.0)  # 1 - 992 / 1024 == 0.03125
    assert [e.detail.split(":")[0] for e in events] == ["daily_loss_cap"]

    m = manager(cfg(daily=0.99, drawdown=0.125), equity=1024.0)
    events = m.on_mark(D1, 896.0)  # 1 - 896 / 1024 == 0.125
    assert [e.detail.split(":")[0] for e in events] == ["drawdown_halt"]


def test_loss_just_under_the_daily_cap_does_not_trip_it() -> None:
    m = manager(cfg(daily=0.03, drawdown=0.50))
    assert m.on_mark(D1, 9_701.0) == ()
    assert m.state.entries_blocked_for is None


def test_drawdown_halt_trips_once_and_lasts_until_reset() -> None:
    m = manager(cfg(daily=0.99, drawdown=0.15))
    assert m.on_mark(D1, 11_000.0) == ()  # new peak
    assert m.on_mark(D2, 9_500.0) == ()  # 13.6% below the peak
    events = m.on_mark(D3, 9_350.0)  # exactly 15% below 11,000
    assert [e.kind for e in events] == ["entries_halted"]
    assert "drawdown_halt" in events[0].detail
    assert m.state.halted and m.state.peak_equity == 11_000.0

    d4, d5 = D3 + dt.timedelta(days=3), D3 + dt.timedelta(days=4)
    assert m.on_mark(d4, 12_000.0) == ()  # recovering does not lift the halt
    assert m.state.halted
    snapshot, _ = portfolio(12_000.0, as_of=d4)
    review = m.review([buy("AAPL", 1, created_on=d4)], snapshot, {"AAPL": 100.0})
    assert rules(review) == ["drawdown_halt"]

    m.reset_halt()
    assert not m.state.halted and m.state.halt_reason == ""
    review = m.review([buy("AAPL", 1, created_on=d4)], snapshot, {"AAPL": 100.0})
    assert approved(review) == [("AAPL", 1)]
    assert m.on_mark(d5, 11_900.0) == ()  # and it does not trip again straight away


def test_reset_rebases_the_peak_so_the_halt_does_not_return_at_once() -> None:
    m = manager(cfg(daily=0.99, drawdown=0.15))
    m.on_mark(D1, 8_000.0)
    assert m.state.halted
    m.reset_halt()
    assert m.state.peak_equity == 8_000.0
    assert m.on_mark(D2, 7_900.0) == ()
    assert not m.state.halted
    with pytest.raises(RiskStateError, match="no drawdown halt"):
        m.reset_halt()


def test_first_mark_has_no_daily_loss_but_can_halt_on_drawdown() -> None:
    m = manager(cfg(daily=0.01, drawdown=0.15), equity=10_000.0, marked=None)
    events = m.on_mark(D0, 8_000.0)
    assert [e.detail.split(":")[0] for e in events] == ["drawdown_halt"]


def test_repeated_mark_for_a_session_changes_nothing() -> None:
    m = manager(cfg(daily=0.03, drawdown=0.50))
    m.on_mark(D1, 9_000.0)
    state = m.state
    assert state.entries_blocked_for == D1
    assert m.on_mark(D1, 9_000.0) == ()  # a rerun after a crash must not clear the block
    assert m.state == state


def test_marks_out_of_order_raise() -> None:
    m = manager()
    m.on_mark(D1, 10_000.0)
    with pytest.raises(RiskStateError, match="different equity"):
        m.on_mark(D1, 9_999.0)
    with pytest.raises(RiskStateError, match="after"):
        m.on_mark(D0, 10_000.0)
    with pytest.raises(RiskStateError, match="not a number"):
        m.on_mark(D2, float("nan"))


def test_review_needs_the_mark_for_its_session() -> None:
    snapshot, _ = portfolio(10_000, as_of=D1)
    with pytest.raises(RiskStateError, match="without a mark"):
        manager().review([], snapshot, {})
    snapshot, _ = portfolio(10_000)
    with pytest.raises(RiskStateError, match="without a mark"):
        manager(marked=None).review([], snapshot, {})


def test_state_round_trips_and_rebuilds_an_identical_manager() -> None:
    m = manager(cfg(daily=0.03, drawdown=0.10))
    m.on_mark(D1, 8_900.0)
    saved = RiskState.model_validate(m.state.model_dump(mode="json"))
    assert saved == m.state
    rebuilt = RiskManager(cfg(daily=0.03, drawdown=0.10), LOOSE, 0.0, saved)
    snapshot, _ = portfolio(8_900.0, as_of=D1)
    orders = [buy("AAPL", 1, created_on=D1)]
    assert rebuilt.review(orders, snapshot, {"AAPL": 1.0}) == m.review(
        orders, snapshot, {"AAPL": 1.0}
    )


def test_unsupported_orders_and_bad_markup_raise() -> None:
    option = Leg(
        instrument=OptionContract(underlying="AAPL", expiry=D3, strike=100, right="call"),
        side="buy",
    )
    order = Order(created_on=D0, action="enter", legs=(option,), quantity=1, strength=1.0)
    snapshot, _ = portfolio(10_000)
    with pytest.raises(NotImplementedError):
        manager().review([order], snapshot, {})
    with pytest.raises(ValueError, match="entry_markup"):
        RiskManager(cfg(), LOOSE, -0.01, RiskState.initial(1.0))


# --------------------------------------------------------------------------- release


def release(
    m: RiskManager,
    order: Order,
    cash: float,
    holdings: Mapping[str, tuple[int, float]] | None = None,
    *,
    price: float = 102.0,
    reserved: float = 0.0,
    pending: int = 0,
    max_positions: int = 10,
) -> RiskDecision:
    snapshot, marks = portfolio(cash, holdings)
    account = AccountState(cash=cash, positions=snapshot.positions)
    return m.review_release(
        order,
        account,
        marks,
        price=price,
        reserved_cash=reserved,
        pending_entries=pending,
        max_positions=max_positions,
    )


def test_release_passes_an_entry_that_fits() -> None:
    decision = release(manager(), buy("AAPL", 10), 10_000.0)
    assert (decision.approved_quantity, decision.rule) == (10, None)


@pytest.mark.parametrize(
    ("cash", "reserved", "approved"),
    [
        (1_020.0, 0.0, 10),  # exactly enough
        (1_019.99, 0.0, 9),  # one cent short of the tenth share
        (1_020.0, 102.0, 9),  # cash an earlier entry has reserved is not there
        (101.99, 0.0, 0),
        (5_000.0, 5_000.0, 0),
        (0.0, 0.0, 0),
        (-50.0, 0.0, 0),
    ],
)
def test_release_never_spends_cash_that_is_not_there(
    cash: float, reserved: float, approved: int
) -> None:
    decision = release(manager(), buy("AAPL", 10), cash, {"MSFT": (10, 300.0)}, reserved=reserved)
    assert decision.approved_quantity == approved
    assert decision.rule == (None if approved == 10 else "no_leverage")
    assert approved * 102.0 <= max(cash - reserved, 0.0)


def test_release_holds_the_position_cap() -> None:
    held = {"MSFT": (10, 300.0), "NVDA": (10, 50.0)}
    m = manager()
    assert release(m, buy("AAPL", 1), 5_000.0, held, max_positions=3).approved_quantity == 1
    full = release(m, buy("AAPL", 1), 5_000.0, held, max_positions=2)
    assert (full.approved_quantity, full.rule) == (0, "max_positions")
    # An entry released earlier and not filled yet takes a slot too.
    pending = release(m, buy("AAPL", 1), 5_000.0, held, max_positions=3, pending=1)
    assert (pending.approved_quantity, pending.rule) == (0, "max_positions")


def test_release_never_adds_to_a_held_position() -> None:
    decision = release(manager(), buy("MSFT", 1), 5_000.0, {"MSFT": (10, 300.0)})
    assert (decision.approved_quantity, decision.rule) == (0, "max_positions")
    assert "already held" in decision.detail


def test_release_applies_the_position_and_exposure_limits_to_actual_equity() -> None:
    # Equity is 10,000: 4,000 in cash and 6,000 in MSFT.
    held = {"MSFT": (20, 300.0)}
    by_position = release(manager(cfg(position=0.1)), buy("AAPL", 50), 4_000.0, held)
    assert (by_position.approved_quantity, by_position.rule) == (9, "max_position")  # 1,000 / 102
    by_exposure = release(manager(cfg(position=0.8, exposure=0.8)), buy("AAPL", 50), 4_000.0, held)
    assert (by_exposure.approved_quantity, by_exposure.rule) == (19, "max_exposure")  # 2,000 / 102
    # Exposure already reserved by an earlier entry counts.
    reserved = release(
        manager(cfg(position=0.8, exposure=0.8)), buy("AAPL", 50), 4_000.0, held, reserved=1_000
    )
    assert (reserved.approved_quantity, reserved.rule) == (9, "max_exposure")
    limits = OrderLimits(max_order_notional=500.0, max_orders_per_session=10)
    by_notional = release(manager(limits=limits), buy("AAPL", 50), 4_000.0, held)
    assert (by_notional.approved_quantity, by_notional.rule) == (4, "max_order_notional")


def test_release_refuses_entries_while_halted_or_capped() -> None:
    m = manager()
    m.on_mark(D1, 8_000.0)  # a 20% fall: daily cap and drawdown halt
    halted = release(m, buy("AAPL", 1, D1), 8_000.0)
    assert (halted.approved_quantity, halted.rule) == (0, "drawdown_halt")
    m.reset_halt()
    capped = release(m, buy("AAPL", 1, D1), 8_000.0)
    assert (capped.approved_quantity, capped.rule) == (0, "daily_loss_cap")


def test_release_rejects_what_it_cannot_judge() -> None:
    m = manager()
    for price in (0.0, -1.0, float("nan"), float("inf")):
        assert release(m, buy("AAPL", 1), 5_000.0, price=price).rule == "no_price"
    snapshot, _ = portfolio(5_000.0, {"MSFT": (10, 300.0)})
    account = AccountState(cash=5_000.0, positions=snapshot.positions)
    for marks in ({}, {"MSFT": 0.0}, {"MSFT": float("nan")}):
        unvalued = m.review_release(
            buy("AAPL", 1), account, marks,
            price=102.0, reserved_cash=0.0, pending_entries=0, max_positions=10,
        )  # fmt: skip
        assert (unvalued.approved_quantity, unvalued.rule) == (0, "no_price")
    with pytest.raises(ValueError, match="exits are never gated"):
        release(m, sell("MSFT", 1), 5_000.0, {"MSFT": (10, 300.0)})
    option = Order(
        created_on=D0, action="enter", quantity=1, strength=1.0,
        legs=(Leg(instrument=OptionContract(underlying="AAPL", expiry=D3, strike=1, right="call"),
                  side="buy"),),
    )  # fmt: skip
    with pytest.raises(NotImplementedError):
        release(m, option, 5_000.0)


def test_random_releases_never_breach_a_limit() -> None:
    rng = random.Random(7)
    for _ in range(2_000):
        exposure_pct = rng.uniform(0.3, 1.0)
        config = cfg(position=rng.uniform(0.05, exposure_pct), exposure=exposure_pct)
        limits = OrderLimits(max_order_notional=rng.uniform(200, 20_000), max_orders_per_session=9)
        m = manager(config, limits)
        held = {s: (rng.randint(1, 40), rng.uniform(5, 400)) for s in rng.sample("BCDEF", 3)}
        cash = rng.uniform(-500, 20_000)
        reserved = rng.uniform(0, 3_000)
        price = round(rng.uniform(1, 500), 2)
        quantity = rng.randint(1, 300)
        decision = release(m, buy("AAPL", quantity), cash, held, price=price, reserved=reserved)
        approved = decision.approved_quantity
        assert 0 <= approved <= quantity
        exposure = sum(q * mark for q, mark in held.values())
        equity = cash + exposure
        cost = approved * price
        if approved:
            assert cost <= cash - reserved
            assert cost <= config.max_position_pct * equity
            assert exposure + reserved + cost <= config.max_exposure_pct * equity
            assert cost <= limits.max_order_notional
        if approved < quantity:
            assert decision.rule is not None and decision.detail
            # One more share would breach the limit that is named.
            more = (approved + 1) * price
            rooms = {
                "no_leverage": cash - reserved,
                "max_position": config.max_position_pct * equity,
                "max_exposure": config.max_exposure_pct * equity - exposure - reserved,
                "max_order_notional": limits.max_order_notional,
            }
            if decision.rule in rooms:
                assert more > rooms[decision.rule]


# --------------------------------------------------------------------------- properties


def test_whole_units_never_exceeds_the_room() -> None:
    rng = random.Random(7)
    for _ in range(5_000):
        price = rng.choice([rng.uniform(0.01, 5_000), round(rng.uniform(0.01, 500), 2)])
        room = rng.choice([rng.uniform(-10, 1e6), price * rng.randint(0, 500)])
        n = _whole_units(room, price)
        assert n >= 0
        assert n * price <= max(room, 0)
        assert (n + 1) * price > room or room <= 0
    assert _whole_units(float("nan"), 1.0) == 0
    assert _whole_units(float("inf"), 1.0) == 0  # not finite: fail closed


SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]


def _random_case(rng: random.Random):  # type: ignore[no-untyped-def]
    config = cfg(
        position=rng.uniform(0.05, 1.0),
        exposure=1.0,
        daily=rng.uniform(0.01, 0.2),
        drawdown=rng.uniform(0.05, 0.5),
    )
    config = config.model_copy(
        update={"max_exposure_pct": rng.uniform(config.max_position_pct, 1.0)}
    )
    limits = OrderLimits(
        max_order_notional=rng.uniform(100, 50_000), max_orders_per_session=rng.randint(1, 12)
    )
    marks = {s: round(rng.uniform(1, 800), 2) for s in SYMBOLS}
    holdings = {s: (rng.randint(1, 60), marks[s]) for s in rng.sample(SYMBOLS, rng.randint(0, 5))}
    snapshot, _ = portfolio(rng.uniform(0, 30_000), holdings, as_of=D1)
    # Yesterday's equity and the peak sit near today's, so the daily cap and the halt
    # each trip in some cases and not in others.
    before = snapshot.equity * rng.uniform(0.9, 1.15) + 1.0
    m = RiskManager(
        config, limits, rng.choice([0.0, 0.0005, 0.02]), RiskState.initial(before * 1.02)
    )
    m.on_mark(D0, before)
    m.on_mark(D1, snapshot.equity)  # may trip the cap, the halt, both, or neither
    orders = []
    for s in rng.sample(SYMBOLS, rng.randint(0, len(SYMBOLS))):
        if s in holdings and rng.random() < 0.7:
            orders.append(sell(s, rng.randint(1, holdings[s][0] + 20), created_on=D1))
        else:
            orders.append(buy(s, rng.randint(1, 80), created_on=D1))
    if rng.random() < 0.2:
        marks.pop(rng.choice(SYMBOLS))
    return m, config, limits, snapshot, marks, orders


def test_random_reviews_keep_every_guarantee() -> None:
    rng = random.Random(20261113)
    seen = {"sell_in_trouble": 0, "shrunk": 0, "rejected": 0, "full": 0}
    for _ in range(2_000):
        m, config, limits, snapshot, marks, orders = _random_case(rng)
        state = m.state
        review = m.review(orders, snapshot, marks)

        assert m.state == state  # a review never changes state
        assert review == m.review(orders, snapshot, marks)  # deterministic
        assert [d.order for d in review.decisions] == orders  # input order, one each
        assert [o.symbol for o in review.orders] == [
            d.order.symbol for d in review.decisions if d.approved_quantity > 0
        ]

        markup = m._markup
        n_sells = 0
        exposure = snapshot.equity - snapshot.cash
        for d in review.decisions:  # sells are credited first, wherever they appear
            order, q = d.order, d.approved_quantity
            assert 0 <= q <= order.quantity
            if order.legs[0].side != "sell":
                continue
            held = snapshot.positions.get(order.symbol)
            # Risk-reducing sells pass in full, whatever the state.
            assert q == min(order.quantity, held.quantity if held else 0)
            n_sells += q > 0
            if q and order.symbol in marks:
                exposure -= q * marks[order.symbol]
            if q and (state.halted or state.entries_blocked_for == D1):
                seen["sell_in_trouble"] += 1
        for d in review.decisions:
            order, q = d.order, d.approved_quantity
            if order.legs[0].side != "buy":
                continue
            if q == 0:
                seen["rejected"] += 1
                continue
            seen["shrunk" if q < order.quantity else "full"] += 1
            # An approved buy breaches nothing, measured the way the manager promises.
            assert not state.halted and state.entries_blocked_for != D1
            held = snapshot.positions.get(order.symbol)
            price = marks[order.symbol] * (1 + markup)
            exposure += q * price
            assert q * price <= limits.max_order_notional
            held_value = held.quantity * marks[order.symbol] if held else 0.0
            assert held_value + q * price <= config.max_position_pct * snapshot.equity
            assert exposure <= config.max_exposure_pct * snapshot.equity
        n_buys = sum(1 for o in review.orders if o.legs[0].side == "buy")
        assert n_buys <= max(0, limits.max_orders_per_session - n_sells)
    assert all(count > 100 for count in seen.values()), seen  # every branch was exercised


# --------------------------------------------------------------------------- in the engine

ENGINE_SYMBOLS = ["SPY", "AAPL", "MSFT", "NVDA"]


def _backtest(strategy, panel, risk: RiskManager | None, max_positions: int = 5):  # type: ignore[no-untyped-def]
    return simulate(
        strategy,
        panel,
        SimBroker(10_000, COSTS),
        panel.symbols,
        construction(max_positions),
        COSTS,
        risk=risk,
    )


def _risk(config: RiskConfig, limits: OrderLimits = LOOSE) -> RiskManager:
    return RiskManager(config, limits, COSTS.slippage_bps / 10_000, RiskState.initial(10_000.0))


def test_limits_that_never_bind_leave_a_backtest_unchanged() -> None:
    panel = make_panel(ENGINE_SYMBOLS, 80)
    plain = _backtest(Churn(), panel, None)
    loose = _backtest(Churn(), panel, _risk(cfg(daily=0.99, drawdown=0.99)))
    assert plain.fills and loose == plain


def test_position_limit_shrinks_entries_in_a_backtest() -> None:
    panel = make_panel(ENGINE_SYMBOLS, 80)
    plain = _backtest(Churn(), panel, None)
    result = _backtest(Churn(), panel, _risk(cfg(position=0.10, daily=0.99, drawdown=0.99)))
    shrunk = [e for e in result.events if e.kind == "risk_shrunk"]
    assert shrunk and all("max_position" in e.detail for e in shrunk)
    # Every entry was sized against that session's equity, at the close plus slippage.
    equity = {p.date: p.equity for p in result.equity}
    closes = {(d, s): panel.field("close")[i, panel.column(s)] for i, d in enumerate(panel.dates)
              for s in panel.symbols}  # fmt: skip
    entries = [o for o in result.orders.values() if o.action == "enter"]
    assert entries
    for order in entries:
        price = closes[(order.created_on, order.symbol)] * (1 + COSTS.slippage_bps / 10_000)
        assert order.quantity * price <= 0.10 * equity[order.created_on]
    assert sum(o.quantity for o in entries) < sum(
        o.quantity for o in plain.orders.values() if o.action == "enter"
    )


def test_drawdown_halt_stops_entries_at_the_expected_session() -> None:
    # SPY is bought on session 0 and falls 2% of its start price each session.
    closes = [100.0 - 2.0 * t for t in range(30)]
    panel = make_panel(["SPY"], 30, opens={"SPY": closes}, closes={"SPY": closes})
    result = _backtest(BuyAndHold(), panel, _risk(cfg(daily=0.99, drawdown=0.10)), max_positions=1)
    halts = [e for e in result.events if e.kind == "entries_halted"]
    assert len(halts) == 1 and "drawdown_halt" in halts[0].detail
    first = next(p for p in result.equity if p.equity <= 9_000.0)  # 10% below the 10,000 peak
    assert halts[0].date == first.date
    assert len(result.fills) == 1  # the position is kept: a halt never sells


def test_halted_run_rejects_later_entries_and_still_exits() -> None:
    # A halt from the first session: Churn keeps signalling, nothing is ever bought.
    panel = make_panel(ENGINE_SYMBOLS, 40)
    risk = RiskManager(
        cfg(daily=0.99, drawdown=0.10), LOOSE, 0.0, RiskState.initial(20_000.0)
    )  # the account is already 50% below its recorded peak
    result = _backtest(Churn(), panel, risk)
    assert result.fills == ()
    assert [e.kind for e in result.events if e.kind == "entries_halted"] == ["entries_halted"]
    rejected = [e for e in result.events if e.kind == "risk_rejected"]
    assert rejected and all(e.detail.startswith("drawdown_halt") for e in rejected)
