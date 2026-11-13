"""The fake venue itself. Order-manager tests trust it, so its behaviors are pinned here."""

from __future__ import annotations

import datetime as dt

import pytest

from harness.broker.venue import (
    KillSwitchEngaged,
    Venue,
    VenueOrder,
    VenueRejected,
    VenueUnavailable,
)
from tests.fake_venue import FakeVenue, FillStep, OrderScript, at

DAY = dt.date(2024, 1, 3)  # a Wednesday
PRE_OPEN = at(DAY, 8, 0)
OPEN = at(DAY, 9, 30)


def venue(when: dt.datetime = PRE_OPEN, **kw) -> FakeVenue:  # type: ignore[no-untyped-def]
    v = FakeVenue(clock=when, **kw)
    v.set_price("AAPL", 100.0)
    v.set_price("MSFT", 300.0)
    return v


def buy(symbol: str = "AAPL", quantity: int = 10, limit: float = 102.0, n: int = 1) -> VenueOrder:
    return VenueOrder(
        client_order_id=f"{symbol}-e-a{n}",
        symbol=symbol,
        side="buy",
        quantity=quantity,
        type="limit",
        limit_price=limit,
        time_in_force="day",
    )


def sell(symbol: str = "AAPL", quantity: int = 10, tif: str = "opg", n: int = 1) -> VenueOrder:
    return VenueOrder(
        client_order_id=f"{symbol}-x-a{n}",
        symbol=symbol,
        side="sell",
        quantity=quantity,
        type="market",
        time_in_force=tif,  # type: ignore[arg-type]
    )


def test_fake_venue_satisfies_the_venue_protocol() -> None:
    v: Venue = venue()  # checked by pyright
    assert v.now() == PRE_OPEN


# --------------------------------------------------------------------------- default behavior


def test_order_placed_before_the_open_waits_for_it_then_fills() -> None:
    v = venue()
    placed = v.place(buy())
    assert (placed.status, placed.filled_quantity) == ("pending", 0)
    v.set_now(at(DAY, 9, 29, 59))
    assert v.get_order("AAPL-e-a1").status == "pending"  # type: ignore[union-attr]
    v.set_now(OPEN)
    filled = v.get_order("AAPL-e-a1")
    assert filled is not None
    assert (filled.status, filled.filled_quantity, filled.filled_avg_price) == ("filled", 10, 100.0)
    assert filled.updated_at == OPEN
    assert v.cash == 9_000.0
    (position,) = v.positions()
    assert (position.symbol, position.quantity, position.avg_entry_price) == ("AAPL", 10, 100.0)
    assert v.open_orders() == []


def test_order_placed_during_the_session_fills_at_once() -> None:
    v = venue(at(DAY, 9, 45))
    assert v.place(buy()).status == "filled"


def test_limit_buy_above_its_cap_waits_and_expires_at_the_close() -> None:
    v = venue(at(DAY, 9, 45))
    v.set_price("AAPL", 105.0)
    assert v.place(buy(limit=102.0)).status == "pending"
    v.set_now(at(DAY, 15, 59))
    assert v.get_order("AAPL-e-a1").status == "pending"  # type: ignore[union-attr]
    v.set_now(at(DAY, 16, 0))
    done = v.get_order("AAPL-e-a1")
    assert done is not None and (done.status, done.filled_quantity) == ("expired", 0)
    assert v.cash == 10_000.0


def test_limit_buy_fills_when_the_price_comes_down_to_it() -> None:
    v = venue(at(DAY, 9, 45))
    v.set_price("AAPL", 105.0)
    v.place(buy(limit=102.0))
    v.advance(120)
    v.set_price("AAPL", 101.5)
    done = v.get_order("AAPL-e-a1")
    assert done is not None
    assert (done.status, done.filled_avg_price) == ("filled", 101.5)
    assert done.updated_at == at(DAY, 9, 47)  # when the price got there, not when it was placed


def test_opg_sell_fills_at_the_open_and_is_refused_after_it() -> None:
    v = venue()
    v.hold("AAPL", 10, 90.0)
    v.place(sell())
    v.set_now(OPEN)
    done = v.get_order("AAPL-x-a1")
    assert done is not None and (done.status, done.filled_avg_price) == ("filled", 100.0)
    assert v.cash == 11_000.0 and v.positions() == []

    v.hold("MSFT", 5, 300.0)
    with pytest.raises(VenueRejected, match="before the open"):
        v.place(sell("MSFT", 5))
    assert v.place(sell("MSFT", 5, tif="day")).status == "filled"  # the fallback path


def test_opg_order_for_a_halted_symbol_expires_soon_after_the_open() -> None:
    v = venue()
    v.hold("AAPL", 10, 90.0)
    v.set_price("AAPL", None)
    v.place(sell())
    v.set_now(at(DAY, 9, 30, 59))
    assert v.get_order("AAPL-x-a1").status == "pending"  # type: ignore[union-attr]
    v.set_now(at(DAY, 9, 31))
    assert v.get_order("AAPL-x-a1").status == "expired"  # type: ignore[union-attr]
    assert v.positions()[0].quantity == 10


def test_half_day_and_holiday_sessions() -> None:
    friday = dt.date(2024, 1, 5)
    v = venue(holidays={dt.date(2024, 1, 4)}, half_days={friday})
    sessions = v.sessions(DAY, dt.date(2024, 1, 8))
    assert [s.date for s in sessions] == [DAY, friday, dt.date(2024, 1, 8)]  # no holiday, weekend
    assert sessions[1].close == at(friday, 13, 0)
    assert sessions[0].open == OPEN and sessions[0].close == at(DAY, 16, 0)
    # An order placed after the close waits for the next session's open.
    v.set_now(at(DAY, 17, 0))
    v.place(buy())
    v.set_now(at(friday, 9, 29))
    assert v.get_order("AAPL-e-a1").status == "pending"  # type: ignore[union-attr]
    v.set_now(at(friday, 9, 30))
    assert v.get_order("AAPL-e-a1").status == "filled"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- the account


def test_buying_power_is_reserved_by_open_buys() -> None:
    v = venue()
    v.place(buy("AAPL", 50, limit=102.0))  # reserves 5,100
    assert v.account().buying_power == pytest.approx(4_900.0)
    with pytest.raises(VenueRejected, match="buying power"):
        v.place(buy("MSFT", 17, limit=300.0))  # 5,100 > 4,900
    v.place(buy("MSFT", 16, limit=300.0))
    v.set_now(OPEN)
    account = v.account()
    assert account.cash == pytest.approx(10_000 - 50 * 100 - 16 * 300)
    assert account.equity == pytest.approx(10_000.0)
    assert (account.shorting_enabled, account.margin_multiplier, account.trading_blocked) == (
        False,
        1.0,
        False,
    )


def test_shares_are_reserved_by_open_sells_so_the_fake_cannot_go_short() -> None:
    v = venue()
    v.hold("AAPL", 10, 90.0)
    with pytest.raises(VenueRejected, match="insufficient quantity"):
        v.place(sell(quantity=11))
    v.place(sell(quantity=6))
    with pytest.raises(VenueRejected, match="insufficient quantity"):
        v.place(sell(quantity=5, n=2))
    v.place(sell(quantity=4, n=2))
    with pytest.raises(VenueRejected, match="insufficient quantity"):
        v.place(sell("MSFT", 1))


def test_blocked_account_rejects_orders() -> None:
    v = venue(trading_blocked=True)
    with pytest.raises(VenueRejected, match="blocked"):
        v.place(buy())


# --------------------------------------------------------------------------- scripts


def test_scripted_rejection_creates_no_order() -> None:
    v = venue()
    v.script("AAPL", OrderScript(reject="symbol not tradable"))
    with pytest.raises(VenueRejected, match="not tradable"):
        v.place(buy())
    assert v.get_order("AAPL-e-a1") is None
    assert v.place(buy(n=2)).status == "pending"  # the script applied to one order only


def test_timeout_before_accept_leaves_no_order_and_a_retry_succeeds() -> None:
    v = venue()
    v.script("AAPL", OrderScript(timeouts_before_accept=2))
    for _ in range(2):
        with pytest.raises(VenueUnavailable, match="not accepted"):
            v.place(buy())
        assert v.get_order("AAPL-e-a1") is None
    assert v.place(buy()).status == "pending"
    assert len(v.place_attempts) == 3


def test_timeout_after_accept_leaves_an_order_the_caller_did_not_see() -> None:
    v = venue()
    v.script("AAPL-e-a1", OrderScript(timeout_after_accept=True))  # keyed by client order id
    with pytest.raises(VenueUnavailable, match="accepted"):
        v.place(buy())
    found = v.get_order("AAPL-e-a1")
    assert found is not None and found.status == "pending"
    # Placing again with the same client id reports that order; it never makes a second one.
    again = v.place(buy())
    assert again.venue_order_id == found.venue_order_id
    assert len(v.open_orders()) == 1


def test_scripted_partial_and_slow_fills() -> None:
    v = venue()
    v.script(
        "AAPL",
        OrderScript(fills=(FillStep(5, 3, 100.5), FillStep(60, 4), FillStep(600, None, 101.0))),
    )
    v.place(buy())
    v.set_now(OPEN + dt.timedelta(seconds=4))
    assert v.get_order("AAPL-e-a1").filled_quantity == 0  # type: ignore[union-attr]
    v.advance(1)
    first = v.get_order("AAPL-e-a1")
    assert first is not None
    assert (first.status, first.filled_quantity, first.filled_avg_price) == (
        "partially_filled",
        3,
        100.5,
    )
    v.advance(55)
    second = v.get_order("AAPL-e-a1")
    assert second is not None and second.filled_quantity == 7
    assert second.filled_avg_price == pytest.approx((3 * 100.5 + 4 * 100.0) / 7)
    v.advance(540)
    done = v.get_order("AAPL-e-a1")
    assert done is not None and (done.status, done.filled_quantity) == ("filled", 10)
    assert v.cash == pytest.approx(10_000 - 3 * 100.5 - 4 * 100.0 - 3 * 101.0)


def test_scripted_order_that_never_fills_expires() -> None:
    v = venue()
    v.script("AAPL", OrderScript(fills=()))
    v.place(buy())
    v.set_now(at(DAY, 16, 0))
    assert v.get_order("AAPL-e-a1").status == "expired"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- cancel


def test_cancel_is_a_request_confirmed_only_after_the_latency() -> None:
    v = venue(cancel_latency=2.0)
    v.script("AAPL", OrderScript(fills=()))
    v.place(buy())
    v.set_now(OPEN)
    v.cancel("AAPL-e-a1")
    assert v.get_order("AAPL-e-a1").status == "pending"  # type: ignore[union-attr]
    v.advance(1.9)
    assert v.get_order("AAPL-e-a1").status == "pending"  # type: ignore[union-attr]
    v.advance(0.1)
    done = v.get_order("AAPL-e-a1")
    assert done is not None and (done.status, done.filled_quantity) == ("canceled", 0)
    v.cancel("AAPL-e-a1")  # canceling a finished order is harmless
    with pytest.raises(VenueRejected, match="no order"):
        v.cancel("unknown")


def test_fill_can_arrive_while_a_cancel_is_in_flight() -> None:
    v = venue()
    v.script("AAPL", OrderScript(fills=(FillStep(0, 4),), fill_on_cancel=3))
    v.place(buy())
    v.set_now(OPEN)
    v.cancel("AAPL-e-a1")
    v.advance(1)
    done = v.get_order("AAPL-e-a1")
    assert done is not None and (done.status, done.filled_quantity) == ("canceled", 7)
    assert v.positions()[0].quantity == 7

    v.script("MSFT", OrderScript(fills=(), fill_on_cancel=99))  # the whole order fills instead
    v.place(buy("MSFT", 5, limit=310.0))
    v.cancel("MSFT-e-a1")
    v.advance(1)
    assert v.get_order("MSFT-e-a1").status == "filled"  # type: ignore[union-attr]


def test_no_fill_after_a_cancel_took_effect() -> None:
    v = venue()
    v.script("AAPL", OrderScript(fills=(FillStep(30, None),)))
    v.place(buy())
    v.set_now(OPEN)
    v.cancel("AAPL-e-a1")
    v.advance(120)  # past both the cancel and the scripted fill time
    done = v.get_order("AAPL-e-a1")
    assert done is not None and (done.status, done.filled_quantity) == ("canceled", 0)
    assert done.updated_at == OPEN + dt.timedelta(seconds=1)


def test_cancel_all_cancels_every_open_order() -> None:
    v = venue()
    v.hold("MSFT", 5, 300.0)
    v.place(buy())
    v.place(sell("MSFT", 5))
    v.cancel_all()
    v.advance(1)
    assert v.open_orders() == []
    assert {o.status for o in (v.get_order("AAPL-e-a1"), v.get_order("MSFT-x-a1")) if o} == {
        "canceled"
    }


# --------------------------------------------------------------------------- failures and the log


@pytest.mark.parametrize(
    "method", ["now", "account", "positions", "open_orders", "get_order", "cancel", "cancel_all"]
)
def test_any_method_can_be_made_to_time_out(method: str) -> None:
    v = venue()
    v.place(buy())
    v.fail_next(method, times=2)
    args = ("AAPL-e-a1",) if method in ("get_order", "cancel") else ()
    for _ in range(2):
        with pytest.raises(VenueUnavailable, match=method):
            getattr(v, method)(*args)
    getattr(v, method)(*args)  # then it works again
    if method == "cancel":
        v.advance(1)
        assert v.open_orders() == []


def test_timed_out_cancel_did_not_reach_the_venue() -> None:
    v = venue()
    v.script("AAPL", OrderScript(fills=()))
    v.place(buy())
    v.fail_next("cancel")
    with pytest.raises(VenueUnavailable):
        v.cancel("AAPL-e-a1")
    v.advance(60)
    assert v.get_order("AAPL-e-a1").status == "pending"  # type: ignore[union-attr]


def test_killed_venue_sends_nothing() -> None:
    v = venue(killed=True)
    with pytest.raises(KillSwitchEngaged):
        v.place(buy())
    assert v.get_order("AAPL-e-a1") is None
    assert len(v.place_attempts) == 1  # the attempt is still visible to the test


def test_calls_are_logged_in_order() -> None:
    v = venue()
    v.place(buy())
    v.get_order("AAPL-e-a1")
    v.cancel("AAPL-e-a1")
    assert v.calls == [("place", "AAPL-e-a1"), ("get_order", "AAPL-e-a1"), ("cancel", "AAPL-e-a1")]


def test_clock_only_moves_forward() -> None:
    v = venue()
    with pytest.raises(ValueError):
        v.advance(-1)
    with pytest.raises(ValueError):
        v.set_now(PRE_OPEN - dt.timedelta(seconds=1))
    with pytest.raises(ValueError):
        FakeVenue(clock=dt.datetime(2024, 1, 3, 8, 0))
