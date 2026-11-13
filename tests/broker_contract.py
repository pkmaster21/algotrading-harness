"""The Broker contract: what the engine may rely on from any Broker implementation.

The engine is shared by backtest and paper trading, but the broker behind it is not
(`SimBroker` for backtests, the order manager over a venue for paper). These tests state
the behavior both must share, so the two cannot drift apart unnoticed. They assert
outcomes the engine depends on (who got filled, when, and what the account says), never
exact prices or order ids, which are each broker's own business.

To run the suite against a broker, subclass `BrokerContract` in a test module and
provide the `make_rig` fixture:

    class TestSimBroker(BrokerContract):
        @pytest.fixture
        def make_rig(self) -> RigFactory: ...

Rules a broker may be stricter about (and that are tested per broker, not here):
session order, which orders it supports, how a buy that no longer fits is handled
beyond "cash never goes negative".
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import pytest

from harness.broker.base import Broker, LookAheadError
from harness.models import AccountState, Action, Leg, Order, OrderUpdate, Side, Stock

D0, D1, D2, D3 = (dt.date(2024, 1, d) for d in (2, 3, 4, 5))
CASH = 1_000.0
TERMINAL = frozenset({"filled", "canceled", "rejected", "expired"})


class BrokerRig(ABC):
    """A broker under test together with the market it trades in."""

    broker: Broker

    @abstractmethod
    def open_session(self, session: dt.date, opens: Mapping[str, float]) -> None:
        """Make `session` open at these prices and run the broker's `on_session_open`.

        A symbol missing from `opens` does not trade that session. Prices are chosen by
        the tests so that an order sized at the prior close still fits unless the test
        says otherwise.
        """


RigFactory = Callable[[float], BrokerRig]  # starting cash -> a fresh rig


def order(symbol: str, side: Side, quantity: int, created_on: dt.date = D0) -> Order:
    action: Action = "enter" if side == "buy" else "exit"
    return Order(
        created_on=created_on,
        action=action,
        legs=(Leg(instrument=Stock(symbol=symbol), side=side),),
        quantity=quantity,
        strength=1.0 if action == "enter" else None,
    )


def settle(
    rig: BrokerRig,
    session: dt.date,
    opens: Mapping[str, float],
    submitted: Mapping[str, Order],
) -> list[OrderUpdate]:
    """Open `session`, collect the updates, and check what must hold for every update."""
    before = rig.broker.account()
    rig.open_session(session, opens)
    updates = rig.broker.poll_updates()
    for update in updates:
        placed = submitted[update.order_id]  # only orders the caller submitted
        leg = placed.legs[0]
        for fill in update.fills:
            assert fill.order_id == update.order_id
            assert (fill.instrument, fill.side) == (leg.instrument, leg.side)
            assert fill.signal_date == placed.created_on
            assert fill.fill_date == session
            assert fill.signal_date < fill.fill_date  # never the signal session
    for order_id, placed in submitted.items():
        filled = sum(f.quantity for u in updates if u.order_id == order_id for f in u.fills)
        assert filled <= placed.quantity
    check_account(before, rig.broker.account(), updates)
    return updates


def check_account(
    before: AccountState, after: AccountState, updates: Sequence[OrderUpdate]
) -> None:
    """The account moved by exactly what the reported fills say, and nothing else."""
    fills = [f for u in updates for f in u.fills]
    assert after.cash == pytest.approx(before.cash + sum(f.cash_delta for f in fills))
    expected = {s: p.quantity for s, p in before.positions.items()}
    for fill in fills:
        sign = 1 if fill.side == "buy" else -1
        key = fill.instrument.key
        expected[key] = expected.get(key, 0) + sign * fill.quantity
    assert {s: p.quantity for s, p in after.positions.items()} == {
        s: q for s, q in expected.items() if q != 0
    }


def final_status(updates: Sequence[OrderUpdate], order_id: str) -> str:
    """The status of the last update for an order (it must have one)."""
    return [u.status for u in updates if u.order_id == order_id][-1]


def filled_quantity(updates: Sequence[OrderUpdate], order_id: str) -> int:
    return sum(f.quantity for u in updates if u.order_id == order_id for f in u.fills)


class BrokerContract:
    """Subclass in a test module and provide `make_rig`."""

    @pytest.fixture
    def make_rig(self) -> RigFactory:
        raise NotImplementedError("provide a make_rig fixture")

    # -------------------------------------------------------------- submit

    def test_submit_returns_one_id_per_order_in_input_order(self, make_rig: RigFactory) -> None:
        broker = make_rig(CASH).broker
        assert broker.submit([]) == []
        orders = [order("AAPL", "buy", 2), order("MSFT", "buy", 3)]
        ids = broker.submit(orders)
        assert len(ids) == 2 and len(set(ids)) == 2
        assert all(isinstance(i, str) and i for i in ids)
        # Input order: `settle` checks each fill against the order its id was paired with.
        rig = make_rig(CASH)
        ids = rig.broker.submit(orders)
        updates = settle(
            rig, D1, {"AAPL": 100.0, "MSFT": 100.0}, dict(zip(ids, orders, strict=True))
        )
        assert [filled_quantity(updates, i) for i in ids] == [2, 3]

    def test_submit_never_fills_or_changes_the_account(self, make_rig: RigFactory) -> None:
        broker = make_rig(CASH).broker
        before = broker.account()
        broker.submit([order("AAPL", "buy", 2)])
        assert broker.poll_updates() == []
        assert broker.account() == before
        assert before == AccountState(cash=CASH)

    # -------------------------------------------------------------- fill timing

    def test_order_fills_in_the_session_after_its_signal(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        placed = order("AAPL", "buy", 5)
        (order_id,) = rig.broker.submit([placed])
        updates = settle(rig, D1, {"AAPL": 100.0}, {order_id: placed})
        assert final_status(updates, order_id) == "filled"
        assert filled_quantity(updates, order_id) == 5
        position = rig.broker.account().positions["AAPL"]
        assert (position.quantity, position.entry_date) == (5, D1)
        assert position.entry_order_id == order_id

    def test_order_is_never_filled_in_its_signal_session(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        (order_id,) = rig.broker.submit([order("AAPL", "buy", 5, created_on=D1)])
        try:
            rig.open_session(D1, {"AAPL": 100.0})
        except LookAheadError:  # refusing outright is the strictest way to honor the rule
            assert rig.broker.poll_updates() == []
        else:
            # Not refused: then the order must be finished without a fill, not left open
            # to fill in a later session (orders are day orders).
            updates = rig.broker.poll_updates()
            assert final_status(updates, order_id) in TERMINAL - {"filled"}
            assert filled_quantity(updates, order_id) == 0
        assert rig.broker.account() == AccountState(cash=CASH)

    def test_every_order_is_terminal_when_the_session_open_returns(
        self, make_rig: RigFactory
    ) -> None:
        rig = make_rig(CASH)
        orders = [
            order("AAPL", "buy", 2),  # fills
            order("MSFT", "buy", 2),  # MSFT does not trade
            order("NVDA", "sell", 2),  # not held
        ]
        ids = rig.broker.submit(orders)
        submitted = dict(zip(ids, orders, strict=True))
        updates = settle(rig, D1, {"AAPL": 100.0, "NVDA": 100.0}, submitted)
        for order_id in ids:
            assert final_status(updates, order_id) in TERMINAL
        # Day orders: nothing is left to act on in a later session.
        settled = rig.broker.account()
        assert settle(rig, D2, {"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0}, submitted) == []
        assert rig.broker.account() == settled
        assert rig.broker.cancel_all("end_of_data") == []

    def test_symbol_that_does_not_trade_is_not_filled(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        placed = order("AAPL", "buy", 2)
        (order_id,) = rig.broker.submit([placed])
        updates = settle(rig, D1, {}, {order_id: placed})
        assert final_status(updates, order_id) in TERMINAL - {"filled"}
        assert filled_quantity(updates, order_id) == 0
        assert rig.broker.account() == AccountState(cash=CASH)

    # -------------------------------------------------------------- updates

    def test_poll_updates_returns_each_update_once(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        rig.broker.submit([order("AAPL", "buy", 2)])
        rig.open_session(D1, {"AAPL": 100.0})
        assert rig.broker.poll_updates() != []
        assert rig.broker.poll_updates() == []

    def test_same_inputs_give_the_same_updates(self, make_rig: RigFactory) -> None:
        def run() -> tuple[list[str], list[OrderUpdate], AccountState]:
            rig = make_rig(CASH)
            orders = [order("AAPL", "buy", 4), order("MSFT", "buy", 4), order("NVDA", "sell", 1)]
            ids = rig.broker.submit(orders)
            updates = settle(
                rig, D1, {"AAPL": 100.0, "NVDA": 50.0}, dict(zip(ids, orders, strict=True))
            )
            return ids, updates, rig.broker.account()

        assert run() == run()

    # -------------------------------------------------------------- cash and positions

    def test_buy_never_takes_cash_negative(self, make_rig: RigFactory) -> None:
        # Sized for a $100 close; the stock opens at $150 and 10 shares no longer fit.
        rig = make_rig(CASH)
        placed = order("AAPL", "buy", 10)
        (order_id,) = rig.broker.submit([placed])
        updates = settle(rig, D1, {"AAPL": 150.0}, {order_id: placed})
        assert final_status(updates, order_id) in TERMINAL
        assert filled_quantity(updates, order_id) < 10
        assert rig.broker.account().cash >= 0

    def test_sell_for_a_symbol_not_held_never_fills(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        placed = order("AAPL", "sell", 3)
        (order_id,) = rig.broker.submit([placed])
        updates = settle(rig, D1, {"AAPL": 100.0}, {order_id: placed})
        assert final_status(updates, order_id) in TERMINAL - {"filled"}
        assert rig.broker.account() == AccountState(cash=CASH)

    def test_exit_closes_the_position(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        buy = order("AAPL", "buy", 5)
        (buy_id,) = rig.broker.submit([buy])
        settle(rig, D1, {"AAPL": 100.0}, {buy_id: buy})
        sell = order("AAPL", "sell", 5, created_on=D1)
        (sell_id,) = rig.broker.submit([sell])
        updates = settle(rig, D2, {"AAPL": 110.0}, {sell_id: sell})
        assert final_status(updates, sell_id) == "filled"
        account = rig.broker.account()
        assert account.positions == {}
        assert account.cash > CASH  # bought near 100, sold near 110

    def test_exit_proceeds_fund_entries_in_the_same_session(self, make_rig: RigFactory) -> None:
        # Nearly all cash goes into AAPL. The next day's entry into MSFT only fits if the
        # AAPL exit is filled first and its proceeds are counted.
        rig = make_rig(CASH)
        first = order("AAPL", "buy", 9)
        (first_id,) = rig.broker.submit([first])
        settle(rig, D1, {"AAPL": 100.0}, {first_id: first})
        assert rig.broker.account().cash < 150

        entry = order("MSFT", "buy", 9, created_on=D1)  # submitted ahead of the exit
        exit_ = order("AAPL", "sell", 9, created_on=D1)
        entry_id, exit_id = rig.broker.submit([entry, exit_])
        updates = settle(rig, D2, {"AAPL": 100.0, "MSFT": 100.0}, {entry_id: entry, exit_id: exit_})
        assert final_status(updates, exit_id) == "filled"
        assert final_status(updates, entry_id) == "filled"
        assert filled_quantity(updates, entry_id) == 9
        account = rig.broker.account()
        assert sorted(account.positions) == ["MSFT"]
        assert account.cash >= 0

    # -------------------------------------------------------------- cancel_all

    def test_cancel_all_cancels_queued_orders_with_the_reason(self, make_rig: RigFactory) -> None:
        rig = make_rig(CASH)
        assert rig.broker.cancel_all("end_of_data") == []
        orders = [order("AAPL", "buy", 2), order("MSFT", "buy", 2)]
        ids = rig.broker.submit(orders)
        assert sorted(rig.broker.cancel_all("end_of_data")) == sorted(ids)
        updates = rig.broker.poll_updates()
        assert sorted(u.order_id for u in updates) == sorted(ids)
        assert {(u.status, u.reason, u.fills) for u in updates} == {("canceled", "end_of_data", ())}
        # Canceled means canceled: the next session fills nothing.
        assert (
            settle(rig, D1, {"AAPL": 100.0, "MSFT": 100.0}, dict(zip(ids, orders, strict=True)))
            == []
        )
        assert rig.broker.account() == AccountState(cash=CASH)

    # -------------------------------------------------------------- force_close

    def test_force_close_closes_the_position_or_leaves_it_untouched(
        self, make_rig: RigFactory
    ) -> None:
        rig = make_rig(CASH)
        buy = order("AAPL", "buy", 5)
        (buy_id,) = rig.broker.submit([buy])
        settle(rig, D1, {"AAPL": 100.0}, {buy_id: buy})
        held = rig.broker.account()

        order_id = rig.broker.force_close("AAPL", 90.0, D1, D3)
        updates = rig.broker.poll_updates()
        if order_id is None:  # a broker that cannot sell at a stale price invents nothing
            assert updates == []
            assert rig.broker.account() == held
            return
        (update,) = updates
        (fill,) = update.fills
        assert (update.order_id, update.status, update.reason) == (order_id, "filled", "no_data")
        assert (fill.kind, fill.side, fill.quantity) == ("forced_close", "sell", 5)
        assert (fill.price_date, fill.fill_date, fill.reference_price) == (D1, D3, 90.0)
        check_account(held, rig.broker.account(), updates)
        assert rig.broker.account().positions == {}
