"""The Broker interface: the only path from Orders to Fills.

SimBroker implements it for backtests; the order manager will implement it for paper
trading in slice 2. The engine talks only to this interface, so backtest and paper share
the engine. What every implementation must do is pinned by `tests/broker_contract.py`.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from harness.models import AccountState, Order, OrderUpdate, OrderUpdateReason


class LookAheadError(RuntimeError):
    """An order was about to fill using a price from its own signal bar or earlier."""


class Broker(ABC):
    """An execution venue."""

    @abstractmethod
    def submit(self, orders: Sequence[Order]) -> list[str]:
        """Queue orders for execution and return broker order ids, in input order.

        Never fills synchronously: a queued order can fill at the earliest in the next
        session after its `created_on` date.
        """

    @abstractmethod
    def on_session_open(self, session: dt.date, opens: Mapping[str, float]) -> None:
        """Called by the engine at the start of every traded session, in date order.
        Not called for warm-up sessions.

        Orders are day orders: when this returns, every order submitted before it has
        reached a final state (filled, canceled, rejected or expired) and is reported
        through `poll_updates`. Nothing carries over to a later session.

        SimBroker executes its queue here using only `opens` (this session's opening
        prices). A live adapter ignores it, because real fills come from the venue.
        """

    @abstractmethod
    def force_close(
        self, symbol: str, price: float, price_date: dt.date, session: dt.date
    ) -> str | None:
        """Close the whole position in `symbol` at a stale `price` from `price_date`.

        Used when a held symbol has stopped trading (delisted or long halt). Returns the
        order id; the fill is reported through `poll_updates`.

        A real venue cannot sell at a stale price, so a live broker returns None and
        leaves the position alone: it never invents a fill. The engine logs
        `position_stale_no_data` and reconciliation picks up whatever the venue did.
        """

    @abstractmethod
    def poll_updates(self) -> list[OrderUpdate]:
        """Return and clear order updates since the last call, in execution order.

        An update's fills are the fills since the previous update for that order.
        """

    @abstractmethod
    def account(self) -> AccountState:
        """Cash and positions as the broker sees them now."""

    @abstractmethod
    def cancel_all(self, reason: OrderUpdateReason) -> list[str]:
        """Cancel every queued order; returns their ids. Updates report `reason`."""
