"""The venue protocol: the narrow surface a real broker's API is reduced to.

A venue places, reports, and cancels single orders, and reports the account. It makes no
decisions: when an order is sent, at what limit, and what happens after a timeout all
belong to the order manager, which is the only code that builds a `VenueOrder`.

The models here refuse shapes the harness never sends, so a bug upstream cannot reach a
real venue as a valid-looking order:

- A buy is always a limit order. Only sells go to market (exits must fill).
- `opg` (opening auction) is for market sells only.
- Whole shares, limit prices in whole cents.

`VenueOrderState` quantities are cumulative, as venues report them. Turning them into
the incremental fills the engine expects is the order manager's job.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from harness.models import MARKET_TZ, FrozenModel, Side, Stock

VenueOrderType = Literal["market", "limit"]
TimeInForce = Literal["opg", "day"]
VenueStatus = Literal["pending", "partially_filled", "filled", "canceled", "rejected", "expired"]
TERMINAL_STATUSES: frozenset[VenueStatus] = frozenset({"filled", "canceled", "rejected", "expired"})


def _require_aware(value: dt.datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


class VenueOrder(FrozenModel):
    """One order as sent to a venue. Built only by the order manager."""

    client_order_id: str = Field(min_length=1, max_length=48)
    symbol: str
    side: Side
    quantity: int = Field(gt=0)  # whole shares
    type: VenueOrderType
    limit_price: float | None = None  # required for limit, forbidden for market
    time_in_force: TimeInForce

    @model_validator(mode="after")
    def _check(self) -> Self:
        Stock(symbol=self.symbol)  # validates the ticker format
        if self.type == "limit":
            if self.limit_price is None or not self.limit_price > 0:
                raise ValueError("a limit order needs a positive limit_price")
            cents = self.limit_price * 100
            if abs(cents - round(cents)) > 1e-6:
                raise ValueError(f"limit_price {self.limit_price} is not in whole cents")
        elif self.limit_price is not None:
            raise ValueError("a market order has no limit_price")
        if self.side == "buy" and self.type != "limit":
            raise ValueError("a buy must be a limit order")
        if self.time_in_force == "opg" and (self.side, self.type) != ("sell", "market"):
            raise ValueError("opg is for market sells only")
        return self


class VenueOrderState(FrozenModel):
    """A venue's current view of one order. Quantities are cumulative."""

    client_order_id: str
    venue_order_id: str
    status: VenueStatus
    quantity: int = Field(gt=0)
    filled_quantity: int = Field(ge=0)
    filled_avg_price: float | None  # average over the filled shares; None until a fill
    updated_at: dt.datetime
    reject_reason: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_aware(self.updated_at, "updated_at")
        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity exceeds quantity")
        if self.status == "filled" and self.filled_quantity != self.quantity:
            raise ValueError("a filled order must be filled in full")
        if self.status == "partially_filled" and not 0 < self.filled_quantity < self.quantity:
            raise ValueError("a partially filled order has some, not all, of its quantity")
        if self.status == "rejected" and self.filled_quantity:
            raise ValueError("a rejected order cannot have fills")
        if self.filled_quantity:
            if self.filled_avg_price is None or not self.filled_avg_price > 0:
                raise ValueError("fills need a positive filled_avg_price")
        elif self.filled_avg_price is not None:
            raise ValueError("filled_avg_price without fills")
        return self

    @property
    def is_terminal(self) -> bool:
        """True once the venue will never change this order again."""
        return self.status in TERMINAL_STATUSES


class VenueAccount(FrozenModel):
    account_id: str
    cash: float
    buying_power: float
    equity: float
    trading_blocked: bool
    shorting_enabled: bool
    margin_multiplier: float


class VenuePosition(FrozenModel):
    symbol: str
    quantity: int  # signed; a negative value is a short, which the harness never opens
    avg_entry_price: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        Stock(symbol=self.symbol)
        if self.quantity == 0:
            raise ValueError("a position has a nonzero quantity")
        return self


class MarketSession(FrozenModel):
    """One trading session. A half day differs only in `close`."""

    date: dt.date
    open: dt.datetime
    close: dt.datetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        _require_aware(self.open, "open")
        _require_aware(self.close, "close")
        if not self.open < self.close:
            raise ValueError("a session opens before it closes")
        if self.open.astimezone(MARKET_TZ).date() != self.date:
            raise ValueError("date must be the New York date of the open")
        return self


class VenueRejected(Exception):
    """Definitive: the venue refused the request. A refused order does not exist and is
    never retried."""


class VenueUnavailable(Exception):
    """Outcome unknown (timeout, connection error, server error). An order being placed
    may or may not exist."""


class KillSwitchEngaged(Exception):
    """The kill latch is set (or unreadable). No order was sent."""


class Venue(Protocol):
    """What the order manager needs from a broker. Every method may raise
    VenueUnavailable; a caller must never read that as "nothing happened"."""

    def now(self) -> dt.datetime:
        """The venue's clock, timezone-aware. All scheduling uses this, not local time."""
        ...

    def sessions(self, start: dt.date, end: dt.date) -> list[MarketSession]:
        """Trading sessions from `start` to `end` inclusive, in date order."""
        ...

    def account(self) -> VenueAccount: ...

    def positions(self) -> list[VenuePosition]: ...

    def open_orders(self) -> list[VenueOrderState]:
        """Every order that is not terminal, whoever placed it."""
        ...

    def get_order(self, client_order_id: str) -> VenueOrderState | None:
        """The order with this client id, or None if the venue has never seen it."""
        ...

    def place(self, order: VenueOrder) -> VenueOrderState:
        """Send one order. Raises VenueRejected, VenueUnavailable, or KillSwitchEngaged.

        A duplicate client id is reported as the existing order, never as a second order.
        """
        ...

    def cancel(self, client_order_id: str) -> None:
        """Request cancellation. Success means requested, not canceled: poll to confirm.
        A fill can still arrive before the cancel takes effect.

        Canceling an order that is already final does nothing. An order the venue has
        never seen raises VenueRejected.
        """
        ...

    def cancel_all(self) -> None:
        """Request cancellation of every open order."""
        ...
