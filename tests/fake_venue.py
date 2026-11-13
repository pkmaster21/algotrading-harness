"""An in-memory `Venue` with a clock the test controls and a script per order.

Nothing happens on its own. The test (or the code under test, through `sleep`) moves the
clock, and every call first brings the venue up to the current time: orders become
eligible at the open, fill at the prices the test set, expire, and cancel.

Default behavior of an order with no script:

- It can fill once the market is open. `opg` orders are only accepted before the open
  and fill at it; an order placed before the open waits for it.
- A market order fills in full at the symbol's current price. A limit buy fills in full
  at the current price if that is at or below its limit. No price set: no fill yet.
- A `day` order still open at the close expires. An `opg` order that did not fill at the
  open expires shortly after it.
- A cancel is a request: the order becomes canceled only `cancel_latency` seconds later,
  so code that does not poll for confirmation fails.
- Buying power and held shares are reserved by open orders. A buy beyond buying power or
  a sell beyond the free shares is rejected, so the fake cannot go short or borrow.

Scripts (`script(key, ...)`, keyed by client order id or by symbol) override that for one
order: reject it, time out before or after accepting it, fill it in pieces on a schedule,
or fill part of it while a cancel is in flight. `fail_next` makes any method time out.

This is a test double, not a simulator: it models the venue behaviors the order manager
must survive, not prices or queues.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

from harness.broker.venue import (
    KillSwitchEngaged,
    MarketSession,
    VenueAccount,
    VenueOrder,
    VenueOrderState,
    VenuePosition,
    VenueRejected,
    VenueStatus,
    VenueUnavailable,
)
from harness.models import MARKET_TZ

OPEN = dt.time(9, 30)
CLOSE = dt.time(16, 0)
HALF_DAY_CLOSE = dt.time(13, 0)
OPG_GRACE_SECONDS = 60.0  # an opg order not filled this long after the open expires


def at(day: dt.date, hour: int, minute: int = 0, second: int = 0) -> dt.datetime:
    """A New York wall-clock time on `day`."""
    return dt.datetime.combine(day, dt.time(hour, minute, second), MARKET_TZ)


@dataclass(frozen=True)
class FillStep:
    """Fill `quantity` shares (None: all that is left) `after_seconds` after the order
    became eligible, at `price` (None: the symbol's current price)."""

    after_seconds: float = 0.0
    quantity: int | None = None
    price: float | None = None


@dataclass(frozen=True)
class OrderScript:
    reject: str = ""  # place raises VenueRejected with this reason; the order never exists
    timeouts_before_accept: int = 0  # this many place calls time out with no order created
    timeout_after_accept: bool = False  # the order is created, but place still times out
    fills: Sequence[FillStep] | None = None  # None: default fill; (): never fills
    fill_on_cancel: int = 0  # shares that fill while a cancel is in flight


@dataclass
class _Order:
    order: VenueOrder
    venue_order_id: str
    eligible_at: dt.datetime
    expires_at: dt.datetime
    steps: list[FillStep]
    fill_on_cancel: int
    updated_at: dt.datetime
    status: VenueStatus = "pending"
    filled: int = 0
    filled_value: float = 0.0
    cancel_at: dt.datetime | None = None

    @property
    def open(self) -> bool:
        return self.status in ("pending", "partially_filled")

    @property
    def remaining(self) -> int:
        return self.order.quantity - self.filled

    def state(self) -> VenueOrderState:
        return VenueOrderState(
            client_order_id=self.order.client_order_id,
            venue_order_id=self.venue_order_id,
            status=self.status,
            quantity=self.order.quantity,
            filled_quantity=self.filled,
            filled_avg_price=self.filled_value / self.filled if self.filled else None,
            updated_at=self.updated_at,
        )


@dataclass
class _Holding:
    quantity: int
    cost: float  # total paid for the shares held


@dataclass
class FakeVenue:
    clock: dt.datetime  # timezone-aware; the venue's "now"
    cash: float = 10_000.0
    cancel_latency: float = 1.0  # seconds between a cancel request and the cancel
    holidays: set[dt.date] = field(default_factory=set)
    half_days: set[dt.date] = field(default_factory=set)
    # Account flags a setup check reads. Defaults are the safe paper configuration.
    trading_blocked: bool = False
    shorting_enabled: bool = False
    margin_multiplier: float = 1.0
    killed: bool = False  # True: place raises KillSwitchEngaged, as a real adapter would

    def __post_init__(self) -> None:
        if self.clock.tzinfo is None:
            raise ValueError("the fake venue clock must be timezone-aware")
        self.prices: dict[str, float] = {}
        self._price_set_at: dict[str, dt.datetime] = {}
        self.calls: list[tuple[str, str]] = []  # (method, client order id or "")
        self.place_attempts: list[VenueOrder] = []  # every place call, even ones that failed
        self._orders: dict[str, _Order] = {}  # by client order id, in placement order
        self._holdings: dict[str, _Holding] = {}
        self._scripts: dict[str, OrderScript] = {}
        self._timeouts_left: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ test controls

    def advance(self, seconds: float) -> None:
        """Move the clock forward. Also usable as the `sleep` of the code under test."""
        if seconds < 0:
            raise ValueError("the clock only moves forward")
        self.clock += dt.timedelta(seconds=seconds)

    def set_now(self, when: dt.datetime) -> None:
        if when < self.clock:
            raise ValueError("the clock only moves forward")
        self.clock = when

    def set_price(self, symbol: str, price: float | None) -> None:
        """The price `symbol` trades at from now on. None: it stops trading (a halt)."""
        self._sync()  # fills up to now happen at the old price
        self._price_set_at[symbol] = self.clock
        if price is None:
            self.prices.pop(symbol, None)
        else:
            self.prices[symbol] = price

    def hold(self, symbol: str, quantity: int, avg_price: float) -> None:
        """Start with a position, as if bought earlier."""
        self._holdings[symbol] = _Holding(quantity, quantity * avg_price)

    def script(self, key: str, script: OrderScript) -> None:
        """Script the next order whose client order id, or else symbol, is `key`."""
        self._scripts[key] = script
        self._timeouts_left[key] = script.timeouts_before_accept

    def fail_next(self, method: str, times: int = 1) -> None:
        """Make the next `times` calls to `method` raise VenueUnavailable before acting."""
        self._failures[method] = self._failures.get(method, 0) + times

    # ------------------------------------------------------------------ Venue

    def now(self) -> dt.datetime:
        self._enter("now")
        return self.clock

    def sessions(self, start: dt.date, end: dt.date) -> list[MarketSession]:
        self._enter("sessions")
        out = []
        day = start
        while day <= end:
            if day.weekday() < 5 and day not in self.holidays:
                out.append(self._session(day))
            day += dt.timedelta(days=1)
        return out

    def account(self) -> VenueAccount:
        self._enter("account")
        market_value = sum(
            h.quantity * self.prices.get(s, h.cost / h.quantity) for s, h in self._holdings.items()
        )
        return VenueAccount(
            account_id="FAKE-PAPER",
            cash=self.cash,
            buying_power=self.cash - self._reserved_cash(),
            equity=self.cash + market_value,
            trading_blocked=self.trading_blocked,
            shorting_enabled=self.shorting_enabled,
            margin_multiplier=self.margin_multiplier,
        )

    def positions(self) -> list[VenuePosition]:
        self._enter("positions")
        return [
            VenuePosition(symbol=s, quantity=h.quantity, avg_entry_price=h.cost / h.quantity)
            for s, h in sorted(self._holdings.items())
        ]

    def open_orders(self) -> list[VenueOrderState]:
        self._enter("open_orders")
        return [o.state() for o in self._orders.values() if o.open]

    def get_order(self, client_order_id: str) -> VenueOrderState | None:
        self._enter("get_order", client_order_id)
        found = self._orders.get(client_order_id)
        return found.state() if found is not None else None

    def place(self, order: VenueOrder) -> VenueOrderState:
        self.place_attempts.append(order)
        if self.killed:
            self.calls.append(("place", order.client_order_id))
            raise KillSwitchEngaged("kill latch is set")
        self._enter("place", order.client_order_id)
        existing = self._orders.get(order.client_order_id)
        if existing is not None:
            return existing.state()  # a duplicate client id is never a second order

        key = order.client_order_id if order.client_order_id in self._scripts else order.symbol
        script = self._scripts.get(key, OrderScript())
        if self._timeouts_left.get(key, 0) > 0:
            self._timeouts_left[key] -= 1
            raise VenueUnavailable(f"timeout placing {order.client_order_id} (not accepted)")
        if script.reject:
            self._scripts.pop(key, None)
            raise VenueRejected(script.reject)
        self._check_acceptable(order)

        eligible_at, expires_at = self._window(order)
        steps = [FillStep()] if script.fills is None else list(script.fills)
        placed = _Order(
            order=order,
            venue_order_id=f"v-{self._next_id:06d}",
            eligible_at=eligible_at,
            expires_at=expires_at,
            steps=steps,
            fill_on_cancel=script.fill_on_cancel,
            updated_at=self.clock,
        )
        self._next_id += 1
        self._orders[order.client_order_id] = placed
        self._scripts.pop(key, None)  # a script applies to one order
        self._sync()
        if script.timeout_after_accept:
            raise VenueUnavailable(f"timeout placing {order.client_order_id} (accepted)")
        return placed.state()

    def cancel(self, client_order_id: str) -> None:
        self._enter("cancel", client_order_id)
        found = self._orders.get(client_order_id)
        if found is None:
            raise VenueRejected(f"no order {client_order_id}")
        self._request_cancel(found)

    def cancel_all(self) -> None:
        self._enter("cancel_all")
        for order in self._orders.values():
            self._request_cancel(order)

    # ------------------------------------------------------------------ internals

    def _enter(self, method: str, client_order_id: str = "") -> None:
        """Log the call, apply a scripted failure, and bring the venue up to now."""
        self.calls.append((method, client_order_id))
        if self._failures.get(method, 0) > 0:
            self._failures[method] -= 1
            raise VenueUnavailable(f"timeout in {method}")
        self._sync()

    def _session(self, day: dt.date) -> MarketSession:
        close = HALF_DAY_CLOSE if day in self.half_days else CLOSE
        return MarketSession(
            date=day,
            open=dt.datetime.combine(day, OPEN, MARKET_TZ),
            close=dt.datetime.combine(day, close, MARKET_TZ),
        )

    def _session_for(self, when: dt.datetime) -> MarketSession:
        """The session `when` falls in, or the next one to open."""
        day = when.astimezone(MARKET_TZ).date()
        for _ in range(14):
            if day.weekday() < 5 and day not in self.holidays:
                session = self._session(day)
                if when < session.close:
                    return session
            day += dt.timedelta(days=1)
        raise AssertionError("no session within two weeks")

    def _window(self, order: VenueOrder) -> tuple[dt.datetime, dt.datetime]:
        """When the order can start filling and when it expires."""
        session = self._session_for(self.clock)
        if order.time_in_force == "opg":
            return session.open, session.open + dt.timedelta(seconds=OPG_GRACE_SECONDS)
        return max(self.clock, session.open), session.close

    def _check_acceptable(self, order: VenueOrder) -> None:
        if self.trading_blocked:
            raise VenueRejected("account is blocked from trading")
        if order.time_in_force == "opg":
            session = self._session_for(self.clock)
            if self.clock >= session.open:
                raise VenueRejected("opg orders are only accepted before the open")
        if order.side == "sell":
            held = self._holdings.get(order.symbol)
            reserved = sum(
                o.remaining
                for o in self._orders.values()
                if o.open and o.order.side == "sell" and o.order.symbol == order.symbol
            )
            if order.quantity > (held.quantity if held else 0) - reserved:
                raise VenueRejected(f"insufficient quantity of {order.symbol}")
        else:
            limit = order.limit_price
            assert limit is not None  # VenueOrder: a buy is a limit order
            if order.quantity * limit > self.cash - self._reserved_cash() + 1e-9:
                raise VenueRejected("insufficient buying power")

    def _reserved_cash(self) -> float:
        return sum(
            o.remaining * (o.order.limit_price or 0.0)
            for o in self._orders.values()
            if o.open and o.order.side == "buy"
        )

    def _request_cancel(self, order: _Order) -> None:
        if order.open and order.cancel_at is None:
            order.cancel_at = self.clock + dt.timedelta(seconds=self.cancel_latency)

    def _sync(self) -> None:
        """Apply everything due by now, in placement order."""
        now = self.clock
        for order in self._orders.values():
            if not order.open:
                continue
            cutoff = min(c for c in (order.cancel_at, order.expires_at, now) if c is not None)
            while order.open and order.steps:
                step = order.steps[0]
                due = order.eligible_at + dt.timedelta(seconds=step.after_seconds)
                if due > cutoff or not self._fill(order, step.quantity, step.price, due):
                    break
                order.steps.pop(0)
            if not order.open:
                continue
            if order.cancel_at is not None and order.cancel_at <= now:
                if order.fill_on_cancel:
                    self._fill(order, order.fill_on_cancel, None, order.cancel_at)
                if order.open:
                    self._close(order, "canceled", order.cancel_at)
            elif order.expires_at <= now:
                self._close(order, "expired", order.expires_at)

    def _fill(
        self, order: _Order, quantity: int | None, price: float | None, when: dt.datetime
    ) -> bool:
        """Fill up to `quantity` shares. False if the order cannot trade at this price."""
        spec = order.order
        if price is None:
            price = self.prices.get(spec.symbol)
            if price is None:
                return False  # not trading
            # A fill at the market price cannot predate that price.
            when = max(when, self._price_set_at.get(spec.symbol, when))
        if spec.limit_price is not None and price > spec.limit_price:
            return False  # limit buy above its cap
        shares = min(order.remaining if quantity is None else quantity, order.remaining)
        if shares <= 0:
            return True
        holding = self._holdings.setdefault(spec.symbol, _Holding(0, 0.0))
        if spec.side == "buy":
            self.cash -= shares * price
            holding.quantity += shares
            holding.cost += shares * price
        else:
            self.cash += shares * price
            holding.cost -= shares * (holding.cost / holding.quantity)
            holding.quantity -= shares
        if holding.quantity == 0:
            del self._holdings[spec.symbol]
        order.filled += shares
        order.filled_value += shares * price
        order.status = "filled" if order.remaining == 0 else "partially_filled"
        order.updated_at = when
        return True

    @staticmethod
    def _close(order: _Order, status: VenueStatus, when: dt.datetime) -> None:
        order.status = status
        order.updated_at = when
