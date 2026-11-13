"""SimBroker: simulated market-on-open fills for backtests."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence

from harness.broker.base import Broker, LookAheadError
from harness.config import CostConfig
from harness.costs import apply_costs
from harness.models import (
    AccountState,
    Fill,
    Order,
    OrderUpdate,
    OrderUpdateReason,
    Position,
    Stock,
)


class SimBroker(Broker):
    """Fills queued orders at the next session's open, plus slippage and commission.

    Rules:
    - An order created on date d fills only in a session s > d; anything else raises
      LookAheadError. Sessions must be passed in strictly increasing order.
    - Each session it receives only that session's opening prices: never highs, lows,
      or closes, so it cannot fill at any other price.
    - Orders are day orders for the next session. If the symbol has no open that
      session, the order is canceled ("no_bar"); the strategy can signal again.
    - Sells execute before buys, so exit proceeds fund same-session entries. Within each
      group, orders execute in submission order (construction's rank order).
    - Buys never take cash negative: the quantity is cut to the whole shares that fit at
      the fill price ("reduced_for_cash"), or the order is canceled if none fit. The
      guarantee covers buys only. A sell (or forced close) whose commission exceeds its
      proceeds reduces cash and can take it below zero; that needs a nonzero
      `commission_per_order`, so it is unreachable at the configured $0.
    - A sell for a symbol not held is canceled ("not_held"); a sell larger than the
      position sells the position.
    - Slice 1 is long-only stock: multi-leg or option orders raise, and a buy for a
      symbol already held raises (construction never adds to a position).

    Order ids are sequential ("sim-000001"), so runs are deterministic.
    """

    def __init__(self, starting_cash: float, costs: CostConfig) -> None:
        if not math.isfinite(starting_cash) or starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self._cash = float(starting_cash)
        self._costs = costs
        self._positions: dict[str, Position] = {}
        self._queue: list[tuple[str, Order]] = []
        self._updates: list[OrderUpdate] = []
        self._next_id = 1
        self._last_session: dt.date | None = None

    # ------------------------------------------------------------------ interface

    def submit(self, orders: Sequence[Order]) -> list[str]:
        ids = []
        for order in orders:
            self._check_supported(order)
            if self._last_session is not None and order.created_on < self._last_session:
                raise ValueError(
                    f"order created on {order.created_on} submitted after session "
                    f"{self._last_session}"
                )
            order_id = self._new_id()
            self._queue.append((order_id, order))
            ids.append(order_id)
        return ids

    def on_session_open(self, session: dt.date, opens: Mapping[str, float]) -> None:
        if self._last_session is not None and session <= self._last_session:
            raise ValueError(f"session {session} is not after {self._last_session}")
        queue, self._queue = self._queue, []
        for order_id, order in queue:
            if order.created_on >= session:
                raise LookAheadError(
                    f"order {order_id} created on {order.created_on} cannot fill in "
                    f"session {session}"
                )
        sells = [(i, o) for i, o in queue if o.legs[0].side == "sell"]
        buys = [(i, o) for i, o in queue if o.legs[0].side == "buy"]
        for order_id, order in sells + buys:
            self._execute(order_id, order, session, opens.get(order.symbol))
        self._last_session = session

    def force_close(self, symbol: str, price: float, price_date: dt.date, session: dt.date) -> str:
        position = self._positions.get(symbol)
        if position is None:
            raise ValueError(f"force_close: {symbol} is not held")
        if self._last_session is not None and session < self._last_session:
            raise ValueError(f"force_close session {session} is before {self._last_session}")
        order_id = self._new_id()
        cost = apply_costs("sell", price, position.quantity, self._costs)
        fill = Fill(
            order_id=order_id,
            kind="forced_close",
            signal_date=session,
            fill_date=session,
            price_date=price_date,
            instrument=position.instrument,
            side="sell",
            quantity=position.quantity,
            reference_price=price,
            price=cost.fill_price,
            slippage_cost=cost.slippage_cost,
            commission=cost.commission,
        )
        self._apply(fill)
        self._updates.append(
            OrderUpdate(order_id=order_id, status="filled", fills=(fill,), reason="no_data")
        )
        return order_id

    def poll_updates(self) -> list[OrderUpdate]:
        updates, self._updates = self._updates, []
        return updates

    def account(self) -> AccountState:
        return AccountState(cash=self._cash, positions=dict(self._positions))

    def cancel_all(self, reason: OrderUpdateReason) -> list[str]:
        queue, self._queue = self._queue, []
        for order_id, _ in queue:
            self._cancel(order_id, reason)
        return [order_id for order_id, _ in queue]

    # ------------------------------------------------------------------ internals

    def _new_id(self) -> str:
        order_id = f"sim-{self._next_id:06d}"
        self._next_id += 1
        return order_id

    @staticmethod
    def _check_supported(order: Order) -> None:
        if len(order.legs) != 1 or not isinstance(order.legs[0].instrument, Stock):
            raise NotImplementedError("SimBroker supports one-leg stock orders only (slice 1)")
        if order.legs[0].ratio != 1:
            raise NotImplementedError("stock legs must have ratio 1")

    def _cancel(self, order_id: str, reason: OrderUpdateReason) -> None:
        self._updates.append(OrderUpdate(order_id=order_id, status="canceled", reason=reason))

    def _execute(
        self, order_id: str, order: Order, session: dt.date, open_price: float | None
    ) -> None:
        if open_price is None or not math.isfinite(open_price) or open_price <= 0:
            self._cancel(order_id, "no_bar")
            return
        leg = order.legs[0]
        symbol = order.symbol
        reason: OrderUpdateReason = ""
        if leg.side == "sell":
            held = self._positions.get(symbol)
            if held is None:
                self._cancel(order_id, "not_held")
                return
            quantity = min(order.quantity, held.quantity)
        else:
            if symbol in self._positions:
                raise ValueError(f"buy for {symbol}, already held: no pyramiding in slice 1")
            quantity = order.quantity
            affordable = self._affordable(open_price)
            if affordable < quantity:
                if affordable < 1:
                    self._cancel(order_id, "insufficient_cash")
                    return
                quantity, reason = affordable, "reduced_for_cash"

        cost = apply_costs(leg.side, open_price, quantity, self._costs)
        fill = Fill(
            order_id=order_id,
            kind="next_open",
            signal_date=order.created_on,
            fill_date=session,
            price_date=session,
            instrument=leg.instrument,
            side=leg.side,
            quantity=quantity,
            reference_price=open_price,
            price=cost.fill_price,
            slippage_cost=cost.slippage_cost,
            commission=cost.commission,
        )
        self._apply(fill)
        self._updates.append(
            OrderUpdate(order_id=order_id, status="filled", fills=(fill,), reason=reason)
        )

    def _affordable(self, open_price: float) -> int:
        """Whole shares whose cost plus commission fits in cash at this open."""
        price = apply_costs("buy", open_price, 1, self._costs).fill_price
        commission = self._costs.commission_per_order
        quantity = max(0, math.floor((self._cash - commission) / price))
        while quantity > 0 and quantity * price + commission > self._cash:
            quantity -= 1  # float rounding guard
        return quantity

    def _apply(self, fill: Fill) -> None:
        symbol = fill.instrument.key
        self._cash += fill.cash_delta
        if fill.side == "buy":
            self._positions[symbol] = Position(
                instrument=fill.instrument,
                quantity=fill.quantity,
                avg_price=fill.price,
                entry_date=fill.fill_date,
                entry_order_id=fill.order_id,
            )
            return
        held = self._positions[symbol]
        remaining = held.quantity - fill.quantity
        if remaining == 0:
            del self._positions[symbol]
        else:
            self._positions[symbol] = held.model_copy(update={"quantity": remaining})
