"""Hard risk limits: the last check between construction and the broker.

Construction decides what to trade and how much. The risk manager only ever makes that
smaller: it can shrink an order or reject it, never enlarge or add one.

Two rules shape everything here:

- **A risk-reducing order always passes.** Risk-reducing is decided by effect, not by
  label: a sell of at most the held quantity. It is approved in full whatever the state
  (entries halted, daily cap hit, over every limit), so no limit can trap a position.
  A sell larger than the position is cut to the position, so it can never open a short.
- **Buys fail closed.** A buy is cut to the largest whole quantity that satisfies every
  limit, or rejected. Anything the manager cannot evaluate (no price) is rejected.

Limits on a buy, all measured at the worst-case entry price (the last close marked up by
`entry_markup`: estimated slippage in backtests, the entry limit cap in paper trading):

- `max_position`: the resulting position is at most `max_position_pct` of equity.
- `max_exposure`: long market value after today's exits, plus entries approved so far, is
  at most `max_exposure_pct` of equity.
- `max_order_notional`, `max_orders`: fat-finger limits from fixed settings.
- `daily_loss_cap`: after a close-to-close loss of `daily_loss_cap_pct`, that session's
  entries are rejected. This is an entry brake, not a stop: nothing watches prices
  during the day.
- `drawdown_halt`: once equity is `max_drawdown_halt_pct` below its peak, entries are
  rejected until an operator resets the halt.

The manager is deterministic and does no I/O. `RiskState` is everything it remembers
between sessions, so the caller can persist it and rebuild the manager later.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field

from harness.config import OrderLimits, RiskConfig
from harness.models import EngineEvent, EventKind, FrozenModel, Order, PortfolioSnapshot, Stock

RiskRule = Literal[
    "max_position",
    "max_exposure",
    "max_order_notional",
    "max_orders",
    "daily_loss_cap",
    "drawdown_halt",
    "not_held",
    "no_price",
]


class RiskStateError(RuntimeError):
    """The risk manager was driven out of order (sessions backwards, bad reset)."""


class RiskState(FrozenModel):
    """Everything risk remembers between sessions."""

    peak_equity: float = Field(gt=0)
    prior_equity: float | None = None  # equity at the last mark
    marked_on: dt.date | None = None  # session of the last mark
    # Signal date whose entries the daily loss cap rejects (the session of the bad close).
    entries_blocked_for: dt.date | None = None
    halted: bool = False  # drawdown halt: entries rejected until an operator resets
    halt_reason: str = ""

    @classmethod
    def initial(cls, equity: float) -> RiskState:
        """State for a fresh account worth `equity`, before its first session."""
        return cls(peak_equity=equity)


class RiskDecision(FrozenModel):
    order: Order  # as proposed
    approved_quantity: int = Field(ge=0)  # 0 means rejected
    rule: RiskRule | None = None  # the limit that bound, if any
    detail: str = ""


class RiskReview(FrozenModel):
    decisions: tuple[RiskDecision, ...]  # one per proposed order, input order
    orders: tuple[Order, ...]  # approved orders with their approved quantities, input order
    events: tuple[EngineEvent, ...]


def _whole_units(room: float, unit_price: float) -> int:
    """The largest whole n with n * unit_price <= room (0 if none).

    The division is only a first guess: float rounding can leave it one off in either
    direction. The answer is settled by the products, so it never breaches the limit it
    was computed from and never gives up a unit that fits.
    """
    if not math.isfinite(room) or room <= 0:
        return 0
    n = max(math.floor(room / unit_price), 0)
    while n > 0 and n * unit_price > room:
        n -= 1
    while (n + 1) * unit_price <= room:
        n += 1
    return n


class RiskManager:
    """Applies one risk preset and the fixed order limits. See the module docstring."""

    def __init__(
        self, cfg: RiskConfig, limits: OrderLimits, entry_markup: float, state: RiskState
    ) -> None:
        if not math.isfinite(entry_markup) or entry_markup < 0:
            raise ValueError("entry_markup must be zero or positive")
        self._cfg = cfg
        self._limits = limits
        self._markup = entry_markup
        self._state = state

    @property
    def state(self) -> RiskState:
        return self._state

    # ------------------------------------------------------------------ session mark

    def on_mark(self, as_of: dt.date, equity: float) -> tuple[EngineEvent, ...]:
        """Record the equity at the close of `as_of`; trip the daily cap or the halt.

        Call once per session, in date order, before `review`. A repeated call for the
        same session and equity changes nothing (so a rerun after a crash cannot erase a
        tripped daily cap); a different equity for the same session, or an earlier
        session, raises.
        """
        state = self._state
        if state.marked_on is not None:
            if as_of < state.marked_on:
                raise RiskStateError(f"mark for {as_of} after {state.marked_on}")
            if as_of == state.marked_on:
                if equity != state.prior_equity:
                    raise RiskStateError(
                        f"second mark for {as_of} with a different equity "
                        f"({equity} after {state.prior_equity})"
                    )
                return ()
        if not math.isfinite(equity):
            raise RiskStateError(f"equity for {as_of} is not a number")

        events: list[EngineEvent] = []
        blocked: dt.date | None = None
        prior = state.prior_equity
        if prior is not None and prior > 0:
            loss = 1 - equity / prior
            if loss >= self._cfg.daily_loss_cap_pct:
                blocked = as_of
                events.append(
                    EngineEvent(
                        date=as_of,
                        kind="entries_halted",
                        symbol="",
                        detail=(
                            f"daily_loss_cap: equity fell {loss:.2%} in one session "
                            f"(cap {self._cfg.daily_loss_cap_pct:.2%}); entries for this "
                            "session rejected"
                        ),
                    )
                )
        peak = max(state.peak_equity, equity)
        halted, halt_reason = state.halted, state.halt_reason
        drawdown = 1 - equity / peak
        if not halted and drawdown >= self._cfg.max_drawdown_halt_pct:
            halted = True
            halt_reason = (
                f"equity {equity:.2f} is {drawdown:.2%} below its peak {peak:.2f} "
                f"(limit {self._cfg.max_drawdown_halt_pct:.2%}) on {as_of}"
            )
            events.append(
                EngineEvent(
                    date=as_of,
                    kind="entries_halted",
                    symbol="",
                    detail=f"drawdown_halt: {halt_reason}; entries rejected until reset",
                )
            )
        self._state = RiskState(
            peak_equity=peak,
            prior_equity=equity,
            marked_on=as_of,
            entries_blocked_for=blocked,
            halted=halted,
            halt_reason=halt_reason,
        )
        return tuple(events)

    def reset_halt(self) -> None:
        """Operator action: lift the drawdown halt.

        The peak is rebased to the last marked equity. Otherwise the next mark, still far
        below the old peak, would halt again at once. The caller records who reset it and
        why.
        """
        state = self._state
        if not state.halted:
            raise RiskStateError("no drawdown halt to reset")
        if state.prior_equity is None or state.prior_equity <= 0:
            raise RiskStateError("cannot rebase the peak: no positive equity marked")
        self._state = state.model_copy(
            update={"halted": False, "halt_reason": "", "peak_equity": state.prior_equity}
        )

    # ------------------------------------------------------------------ review

    def review(
        self,
        orders: Sequence[Order],
        portfolio: PortfolioSnapshot,
        marks: Mapping[str, float],
    ) -> RiskReview:
        """Check the orders built at the close of `portfolio.as_of`.

        `marks` is the latest close per symbol: it values held positions and, marked up
        by `entry_markup`, gives each buy its worst-case price. Sells are decided first,
        wherever they are in `orders`, so every buy is judged against the portfolio as it
        will be after today's approved exits. Decisions come back in input order.
        """
        if self._state.marked_on != portfolio.as_of:
            raise RiskStateError(
                f"review for {portfolio.as_of} without a mark for that session "
                f"(last mark {self._state.marked_on})"
            )
        for order in orders:
            leg = order.legs[0] if len(order.legs) == 1 else None
            if leg is None or not isinstance(leg.instrument, Stock) or leg.ratio != 1:
                raise NotImplementedError("risk review supports one-leg stock orders only")

        decisions: dict[int, RiskDecision] = {}
        held = {s: p.quantity for s, p in portfolio.positions.items()}
        exposure = portfolio.equity - portfolio.cash  # long market value at the marks

        # Sells first: they free exposure for the buys, and they always pass.
        for i, order in enumerate(orders):
            if order.legs[0].side != "sell":
                continue
            symbol = order.symbol
            available = held.get(symbol, 0)
            quantity = min(order.quantity, available)
            if quantity == 0:
                decisions[i] = RiskDecision(
                    order=order,
                    approved_quantity=0,
                    rule="not_held",
                    detail=f"sell {order.quantity}, none held",
                )
                continue
            held[symbol] = available - quantity
            mark = marks.get(symbol)
            if mark is not None and math.isfinite(mark) and mark > 0:
                exposure -= quantity * mark
            # No usable mark: the exit is not credited, so buys are judged conservatively.
            cut = quantity < order.quantity
            decisions[i] = RiskDecision(
                order=order,
                approved_quantity=quantity,
                rule="not_held" if cut else None,
                detail=f"sell {order.quantity} cut to the {quantity} held" if cut else "",
            )

        orders_left = self._limits.max_orders_per_session - sum(
            1 for d in decisions.values() if d.approved_quantity > 0
        )
        for i, order in enumerate(orders):
            if order.legs[0].side != "buy":
                continue
            decision = self._review_buy(order, portfolio, marks, held, exposure, orders_left)
            decisions[i] = decision
            if decision.approved_quantity > 0:
                orders_left -= 1
                exposure += decision.approved_quantity * self._entry_price(marks[order.symbol])
                held[order.symbol] = held.get(order.symbol, 0) + decision.approved_quantity

        ordered = tuple(decisions[i] for i in range(len(orders)))
        return RiskReview(
            decisions=ordered,
            orders=tuple(
                d.order
                if d.approved_quantity == d.order.quantity
                else d.order.model_copy(update={"quantity": d.approved_quantity})
                for d in ordered
                if d.approved_quantity > 0
            ),
            events=tuple(e for d in ordered if (e := self._event(d, portfolio.as_of)) is not None),
        )

    # ------------------------------------------------------------------ helpers

    def _entry_price(self, mark: float) -> float:
        return mark * (1 + self._markup)

    def _review_buy(
        self,
        order: Order,
        portfolio: PortfolioSnapshot,
        marks: Mapping[str, float],
        held: Mapping[str, int],
        exposure: float,
        orders_left: int,
    ) -> RiskDecision:
        def reject(rule: RiskRule, detail: str) -> RiskDecision:
            return RiskDecision(order=order, approved_quantity=0, rule=rule, detail=detail)

        state, cfg = self._state, self._cfg
        if state.halted:
            return reject("drawdown_halt", state.halt_reason)
        if state.entries_blocked_for == order.created_on:
            return reject("daily_loss_cap", f"daily loss cap hit on {order.created_on}")
        if orders_left <= 0:
            return reject(
                "max_orders", f"more than {self._limits.max_orders_per_session} orders in a session"
            )
        symbol = order.symbol
        mark = marks.get(symbol)
        if mark is None or not math.isfinite(mark) or mark <= 0:
            return reject("no_price", f"no close to price {symbol}")
        equity = portfolio.equity
        if not math.isfinite(equity) or equity <= 0:
            return reject("max_exposure", f"equity is {equity}")

        price = self._entry_price(mark)
        # Room left under each limit, in dollars at the worst-case price. The first of
        # the tightest limits names the decision.
        rooms: tuple[tuple[RiskRule, float], ...] = (
            ("max_position", cfg.max_position_pct * equity - held.get(symbol, 0) * mark),
            ("max_exposure", cfg.max_exposure_pct * equity - exposure),
            ("max_order_notional", self._limits.max_order_notional),
        )
        quantity, rule, room = order.quantity, None, 0.0
        for name, limit_room in rooms:
            allowed = _whole_units(limit_room, price)
            if allowed < quantity:
                quantity, rule, room = allowed, name, limit_room
        if rule is None:
            return RiskDecision(order=order, approved_quantity=quantity)
        return RiskDecision(
            order=order,
            approved_quantity=quantity,
            rule=rule,
            detail=f"buy {order.quantity} cut to {quantity}: {room:.2f} of room at {price:.4f}",
        )

    @staticmethod
    def _event(decision: RiskDecision, as_of: dt.date) -> EngineEvent | None:
        if decision.approved_quantity == decision.order.quantity:
            return None
        kind: EventKind = "risk_rejected" if decision.approved_quantity == 0 else "risk_shrunk"
        return EngineEvent(
            date=as_of,
            kind=kind,
            symbol=decision.order.symbol,
            detail=f"{decision.rule}: {decision.detail}",
        )
