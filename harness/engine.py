"""The daily engine loop, shared by backtest now and paper/live later.

Each session has two phases:

- **Open:** the broker executes orders queued at the previous close, at this session's
  open. The engine records the resulting fills and cancellations.
- **Close:** the engine appends this session's bars to the strategy's view, force-closes
  positions in symbols that stopped trading, marks the portfolio to market, asks the
  strategy for signals, turns them into orders through construction, passes them through
  the risk manager (if the run has one), and submits what is left.

Sessions before the first traded one go through `warm_up` instead: their bars enter the
view and nothing else happens. Everything the engine keeps between sessions about the
market (the view, last closes, how long a held symbol has had no bar) is a function of
the bars it was fed, so an engine rebuilt from the same bars is in the same state.

`simulate` drives both phases over a BarPanel for backtests. The engine never compares
dates against any cutoff: it runs whatever panel the data loader produced, which is where
the holdout boundary is enforced. It reads no clock, uses no randomness, and never
iterates over a set, so identical inputs give identical results.

Every call into strategy code (`on_start`, `on_bar`) runs inside `harness.guard`, which
rejects file opens, network use, and process starts for the duration of the call.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from typing import assert_never

import numpy as np
from pydantic import Field

from harness.broker.base import Broker
from harness.config import ConstructionConfig, CostConfig
from harness.construction import build_orders
from harness.data import BarPanel
from harness.guard import strategy_code
from harness.models import (
    EngineEvent,
    EventKind,
    Fill,
    FrozenModel,
    Order,
    OrderUpdate,
    PortfolioSnapshot,
    Position,
    Signal,
    Stock,
)
from harness.risk import RiskManager
from harness.view import PRICE_FIELDS, BarBuffer
from strategies.base import Strategy


class StrategyError(RuntimeError):
    """A strategy returned something other than valid slice 1 signals."""


class EquityPoint(FrozenModel):
    date: dt.date
    cash: float
    market_value: float
    equity: float
    n_positions: int = Field(ge=0)


class BacktestResult(FrozenModel):
    """Everything a run produced. The report is built from this alone."""

    strategy: str
    start: dt.date  # first traded session (after warm-up)
    end: dt.date  # last session
    warmup_sessions: int = Field(ge=0)
    starting_cash: float
    equity: tuple[EquityPoint, ...]
    orders: dict[str, Order]  # broker order id -> order, in submission order
    updates: tuple[OrderUpdate, ...]
    events: tuple[EngineEvent, ...]
    final_positions: dict[str, Position]
    final_marks: dict[str, float]  # last close used to value each final position
    data_fingerprint: str

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(f for u in self.updates for f in u.fills)


class Engine:
    """Two-phase session logic. Holds the strategy's view, the broker, and the run log."""

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        universe: Sequence[str],
        construction: ConstructionConfig,
        costs: CostConfig,
        capacity: int,
        risk: RiskManager | None = None,
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.risk = risk  # None: orders go from construction to the broker unchecked
        self.universe = tuple(universe)
        self.construction = construction
        self.costs = costs
        self._buffer = BarBuffer(self.universe, capacity)
        self._universe_set = frozenset(self.universe)  # membership checks only
        self._last_close: dict[str, float] = {}
        self._last_close_date: dict[str, dt.date] = {}
        self.orders: dict[str, Order] = {}
        self.updates: list[OrderUpdate] = []
        self.events: list[EngineEvent] = []
        self.equity: list[EquityPoint] = []
        with strategy_code():
            strategy.on_start(self.universe)

    # ------------------------------------------------------------------ phases

    def on_session_open(self, session: dt.date, opens: Mapping[str, float]) -> None:
        """Let the broker execute orders queued at the previous close."""
        self.broker.on_session_open(session, opens)
        self._collect_updates(session)

    def warm_up(self, session: dt.date, row: Mapping[str, np.ndarray]) -> None:
        """Add the bars of a session that is not traded (history before the first traded
        session) to the view. The broker is not touched: no orders, no forced closes, no
        equity point, and the strategy is not called."""
        self._buffer.append(session, row)
        self._update_closes(session, row["close"])

    def on_session_close(self, session: dt.date, row: Mapping[str, np.ndarray]) -> None:
        """Process the close of `session`: bars, forced closes, mark, signals, orders."""
        self._buffer.append(session, row)
        closes = self._update_closes(session, row["close"])
        self._force_close_stale(session)
        snapshot = self._mark(session)
        if self.risk is not None:
            self.events.extend(self.risk.on_mark(session, snapshot.equity))
        view = self._buffer.view()
        # The strategy gets its own deep copy: the snapshot's positions dict is mutable, and
        # construction below must size and count slots from what the broker reported, not
        # from whatever the strategy did to its copy.
        with strategy_code():  # no file or network access from strategy code
            emitted = self.strategy.on_bar(view, snapshot.model_copy(deep=True))
        signals = self._validated(emitted, session)
        result = build_orders(signals, snapshot, closes, self.construction, self.costs)
        self.events.extend(result.events)
        orders = result.orders
        if self.risk is not None:
            # Judged against the engine's own snapshot (what the broker reported), with
            # the closes positions are marked at. Risk can only shrink or drop orders.
            review = self.risk.review(orders, snapshot, self._last_close)
            self.events.extend(review.events)
            orders = review.orders
        ids = self.broker.submit(orders)
        for order_id, order in zip(ids, orders, strict=True):
            self.orders[order_id] = order

    def finish(self, session: dt.date) -> None:
        """Cancel orders that can no longer fill because the data ended."""
        self.broker.cancel_all("end_of_data")
        self._collect_updates(session)

    def last_close(self, symbol: str) -> float:
        """The most recent close seen for `symbol` (what positions are marked at)."""
        return self._last_close[symbol]

    # ------------------------------------------------------------------ helpers

    def _log(self, session: dt.date, kind: EventKind, symbol: str, detail: str = "") -> None:
        self.events.append(EngineEvent(date=session, kind=kind, symbol=symbol, detail=detail))

    def _collect_updates(self, session: dt.date) -> None:
        for update in self.broker.poll_updates():
            self.updates.append(update)
            order = self.orders.get(update.order_id)
            if order is not None:
                symbol = order.symbol
            elif update.fills:  # a forced close: the broker's own order, not one of ours
                symbol = update.fills[0].instrument.key
            else:
                raise RuntimeError(
                    f"broker reported {update.status} for unknown order {update.order_id}"
                )
            filled = sum(f.quantity for f in update.fills)
            ordered = order.quantity if order is not None else "?"
            detail = f"{update.order_id}: {update.reason}"
            match update.status:
                case "canceled":
                    kind: EventKind = (
                        "order_unfilled_end" if update.reason == "end_of_data" else "order_canceled"
                    )
                    self._log(session, kind, symbol, detail)
                case "rejected":
                    self._log(session, "order_rejected", symbol, detail)
                case "expired":
                    self._log(session, "order_expired", symbol, detail)
                case "partially_filled":
                    self._log(
                        session, "order_partial", symbol,
                        f"{update.order_id}: {filled} of {ordered} shares in this update",
                    )  # fmt: skip
                case "filled":
                    if update.reason == "reduced_for_cash":
                        self._log(
                            session, "order_reduced", symbol,
                            f"{update.order_id}: {filled} of {ordered} shares",
                        )  # fmt: skip
                case _:
                    assert_never(update.status)

    def _update_closes(self, session: dt.date, close_row: np.ndarray) -> dict[str, float]:
        closes: dict[str, float] = {}
        for j, symbol in enumerate(self.universe):
            value = float(close_row[j])
            if math.isnan(value):
                continue
            closes[symbol] = value
            self._last_close[symbol] = value
            self._last_close_date[symbol] = session
        return closes

    def _force_close_stale(self, session: dt.date) -> None:
        """Close held positions whose symbol has had no bar for `max_missing_sessions`.

        The count is read from the view (consecutive sessions without a bar, ending now),
        not kept as a counter, so it survives the engine being rebuilt. It can only cover
        sessions the position was held for: a position is opened by a fill, and a fill
        needs a bar that session.

        A broker that cannot close at a stale price returns no order id. The position
        stays, and this is logged again every session until it is gone.
        """
        limit = self.construction.max_missing_sessions
        for symbol in sorted(self.broker.account().positions):
            missing = self._buffer.trailing_missing(symbol)
            if missing < limit:
                continue
            last_date = self._last_close_date[symbol]
            order_id = self.broker.force_close(symbol, self._last_close[symbol], last_date, session)
            if order_id is None:
                self._log(
                    session, "position_stale_no_data", symbol,
                    f"no bar for {missing} sessions; last close {last_date}; not closed",
                )  # fmt: skip
                continue
            self._collect_updates(session)
            self._log(
                session, "position_closed_no_data", symbol,
                f"{order_id}: no bar for {limit} sessions; closed at {last_date} close",
            )  # fmt: skip

    def _mark(self, session: dt.date) -> PortfolioSnapshot:
        account = self.broker.account()
        market_value = sum(
            p.quantity * p.instrument.multiplier * self._last_close[s]
            for s, p in account.positions.items()
        )
        equity = account.cash + market_value
        self.equity.append(
            EquityPoint(
                date=session,
                cash=account.cash,
                market_value=market_value,
                equity=equity,
                n_positions=len(account.positions),
            )
        )
        return PortfolioSnapshot(
            as_of=session, cash=account.cash, equity=equity, positions=dict(account.positions)
        )

    def _validated(self, signals: object, session: dt.date) -> list[Signal]:
        name = type(self.strategy).__name__
        if not isinstance(signals, Sequence) or isinstance(signals, str):
            raise StrategyError(
                f"{name}.on_bar returned {type(signals).__name__} on {session}; "
                "return a list of Signals (empty if none)"
            )
        checked: list[Signal] = []
        for sig in signals:
            if not isinstance(sig, Signal):
                raise StrategyError(f"{name} returned {type(sig).__name__}, not a Signal")
            if len(sig.legs) != 1:
                raise StrategyError(f"{name}: multi-leg signals are not supported in slice 1")
            leg = sig.legs[0]
            if not isinstance(leg.instrument, Stock):
                raise StrategyError(f"{name}: option signals are not supported in slice 1")
            expected = "buy" if sig.action == "enter" else "sell"
            if leg.side != expected or leg.ratio != 1:
                raise StrategyError(
                    f"{name}: {sig.action} signal for {sig.symbol} must be a single "
                    f"{expected} leg (long-only)"
                )
            if sig.symbol not in self._universe_set:
                raise StrategyError(f"{name}: {sig.symbol} is not in the universe")
            checked.append(sig)
        return checked


def simulate(
    strategy: Strategy,
    panel: BarPanel,
    broker: Broker,
    universe: Sequence[str],
    construction: ConstructionConfig,
    costs: CostConfig,
    trade_from: int = 0,
    risk: RiskManager | None = None,
) -> BacktestResult:
    """Run `strategy` over every session in `panel`.

    Sessions before index `trade_from` are warm-up: their bars enter the view, but the
    strategy is not called and no equity is recorded. The first `on_bar` call is at
    `panel.dates[trade_from]`, with the whole warm-up visible through `view.history`.
    Orders still queued after the last session are canceled and logged. Open positions
    are not liquidated: they are marked at their last close.
    """
    if not 0 <= trade_from < panel.n_dates:
        raise ValueError(f"trade_from {trade_from} outside 0..{panel.n_dates - 1}")
    cols = [panel.column(s) for s in universe]
    fields = {f: panel.field(f)[:, cols] for f in PRICE_FIELDS}
    starting_cash = broker.account().cash
    engine = Engine(strategy, broker, universe, construction, costs, panel.n_dates, risk)
    for i, session in enumerate(panel.dates):
        row = {f: fields[f][i] for f in PRICE_FIELDS}
        opens = {
            s: float(row["open"][j]) for j, s in enumerate(universe) if not np.isnan(row["open"][j])
        }
        if i < trade_from:
            engine.warm_up(session, row)
            continue
        engine.on_session_open(session, opens)
        engine.on_session_close(session, row)
    engine.finish(panel.dates[-1])
    return BacktestResult(
        strategy=strategy.strategy_name(),
        start=panel.dates[trade_from],
        end=panel.dates[-1],
        warmup_sessions=trade_from,
        starting_cash=starting_cash,
        equity=tuple(engine.equity),
        orders=engine.orders,
        updates=tuple(engine.updates),
        events=tuple(engine.events),
        final_positions=dict(broker.account().positions),
        final_marks={s: engine.last_close(s) for s in broker.account().positions},
        data_fingerprint=panel.fingerprint(),
    )
