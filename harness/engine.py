"""The daily engine loop, shared by backtest now and paper/live later.

Each session has two phases:

- **Open:** the broker executes orders queued at the previous close, at this session's
  open. The engine records the resulting fills and cancellations.
- **Close:** the engine appends this session's bars to the strategy's view, force-closes
  positions in symbols that stopped trading, marks the portfolio to market, asks the
  strategy for signals, turns them into orders through construction, and submits them.

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
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.universe = tuple(universe)
        self.construction = construction
        self.costs = costs
        self._buffer = BarBuffer(self.universe, capacity)
        self._universe_set = frozenset(self.universe)  # membership checks only
        self._last_close: dict[str, float] = {}
        self._last_close_date: dict[str, dt.date] = {}
        self._missing: dict[str, int] = {}  # held symbol -> consecutive sessions without a bar
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

    def on_session_close(
        self, session: dt.date, row: Mapping[str, np.ndarray], trade: bool = True
    ) -> None:
        """Process the close of `session`. With `trade=False` (warm-up) the bars are added
        to the view but the strategy is not called and no equity point is recorded."""
        self._buffer.append(session, row)
        closes = self._update_closes(session, row["close"])
        self._force_close_stale(session)
        if not trade:
            return
        snapshot = self._mark(session)
        view = self._buffer.view()
        # The strategy gets its own deep copy: the snapshot's positions dict is mutable, and
        # construction below must size and count slots from what the broker reported, not
        # from whatever the strategy did to its copy.
        with strategy_code():  # no file or network access from strategy code
            emitted = self.strategy.on_bar(view, snapshot.model_copy(deep=True))
        signals = self._validated(emitted, session)
        result = build_orders(signals, snapshot, closes, self.construction, self.costs)
        self.events.extend(result.events)
        ids = self.broker.submit(result.orders)
        for order_id, order in zip(ids, result.orders, strict=True):
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
            symbol = order.symbol if order is not None else update.fills[0].instrument.key
            if update.status == "canceled":
                kind: EventKind = (
                    "order_unfilled_end" if update.reason == "end_of_data" else "order_canceled"
                )
                self._log(session, kind, symbol, f"{update.order_id}: {update.reason}")
            elif update.reason == "reduced_for_cash":
                filled = sum(f.quantity for f in update.fills)
                ordered = order.quantity if order is not None else "?"
                self._log(
                    session, "order_reduced", symbol,
                    f"{update.order_id}: {filled} of {ordered} shares",
                )  # fmt: skip

    def _update_closes(self, session: dt.date, close_row: np.ndarray) -> dict[str, float]:
        closes: dict[str, float] = {}
        for j, symbol in enumerate(self.universe):
            value = float(close_row[j])
            if math.isnan(value):
                continue
            closes[symbol] = value
            self._last_close[symbol] = value
            self._last_close_date[symbol] = session
        held = self.broker.account().positions
        for symbol in held:
            self._missing[symbol] = 0 if symbol in closes else self._missing.get(symbol, 0) + 1
        for symbol in [s for s in self._missing if s not in held]:
            del self._missing[symbol]
        return closes

    def _force_close_stale(self, session: dt.date) -> None:
        limit = self.construction.max_missing_sessions
        for symbol in sorted(s for s, n in self._missing.items() if n >= limit):
            last_date = self._last_close_date[symbol]
            order_id = self.broker.force_close(symbol, self._last_close[symbol], last_date, session)
            del self._missing[symbol]
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
    engine = Engine(strategy, broker, universe, construction, costs, panel.n_dates)
    for i, session in enumerate(panel.dates):
        row = {f: fields[f][i] for f in PRICE_FIELDS}
        opens = {
            s: float(row["open"][j]) for j, s in enumerate(universe) if not np.isnan(row["open"][j])
        }
        engine.on_session_open(session, opens)
        engine.on_session_close(session, row, trade=i >= trade_from)
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
