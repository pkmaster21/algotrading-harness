"""The daily engine loop, shared by backtest now and paper/live later.

Each session has two phases:

- **Open:** the broker executes orders queued at the previous close, at this session's
  open. The engine records the resulting fills and cancellations.
- **Close:** the engine appends this session's bars to the strategy's view, force-closes
  positions in symbols that stopped trading, marks the portfolio to market, asks the
  strategy for signals, turns them into orders through construction, passes them through
  the risk manager (if the run has one), and submits what is left.

With a point-in-time universe (`members`, a MemberMask), the close phase also enforces
membership (docs/pit-universe-design.md):

- Entries for a symbol the strategy may not enter today (not a member, or a member whose
  last member session is today) are dropped and logged (`entry_skipped_not_member`).
- At the close of a held symbol's last member session, the engine adds an exit for it
  (`forced_reason="index_removal"`), merged with any strategy exit for it that day. It
  fills at the next open, the first session out, at that session's real bar.
- If a held symbol is no longer a member and has no bar today (so that exit was
  canceled at the open), it is closed at today's close at its last close minus slippage
  (`force_close` with `reason="index_removal"`), without waiting for
  `max_missing_sessions`. The proceeds arrive at the close, so unlike a real open fill
  they cannot fund that session's opening buys. A held non-member that does have a bar
  today gets another removal exit for the next open.
- The strategy's view hides a symbol's bars from the end of its last interval onward.
  Fills and marks use the real bars.

`index_removal` means the ticker left the membership list. That includes ticker changes
the data source keeps as two series (CBS to VIAC, MYL to VTRS, ...) and VIAC, whose
successor PARA has no usable bars: the company did not leave the index, but its old
ticker's series ended, and a strategy that buys the new ticker back pays a round trip.

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
from harness.view import PRICE_FIELDS, BarBuffer, MemberMask
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
        members: MemberMask | None = None,
    ) -> None:
        if members is not None and members.symbols != tuple(universe):
            raise ValueError("member mask symbols must be the universe, in order")
        self.strategy = strategy
        self.broker = broker
        self.risk = risk  # None: orders go from construction to the broker unchecked
        self.universe = tuple(universe)
        self.construction = construction
        self.costs = costs
        self.members = members  # None: a fixed universe, every symbol a member every session
        self._buffer = BarBuffer(self.universe, capacity)
        self._universe_set = frozenset(self.universe)  # membership checks only
        self._column = {s: j for j, s in enumerate(self.universe)}
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
        self._append(session, row)
        self._update_closes(session, row["close"])

    def on_session_close(self, session: dt.date, row: Mapping[str, np.ndarray]) -> None:
        """Process the close of `session`: bars, forced closes, mark, signals, orders."""
        self._append(session, row)
        closes = self._update_closes(session, row["close"])
        self._force_close_stale(session)
        removals = self._close_removed(session, closes)
        snapshot = self._mark(session)
        if self.risk is not None:
            self.events.extend(self.risk.on_mark(session, snapshot.equity))
        view = self._buffer.view()
        # The strategy gets its own deep copy: the snapshot's positions dict is mutable, and
        # construction below must size and count slots from what the broker reported, not
        # from whatever the strategy did to its copy.
        with strategy_code():  # no file or network access from strategy code
            emitted = self.strategy.on_bar(view, snapshot.model_copy(deep=True))
        signals = self._enforce_membership(self._validated(emitted, session), session, removals)
        result = build_orders(signals, snapshot, closes, self.construction, self.costs)
        self.events.extend(result.events)
        orders = [
            o.model_copy(update={"forced_reason": "index_removal"})
            if o.action == "exit" and o.symbol in removals
            else o
            for o in result.orders
        ]
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

    def _append(self, session: dt.date, row: Mapping[str, np.ndarray]) -> None:
        """Add the session to the strategy's view: bars of symbols past their last
        membership hidden, and which symbols may be entered today."""
        if self.members is None:
            self._buffer.append(session, row)
            return
        hidden = self.members.hidden(session)
        shown = {f: np.where(hidden, np.nan, row[f]) for f in PRICE_FIELDS}
        enterable = self.members.members(session) & ~self.members.leaving(session)
        self._buffer.append(session, shown, enterable)

    def _close_removed(self, session: dt.date, closes: Mapping[str, float]) -> frozenset[str]:
        """Handle held symbols that are leaving or have left the membership.

        Returns the held symbols that get a removal exit at this close: today is their
        last member session, or they are past it with a bar today (the earlier removal
        exit did not fill). Held non-members without a bar today are closed here at their
        last close; a broker that cannot do that leaves them, and it is logged.
        """
        if self.members is None:
            return frozenset()
        member = self.members.members(session)
        leaving = self.members.leaving(session)
        exits: set[str] = set()
        for symbol in sorted(self.broker.account().positions):
            j = self._column[symbol]
            if leaving[j] or (not member[j] and symbol in closes):
                exits.add(symbol)
                continue
            if member[j]:
                continue
            last_date = self._last_close_date[symbol]
            order_id = self.broker.force_close(
                symbol, self._last_close[symbol], last_date, session, reason="index_removal"
            )
            if order_id is None:
                self._log(
                    session, "position_stale_no_data", symbol,
                    f"left the index; no bar today; last close {last_date}; not closed",
                )  # fmt: skip
                continue
            self._collect_updates(session)
            self._log(
                session, "position_closed_index_removal", symbol,
                f"{order_id}: left the index, no bar today; closed at {last_date} close",
            )  # fmt: skip
        return frozenset(exits)

    def _enforce_membership(
        self, signals: list[Signal], session: dt.date, removals: frozenset[str]
    ) -> list[Signal]:
        """Drop entries for symbols that may not be entered today, fold strategy exits for
        removed symbols into the engine's removal exit, and add that exit."""
        if self.members is None:
            return signals
        enterable = self.members.members(session) & ~self.members.leaving(session)
        kept: list[Signal] = []
        for sig in signals:
            if sig.action == "enter" and not enterable[self._column[sig.symbol]]:
                self._log(session, "entry_skipped_not_member", sig.symbol, sig.reason)
            elif sig.action == "exit" and sig.symbol in removals:
                continue  # merged into the removal exit below
            else:
                kept.append(sig)
        kept.extend(Signal.exit(s, reason="index_removal") for s in sorted(removals))
        return kept

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
            if not self._is_member(symbol, session):
                continue  # `_close_removed` owns held non-members
            missing = self._buffer.trailing_missing(symbol)
            if missing < limit:
                continue
            last_date = self._last_close_date[symbol]
            order_id = self.broker.force_close(
                symbol, self._last_close[symbol], last_date, session, reason="no_data"
            )
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

    def _is_member(self, symbol: str, session: dt.date) -> bool:
        if self.members is None:
            return True
        return bool(self.members.members(session)[self._column[symbol]])

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
    members: MemberMask | None = None,
) -> BacktestResult:
    """Run `strategy` over every session in `panel`.

    Sessions before index `trade_from` are warm-up: their bars enter the view, but the
    strategy is not called and no equity is recorded. The first `on_bar` call is at
    `panel.dates[trade_from]`, with the whole warm-up visible through `view.history`.
    Orders still queued after the last session are canceled and logged. Open positions
    are not liquidated: they are marked at their last close.

    A point-in-time universe needs `members` (and its bad-bars windows removed from
    `panel`): without them every ticker that was ever a member is tradable on every
    session. `harness/run.py` builds both; code calling `simulate` directly must too.
    """
    if not 0 <= trade_from < panel.n_dates:
        raise ValueError(f"trade_from {trade_from} outside 0..{panel.n_dates - 1}")
    cols = [panel.column(s) for s in universe]
    fields = {f: panel.field(f)[:, cols] for f in PRICE_FIELDS}
    starting_cash = broker.account().cash
    engine = Engine(strategy, broker, universe, construction, costs, panel.n_dates, risk, members)
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
