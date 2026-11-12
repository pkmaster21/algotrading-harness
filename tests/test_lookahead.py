"""Acceptance: a strategy that tries to access a future bar fails by design.

Each "prophet" below tries a different route to tomorrow's data. Every route either raises
or yields nothing from the future, because the future has not been copied into the
strategy's buffer yet.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from harness.broker.sim import SimBroker
from harness.data import BarPanel
from harness.engine import simulate
from harness.models import PortfolioSnapshot, Signal
from harness.view import PRICE_FIELDS, BarBuffer, MarketView
from strategies.base import Strategy
from tests.synthetic import COSTS, construction, make_panel

SYMBOLS = ["SPY", "AAPL", "MSFT"]
N = 40


def _run(strategy: Strategy, panel: BarPanel | None = None):
    panel = panel or make_panel(SYMBOLS, N)
    return simulate(strategy, panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS)


class OversizedLookback(Strategy):
    """Asks for far more history than exists, hoping for rows past today."""

    def __init__(self) -> None:
        super().__init__()
        self.last_dates: list[tuple] = []

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        for field in PRICE_FIELDS:
            frame = view.history(field, lookback=N * 10)
            self.last_dates.append((view.today, frame.index[-1].date(), len(frame), view.n_bars))
        return []


class InternalsProphet(Strategy):
    """Reaches into the view's private buffer for row t+1 and trades on it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_future: list[float] = []

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        buffer: BarBuffer = view._buffer  # deliberate misuse
        t = view.n_bars - 1
        signals = []
        for j, symbol in enumerate(view.symbols):
            for field in PRICE_FIELDS:
                rows = buffer._data[field][t + 1 :]
                self.seen_future.extend(rows[~np.isnan(rows)].tolist())
            tomorrow = (
                buffer._data["close"][t + 1, j] if t + 1 < len(buffer._data["close"]) else np.nan
            )
            if tomorrow > buffer._data["close"][t, j]:  # NaN comparison is always False
                signals.append(Signal.enter(symbol, strength=1.0, reason="knows tomorrow"))
        return signals


class Mutator(Strategy):
    """Checks that what it sees matches the true data, then scribbles over everything it
    was handed, trying to corrupt what later sessions see."""

    def __init__(self, panel: BarPanel) -> None:
        super().__init__()
        self.panel = panel
        self.checked = 0

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        frame = view.history("close", lookback=view.n_bars)
        truth = self.panel.field("close")[: view.n_bars]
        np.testing.assert_array_equal(frame.to_numpy(), truth)
        self.checked += 1

        frame.iloc[:, :] = -1.0
        series = view.series("SPY", "open", 5)
        series.iloc[:] = -1.0
        portfolio.positions["FAKE"] = None  # type: ignore[assignment]
        with pytest.raises(ValueError, match="read-only"):
            view._buffer._data["close"][0, 0] = -1.0  # deliberate misuse
        return []


class Hoarder(Strategy):
    """Keeps every view it is given, then later checks whether old views grew."""

    def __init__(self) -> None:
        super().__init__()
        self.views: list[tuple[MarketView, object, int]] = []

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        self.views.append((view, view.today, view.n_bars))
        return []


def test_history_never_extends_past_today() -> None:
    strategy = OversizedLookback()
    _run(strategy)
    assert len(strategy.last_dates) == N * len(PRICE_FIELDS)
    for today, last, rows, n_bars in strategy.last_dates:
        assert last == today
        assert rows == n_bars


def test_future_rows_are_absent_from_the_buffer_itself() -> None:
    strategy = InternalsProphet()
    result = _run(strategy)
    assert strategy.seen_future == []  # every future row was still NaN when it looked
    assert result.fills == ()  # so the prophet never found a reason to trade


def test_buffer_never_shares_memory_with_the_full_panel() -> None:
    panel = make_panel(SYMBOLS, N)
    captured: list[MarketView] = []

    class Capture(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            captured.append(view)
            return []

    _run(Capture(), panel)
    buffer = captured[0]._buffer
    for field in PRICE_FIELDS:
        assert not np.shares_memory(buffer._data[field], panel.field(field))
    assert not any(isinstance(getattr(buffer, a), BarPanel) for a in BarBuffer.__slots__)


def test_mutating_returned_data_does_not_change_later_sessions() -> None:
    panel = make_panel(SYMBOLS, N)
    strategy = Mutator(panel)
    result = _run(strategy, panel)
    assert strategy.checked == N  # every session still saw the true history
    assert len(result.equity) == N


def test_retained_views_do_not_grow() -> None:
    strategy = Hoarder()
    _run(strategy)
    for view, today, n_bars in strategy.views:
        assert view.today == today
        assert view.n_bars == n_bars
        assert view.history("close", lookback=1000).index[-1].date() == today


def test_today_bar_is_the_last_visible_bar() -> None:
    panel = make_panel(SYMBOLS, N)
    seen = []

    class Peek(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            bar = view.bar("AAPL")
            assert bar is not None
            seen.append((view.today, bar.date, bar.close, view.series("AAPL", "close", 1).iloc[-1]))
            return []

    _run(Peek(), panel)
    closes = panel.field("close")[:, panel.column("AAPL")]
    for i, (today, bar_date, close, last_close) in enumerate(seen):
        assert today == bar_date == panel.dates[i]
        assert close == last_close == closes[i]


def test_buffer_rejects_out_of_order_and_overflow() -> None:
    buffer = BarBuffer(["SPY"], capacity=2)
    row = {f: np.array([1.0]) for f in PRICE_FIELDS}
    d0, d1 = make_panel(["SPY"], 2).dates
    buffer.append(d1, row)
    with pytest.raises(ValueError):
        buffer.append(d0, row)
    buffer.append(d1.replace(day=d1.day + 1), row)
    with pytest.raises(IndexError):
        buffer.append(d1.replace(day=d1.day + 2), row)


def test_view_rejects_bad_requests() -> None:
    buffer = BarBuffer(["SPY"], capacity=3)
    buffer.append(make_panel(["SPY"], 1).dates[0], {f: np.array([1.0]) for f in PRICE_FIELDS})
    view = buffer.view()
    with pytest.raises(ValueError):
        view.history("close", lookback=0)
    with pytest.raises(ValueError):
        view.history("adj_close", lookback=1)  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        view.history("close", lookback=1, symbols=["TSLA"])
    with pytest.raises(ValueError):
        MarketView(buffer, 2)  # cannot build a view longer than what was appended
