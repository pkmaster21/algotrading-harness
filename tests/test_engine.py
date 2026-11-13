"""Engine behavior beyond the acceptance tests: warm-up, signal validation, event logging."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence

import pytest

from harness.broker.sim import SimBroker
from harness.engine import BacktestResult, StrategyError, simulate
from harness.models import (
    MARKET_TZ,
    Fill,
    Leg,
    OptionContract,
    Order,
    OrderUpdate,
    OrderUpdateReason,
    PortfolioSnapshot,
    Signal,
    Stock,
)
from harness.view import MarketView
from strategies.base import Strategy
from tests.fixture_strategies import BuyAndHold, Recorder
from tests.synthetic import COSTS, construction, make_panel

SYMBOLS = ["SPY", "AAPL"]


def _sim(strategy: Strategy, n: int = 30, trade_from: int = 0, symbols=SYMBOLS, **panel_kw):
    panel = make_panel(symbols, n, **panel_kw)
    result = simulate(
        strategy, panel, SimBroker(10_000, COSTS), symbols, construction(), COSTS, trade_from
    )
    return panel, result


class Returns(Strategy):
    def __init__(self, value: object) -> None:
        super().__init__()
        self.value = value

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        return self.value  # type: ignore[return-value]


# --------------------------------------------------------------------------- warm-up


def test_warmup_sessions_fill_the_view_without_calling_the_strategy() -> None:
    recorder = Recorder()
    panel, result = _sim(recorder, n=30, trade_from=12)
    first_today, first_n_bars, first_last, first_rows = recorder.calls[0]
    assert len(recorder.calls) == 30 - 12
    assert first_today == panel.dates[12]
    assert first_n_bars == first_rows == 13  # warm-up history visible from the first call
    assert first_last == panel.dates[12]
    assert result.start == panel.dates[12]
    assert result.warmup_sessions == 12
    assert [p.date for p in result.equity] == list(panel.dates[12:])
    assert result.equity[0].equity == 10_000


def test_trade_from_out_of_range() -> None:
    with pytest.raises(ValueError):
        _sim(Recorder(), n=10, trade_from=10)


def test_first_trade_fills_the_session_after_trade_from() -> None:
    panel, result = _sim(BuyAndHold(), n=20, trade_from=5)
    (fill,) = result.fills
    assert (fill.signal_date, fill.fill_date) == (panel.dates[5], panel.dates[6])


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "bad, match",
    [
        (None, "return a list"),
        ("AAPL", "return a list"),
        ([42], "not a Signal"),
        ([Signal.enter("TSLA", 1.0)], "not in the universe"),
        (
            [
                Signal(
                    action="enter",
                    legs=(Leg(instrument=Stock(symbol="AAPL"), side="sell"),),
                    strength=1.0,
                )
            ],
            "long-only",
        ),  # fmt: skip
        (
            [Signal(action="exit", legs=(Leg(instrument=Stock(symbol="AAPL"), side="buy"),))],
            "long-only",
        ),  # fmt: skip
        (
            [
                Signal(
                    action="enter",
                    strength=1.0,
                    legs=(
                        Leg(
                            instrument=OptionContract(
                                underlying="AAPL",
                                expiry=make_panel(["SPY"], 1).dates[0],
                                strike=1,
                                right="call",
                            ),
                            side="buy",
                        ),
                    ),
                )
            ],
            "option",
        ),  # fmt: skip
        (
            [
                Signal(
                    action="enter",
                    strength=1.0,
                    legs=(
                        Leg(instrument=Stock(symbol="AAPL"), side="buy"),
                        Leg(instrument=Stock(symbol="SPY"), side="buy"),
                    ),
                )
            ],
            "multi-leg",
        ),  # fmt: skip
    ],
)
def test_invalid_strategy_output_raises(bad: object, match: str) -> None:
    with pytest.raises(StrategyError, match=match):
        _sim(Returns(bad))


def test_generator_output_rejected() -> None:
    with pytest.raises(StrategyError):
        _sim(Returns(s for s in []))


# --------------------------------------------------------------------------- events


def test_ignored_exit_and_unfilled_end_orders_are_logged() -> None:
    class ExitThenEnterLast(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            if view.n_bars == 1:
                return [Signal.exit("AAPL", reason="not held")]
            if view.n_bars == 10:
                return [Signal.enter("AAPL", 1.0)]
            return []

    panel, result = _sim(ExitThenEnterLast(), n=10)
    kinds = [(e.kind, e.symbol, e.date) for e in result.events]
    assert ("exit_ignored_not_held", "AAPL", panel.dates[0]) in kinds
    assert ("order_unfilled_end", "AAPL", panel.dates[-1]) in kinds
    assert result.fills == ()


def test_order_for_symbol_without_next_open_is_canceled_and_logged() -> None:
    class EnterDay3(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            return [Signal.enter("AAPL", 1.0)] if view.n_bars == 3 else []

    panel, result = _sim(EnterDay3(), n=10, missing={"AAPL": [3]})
    (event,) = [e for e in result.events if e.kind == "order_canceled"]
    assert (event.symbol, event.date) == ("AAPL", panel.dates[3])
    assert "no_bar" in event.detail


class ScriptedBroker(SimBroker):
    """A SimBroker whose update stream can be replaced, to feed the engine statuses that
    only a live broker reports."""

    def __init__(self, script: Callable[[list[str]], list[OrderUpdate]]) -> None:
        super().__init__(10_000, COSTS)
        self._script = script
        self._ids: list[str] = []
        self._pending = False

    def submit(self, orders: Sequence[Order]) -> list[str]:
        ids = super().submit(orders)
        self._ids.extend(ids)
        return ids

    def on_session_open(self, session: dt.date, opens: Mapping[str, float]) -> None:
        self._pending = bool(self._ids)

    def poll_updates(self) -> list[OrderUpdate]:
        if not self._pending:
            return []
        self._pending = False
        ids, self._ids = self._ids, []
        return self._script(ids)

    def cancel_all(self, reason: OrderUpdateReason) -> list[str]:
        return []


def _run_scripted(script: Callable[[list[str]], list[OrderUpdate]]) -> BacktestResult:
    class EnterDay3(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            return [Signal.enter("AAPL", 1.0)] if view.n_bars == 3 else []

    panel = make_panel(SYMBOLS, 6)
    return simulate(EnterDay3(), panel, ScriptedBroker(script), SYMBOLS, construction(), COSTS)


@pytest.mark.parametrize(
    ("status", "reason", "kind"),
    [
        ("rejected", "broker_rejected", "order_rejected"),
        ("expired", "limit_not_reached", "order_expired"),
        ("canceled", "session_deadline", "order_canceled"),
    ],
)
def test_live_terminal_statuses_are_logged(status: str, reason: str, kind: str) -> None:
    result = _run_scripted(
        lambda ids: [OrderUpdate(order_id=i, status=status, reason=reason) for i in ids]  # type: ignore[arg-type]
    )
    (event,) = result.events
    assert (event.kind, event.symbol) == (kind, "AAPL")
    assert reason in event.detail
    assert result.fills == ()


def test_partial_fills_are_logged_and_collected() -> None:
    day4 = make_panel(SYMBOLS, 6).dates[3]

    def fill(order_id: str, quantity: int) -> Fill:
        return Fill(
            order_id=order_id,
            kind="venue",
            signal_date=make_panel(SYMBOLS, 6).dates[2],
            fill_date=day4,
            price_date=day4,
            instrument=Stock(symbol="AAPL"),
            side="buy",
            quantity=quantity,
            reference_price=60.0,
            price=60.0,
            slippage_cost=0.0,
            commission=0.0,
            filled_at=dt.datetime.combine(day4, dt.time(9, 31), MARKET_TZ),
        )

    def script(ids: list[str]) -> list[OrderUpdate]:
        (order_id,) = ids
        return [
            OrderUpdate(order_id=order_id, status="partially_filled", fills=(fill(order_id, 3),)),
            OrderUpdate(order_id=order_id, status="canceled", reason="session_deadline"),
        ]

    result = _run_scripted(script)
    assert [e.kind for e in result.events] == ["order_partial", "order_canceled"]
    assert "3 of" in result.events[0].detail
    assert [f.quantity for f in result.fills] == [3]


def test_update_without_fills_for_an_unknown_order_raises() -> None:
    with pytest.raises(RuntimeError, match="unknown order"):
        _run_scripted(lambda ids: [OrderUpdate(order_id="nope", status="rejected")])


def test_equity_marks_held_symbol_at_last_close_while_bar_missing() -> None:
    panel, result = _sim(
        BuyAndHold(BuyAndHold.Params(symbol="AAPL")), n=12, missing={"AAPL": [6, 7]}
    )
    qty = result.final_positions["AAPL"].quantity
    close5 = float(panel.field("close")[5, panel.column("AAPL")])
    assert result.equity[6].market_value == pytest.approx(qty * close5)
    assert result.equity[7].market_value == pytest.approx(qty * close5)


def test_strategy_name_and_fingerprint_recorded() -> None:
    panel, result = _sim(BuyAndHold())
    assert result.strategy == "test_buy_and_hold"
    assert result.data_fingerprint == panel.fingerprint()


def test_strategy_params_validated() -> None:
    assert BuyAndHold.from_params({"symbol": "AAPL"}).params.symbol == "AAPL"  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        BuyAndHold.from_params({"symbol": "AAPL", "extra": 1})
    with pytest.raises(TypeError):
        BuyAndHold(Recorder.Params())


# --------------------------------------------------------------------------- snapshot isolation


def test_strategy_mutating_its_snapshot_does_not_change_construction() -> None:
    # Two slots, both filled. The strategy then drops a held symbol from the positions
    # dict it was handed (without exiting it) and asks for a third position. Construction
    # must still see both holdings and refuse the entry for lack of a slot.
    class PopThenEnter(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            if view.n_bars == 1:
                return [Signal.enter("AAA", 2.0), Signal.enter("BBB", 1.0)]
            if view.n_bars == 5:
                portfolio.positions.pop("AAA")
                return [Signal.enter("CCC", 1.0)]
            return []

    symbols = ["AAA", "BBB", "CCC"]
    cheap = [5.0] * 12  # CCC is cheap enough to buy with leftover cash
    panel = make_panel(symbols, 12, opens={"CCC": cheap}, closes={"CCC": cheap})
    result = simulate(
        PopThenEnter(), panel, SimBroker(10_000, COSTS), symbols, construction(max_positions=2),
        COSTS,
    )  # fmt: skip
    assert sorted(result.final_positions) == ["AAA", "BBB"]
    assert max(p.n_positions for p in result.equity) == 2
    assert ("entry_skipped_no_slot", "CCC", panel.dates[4]) in [
        (e.kind, e.symbol, e.date) for e in result.events
    ]


def test_strategy_clearing_its_snapshot_cannot_cause_a_second_buy_of_a_held_symbol() -> None:
    class ClearThenReenter(Strategy):
        def on_bar(self, view, portfolio):  # type: ignore[no-untyped-def]
            if view.n_bars == 1:
                return [Signal.enter("AAPL", 1.0)]
            if view.n_bars == 5:
                portfolio.positions.clear()
                return [Signal.enter("AAPL", 1.0)]
            return []

    panel, result = _sim(ClearThenReenter(), n=12)  # used to abort inside SimBroker
    assert len(result.fills) == 1
    assert ("entry_ignored_held", "AAPL", panel.dates[4]) in [
        (e.kind, e.symbol, e.date) for e in result.events
    ]
