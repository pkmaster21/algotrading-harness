"""Report: trade pairing, metrics, output files, and the run index."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from harness.broker.sim import SimBroker
from harness.engine import BacktestResult, EquityPoint, simulate
from harness.models import MARKET_TZ, Fill, FillKind, OrderUpdate, Position, Side, Stock
from harness.report import (
    INDEX_COLUMNS,
    TRADE_COLUMNS,
    ReportError,
    append_index,
    return_metrics,
    round_trips,
    run_benchmark,
    write_run,
)
from tests.fake_alpaca import recipe, tmp_config
from tests.fixture_strategies import BuyAndHold, Churn
from tests.synthetic import COSTS, construction, make_panel


def _result_with_equity(values: list[float], start_cash: float = 100.0) -> BacktestResult:
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(values))]
    return BacktestResult(
        strategy="x", start=days[0], end=days[-1], warmup_sessions=0, starting_cash=start_cash,
        equity=tuple(EquityPoint(date=d, cash=v, market_value=0.0, equity=v, n_positions=0)
                     for d, v in zip(days, values, strict=True)),
        orders={}, updates=(), events=(), final_positions={}, final_marks={},
        data_fingerprint="f",
    )  # fmt: skip


def test_metrics_on_a_known_curve() -> None:
    m = return_metrics(_result_with_equity([100, 110, 99, 121]), risk_free_rate=0.0)
    assert m["total_return"] == pytest.approx(0.21)
    assert m["max_drawdown"] == pytest.approx(99 / 110 - 1)
    returns = np.array([0.10, 99 / 110 - 1, 121 / 99 - 1])
    assert m["sharpe"] == pytest.approx(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
    assert m["cagr"] == pytest.approx(1.21 ** (365.25 / 3) - 1)


def test_flat_curve_has_no_sharpe() -> None:
    m = return_metrics(_result_with_equity([100, 100, 100]), risk_free_rate=0.0)
    assert m["sharpe"] is None
    assert m["max_drawdown"] == 0.0


def test_drawdown_counts_from_starting_cash() -> None:
    m = return_metrics(_result_with_equity([90, 95]), risk_free_rate=0.0)
    assert m["max_drawdown"] == pytest.approx(-0.10)


def test_round_trips_pair_fills_and_mark_open_positions() -> None:
    panel = make_panel(["SPY", "AAPL", "MSFT", "NVDA"], 80)
    result = simulate(Churn(), panel, SimBroker(10_000, COSTS), panel.symbols, construction(),
                      COSTS)  # fmt: skip
    trades = round_trips(result)
    closed = [t for t in trades if not t["is_open"]]
    open_ = [t for t in trades if t["is_open"]]
    assert len(closed) > 10
    assert {t["symbol"] for t in open_} == set(result.final_positions)
    fills = [f for f in result.fills]
    assert len(closed) * 2 + len(open_) == len(fills)
    for t in closed:
        expected = t["quantity"] * (t["exit_price"] - t["entry_price"])
        assert t["pnl"] == pytest.approx(expected)  # zero commission
        assert t["holding_days"] >= 1
        assert t["exit_reason"] == "signal"
    # Realized + unrealized P&L reconciles to the equity change.
    total = sum(t["pnl"] for t in trades)
    assert total == pytest.approx(result.equity[-1].equity - result.starting_cash)


def test_forced_close_trade_is_labeled_no_data() -> None:
    panel = make_panel(["AAPL"], 30, missing={"AAPL": range(11, 30)})
    result = simulate(BuyAndHold(BuyAndHold.Params(symbol="AAPL")), panel,
                      SimBroker(10_000, COSTS), ["AAPL"], construction(1), COSTS)  # fmt: skip
    (trade,) = round_trips(result)
    assert trade["exit_reason"] == "no_data"
    assert trade["exit_date"] == panel.dates[15].isoformat()


# --------------------------------------------------------------------------- multi-fill trades

DAYS = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]  # five sessions; signals on DAYS[0]


def _fill(
    order_id: str,
    side: Side,
    quantity: int,
    price: float,
    day: int,
    *,
    symbol: str = "AAPL",
    commission: float = 0.0,
    kind: FillKind = "venue",
) -> Fill:
    """A fill on DAYS[day]. Venue fills carry a time; a forced close uses a stale price."""
    date = DAYS[day]
    forced = kind == "forced_close"
    return Fill(
        order_id=order_id,
        kind=kind,
        signal_date=date if forced else DAYS[day - 1],
        fill_date=date,
        price_date=DAYS[day - 1] if forced else date,
        instrument=Stock(symbol=symbol),
        side=side,
        quantity=quantity,
        reference_price=price,
        price=price,
        slippage_cost=0.0,
        commission=commission,
        filled_at=None if forced else dt.datetime.combine(date, dt.time(9, 31), MARKET_TZ),
    )


def _result_with_fills(
    fills: list[Fill], final: dict[str, tuple[int, float]] | None = None
) -> BacktestResult:
    """A result whose updates carry `fills`, one update each. `final` maps a symbol still
    held at the end to (quantity, last close)."""
    final = final or {}
    positions = {
        s: Position(
            instrument=Stock(symbol=s),
            quantity=q,
            avg_price=mark,
            entry_date=DAYS[1],
            entry_order_id="e1",
        )
        for s, (q, mark) in final.items()
    }
    return _result_with_equity([100.0] * len(DAYS)).model_copy(
        update={
            "start": DAYS[0],
            "end": DAYS[-1],
            "equity": tuple(
                EquityPoint(date=d, cash=100.0, market_value=0.0, equity=100.0, n_positions=0)
                for d in DAYS
            ),
            "updates": tuple(
                OrderUpdate(order_id=f.order_id, status="partially_filled", fills=(f,))
                for f in fills
            ),
            "final_positions": positions,
            "final_marks": {s: mark for s, (_, mark) in final.items()},
        }
    )


def test_entry_filled_in_pieces_is_one_trade_at_the_average_price() -> None:
    result = _result_with_fills(
        [
            _fill("e1", "buy", 3, 100.0, 1, commission=1.0),
            _fill("e1", "buy", 7, 101.0, 1, commission=1.0),
            _fill("x1", "sell", 10, 110.0, 3, commission=0.5),
        ]
    )
    (trade,) = round_trips(result)
    assert trade["quantity"] == 10
    assert trade["entry_price"] == pytest.approx(100.7)  # (3 * 100 + 7 * 101) / 10
    assert trade["exit_price"] == 110.0
    assert (trade["entry_cost"], trade["exit_cost"]) == (2.0, 0.5)
    assert trade["pnl"] == pytest.approx(1100 - 0.5 - (1007 + 2))
    assert trade["return_pct"] == pytest.approx(trade["pnl"] / 1009)
    assert (trade["entry_date"], trade["exit_date"]) == ("2024-01-03", "2024-01-05")
    assert (trade["holding_days"], trade["exit_reason"], trade["is_open"]) == (2, "signal", False)


def test_exit_spread_over_two_sessions_closes_when_flat() -> None:
    result = _result_with_fills(
        [
            _fill("e1", "buy", 10, 100.0, 1),
            _fill("x1", "sell", 4, 110.0, 2),  # partly filled exit; remainder sold next day
            _fill("x2", "sell", 6, 105.0, 3, commission=1.0),
        ]
    )
    (trade,) = round_trips(result)
    assert trade["quantity"] == 10
    assert trade["exit_price"] == pytest.approx(107.0)  # (4 * 110 + 6 * 105) / 10
    assert trade["pnl"] == pytest.approx(440 + 630 - 1.0 - 1000)
    assert trade["exit_date"] == "2024-01-05"  # the fill that made the position flat
    assert (trade["holding_days"], trade["is_open"]) == (2, False)


def test_partly_exited_position_still_open_at_the_end() -> None:
    result = _result_with_fills(
        [_fill("e1", "buy", 10, 100.0, 1), _fill("x1", "sell", 4, 110.0, 2, commission=1.0)],
        final={"AAPL": (6, 120.0)},
    )
    (trade,) = round_trips(result)
    assert (trade["is_open"], trade["exit_reason"], trade["quantity"]) == (True, "open", 10)
    assert trade["exit_price"] == pytest.approx(116.0)  # (4 sold at 110 + 6 marked at 120) / 10
    assert trade["exit_cost"] == 1.0  # only what was actually paid
    assert trade["pnl"] == pytest.approx(440 + 720 - 1.0 - 1000)
    assert trade["exit_date"] == DAYS[-1].isoformat()


def test_forced_close_of_a_remainder_is_labeled_no_data() -> None:
    result = _result_with_fills(
        [
            _fill("e1", "buy", 10, 100.0, 1),
            _fill("x1", "sell", 4, 110.0, 2),
            _fill("f1", "sell", 6, 90.0, 4, kind="forced_close"),
        ]
    )
    (trade,) = round_trips(result)
    assert (trade["exit_reason"], trade["exit_date"]) == ("no_data", DAYS[4].isoformat())
    assert trade["pnl"] == pytest.approx(440 + 540 - 1000)


def test_reentry_after_flat_is_a_new_trade_and_symbols_are_independent() -> None:
    result = _result_with_fills(
        [
            _fill("e1", "buy", 5, 100.0, 1),
            _fill("m1", "buy", 2, 300.0, 1, symbol="MSFT"),
            _fill("x1", "sell", 5, 101.0, 2),
            _fill("e2", "buy", 5, 102.0, 3),
        ],
        final={"AAPL": (5, 104.0), "MSFT": (2, 310.0)},
    )
    trades = round_trips(result)
    assert [(t["symbol"], t["entry_date"], t["is_open"]) for t in trades] == [
        ("AAPL", "2024-01-03", False),
        ("MSFT", "2024-01-03", True),
        ("AAPL", "2024-01-05", True),
    ]
    assert [t["pnl"] for t in trades] == pytest.approx([5.0, 20.0, 10.0])


@pytest.mark.parametrize(
    ("fills", "final", "match"),
    [
        ([("e1", "buy", 5, 1), ("e2", "buy", 5, 2)], {"AAPL": (10, 100.0)}, "second entry"),
        ([("x1", "sell", 5, 1)], {}, "without an entry"),
        ([("e1", "buy", 5, 1), ("x1", "sell", 6, 2)], {}, "more than the position"),
        ([("e1", "buy", 5, 1), ("x1", "sell", 2, 2)], {}, "not in final positions"),
        ([("e1", "buy", 5, 1), ("x1", "sell", 2, 2)], {"AAPL": (5, 100.0)}, "leave 3 shares"),
    ],
)
def test_fills_that_do_not_add_up_are_an_error(
    fills: list[tuple[str, Side, int, int]], final: dict[str, tuple[int, float]], match: str
) -> None:
    result = _result_with_fills([_fill(i, s, q, 100.0, day) for i, s, q, day in fills], final)
    with pytest.raises(ReportError, match=match):
        round_trips(result)


def test_benchmark_runs_through_the_same_engine(tmp_path: Path) -> None:
    r = recipe(tmp_config(tmp_path))
    panel = make_panel(["SPY", "AAPL"], 40)
    bench = run_benchmark(panel, r, trade_from=0)
    (fill,) = bench.fills
    assert fill.instrument.key == "SPY"
    assert fill.fill_date == panel.dates[1]
    assert fill.price == pytest.approx(panel.field("open")[1, 0] * 1.0005)


def test_write_run_files_and_index(tmp_path: Path) -> None:
    r = recipe(tmp_config(tmp_path))
    panel = make_panel(["SPY", "AAPL", "MSFT"], 60)
    result = simulate(Churn(), panel, SimBroker(10_000, COSTS), ["SPY", "AAPL", "MSFT"],
                      construction(), COSTS)  # fmt: skip
    bench = run_benchmark(panel, r, 0)
    run_dir = tmp_path / "backtests" / "20261112T000000Z_test"
    summary = write_run(run_dir, result, bench, r, holdout=False, run_info={"run_id": "x"})

    assert sorted(p.name for p in run_dir.iterdir()) == [
        "config_snapshot.yaml", "equity.csv", "events.csv", "summary.json", "trades.csv",
    ]  # fmt: skip
    with (run_dir / "trades.csv").open() as fh:
        assert next(csv.reader(fh)) == TRADE_COLUMNS
    equity_rows = (run_dir / "equity.csv").read_text().splitlines()
    assert len(equity_rows) == 61
    loaded = json.loads((run_dir / "summary.json").read_text())
    for key in ("total_return", "cagr", "max_drawdown", "sharpe", "trade_count", "win_rate",
                "avg_holding_days", "pct_time_invested"):  # fmt: skip
        assert key in loaded
    assert set(loaded["benchmark"]) == {"symbol", "total_return", "cagr", "max_drawdown", "sharpe"}
    assert loaded["risk_preset"] == "none"
    snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text())
    assert snapshot["recipe"]["costs"]["slippage_bps"] == 5.0
    assert snapshot["recipe"]["strategy"]["strategy"] == "example_buy_and_hold"
    assert snapshot["data"]["fingerprint"] == panel.fingerprint()

    index = tmp_path / "backtests" / "index.csv"
    append_index(index, summary, "2026-11-12T00:00:00Z", run_dir)
    append_index(index, summary, "2026-11-12T00:00:01Z", run_dir)
    rows = list(csv.reader(index.read_text().splitlines()))
    assert rows[0] == INDEX_COLUMNS
    assert len(rows) == 3
    assert rows[1][INDEX_COLUMNS.index("run_folder")] == run_dir.name


def test_write_run_refuses_to_overwrite(tmp_path: Path) -> None:
    r = recipe(tmp_config(tmp_path))
    panel = make_panel(["SPY"], 5)
    result = run_benchmark(panel, r, 0)
    run_dir = tmp_path / "run"
    write_run(run_dir, result, result, r, holdout=False, run_info={})
    with pytest.raises(FileExistsError):
        write_run(run_dir, result, result, r, holdout=False, run_info={})
