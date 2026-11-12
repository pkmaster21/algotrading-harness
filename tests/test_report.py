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
from harness.report import (
    INDEX_COLUMNS,
    TRADE_COLUMNS,
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
