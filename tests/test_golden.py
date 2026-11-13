"""Golden run: one fixed backtest must keep producing exactly the outputs checked in
under tests/golden/.

`test_determinism.py` proves two runs agree with each other. This proves a run agrees
with the past, so a change to the engine, SimBroker, construction, costs, or the report
that moves any fill, price, quantity, event, or equity point fails here.

The scenario is built to reach the paths that matter: prices that rise and fall (losing
trades, a real drawdown), overnight gaps in both directions (buys reduced for cash),
symbols with no bar (canceled orders), and symbols that stop trading while held (forced
closes). Prices are multiples of 0.25 made with integer arithmetic, so the inputs are
exact on every platform.

If a change is meant to alter results, regenerate the files and review their diff:

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from harness.broker.sim import SimBroker
from harness.engine import BacktestResult, simulate
from harness.report import (
    EQUITY_COLUMNS,
    EVENT_COLUMNS,
    TRADE_COLUMNS,
    _write_csv,
    return_metrics,
    round_trips,
)
from tests.fixture_strategies import Churn
from tests.synthetic import COSTS, construction, make_panel

GOLDEN = Path(__file__).resolve().parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"
CSV_FILES = ("trades.csv", "equity.csv", "events.csv")

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]
N = 200


def golden_run() -> BacktestResult:
    t = np.arange(N)
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}
    for k, symbol in enumerate(SYMBOLS):
        # Close: a sawtooth (period 23 + k sessions, 12 dollars peak to trough) on a slow
        # drift that is up for even k and down for odd k.
        saw = (t * (5 + k % 3) + 7 * k) % (23 + k)
        drift = t // 8 if k % 2 == 0 else -(t // 16)
        close = 60.0 + 10.0 * k + 0.5 * saw + 0.25 * drift
        # Open: the previous close plus a gap between -3.00 and +3.00 dollars.
        gap = ((t * 7 + 11 * k) % 9 - 4) * 0.75
        closes[symbol] = close
        opens[symbol] = np.concatenate(([close[0]], close[:-1])) + gap
    missing = {
        "META": range(0, 30),  # lists late
        "NVDA": [70, 71],  # short halt: orders canceled, no forced close
        "QQQ": range(150, N),  # these three stop trading on the same session
        "MSFT": range(150, N),
        "JPM": range(150, N),
    }
    panel = make_panel(SYMBOLS, N, opens=opens, closes=closes, missing=missing)
    return simulate(
        Churn(), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS, trade_from=20
    )


def _write_outputs(result: BacktestResult, out: Path) -> None:
    """The same CSV files a run folder gets, written by the report's own writer, plus
    the metrics and a hash of the complete result (orders, updates, fills, ids)."""
    _write_csv(out / "trades.csv", TRADE_COLUMNS, round_trips(result))
    _write_csv(out / "equity.csv", EQUITY_COLUMNS, [p.model_dump() for p in result.equity])
    _write_csv(out / "events.csv", EVENT_COLUMNS, [e.model_dump() for e in result.events])
    metrics = return_metrics(result, risk_free_rate=0.0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    digest = hashlib.sha256(result.model_dump_json().encode()).hexdigest()
    (out / "result.sha256").write_text(digest + "\n")


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    result = golden_run()
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        _write_outputs(result, GOLDEN)
    out = tmp_path_factory.mktemp("golden")
    _write_outputs(result, out)
    return out


@pytest.mark.parametrize("name", CSV_FILES)
def test_csv_outputs_match_golden(outputs: Path, name: str) -> None:
    # Exact text: these values come from +, -, *, / on doubles, which give the same
    # result on every platform, and floats are written with repr.
    assert (outputs / name).read_text() == (GOLDEN / name).read_text()


def test_metrics_match_golden(outputs: Path) -> None:
    # CAGR and Sharpe use pow and a standard deviation, which can differ in the last bit
    # between platforms, so these are compared to 12 significant digits.
    actual = json.loads((outputs / "metrics.json").read_text())
    expected = json.loads((GOLDEN / "metrics.json").read_text())
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-12), key


def test_full_result_matches_golden_hash(outputs: Path) -> None:
    # Everything the CSV files leave out: order ids, the update stream, every fill field.
    assert (outputs / "result.sha256").read_text() == (GOLDEN / "result.sha256").read_text()


def test_golden_run_is_not_trivial() -> None:
    """Guards the scenario itself: if it stopped reaching these paths, matching the
    golden files would prove much less."""
    result = golden_run()
    trades = round_trips(result)
    closed = [t for t in trades if not t["is_open"]]
    assert len(closed) > 100
    assert sum(t["pnl"] < 0 for t in closed) > 20  # losing trades
    assert sum(t["pnl"] > 0 for t in closed) > 20
    assert sum(t["exit_reason"] == "no_data" for t in trades) >= 2  # forced closes
    assert any(t["is_open"] for t in trades)
    kinds = {e.kind for e in result.events}
    assert {"order_canceled", "order_reduced", "entry_skipped_no_slot",
            "exit_ignored_not_held", "position_closed_no_data"} <= kinds  # fmt: skip
    metrics = return_metrics(result, risk_free_rate=0.0)
    assert metrics["max_drawdown"] is not None and metrics["max_drawdown"] < -0.02
