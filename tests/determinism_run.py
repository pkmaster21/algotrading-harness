"""Runs one fixed multi-symbol backtest and prints the result as JSON.

Used by test_determinism.py in subprocesses with different PYTHONHASHSEED values, which
would expose any result that depends on set or hash ordering.
"""

from __future__ import annotations

from harness.broker.sim import SimBroker
from harness.engine import BacktestResult, simulate
from tests.fixture_strategies import Churn
from tests.synthetic import COSTS, construction, make_panel

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]
GONE_TOGETHER = ["QQQ", "MSFT", "AMZN", "GOOGL", "JPM", "XOM"]


def run() -> BacktestResult:
    # GONE_TOGETHER stop trading on the same session while several are held, so they are
    # force-closed on one session: the only place the engine picks an order among symbols
    # itself (sorted, in `_force_close_stale`). Order ids record that order.
    gone = {s: range(150, 200) for s in GONE_TOGETHER}
    panel = make_panel(SYMBOLS, 200, missing={"META": range(0, 30), "NVDA": [70, 71], **gone})
    return simulate(
        Churn(), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS, trade_from=20
    )


if __name__ == "__main__":
    print(run().model_dump_json())
