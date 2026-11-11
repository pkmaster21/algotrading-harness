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


def run() -> BacktestResult:
    panel = make_panel(
        SYMBOLS, 200, missing={"META": range(0, 30), "XOM": range(150, 200), "NVDA": [70, 71]}
    )
    return simulate(
        Churn(), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS, trade_from=20
    )


if __name__ == "__main__":
    print(run().model_dump_json())
