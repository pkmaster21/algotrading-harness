"""Runtime guard: strategy code cannot open files or reach the network.

The static check in `harness.run` only sees a strategy module's source. This guard is the
second layer: it is active whenever the engine calls `on_start` or `on_bar`, for every
strategy (including test strategies, which skip the static check).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.broker.sim import SimBroker
from harness.engine import simulate
from harness.guard import StrategyIOError, strategy_code
from harness.models import PortfolioSnapshot, Signal
from harness.view import MarketView
from strategies.base import Strategy
from tests.synthetic import COSTS, construction, make_panel


class Does(Strategy):
    """Runs `action` on the chosen hook."""

    def __init__(self, action: Callable[[], object], hook: str = "on_bar") -> None:
        super().__init__()
        self.action = action
        self.hook = hook
        self.results: list[object] = []

    def on_start(self, symbols: tuple[str, ...]) -> None:
        if self.hook == "on_start":
            self.results.append(self.action())

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        if self.hook == "on_bar":
            self.results.append(self.action())
        return []


def _run(strategy: Strategy) -> None:
    simulate(strategy, make_panel(["SPY"], 3), SimBroker(10_000, COSTS), ["SPY"], construction(),
             COSTS)  # fmt: skip


@pytest.fixture
def secret(tmp_path: Path) -> Path:
    path = tmp_path / "secret.csv"
    path.write_text("a,b\n1,2\n")
    return path


@pytest.mark.parametrize("hook", ["on_start", "on_bar"])
def test_strategy_cannot_open_a_file(secret: Path, hook: str) -> None:
    with pytest.raises(StrategyIOError, match="secret.csv"):
        _run(Does(lambda: open(secret).read(), hook))


@pytest.mark.parametrize(
    "read",
    [
        lambda p: pd.read_csv(p),
        lambda p: np.loadtxt(p, delimiter=",", skiprows=1),
        lambda p: np.fromfile(p),
        lambda p: os.open(p, os.O_RDONLY),
        lambda p: Path(p).read_text(),
    ],
    ids=["pd.read_csv", "np.loadtxt", "np.fromfile", "os.open", "Path.read_text"],
)
def test_library_file_reads_are_blocked_too(secret: Path, read: Callable[[Path], object]) -> None:
    with pytest.raises(StrategyIOError):
        _run(Does(lambda: read(secret)))


def test_strategy_cannot_write_a_file(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    with pytest.raises(StrategyIOError):
        _run(Does(lambda: pd.DataFrame({"a": [1]}).to_csv(target)))
    assert not target.exists()


def test_strategy_cannot_open_a_network_connection() -> None:
    def connect() -> None:
        with socket.socket() as sock:
            sock.connect(("127.0.0.1", 9))  # rejected before any packet is sent

    with pytest.raises(StrategyIOError, match="network"):
        _run(Does(connect))


@pytest.mark.parametrize(
    "spawn",
    [
        lambda: subprocess.run(["true"], check=False),
        lambda: os.system("true"),
        lambda: os.posix_spawn("/usr/bin/true", ["true"], {}),
    ],
    ids=["subprocess.run", "os.system", "os.posix_spawn"],
)  # os.exec* is refused as well, but not tested: a regression would replace the test process
def test_strategy_cannot_start_a_process(spawn: Callable[[], object]) -> None:
    # A child process could do any of the I/O the guard refuses, so it is refused too.
    with pytest.raises(StrategyIOError, match="process"):
        _run(Does(spawn))


def test_guard_is_off_outside_strategy_code(secret: Path) -> None:
    _run(Does(lambda: None))
    assert secret.read_text().startswith("a,b")  # the harness itself can still read files


def test_guard_is_off_again_after_a_strategy_raises(secret: Path) -> None:
    def boom() -> None:
        raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        _run(Does(boom))
    assert secret.read_text().startswith("a,b")


def test_imports_and_library_data_still_work_inside_strategy_code() -> None:
    sys.modules.pop("colorsys", None)  # a stdlib module nothing else here imports

    def work() -> object:
        import colorsys  # reads the module's source or bytecode from the Python install

        stamp = pd.Timestamp("2024-01-02", tz="America/New_York")  # reads the tz database
        return colorsys.rgb_to_hsv(1, 0, 0), stamp.utcoffset()

    strategy = Does(work)
    _run(strategy)
    assert len(strategy.results) == 3


def test_guard_nests_and_reads_of_the_python_install_are_allowed(secret: Path) -> None:
    with strategy_code():
        with strategy_code():
            pass
        with pytest.raises(StrategyIOError):  # still on after the inner block exits
            open(secret)
        with open(os.__file__) as fh:  # stdlib source: needed for imports and tracebacks
            assert fh.read(1)
        with pytest.raises(StrategyIOError):  # but never for writing
            open(os.__file__, "a")
    open(secret).close()
