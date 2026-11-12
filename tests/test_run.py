"""Run entry point: strategy loading and the import allowlist, early param validation,
end-to-end runs of the example strategy, and reproducible outputs."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from harness.config import load_recipe
from harness.data import DataLoader
from harness.run import (
    StrategyLoadError,
    build_strategy,
    check_strategy_source,
    load_strategy_class,
    main,
    run_backtest,
)
from strategies.example_buy_and_hold import ExampleBuyAndHold
from tests.fake_alpaca import NOW, FakeFetcher, recipe, tmp_config


def _loader(config: Path, fetcher: FakeFetcher | None = None) -> DataLoader:
    r = recipe(config)
    fetcher = fetcher or FakeFetcher()
    return DataLoader(r.dates, r.data, fetcher_factory=lambda: fetcher, now=lambda: NOW)


def _run(config: Path, **kw):  # type: ignore[no-untyped-def]
    return run_backtest(
        recipe(config), loader=_loader(config), now=lambda: NOW, git_commit=lambda: None, **kw
    )


# --------------------------------------------------------------------------- allowlist


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "import sys",
        "from pathlib import Path",
        "import requests",
        "from alpaca.trading.client import TradingClient",
        "from harness.data import DataLoader",
        "from harness.broker.sim import SimBroker",
        "import harness.config",
        "from . import helper",
        "data = open('data/bars/SPY.parquet').read()",
        "exec('import os')",
        "__import__('os')",
    ],
)
def test_disallowed_strategy_code_is_rejected(source: str) -> None:
    assert check_strategy_source(source)


def test_allowed_strategy_imports_pass() -> None:
    source = "\n".join(
        [
            "from __future__ import annotations",
            "import math",
            "import numpy as np",
            "import pandas as pd",
            "from pydantic import Field",
            "from harness.models import Signal",
            "from harness.view import MarketView",
            "from harness.indicators import sma",
            "from strategies.base import Strategy",
        ]
    )
    assert check_strategy_source(source) == []


def test_every_shipped_strategy_passes_the_allowlist() -> None:
    for path in (Path(__file__).resolve().parent.parent / "strategies").glob("*.py"):
        if path.name in ("__init__.py", "base.py"):  # base.py is the reviewed interface
            continue
        assert check_strategy_source(path.read_text(), str(path)) == [], path.name


def test_loader_finds_the_one_strategy_class() -> None:
    assert load_strategy_class("example_buy_and_hold") is ExampleBuyAndHold
    with pytest.raises(StrategyLoadError, match="no strategy module"):
        load_strategy_class("does_not_exist")


def test_bad_params_fail_before_any_data_is_loaded(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    (config / "strategies" / "bad.yaml").write_text(
        "strategy: example_buy_and_hold\nparams: {min_history: 0}\n"
    )

    class ExplodingLoader:
        def load(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("data was loaded before params were validated")

    with pytest.raises(StrategyLoadError, match="invalid params"):
        run_backtest(load_recipe("bad", "test-3", config), loader=ExplodingLoader())  # type: ignore[arg-type]


def test_unknown_param_rejected(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    (config / "strategies" / "typo.yaml").write_text(
        "strategy: example_buy_and_hold\nparams: {symbl: SPY}\n"
    )
    with pytest.raises(StrategyLoadError, match="symbl"):
        build_strategy(load_recipe("typo", "test-3", config))


# --------------------------------------------------------------------------- end to end


def test_example_strategy_end_to_end(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    construction = config / "construction.yaml"
    construction.write_text(
        construction.read_text().replace("max_positions: 5", "max_positions: 1")
    )
    summary = _run(config)
    run_dir = Path(summary["run_dir"])
    assert run_dir.parent == recipe(config).backtest.output_dir
    assert run_dir.name == "20260701T150000Z_example_buy_and_hold"
    assert summary["trade_count"] == 1 and summary["open_trades"] == 1
    assert summary["pct_time_invested"] > 0.99
    # With one slot, buying SPY and holding is the benchmark, so they match exactly.
    for key in ("total_return", "cagr", "max_drawdown", "sharpe"):
        assert summary[key] == summary["benchmark"][key]
    (trade,) = csv.DictReader((run_dir / "trades.csv").read_text().splitlines())
    assert trade["symbol"] == "SPY"
    assert trade["entry_date"] == "2025-10-02"  # signal on the first bar, fill next open


def test_holdout_example_buys_on_first_holdout_session(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    summary = _run(config, holdout=True)
    (trade,) = csv.DictReader((Path(summary["run_dir"]) / "trades.csv").read_text().splitlines())
    assert trade["entry_date"] == "2026-04-02"  # signal 2026-04-01 close, fill next open


def test_identical_runs_write_identical_result_files(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    first, second = Path(_run(config)["run_dir"]), Path(_run(config)["run_dir"])
    assert second.name == first.name + "-2"  # same second: suffixed, never overwritten
    for name in ("summary.json", "trades.csv", "equity.csv", "events.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    rows = list(csv.reader((first.parent / "index.csv").read_text().splitlines()))
    assert len(rows) == 3


def test_cli_reports_config_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--strategy", "no-such-preset"]) == 2
    assert "no-such-preset" in capsys.readouterr().err


def test_snapshot_records_holdout_and_commit(tmp_path: Path) -> None:
    import yaml

    config = tmp_config(tmp_path)
    summary = run_backtest(recipe(config), holdout=True, loader=_loader(config),
                           now=lambda: NOW, git_commit=lambda: "deadbeef")  # fmt: skip
    snap = yaml.safe_load((Path(summary["run_dir"]) / "config_snapshot.yaml").read_text())
    assert snap["holdout"] is True
    assert snap["run"]["git_commit"] == "deadbeef"
    assert snap["data"]["start"] == "2026-04-01"
    assert dt.date.fromisoformat(snap["recipe"]["dates"]["holdout_start"]) == dt.date(2026, 4, 1)


def test_cli_reports_missing_keys_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import harness.run as run_module
    from harness.data import DataError

    config = tmp_config(tmp_path)

    def fake_load_recipe(strategy: str, universe: str):  # type: ignore[no-untyped-def]
        return load_recipe(strategy, "test-3", config)

    def fake_run(recipe, holdout=False):  # type: ignore[no-untyped-def]
        raise DataError("ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY must be set")

    monkeypatch.setattr(run_module, "load_recipe", fake_load_recipe)
    monkeypatch.setattr(run_module, "run_backtest", fake_run)
    assert main(["--strategy", "example_buy_and_hold"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ALPACA_PAPER_API_KEY")
    assert "Traceback" not in err


def test_holdout_run_without_holdout_sessions_is_a_clear_error(tmp_path: Path) -> None:
    from harness.run import HoldoutUnavailable

    config = tmp_config(tmp_path)
    before_holdout = dt.datetime(2026, 3, 20, 15, 0, tzinfo=dt.UTC)
    r = recipe(config)
    loader = DataLoader(r.dates, r.data, fetcher_factory=FakeFetcher, now=lambda: before_holdout)
    with pytest.raises(HoldoutUnavailable):
        run_backtest(r, holdout=True, loader=loader, now=lambda: before_holdout,
                     git_commit=lambda: None)  # fmt: skip


def test_harness_bugs_keep_their_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    import harness.run as run_module
    from harness.data import HoldoutViolation

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise HoldoutViolation("bug")

    monkeypatch.setattr(run_module, "load_recipe", boom)
    with pytest.raises(HoldoutViolation):
        main(["--strategy", "example_buy_and_hold"])


def test_snapshot_records_strategy_source_hash(tmp_path: Path) -> None:
    import hashlib

    import yaml

    import strategies.example_buy_and_hold as module

    config = tmp_config(tmp_path)
    summary = _run(config)
    snap = yaml.safe_load((Path(summary["run_dir"]) / "config_snapshot.yaml").read_text())
    expected = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert snap["run"]["strategy_source_sha256"] == expected
