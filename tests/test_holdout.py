"""Acceptance: a run without the holdout flag never touches data on or after 2026-04-01.

"Touches" covers all three ways data could leak: requesting it from Alpaca, reading it
from a cache that already holds it (after an earlier holdout run), and handing it to the
engine or the strategy.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from harness.data import (
    DataLoader,
    HoldoutAccess,
    HoldoutLogError,
    HoldoutViolation,
    read_holdout_log,
)
from harness.run import run_backtest
from tests.fake_alpaca import NOW, FakeFetcher, recipe, tmp_config

HOLDOUT_START = dt.date(2026, 4, 1)
ACCESS = HoldoutAccess(run_id="r1", strategy="example_buy_and_hold", git_commit="abc123")


def _loader(config: Path, fetcher: FakeFetcher) -> DataLoader:
    r = recipe(config)
    return DataLoader(r.dates, r.data, fetcher_factory=lambda: fetcher, now=lambda: NOW)


def test_default_load_never_requests_or_returns_holdout_dates(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    panel = _loader(config, fetcher).load(["SPY", "AAPL"])
    assert fetcher.requests
    assert all(end < HOLDOUT_START for _, _, end in fetcher.requests)
    assert max(panel.dates) == dt.date(2026, 3, 31)
    assert not recipe(config).data.holdout_log.exists()


def test_cached_holdout_rows_are_never_returned_without_the_flag(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    loader = _loader(config, fetcher)
    loader.load(["SPY"], holdout=ACCESS)  # the cache now holds bars through 2026-06-30
    cached = loader._cache._read_unfiltered("SPY")  # test-only bypass, to prove rows exist
    assert max(cached["date"]) == dt.date(2026, 6, 30)

    fetcher.requests.clear()
    panel = _loader(config, fetcher).load(["SPY"])
    assert fetcher.requests == []  # served from cache; nothing new requested
    assert max(panel.dates) == dt.date(2026, 3, 31)
    assert all(d < HOLDOUT_START for d in panel.dates)


def test_fetch_past_cutoff_is_refused_even_if_requested_internally(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    loader = _loader(config, FakeFetcher())
    with pytest.raises(HoldoutViolation):
        loader._fetch(["SPY"], dt.date(2026, 3, 1), HOLDOUT_START, loader.cutoff(None))


def test_holdout_flag_lifts_cutoff_and_logs_first(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    loader = _loader(config, fetcher)
    panel = loader.load(["SPY"], holdout=ACCESS)
    assert max(panel.dates) == dt.date(2026, 6, 30)  # yesterday relative to NOW
    (row,) = read_holdout_log(recipe(config).data.holdout_log)
    assert row["run_id"] == "r1"
    assert row["strategy"] == "example_buy_and_hold"
    assert row["data_end"] == "2026-06-30"
    assert row["git_commit"] == "abc123"


def test_holdout_end_pins_the_holdout_range(tmp_path: Path) -> None:
    config = tmp_config(tmp_path, holdout_end="2026-05-15")
    panel = _loader(config, FakeFetcher()).load(["SPY"], holdout=ACCESS)
    assert max(panel.dates) == dt.date(2026, 5, 15)


def test_unwritable_holdout_log_returns_no_data(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    r = recipe(config)
    r.data.holdout_log.mkdir(parents=True)  # a directory where the log file should be
    fetcher = FakeFetcher()
    with pytest.raises(HoldoutLogError):
        _loader(config, fetcher).load(["SPY"], holdout=ACCESS)
    assert fetcher.requests == []


def test_run_without_holdout_flag_never_shows_holdout_dates(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    loader = _loader(config, fetcher)
    loader.load(["SPY", "AAPL", "MSFT"], holdout=ACCESS)  # cache already holds holdout rows
    log_rows = len(read_holdout_log(recipe(config).data.holdout_log))

    summary = run_backtest(recipe(config), loader=_loader(config, fetcher), now=lambda: NOW,
                           git_commit=lambda: None)  # fmt: skip
    assert summary["end"] == "2026-03-31"
    assert summary["holdout"] is False
    assert summary["benchmark"]["total_return"] is not None
    equity = (Path(summary["run_dir"]) / "equity.csv").read_text().splitlines()[1:]
    assert all(line.split(",")[0] < "2026-04-01" for line in equity)
    assert len(read_holdout_log(recipe(config).data.holdout_log)) == log_rows  # not logged


def test_holdout_run_trades_from_holdout_start_with_warmup_and_is_logged(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    summary = run_backtest(recipe(config), holdout=True, loader=_loader(config, fetcher),
                           now=lambda: NOW, git_commit=lambda: "abc123")  # fmt: skip
    assert summary["holdout"] is True
    assert summary["start"] == "2026-04-01"
    assert summary["warmup_sessions"] > 100
    (row,) = read_holdout_log(recipe(config).data.holdout_log)
    assert row["run_id"] == Path(summary["run_dir"]).name
    index = (recipe(config).backtest.output_dir / "index.csv").read_text().splitlines()
    assert index[1].split(",")[6] == "True"


def test_holdout_log_run_id_is_the_run_folder_even_when_runs_share_a_second(
    tmp_path: Path,
) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    folders = [
        Path(
            run_backtest(
                recipe(config),
                holdout=True,
                loader=_loader(config, fetcher),
                now=lambda: NOW,
                git_commit=lambda: None,
            )["run_dir"]  # fmt: skip
        ).name
        for _ in range(2)
    ]
    assert folders[1] == folders[0] + "-2"
    logged = [row["run_id"] for row in read_holdout_log(recipe(config).data.holdout_log)]
    assert logged == folders


def test_holdout_benchmark_covers_the_same_window_as_the_strategy(tmp_path: Path) -> None:
    # With one slot, the example strategy (buy SPY, hold) is the benchmark, so every metric
    # matches, but only if the benchmark also starts with fresh capital at the holdout start.
    config = tmp_config(tmp_path)
    construction = config / "construction.yaml"
    construction.write_text(
        construction.read_text().replace("max_positions: 5", "max_positions: 1")
    )
    summary = run_backtest(recipe(config), holdout=True, loader=_loader(config, FakeFetcher()),
                           now=lambda: NOW, git_commit=lambda: None)  # fmt: skip
    assert summary["start"] == "2026-04-01"
    assert summary["trade_count"] == 1
    for key in ("total_return", "cagr", "max_drawdown", "sharpe"):
        assert summary[key] is not None
        assert summary[key] == summary["benchmark"][key], key
