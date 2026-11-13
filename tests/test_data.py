"""Data loader and cache: download only what is missing, detect re-adjusted history,
batch requests, and convert Alpaca responses. No network."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.data import (
    AlpacaBarFetcher,
    DataError,
    DataLoader,
    HoldoutAccess,
    RateLimiter,
    frames_from_alpaca_df,
)
from tests.fake_alpaca import NOW, FakeFetcher, recipe, tmp_config

ACCESS = HoldoutAccess(run_id="r", strategy="s")


def _loader(config: Path, fetcher: FakeFetcher, now: dt.datetime = NOW) -> DataLoader:
    r = recipe(config)
    return DataLoader(r.dates, r.data, fetcher_factory=lambda: fetcher, now=lambda: now)


def test_second_load_is_served_from_cache(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    first = _loader(config, fetcher).load(["SPY", "AAPL"])
    n = len(fetcher.requests)
    second = _loader(config, fetcher).load(["SPY", "AAPL"])
    assert len(fetcher.requests) == n
    assert first.fingerprint() == second.fingerprint()


def test_extending_fetches_only_the_overlap_and_new_days(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    _loader(config, fetcher).load(["SPY"])  # through 2026-03-31
    fetcher.requests.clear()
    panel = _loader(config, fetcher).load(["SPY"], holdout=ACCESS)  # through 2026-06-30
    ((symbols, start, end),) = fetcher.requests
    assert symbols == ("SPY",)
    assert start == dt.date(2026, 3, 25)  # last 5 cached sessions: 3/25 .. 3/31
    assert end == dt.date(2026, 6, 30)
    assert panel.dates[0] == dt.date(2025, 10, 1)
    assert panel.dates[-1] == dt.date(2026, 6, 30)


def test_corporate_action_after_cache_end_triggers_full_refetch(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    _loader(config, fetcher).load(["SPY", "AAPL"])
    # A 2:1 split on 2026-05-01: the vendor now reports every earlier AAPL bar halved.
    fetcher.scale_before["AAPL"] = (dt.date(2026, 5, 1), 0.5)
    fetcher.requests.clear()
    panel = _loader(config, fetcher).load(["SPY", "AAPL"], holdout=ACCESS)
    full = [r for r in fetcher.requests if r[1] == dt.date(2025, 10, 1)]
    assert full == [(("AAPL",), dt.date(2025, 10, 1), dt.date(2026, 6, 30))]
    close = panel.field("close")[:, panel.column("AAPL")]
    expected = [fetcher.price("AAPL", d) for d in panel.dates]
    np.testing.assert_allclose(close, expected)  # one consistent scaling, no mixing


def test_stale_cache_is_fully_refetched(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    _loader(config, fetcher).load(["SPY"])
    fetcher.requests.clear()
    _loader(config, fetcher, now=NOW + dt.timedelta(days=29)).load(["SPY"])
    assert fetcher.requests == []
    _loader(config, fetcher, now=NOW + dt.timedelta(days=30)).load(["SPY"])
    assert fetcher.requests == [(("SPY",), dt.date(2025, 10, 1), dt.date(2026, 3, 31))]


def test_symbol_with_no_data_is_not_refetched_every_run(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    fetcher.listed["NEWCO"] = dt.date(2030, 1, 1)
    panel = _loader(config, fetcher).load(["SPY", "NEWCO"])
    assert np.isnan(panel.field("close")[:, panel.column("NEWCO")]).all()
    fetcher.requests.clear()
    _loader(config, fetcher).load(["SPY", "NEWCO"])
    assert fetcher.requests == []


def test_late_listing_and_delisting_leave_nan_gaps(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    fetcher.listed["IPO"] = dt.date(2026, 1, 5)
    fetcher.delisted["GONE"] = dt.date(2025, 12, 1)
    panel = _loader(config, fetcher).load(["SPY", "IPO", "GONE"])
    dates = np.array(panel.dates)
    ipo = panel.field("close")[:, panel.column("IPO")]
    gone = panel.field("close")[:, panel.column("GONE")]
    assert np.isnan(ipo[dates < dt.date(2026, 1, 5)]).all()
    assert not np.isnan(ipo[dates >= dt.date(2026, 1, 5)]).any()
    assert np.isnan(gone[dates >= dt.date(2025, 12, 1)]).all()


def test_requests_are_batched(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    data_yaml = config / "data.yaml"
    data_yaml.write_text(data_yaml.read_text().replace("batch_size: 50", "batch_size: 2"))
    fetcher = FakeFetcher()
    _loader(config, fetcher).load(["SPY", "AAPL", "MSFT", "NVDA", "XOM"])
    assert [len(r[0]) for r in fetcher.requests] == [2, 2, 1]


def test_fetcher_returning_out_of_range_bars_is_rejected(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)

    class Leaky(FakeFetcher):
        def fetch(self, symbols, start, end):  # type: ignore[no-untyped-def]
            return super().fetch(symbols, start, end + dt.timedelta(days=10))

    with pytest.raises(DataError, match="outside"):
        _loader(config, Leaky()).load(["SPY"])


def test_cache_files_are_one_parquet_per_symbol(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    _loader(config, FakeFetcher()).load(["SPY", "AAPL"])
    cache_dir = recipe(config).data.cache_dir
    assert sorted(p.name for p in cache_dir.iterdir()) == ["AAPL.parquet", "SPY.parquet"]


def test_no_fetcher_is_built_when_cache_is_complete(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    _loader(config, FakeFetcher()).load(["SPY"])
    r = recipe(config)

    def no_network() -> FakeFetcher:
        raise AssertionError("fetcher built although the cache was complete")

    DataLoader(r.dates, r.data, fetcher_factory=no_network, now=lambda: NOW).load(["SPY"])


def _future_in_sample_config(tmp_path: Path) -> Path:
    """in_sample_end 2026-12-31, later than the tests' clock (NOW = 2026-07-01)."""
    config = tmp_config(tmp_path)
    dates = config / "dates.yaml"
    dates.write_text(
        dates.read_text().replace("2026-03-31", "2026-12-31").replace("2026-04-01", "2027-01-01")
    )
    return config


def test_load_never_requests_past_the_last_completed_session(tmp_path: Path) -> None:
    config = _future_in_sample_config(tmp_path)
    fetcher = FakeFetcher()
    panel = _loader(config, fetcher).load(["SPY"])
    yesterday = dt.date(2026, 6, 30)
    assert fetcher.latest_end == yesterday  # not 2026-12-31: those sessions have not happened
    assert max(panel.dates) == yesterday
    meta = _loader(config, fetcher)._cache.meta("SPY")
    assert meta is not None and meta.coverage_end == yesterday


def test_cache_extends_as_sessions_complete_before_in_sample_end(tmp_path: Path) -> None:
    config = _future_in_sample_config(tmp_path)
    fetcher = FakeFetcher()
    _loader(config, fetcher).load(["SPY"])
    fetcher.requests.clear()
    later = NOW + dt.timedelta(days=20)  # 2026-07-21; last completed session 2026-07-20
    panel = _loader(config, fetcher, now=later).load(["SPY"])
    ((_, start, end),) = fetcher.requests  # one extend request, not a full refetch
    assert start == dt.date(2026, 6, 24)  # the 5 overlap bars: 6/24 .. 6/30
    assert end == dt.date(2026, 7, 20)
    assert max(panel.dates) == dt.date(2026, 7, 20)


def test_holdout_log_records_the_clamped_end(tmp_path: Path) -> None:
    from harness.data import read_holdout_log

    config = tmp_config(tmp_path, holdout_end="2026-09-30")  # pinned later than the clock
    panel = _loader(config, FakeFetcher()).load(["SPY"], holdout=ACCESS)
    assert max(panel.dates) == dt.date(2026, 6, 30)
    (row,) = read_holdout_log(recipe(config).data.holdout_log)
    assert row["data_end"] == "2026-06-30"


def test_load_before_any_session_has_completed_is_a_clear_error(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)  # data_start 2025-10-01
    early = dt.datetime(2025, 10, 1, 15, 0, tzinfo=dt.UTC)
    fetcher = FakeFetcher()
    with pytest.raises(DataError, match="no completed session"):
        _loader(config, fetcher, now=early).load(["SPY"])
    assert fetcher.requests == []


def test_each_cache_write_uses_its_own_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import harness.data as data_module

    config = tmp_config(tmp_path)
    written: list[Path] = []
    real_write = data_module.pq.write_table

    def recording_write(table, where, *args, **kwargs):  # type: ignore[no-untyped-def]
        written.append(Path(where))
        real_write(table, where, *args, **kwargs)

    monkeypatch.setattr(data_module.pq, "write_table", recording_write)
    _loader(config, FakeFetcher()).load(["SPY"])
    _loader(config, FakeFetcher(), now=NOW + dt.timedelta(days=30)).load(["SPY"])  # refresh
    assert len(written) == 2
    assert written[0] != written[1]  # two runs writing SPY at once cannot share a temp file
    cache_dir = recipe(config).data.cache_dir
    assert all(p.parent == cache_dir for p in written)  # same directory: rename stays atomic
    assert [p.name for p in cache_dir.iterdir()] == ["SPY.parquet"]


def test_failed_cache_write_keeps_the_old_file_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import harness.data as data_module

    config = tmp_config(tmp_path)
    first = _loader(config, FakeFetcher()).load(["SPY"])

    def half_write(table, where, *args, **kwargs):  # type: ignore[no-untyped-def]
        Path(where).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(data_module.pq, "write_table", half_write)
    with pytest.raises(OSError, match="disk full"):
        _loader(config, FakeFetcher(), now=NOW + dt.timedelta(days=30)).load(["SPY"])
    monkeypatch.undo()
    cache_dir = recipe(config).data.cache_dir
    assert [p.name for p in cache_dir.iterdir()] == ["SPY.parquet"]
    assert _loader(config, FakeFetcher()).load(["SPY"]).fingerprint() == first.fingerprint()


# --------------------------------------------------------------------------- pieces


def test_rate_limiter_waits_when_the_window_is_full() -> None:
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    limiter = RateLimiter(3, clock=lambda: clock[0], sleep=sleep)
    limiter.acquire(3)
    assert sleeps == []
    clock[0] = 10.0
    limiter.acquire(1)
    assert sleeps == [50.0]


def test_alpaca_frames_use_new_york_session_dates() -> None:
    index = pd.MultiIndex.from_tuples(
        [("AAPL", pd.Timestamp("2026-03-02 05:00", tz="UTC")),
         ("AAPL", pd.Timestamp("2026-03-03 05:00", tz="UTC")),
         ("SPY", pd.Timestamp("2026-03-02 05:00", tz="UTC"))],
        names=["symbol", "timestamp"],
    )  # fmt: skip
    df = pd.DataFrame(
        {"open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5], "low": [0.5, 1.5, 2.5],
         "close": [1.2, 2.2, 3.2], "volume": [10.0, 20.0, 30.0], "trade_count": [1, 2, 3],
         "vwap": [1.1, 2.1, 3.1]},
        index=index,
    )  # fmt: skip
    frames = frames_from_alpaca_df(df, ["AAPL", "SPY", "MSFT"])
    assert list(frames["AAPL"]["date"]) == [dt.date(2026, 3, 2), dt.date(2026, 3, 3)]
    assert list(frames["AAPL"].columns) == ["date", "open", "high", "low", "close", "volume"]
    assert frames["SPY"]["close"].tolist() == [3.2]
    assert frames["MSFT"].empty


def test_alpaca_frames_handle_empty_response() -> None:
    assert frames_from_alpaca_df(pd.DataFrame(), ["SPY"])["SPY"].empty


def test_missing_credentials_fail_clearly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    config = tmp_config(tmp_path)
    r = recipe(config)
    with pytest.raises(DataError, match="ALPACA_PAPER_API_KEY"):
        AlpacaBarFetcher(r.data, RateLimiter(10), dotenv_path=tmp_path / "no.env")
