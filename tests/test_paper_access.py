"""Paper access: the second way past the holdout cutoff, with its own log.

Paper trading reads yesterday's close every day. That must not flood the holdout log,
must not loosen anything for backtests, and must leave a record of its own.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from harness.data import (
    DataLoader,
    HoldoutAccess,
    HoldoutViolation,
    PaperAccess,
    PaperLogError,
    read_holdout_log,
    read_paper_access_log,
)
from tests.fake_alpaca import NOW, FakeFetcher, recipe, tmp_config

HOLDOUT_START = dt.date(2026, 4, 1)
YESTERDAY = dt.date(2026, 6, 30)  # the last completed session relative to NOW
PAPER = PaperAccess(profile_id="plumbing-0a1b2c3d")
HARNESS = Path(__file__).resolve().parent.parent / "harness"


def _loader(config: Path, fetcher: FakeFetcher, now: dt.datetime = NOW) -> DataLoader:
    r = recipe(config)
    return DataLoader(r.dates, r.data, fetcher_factory=lambda: fetcher, now=lambda: now)


def test_paper_access_reaches_yesterdays_close_and_logs_it(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    panel = _loader(config, fetcher).load(["SPY", "AAPL"], paper=PAPER)
    assert max(panel.dates) == YESTERDAY
    assert fetcher.latest_end == YESTERDAY
    (row,) = read_paper_access_log(recipe(config).data.paper_access_log)
    assert row == {
        "timestamp_utc": "2026-07-01T15:00:00Z",
        "day": "2026-07-01",
        "profile_id": "plumbing-0a1b2c3d",
        "data_start": "2025-10-01",
        "data_end": "2026-06-30",
    }
    # The holdout log is for holdout runs. Paper trading never writes to it.
    assert not recipe(config).data.holdout_log.exists()


def test_paper_access_ignores_a_pinned_holdout_end(tmp_path: Path) -> None:
    # `holdout_end` pins a holdout backtest's window. Paper always needs yesterday.
    config = tmp_config(tmp_path, holdout_end="2026-05-15")
    loader = _loader(config, FakeFetcher())
    assert max(loader.load(["SPY"], paper=PAPER).dates) == YESTERDAY
    holdout = HoldoutAccess(run_id="r1", strategy="example_buy_and_hold")
    assert max(loader.load(["SPY"], holdout=holdout).dates) == dt.date(2026, 5, 15)


def test_one_line_per_profile_per_day(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    log = recipe(config).data.paper_access_log
    loader = _loader(config, fetcher)
    for _ in range(3):  # the runner, a rerun, a restart: still one line
        loader.load(["SPY"], paper=PAPER)
    assert len(read_paper_access_log(log)) == 1
    loader.load(["SPY"], paper=PaperAccess(profile_id="momentum-99999999"))
    assert [r["profile_id"] for r in read_paper_access_log(log)] == [
        "plumbing-0a1b2c3d",
        "momentum-99999999",
    ]
    tomorrow = _loader(config, fetcher, now=NOW + dt.timedelta(days=1))
    tomorrow.load(["SPY"], paper=PAPER)
    rows = read_paper_access_log(log)
    assert [(r["day"], r["data_end"]) for r in rows if r["profile_id"] == PAPER.profile_id] == [
        ("2026-07-01", "2026-06-30"),
        ("2026-07-02", "2026-07-01"),
    ]


def test_the_day_is_the_new_york_day(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    late = dt.datetime(2026, 7, 2, 3, 30, tzinfo=dt.UTC)  # 23:30 on July 1 in New York
    _loader(config, FakeFetcher(), now=late).load(["SPY"], paper=PAPER)
    (row,) = read_paper_access_log(recipe(config).data.paper_access_log)
    assert (row["day"], row["data_end"]) == ("2026-07-01", "2026-06-30")


def test_the_log_is_written_before_any_bar_is_fetched(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    log = recipe(config).data.paper_access_log
    seen: list[int] = []

    class Watching(FakeFetcher):
        def fetch(self, symbols, start, end):  # type: ignore[no-untyped-def]
            seen.append(len(read_paper_access_log(log)))
            return super().fetch(symbols, start, end)

    _loader(config, Watching()).load(["SPY"], paper=PAPER)
    assert seen and all(count == 1 for count in seen)


def test_log_that_cannot_be_written_returns_no_data(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    recipe(config).data.paper_access_log.mkdir(parents=True)  # a directory in its place
    fetcher = FakeFetcher()
    with pytest.raises(PaperLogError, match="could not update the paper access log"):
        _loader(config, fetcher).load(["SPY"], paper=PAPER)
    assert fetcher.requests == []  # nothing past the cutoff was even asked for


def test_log_that_cannot_be_read_returns_no_data(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    log = recipe(config).data.paper_access_log
    log.parent.mkdir(parents=True)
    log.write_bytes(b"timestamp_utc,day,profile_id\n\xff\xfe\x00not text\n")
    fetcher = FakeFetcher()
    with pytest.raises(PaperLogError):
        _loader(config, fetcher).load(["SPY"], paper=PAPER)
    assert fetcher.requests == []


def test_paper_access_does_not_loosen_anything_for_a_backtest(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    loader = _loader(config, fetcher)
    loader.load(["SPY"], paper=PAPER)  # the cache now holds bars through yesterday
    cached = loader._cache._read_unfiltered("SPY")  # test-only bypass, to prove rows exist
    assert max(cached["date"]) == YESTERDAY
    fetcher.requests.clear()
    panel = loader.load(["SPY"])  # an ordinary backtest load
    assert max(panel.dates) == dt.date(2026, 3, 31)
    assert all(end < HOLDOUT_START for _, _, end in fetcher.requests)
    assert loader.cutoff(None) == HOLDOUT_START
    assert len(read_paper_access_log(recipe(config).data.paper_access_log)) == 1  # not logged


def test_paper_access_still_cannot_fetch_past_the_last_completed_session(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    loader = _loader(config, FakeFetcher())
    cutoff = loader.cutoff(None, PAPER)
    assert cutoff == YESTERDAY + dt.timedelta(days=1)
    with pytest.raises(HoldoutViolation):  # today's bar is not complete: never requested
        loader._fetch(["SPY"], dt.date(2026, 6, 1), cutoff, cutoff)


def test_holdout_and_paper_access_cannot_be_combined(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    holdout = HoldoutAccess(run_id="r1", strategy="example_buy_and_hold")
    with pytest.raises(ValueError, match="cannot be combined"):
        _loader(config, fetcher).load(["SPY"], holdout=holdout, paper=PAPER)
    assert fetcher.requests == []
    assert read_holdout_log(recipe(config).data.holdout_log) == []
    assert read_paper_access_log(recipe(config).data.paper_access_log) == []


@pytest.mark.parametrize(
    "profile_id",
    ["plumbing-0a1b2c3d", "my-strategy.v2-0A1B2C3D", "a-12345678", "mean_rev-abcdefgh"],
)
def test_profile_id_of_a_frozen_profile_is_accepted(profile_id: str) -> None:
    assert PaperAccess(profile_id=profile_id).profile_id == profile_id


@pytest.mark.parametrize(
    "profile_id",
    [
        "",
        "plumbing",  # no hash
        "plumbing-123",  # hash too short
        "plumbing-0a1b2c3!",
        "-0a1b2c3d",  # no name
        "0a1b2c3d",
        "plumbing-0a1b2c3d ",
        "../x-0a1b2c3d",  # not something to put in a path
        "plumbing--0a1b2c3d",
    ],
)
def test_anything_else_is_not_a_profile_id(profile_id: str) -> None:
    with pytest.raises(ValueError):
        PaperAccess(profile_id=profile_id)


def test_only_the_profiles_module_builds_paper_access() -> None:
    # One choke point: `harness.profiles.paper_access_for` builds a PaperAccess from a
    # profile it has loaded and verified. No other harness code builds one, and the
    # backtest entry point never passes one.
    builds = re.compile(r"PaperAccess\(")
    allowed = {"profiles.py"}
    for path in HARNESS.rglob("*.py"):
        text = path.read_text().replace("class PaperAccess(FrozenModel)", "")
        if builds.search(text):
            assert path.name in allowed, f"{path.name} builds a PaperAccess"
    assert "paper=" not in (HARNESS / "run.py").read_text()
