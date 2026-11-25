"""The spin-off build (`harness.corporate_actions`, docs/spinoff-design.md decision 8):
resolving the three ways Alpaca serves a spin-off and a same-day dividend, the
all-or-nothing rule for events it cannot measure, holdout-dated events, and the shipped
files. No network: bars and dividends come from a table."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest

from harness import corporate_actions
from harness.adjust import apply_spinoffs
from harness.config import SPINOFFS_FILE, load_data_settings, load_spinoffs
from harness.corporate_actions import (
    EVENTS_FILE,
    REVIEWED_FILE,
    BuildError,
    build,
    load_events,
    load_reviewed,
)
from harness.data import (
    AlpacaDividendFetcher,
    BarPanel,
    DataError,
    DataLoader,
    HoldoutAccess,
    HoldoutViolation,
    _BarCache,  # pyright: ignore[reportPrivateUsage]
    read_holdout_log,
)
from harness.ratelimit import RateLimiter
from tests.fake_alpaca import NOW, REPO_CONFIG, tmp_config

EX = dt.date(2026, 3, 4)  # a Wednesday, in-sample in the test config
BEFORE = dt.date(2026, 3, 3)
HOLDOUT_EX = dt.date(2026, 5, 6)  # a Wednesday in the holdout
F = 0.6  # parent 60 on the ex-date, one child share per parent share worth 40
LATER = 0.9  # adjusted-to-raw ratio on the ex-date, from dividends after it


class TableFetcher:
    """Weekday bars from a table of closes. `raw[s][d]` and `adjusted[s][d]` override the
    default close of 50; `missing` removes a bar from both."""

    def __init__(self) -> None:
        self.raw: dict[str, dict[dt.date, float]] = {}
        self.adjusted: dict[str, dict[dt.date, float]] = {}
        self.missing: set[tuple[str, dt.date]] = set()
        self.unlisted: set[str] = set()
        self.requests: list[tuple[tuple[str, ...], dt.date, dt.date]] = []
        self.dividends: dict[str, list[tuple[dt.date, float]]] = {}

    def spinoff(self, parent: str, ex: dt.date, *, applied: float, child: str = "KID") -> None:
        """The parent at 100 raw before `ex` and 60 from it, the child at 40 on `ex`, and
        Alpaca's adjustment `applied` folded in before `ex`."""
        days = [ex - dt.timedelta(days=i) for i in range(1, 15)]
        self.raw[parent] = {d: 100.0 for d in days} | {ex: 60.0}
        self.adjusted[parent] = {d: 100.0 * LATER * applied for d in days} | {ex: 60.0 * LATER}
        self.raw[child] = {ex: 40.0}
        self.adjusted[child] = {ex: 40.0}

    def fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        self.requests.append((tuple(symbols), start, end))
        return {s: self._frame(s, start, end, self.adjusted) for s in symbols}

    def fetch_raw(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        return {s: self._frame(s, start, end, self.raw) for s in symbols}

    def cash_dividends(
        self, symbol: str, start: dt.date, end: dt.date
    ) -> list[tuple[dt.date, float]]:
        return [(d, r) for d, r in self.dividends.get(symbol, []) if start <= d <= end]

    def _frame(
        self, symbol: str, start: dt.date, end: dt.date, table: dict[str, dict[dt.date, float]]
    ) -> pd.DataFrame:
        """A symbol in `table` has bars on its listed days only; any other trades at 50
        every weekday (the calendar, SPY)."""
        rows = []
        day = start
        while day <= end:
            closes = table.get(symbol)
            close = 50.0 if closes is None else closes.get(day)
            live = day.weekday() < 5 and symbol not in self.unlisted
            if live and close is not None and (symbol, day) not in self.missing:
                rows.append({"date": day, "open": close, "high": close, "low": close,
                             "close": close, "volume": 1e6})  # fmt: skip
            day += dt.timedelta(days=1)
        return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def _config(tmp_path: Path, events: Sequence[str], reviewed: Sequence[str] = ()) -> Path:
    config = tmp_config(tmp_path)
    (config / EVENTS_FILE).write_text(
        "parent,ex_date,children,hand_closes,note\n" + "".join(f"{e}\n" for e in events)
    )
    (config / REVIEWED_FILE).write_text("ticker,date,note\n" + "".join(f"{r}\n" for r in reviewed))
    lines = (config / SPINOFFS_FILE).read_text().splitlines()
    (config / SPINOFFS_FILE).write_text(lines[0] + "\n")  # header only
    return config


def _loader(config: Path, fetcher: TableFetcher) -> DataLoader:
    settings = load_data_settings(config)
    return DataLoader(
        settings.dates,
        settings.data,
        fetcher_factory=lambda: fetcher,
        now=lambda: NOW,
        dividend_factory=lambda: fetcher,
    )


def _build(
    config: Path, fetcher: TableFetcher, **kwargs: HoldoutAccess
) -> corporate_actions.BuildResult:
    return build(config, _loader(config, fetcher), "SPY", **kwargs)


def _event(parent: str = "PAR", ex: dt.date = EX, children: str = "KID:1", hand: str = "") -> str:
    return f"{parent},{ex},{children},{hand},synthetic"


HOLDOUT = HoldoutAccess(run_id="spinoff-build-test", strategy="spinoff-build")


# --------------------------------------------------------------------------- resolving


@pytest.mark.parametrize(
    ("applied", "multiplier"),
    [(F, 1.0), (1.0, F), (2.0, F / 2)],  # applied correctly, not applied, reverse split
    ids=["applied", "not-applied", "reverse-split"],
)
def test_the_three_cases_resolve_to_a_continuous_series(
    tmp_path: Path, applied: float, multiplier: float
) -> None:
    config = _config(tmp_path, [_event()])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=applied)
    result = _build(config, fetcher)
    (row,) = load_spinoffs(config)
    assert result.built == (row,)
    assert (row.parent_close, row.child_closes) == (60.0, (40.0,))
    assert row.factor == pytest.approx(F)
    assert row.dividend_factor == 1.0
    assert row.applied_factor == pytest.approx(applied)
    assert row.multiplier == pytest.approx(multiplier)
    assert row.cached_step == pytest.approx(60 * LATER / (100 * LATER * applied))
    # The cached bars, adjusted by the row, no longer move on the ex-date.
    frames = {"PAR": fetcher.fetch(["PAR"], BEFORE, EX)["PAR"]}
    panel = apply_spinoffs(BarPanel.from_frames(frames), [row.applied()])
    close = panel.field("close")[:, 0]
    assert close[1] / close[0] == pytest.approx(1.0)


def test_a_same_day_cash_dividend_is_kept(tmp_path: Path) -> None:
    # Alpaca folded a 2.00 dividend (d = 0.98 on the close of 100 before) into its
    # correct adjustment: the multiplier is 1, not 1/0.98.
    config = _config(tmp_path, [_event()])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=F * 0.98)
    fetcher.dividends["PAR"] = [(EX, 2.0), (EX - dt.timedelta(days=7), 1.0)]  # one other day
    (row,) = _build(config, fetcher).built
    assert row.dividend_factor == pytest.approx(0.98)
    assert row.multiplier == pytest.approx(1.0)


def test_several_children_and_fraction_ratios(tmp_path: Path) -> None:
    config = _config(tmp_path, [_event(children="KID:1;KID2:1/2")])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=1.0)
    fetcher.raw["KID"] = {EX: 30.0}
    fetcher.raw["KID2"] = {EX: 20.0}  # 30 + 20 / 2 = 40 handed out
    (row,) = _build(config, fetcher).built
    assert row.children == (("KID", 1.0), ("KID2", 0.5))
    assert row.factor == pytest.approx(F)


def test_a_hand_close_stands_in_for_a_child_with_no_bars(tmp_path: Path) -> None:
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=1.0)
    fetcher.unlisted.add("KID")
    config = _config(tmp_path, [_event()])
    with pytest.raises(BuildError, match=r"PAR 2026-03-04: no KID bar on 2026-03-04: give"):
        _build(config, fetcher)
    config = _config(tmp_path / "hand", [_event(hand="KID:40")])
    (row,) = _build(config, fetcher).built
    assert row.child_closes == (40.0,)


def test_an_ex_date_that_is_not_a_session_is_refused(tmp_path: Path) -> None:
    config = _config(tmp_path, [_event(ex=dt.date(2026, 3, 7))])  # a Saturday
    with pytest.raises(BuildError, match="is not a session"):
        _build(config, TableFetcher())


# --------------------------------------------------------------------------- all or nothing


def _unmeasurable(tmp_path: Path, reviewed: Sequence[str] = ()) -> tuple[Path, TableFetcher]:
    """Three events: OK measures; HALT has no bar the session before; GONE none on the
    ex-date."""
    fetcher = TableFetcher()
    for parent in ("OK", "HALT", "GONE"):
        fetcher.spinoff(parent, EX, applied=1.0)
    fetcher.missing |= {("HALT", BEFORE), ("GONE", EX)}
    events = [_event(p) for p in ("OK", "HALT", "GONE")]
    return _config(tmp_path, events, reviewed), fetcher


def test_an_unmeasurable_event_fails_the_build_naming_every_one(tmp_path: Path) -> None:
    config, fetcher = _unmeasurable(tmp_path)
    # A good row from an earlier build stays as it was.
    previous = "OK,2026-03-04,KID:1,60,KID:40,0.6,1.0,1.0,0.6,0.6,earlier\n"
    with (config / SPINOFFS_FILE).open("a") as fh:
        fh.write(previous)
    before = (config / SPINOFFS_FILE).read_bytes()
    with pytest.raises(BuildError) as caught:
        _build(config, fetcher)
    message = str(caught.value)
    assert "2 event(s)" in message
    assert "HALT 2026-03-04: no HALT bar on the session before (2026-03-03)" in message
    assert "GONE 2026-03-04: no GONE bar on the ex-date" in message
    assert "OK 2026" not in message
    assert str(REVIEWED_FILE) in message
    assert (config / SPINOFFS_FILE).read_bytes() == before


def test_the_same_events_pass_once_reviewed(tmp_path: Path) -> None:
    reviewed = ["HALT,2026-03-04,halted that day", "GONE,2026-03-04,halted that day"]
    config, fetcher = _unmeasurable(tmp_path, reviewed)
    result = _build(config, fetcher)
    assert [e.parent for e in result.built] == ["OK"]
    assert sorted(p.parent for p, _ in result.skipped) == ["GONE", "HALT"]
    assert [e.parent for e in load_spinoffs(config)] == ["OK"]


def test_a_review_for_one_event_does_not_excuse_another(tmp_path: Path) -> None:
    config, fetcher = _unmeasurable(tmp_path, ["HALT,2026-03-04,halted that day"])
    with pytest.raises(BuildError, match="1 event") as caught:
        _build(config, fetcher)
    assert "GONE" in str(caught.value) and "HALT 2026" not in str(caught.value)


def test_a_review_does_not_excuse_a_missing_child_close(tmp_path: Path) -> None:
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=1.0)
    fetcher.unlisted.add("KID")
    config = _config(tmp_path, [_event()], ["PAR,2026-03-04,no idea"])
    with pytest.raises(BuildError, match="give its raw close in hand_closes"):
        _build(config, fetcher)


def test_a_reviewed_row_needs_a_note(tmp_path: Path) -> None:
    config = _config(tmp_path, [], ["PAR,2026-03-04, "])
    with pytest.raises(ValueError, match="spinoffs-reviewed.csv line 2"):
        load_reviewed(config)


def test_rows_no_longer_listed_are_removed(tmp_path: Path) -> None:
    config = _config(tmp_path, [_event("PAR")])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=1.0)
    _build(config, fetcher)
    (config / EVENTS_FILE).write_text("parent,ex_date,children,hand_closes,note\n")
    result = _build(config, fetcher)
    assert [e.parent for e in result.removed] == ["PAR"]
    assert load_spinoffs(config) == ()


def test_a_rebuild_is_byte_identical(tmp_path: Path) -> None:
    config = _config(tmp_path, [_event("PAR"), _event("PAR", ex=dt.date(2026, 3, 18))])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=2.0)
    for table, scale in ((fetcher.raw, 1.0), (fetcher.adjusted, LATER)):
        table["PAR"] |= {dt.date(2026, 3, 17): 60.0 * scale, dt.date(2026, 3, 18): 50.0 * scale}
    fetcher.raw["KID"][dt.date(2026, 3, 18)] = 10.0
    _build(config, fetcher)
    first = (config / SPINOFFS_FILE).read_bytes()
    _build(config, fetcher)
    assert (config / SPINOFFS_FILE).read_bytes() == first
    assert len(load_spinoffs(config)) == 2


# --------------------------------------------------------------------------- holdout


def test_a_holdout_dated_event_needs_the_flag_and_is_never_fetched_without_it(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, [_event(ex=HOLDOUT_EX)])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", HOLDOUT_EX, applied=1.0)
    with pytest.raises(BuildError, match="dated in the holdout .* run build --holdout"):
        _build(config, fetcher)
    assert fetcher.requests == []


def test_holdout_built_rows_are_logged_then_carried_over(tmp_path: Path) -> None:
    config = _config(tmp_path, [_event(ex=EX), _event("LATE", ex=HOLDOUT_EX)])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=1.0)
    fetcher.spinoff("LATE", HOLDOUT_EX, applied=1.0, child="KID")
    fetcher.raw["KID"][EX] = 40.0
    result = _build(config, fetcher, holdout=HOLDOUT)
    assert [e.parent for e in result.built] == ["LATE", "PAR"]
    log = read_holdout_log(load_data_settings(config).data.holdout_log)
    assert [(r["run_id"], r["data_end"]) for r in log] == [("spinoff-build-test", str(HOLDOUT_EX))]
    # Without the flag the holdout row is kept as built, and nothing past the cutoff is
    # fetched.
    fetcher.requests.clear()
    result = _build(config, fetcher)
    assert [e.parent for e in result.carried] == ["LATE"]
    assert all(end < HOLDOUT_EX for _, _, end in fetcher.requests)
    assert len(load_spinoffs(config)) == 2
    # A changed listing is not carried over.
    (config / EVENTS_FILE).write_text(
        "parent,ex_date,children,hand_closes,note\n" + _event("LATE", HOLDOUT_EX, "KID:2") + "\n"
    )
    with pytest.raises(BuildError, match="LATE 2026-05-06: dated in the holdout"):
        _build(config, fetcher)


def test_a_window_into_the_holdout_is_refused_without_access(tmp_path: Path) -> None:
    config = _config(tmp_path, [])
    loader = _loader(config, TableFetcher())
    with pytest.raises(HoldoutViolation):
        loader.window(["SPY"], HOLDOUT_EX - dt.timedelta(days=5), HOLDOUT_EX)


# --------------------------------------------------------------------------- CLI


def test_the_cli_exits_1_and_writes_nothing_when_an_event_is_unmeasurable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, fetcher = _unmeasurable(tmp_path)
    before = (config / SPINOFFS_FILE).read_bytes()
    code = corporate_actions.main(["build"], config, loader=_loader(config, fetcher))
    assert code == 1
    assert "HALT" in capsys.readouterr().err
    assert (config / SPINOFFS_FILE).read_bytes() == before


def test_the_cli_prints_each_built_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _config(tmp_path, [_event()])
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", EX, applied=2.0)
    assert corporate_actions.main(["build"], config, loader=_loader(config, fetcher)) == 0
    out = capsys.readouterr().out
    assert "PAR 2026-03-04" in out and "0.3000" in out and "new" in out


# --------------------------------------------------------------------------- shipped files


def test_every_shipped_event_is_built_or_reviewed_as_listed() -> None:
    events = load_events(REPO_CONFIG)
    reviewed = load_reviewed(REPO_CONFIG)
    rows = {(e.parent, e.ex_date): e for e in load_spinoffs(REPO_CONFIG)}
    listed = {(e.parent, e.ex_date) for e in events}
    assert set(rows) <= listed, "a row in spinoffs.csv that is not listed: rerun build"
    for event in events:
        key = (event.parent, event.ex_date)
        if key in rows:
            assert rows[key].children == event.children
            assert rows[key].note == event.note
        else:
            assert key in reviewed, f"{key} is listed but neither built nor reviewed"


RTX_EX = dt.date(2020, 4, 3)


def test_rtx_is_continuous_across_its_spinoff_in_the_cache() -> None:
    settings = load_data_settings(REPO_CONFIG)
    cache = _BarCache(settings.data.cache_dir)
    if cache.meta("RTX") is None:
        pytest.skip("no cached RTX bars")
    frame = cache.read("RTX", dt.date(2020, 3, 1), dt.date(2020, 5, 1))
    panel = BarPanel.from_frames({"RTX": frame})
    events = [e.applied() for e in load_spinoffs(REPO_CONFIG) if e.parent == "RTX"]
    close = panel.field("close")[:, 0]
    i = panel.dates.index(RTX_EX)
    assert close[i] / close[i - 1] < 0.35  # as cached: a fake fall of about 71%
    adjusted = apply_spinoffs(panel, events).field("close")[:, 0]
    # What a UTX holder had that day (RTX, CARR and half an OTIS) rose about 5%.
    assert abs(adjusted[i] / adjusted[i - 1] - 1) < 0.10


# --------------------------------------------------------------------------- failures


class FailingFetcher(TableFetcher):
    """Bar requests that include `fail` raise, as a failed Alpaca request does."""

    def __init__(self, fail: str) -> None:
        super().__init__()
        self.fail = fail

    def fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        if self.fail in symbols:
            raise DataError(f"Alpaca bars request failed for {len(symbols)} symbols: boom")
        return super().fetch(symbols, start, end)


def test_a_failed_request_is_named_with_every_other_fault(tmp_path: Path) -> None:
    fetcher = FailingFetcher("BAD")
    for parent in ("OK", "HALT", "BAD"):
        fetcher.spinoff(parent, EX, applied=1.0)
    fetcher.missing.add(("HALT", BEFORE))
    config = _config(tmp_path, [_event(p) for p in ("BAD", "HALT", "OK")])
    before = (config / SPINOFFS_FILE).read_bytes()
    with pytest.raises(BuildError, match="2 event") as caught:
        _build(config, fetcher)
    assert "BAD 2026-03-04: Alpaca bars request failed" in str(caught.value)
    assert "HALT 2026-03-04" in str(caught.value)
    assert (config / SPINOFFS_FILE).read_bytes() == before


def test_an_event_past_a_pinned_holdout_end_is_a_fault_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, [_event(ex=HOLDOUT_EX)])
    (config / "dates.yaml").write_text(
        (config / "dates.yaml").read_text().replace("holdout_end: null", "holdout_end: 2026-04-30")
    )
    fetcher = TableFetcher()
    fetcher.spinoff("PAR", HOLDOUT_EX, applied=1.0)
    with pytest.raises(BuildError, match="PAR 2026-05-06: after the last loadable day"):
        _build(config, fetcher, holdout=HOLDOUT)
    assert fetcher.requests == []
    code = corporate_actions.main(["build", "--holdout"], config, loader=_loader(config, fetcher))
    assert code == 1 and "after the last loadable day" in capsys.readouterr().err


class _Item:
    def __init__(self, ex_date: dt.date, rate: object) -> None:
        self.symbol, self.ex_date, self.rate = "PAR", ex_date, rate


class _Response:
    def __init__(self, *items: _Item) -> None:
        self.data = {"cash_dividends": list(items)}


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def get_corporate_actions(self, request: object) -> _Response:
        return self.response


def _dividends(response: _Response) -> AlpacaDividendFetcher:
    fetcher = object.__new__(AlpacaDividendFetcher)  # no keys: the client is a stub
    fetcher._limiter = RateLimiter(1000)  # pyright: ignore[reportPrivateUsage]
    fetcher._today = lambda: dt.date(2026, 7, 1)  # pyright: ignore[reportPrivateUsage]
    fetcher._client = _Client(response)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return fetcher


def test_a_dividend_with_no_amount_fails_rather_than_counting_as_none() -> None:
    other = _Item(dt.date(2026, 1, 2), 0.5)  # outside the window: dropped
    good = _dividends(_Response(_Item(EX, 0.25), other))
    assert good.cash_dividends("PAR", BEFORE, EX) == [(EX, 0.25)]
    for rate in (None, float("nan"), -1.0):
        with pytest.raises(DataError, match="no valid amount"):
            _dividends(_Response(_Item(EX, rate))).cash_dividends("PAR", BEFORE, EX)
