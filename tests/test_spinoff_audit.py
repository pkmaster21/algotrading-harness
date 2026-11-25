"""The spin-off audit (`harness.corporate_actions audit`, docs/spinoff-design.md decision
8): each Alpaca source finds its candidates, a candidate is covered by spinoffs.csv or a
reviewed row or else fails the audit, only bars a run can read count, and the holdout
stays closed. No network: bars and feed rows come from tables."""

from __future__ import annotations

import datetime as dt
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from harness import corporate_actions
from harness.config import load_data_settings, load_universe, universe_names
from harness.corporate_actions import EVENTS_FILE, REVIEWED_FILE, audit
from harness.data import (
    Actions,
    AlpacaActionsFetcher,
    DataError,
    DataLoader,
    FeedSpinOff,
    HoldoutViolation,
    Split,
)
from harness.ratelimit import RateLimiter
from tests.fake_alpaca import NOW, REPO_CONFIG, spinoff_row, tmp_config

START = dt.date(2025, 10, 1)  # tmp_config's data start
END = dt.date(2026, 3, 31)  # the day before the holdout start
DAY = dt.date(2026, 1, 14)  # a Wednesday
NEXT = dt.date(2026, 1, 15)


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    days, day = [], start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


class AuditFetcher:
    """Every symbol trades at 50 every weekday, raw and adjusted alike, unless a table
    says otherwise. `adjusted[s]` scales the adjusted bars before a date (an adjustment
    Alpaca applied); `gap[s]` opens and closes `s` lower from a date in both; `listed[s]`
    starts the symbol's bars on a date."""

    def __init__(self) -> None:
        self.adjusted: dict[str, tuple[dt.date, float]] = {}
        self.gap: dict[str, tuple[dt.date, float]] = {}
        self.listed: dict[str, dt.date] = {}
        self.feed = Actions(spinoffs=(), splits=())
        self.ends: list[dt.date] = []

    def _frames(
        self, symbols: Sequence[str], start: dt.date, end: dt.date, *, raw: bool
    ) -> dict[str, pd.DataFrame]:
        self.ends.append(end)
        out: dict[str, pd.DataFrame] = {}
        for s in symbols:
            rows = []
            for day in _weekdays(max(start, self.listed.get(s, start)), end):
                price = 50.0
                if s in self.gap and day >= self.gap[s][0]:
                    price *= self.gap[s][1]
                if not raw and s in self.adjusted and day < self.adjusted[s][0]:
                    price *= self.adjusted[s][1]
                rows.append({"date": day, "open": price, "high": price, "low": price,
                             "close": price, "volume": 1e6})  # fmt: skip
            out[s] = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        return out

    def fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        return self._frames(symbols, start, end, raw=False)

    def fetch_raw(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        return self._frames(symbols, start, end, raw=True)

    def actions(self, start: dt.date, end: dt.date) -> Actions:
        self.ends.append(end)
        return Actions(
            spinoffs=tuple(r for r in self.feed.spinoffs if start <= r.ex_date <= end),
            splits=tuple(r for r in self.feed.splits if start <= r.ex_date <= end),
        )


def _config(
    tmp_path: Path,
    *,
    fixed: Sequence[str] = ("SPY", "AAA"),
    members: Sequence[str] = (),
    renames: Sequence[str] = (),
    bad_bars: Sequence[str] = (),
    reviewed: Sequence[str] = (),
    spinoffs: Sequence[str] = (),
) -> Path:
    """Only two universes: `fixed` as a symbol list, and `members` as membership rows
    (`TICKER,start,end`) of a point-in-time universe."""
    config = tmp_config(tmp_path, spinoffs=spinoffs)
    universe = config / "universe"
    for path in universe.glob("*.yaml"):
        path.unlink()
    (universe / "fixed.yaml").write_text(f"symbols: [{', '.join(fixed)}]\n")
    if members:
        (universe / "mini.yaml").write_text(
            "membership:\n  file: mini-membership.csv\n  corrections: mini-corrections.csv\n"
            "  renames: mini-renames.csv\n  bad_bars: mini-bad-bars.csv\n"
            "  reviewed_multi_interval: []\n"
        )
        files = {
            "mini-membership.csv": ("ticker,start_date,end_date", members),
            "mini-corrections.csv": (
                "ticker,start_date,end_date,new_start_date,new_end_date,reason",
                (),
            ),
            "mini-renames.csv": ("old,new,date,note", renames),
            "mini-bad-bars.csv": ("ticker,from_date,to_date,reason", bad_bars),
        }
        for name, (header, rows) in files.items():
            (universe / name).write_text(header + "\n" + "".join(f"{r}\n" for r in rows))
    (config / EVENTS_FILE).write_text("parent,ex_date,children,hand_closes,note\n")
    (config / REVIEWED_FILE).write_text("ticker,date,note\n" + "".join(f"{r}\n" for r in reviewed))
    return config


def _loader(config: Path, fetcher: AuditFetcher) -> DataLoader:
    settings = load_data_settings(config)
    return DataLoader(
        settings.dates,
        settings.data,
        fetcher_factory=lambda: fetcher,
        now=lambda: NOW,
        actions_factory=lambda: fetcher,
    )


def _audit(config: Path, fetcher: AuditFetcher) -> corporate_actions.AuditResult:
    return audit(config, _loader(config, fetcher), "SPY")


def _found(result: corporate_actions.AuditResult) -> list[tuple[str, dt.date, str]]:
    return [(c.ticker, c.date, c.source) for c in result.candidates]


def _feed(*rows: FeedSpinOff, splits: Sequence[Split] = ()) -> Actions:
    return Actions(spinoffs=rows, splits=tuple(splits))


def _row(source: str, child: str, day: dt.date = DAY, rate: float = 0.5) -> FeedSpinOff:
    return FeedSpinOff(source=source, child=child, ex_date=day, source_rate=1.0, new_rate=rate)


# --------------------------------------------------------------------------- sources


def test_each_source_finds_its_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path, fixed=("SPY", "AAA", "BBB", "CCC"), members=("NEW,2026-02-02,",))
    fetcher = AuditFetcher()
    fetcher.feed = _feed(_row("AAA", "KID"))
    fetcher.adjusted["BBB"] = (DAY, 0.8)  # Alpaca applied 0.8: a step
    fetcher.gap["CCC"] = (DAY, 0.7)  # opens 30% lower, raw and adjusted
    fetcher.listed["NEW"] = dt.date(2026, 1, 28)  # joins 3 sessions after its first bar
    result = _audit(config, fetcher)
    assert _found(result) == [
        ("AAA", DAY, "feed"),
        ("BBB", DAY, "step"),
        ("CCC", DAY, "gap"),
        ("NEW", dt.date(2026, 2, 2), "addition"),
    ]
    assert result.open == tuple((t, d) for t, d, _ in _found(result))
    assert "0.8000" in result.candidates[1].detail
    assert "-30.0%" in result.candidates[2].detail


def test_what_is_not_a_candidate(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        fixed=("SPY", "AAA", "BBB", "CCC", "DDD"),
        members=("OLD,2025-10-01,", "LATE,2026-03-02,"),
    )
    fetcher = AuditFetcher()
    fetcher.feed = _feed(
        _row("AAA", "AAA.WI"),  # when-issued
        _row("AAA", "AAAW"),  # a warrant
        _row("AAA", "123CVR456"),  # a CVR, by CUSIP
        _row("AAA", "AAA"),  # the parent itself
        _row("ZZZ", "KID"),  # not in any universe
        splits=[Split(symbol="BBB", ex_date=DAY, ratio=2.0)],
    )
    fetcher.adjusted["BBB"] = (DAY, 0.5)  # a 2-for-1 split in the split feed
    fetcher.adjusted["CCC"] = (DAY, 0.98)  # a 2% step: a dividend
    fetcher.gap["DDD"] = (DAY, 0.81)  # a 19% gap
    fetcher.listed["LATE"] = dt.date(2026, 1, 2)  # joins 40 sessions after listing
    assert _audit(config, fetcher).candidates == ()


def test_a_spinoff_filed_as_a_split_stays_a_candidate(tmp_path: Path) -> None:
    # Alpaca's split feed holds old spin-offs as "splits" of the factor (DHR 2016 is
    # 1.319) or as reverse splits of the distribution ratio (EQT 2018 is 4 for 5). Only a
    # simple forward split (here 3 for 2) excuses a step.
    config = _config(tmp_path, fixed=("SPY", "FAC", "REV", "OK"))
    fetcher = AuditFetcher()
    fetcher.feed = _feed(
        splits=[
            Split(symbol="FAC", ex_date=DAY, ratio=1.319),
            Split(symbol="REV", ex_date=DAY, ratio=0.8),
            Split(symbol="OK", ex_date=DAY, ratio=1.5),
        ]
    )
    fetcher.adjusted["FAC"] = (DAY, 1 / 1.319)
    fetcher.adjusted["REV"] = (DAY, 1.25)
    fetcher.adjusted["OK"] = (DAY, 1 / 1.5)
    result = _audit(config, fetcher)
    assert _found(result) == [("FAC", DAY, "step"), ("REV", DAY, "step")]
    assert "the split feed lists 1.319" in result.candidates[0].detail


def test_a_split_excuses_only_the_step_it_explains(tmp_path: Path) -> None:
    # Both have a 2-for-1 split in the feed. WRONG was adjusted by 1/4, and BOTH by the
    # split and a spin-off together: neither step is the split's 0.5.
    config = _config(tmp_path, fixed=("SPY", "WRONG", "BOTH", "OK"))
    fetcher = AuditFetcher()
    fetcher.feed = _feed(
        splits=[Split(symbol=s, ex_date=DAY, ratio=2.0) for s in ("WRONG", "BOTH", "OK")]
    )
    fetcher.adjusted["WRONG"] = (DAY, 0.25)
    fetcher.adjusted["BOTH"] = (DAY, 0.5 * 0.8)
    fetcher.adjusted["OK"] = (DAY, 0.5 * 0.99)  # the split and a 1% dividend
    assert _found(_audit(config, fetcher)) == [("BOTH", DAY, "step"), ("WRONG", DAY, "step")]


def test_only_bars_a_run_can_read_are_audited(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        members=("GONE,2025-10-01,2026-01-14", "BAD,2025-10-01,", "BOTH,2025-10-01,2026-01-02"),
        bad_bars=("BAD,2026-01-01,2026-02-01,another security",),
        fixed=("SPY", "BOTH"),
    )
    fetcher = AuditFetcher()
    for symbol in ("GONE", "BAD", "BOTH"):
        fetcher.gap[symbol] = (DAY, 0.5)
    # GONE left the index on DAY, BAD's bars are not the member's then; BOTH is also in
    # a fixed list, so all of its bars count.
    assert _found(_audit(config, fetcher)) == [("BOTH", DAY, "gap")]


def test_a_feed_row_under_an_old_ticker_is_the_new_ones(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        members=("OLD,2025-10-01,2025-12-01", "NEW,2025-12-01,"),
        renames=("OLD,NEW,2025-12-01,renamed",),
    )
    fetcher = AuditFetcher()
    fetcher.feed = _feed(_row("OLD", "KID"))
    assert _found(_audit(config, fetcher)) == [("NEW", DAY, "feed")]


# --------------------------------------------------------------------------- coverage


def test_a_candidate_is_covered_by_a_listed_event_or_a_reviewed_row(tmp_path: Path) -> None:
    row = spinoff_row("AAA", DAY, cached_step=0.6, child="KID")
    config = _config(
        tmp_path,
        fixed=("SPY", "AAA", "BBB", "CCC"),
        members=("KID,2026-01-14,",),
        spinoffs=(row,),
        reviewed=(f"BBB,{DAY},earnings",),
    )
    fetcher = AuditFetcher()
    fetcher.gap["AAA"] = (DAY, 0.6)
    fetcher.gap["BBB"] = (DAY, 0.7)
    fetcher.gap["CCC"] = (DAY, 0.7)
    fetcher.listed["KID"] = DAY  # the child joins the index on its first day
    result = _audit(config, fetcher)
    assert set(result.listed) == {("AAA", DAY), ("KID", DAY)}
    assert result.reviewed == (("BBB", DAY),)
    assert result.open == (("CCC", DAY),)


def test_an_addition_is_covered_only_by_a_child_near_its_ex_date(tmp_path: Path) -> None:
    far = dt.date(2025, 11, 3)
    row = spinoff_row("AAA", far, cached_step=0.6, child="KID")
    config = _config(tmp_path, members=("KID,2026-01-14,",), spinoffs=(row,))
    fetcher = AuditFetcher()
    fetcher.listed["KID"] = DAY
    assert _audit(config, fetcher).open == (("KID", DAY),)


def test_unused_reviewed_rows_are_reported_but_do_not_fail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, reviewed=(f"AAA,{DAY},earnings",))
    fetcher = AuditFetcher()
    code = corporate_actions.main(["audit"], config, loader=_loader(config, fetcher))
    assert code == 0
    assert f"reviewed row AAA {DAY} matches no candidate" in capsys.readouterr().out


def test_the_cli_fails_on_open_candidates_and_writes_the_worksheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, fixed=("SPY", "AAA"))
    fetcher = AuditFetcher()
    fetcher.gap["AAA"] = (DAY, 0.7)
    fetcher.adjusted["AAA"] = (DAY, 0.9)
    sheet = tmp_path / "open.csv"
    code = corporate_actions.main(
        ["audit", "--out", str(sheet)], config, loader=_loader(config, fetcher)
    )
    assert code == 1
    assert f"{DAY} AAA" in capsys.readouterr().err
    lines = sheet.read_text().splitlines()
    assert lines[0] == "ticker,date,sources,detail"
    assert lines[1].startswith(f"AAA,{DAY},gap;step,")
    # Reviewed, it passes.
    (config / REVIEWED_FILE).write_text(f"ticker,date,note\nAAA,{DAY},earnings\n")
    assert corporate_actions.main(["audit"], config, loader=_loader(config, fetcher)) == 0


# --------------------------------------------------------------------------- holdout


def test_the_audit_stops_at_the_holdout_start(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fetcher = AuditFetcher()
    result = _audit(config, fetcher)
    assert (result.start, result.end) == (START, END)
    assert max(fetcher.ends) == END
    loader = _loader(config, fetcher)
    with pytest.raises(HoldoutViolation):
        loader.actions(START, END + dt.timedelta(days=1))


def test_the_audit_has_no_holdout_route(tmp_path: Path) -> None:
    # The audit is not a holdout route (David, 2026-11-25): it takes no holdout access,
    # and the CLI has no flag for one.
    assert "holdout" not in inspect.signature(audit).parameters
    config = _config(tmp_path)
    loader = _loader(config, AuditFetcher())
    with pytest.raises(SystemExit):
        corporate_actions.main(["audit", "--holdout"], config, loader=loader)
    assert corporate_actions.main(["audit"], config, loader=loader) == 0


def test_the_loader_rejects_feed_rows_outside_the_range(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class Leaky(AuditFetcher):
        def actions(self, start: dt.date, end: dt.date) -> Actions:
            return _feed(_row("AAA", "KID", day=end + dt.timedelta(days=1)))

    leaky = Leaky()
    with pytest.raises(DataError, match="outside"):
        _loader(config, leaky).actions(START, END)


# --------------------------------------------------------------------------- Alpaca


class _Stub:
    """alpaca-py's corporate action models, built without validation."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[Any] = []

    def get_corporate_actions(self, request: object) -> object:
        self.requests.append(request)
        return self.response


def _alpaca(
    response: object, today: dt.date = dt.date(2026, 7, 1)
) -> tuple[AlpacaActionsFetcher, _Stub]:
    stub = _Stub(response)
    fetcher = object.__new__(AlpacaActionsFetcher)  # no keys: the client is a stub
    fetcher._limiter = RateLimiter(1000)  # pyright: ignore[reportPrivateUsage]
    fetcher._today = lambda: today  # pyright: ignore[reportPrivateUsage]
    fetcher._client = stub  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return fetcher, stub


def _response(*items: object) -> object:
    class Response:
        data = {"all": list(items)}

    return Response()


def test_alpaca_rows_are_kept_by_ex_date_and_deduplicated() -> None:
    from alpaca.data.models.corporate_actions import ForwardSplit, SpinOff

    spin = SpinOff.model_construct(
        source_symbol="AAA", new_symbol="KID", source_rate=1.0, new_rate=0.5, ex_date=DAY
    )
    late = SpinOff.model_construct(
        source_symbol="AAA", new_symbol="KID", source_rate=1.0, new_rate=0.5,
        ex_date=dt.date(2026, 4, 1),
    )  # fmt: skip
    split = ForwardSplit.model_construct(symbol="BBB", new_rate=4.0, old_rate=1.0, ex_date=DAY)
    fetcher, stub = _alpaca(_response(spin, late, split))
    result = fetcher.actions(START, END)
    # Each year's request returns the same rows; each is kept once, the holdout one not.
    assert result.spinoffs == (_row("AAA", "KID"),)
    assert result.splits == (Split(symbol="BBB", ex_date=DAY, ratio=4.0),)
    # Requests run from a year before the start through today, by process date.
    first, last = stub.requests[0], stub.requests[-1]
    assert first.start == START - dt.timedelta(days=366)
    assert last.end == dt.date(2026, 7, 1)


def test_a_split_with_no_rates_fails() -> None:
    from alpaca.data.models.corporate_actions import ReverseSplit

    bad = ReverseSplit.model_construct(symbol="BBB", new_rate=1.0, old_rate=0.0, ex_date=DAY)
    fetcher, _ = _alpaca(_response(bad))
    with pytest.raises(DataError, match="BBB: Alpaca lists a split"):
        fetcher.actions(START, END)


# --------------------------------------------------------------------------- shipped files


def test_every_shipped_universe_loads() -> None:
    names = universe_names(REPO_CONFIG)
    assert "sp500" in names
    for name in names:
        assert load_universe(name, REPO_CONFIG).symbols
