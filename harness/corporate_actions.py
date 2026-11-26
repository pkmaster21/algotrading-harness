"""Spin-off build: `python -m harness.corporate_actions build [--holdout]`.

Fills `config/corporate-actions/spinoffs.csv`, the file every run reads, from the
reviewed list of events in `spinoff-events.csv` (docs/spinoff-design.md, decisions 2 to
4, 6 and 8). For each event it fetches the parent's adjusted and raw bars on the ex-date
and the session before, each child's raw ex-date close, and any cash dividend of the
parent going ex that day, then resolves:

- `factor` f = P / (P + sum r_i * C_i), from the raw ex-date closes (decision 3);
- `dividend_factor` d = 1 - D / (the parent's raw close the session before), D the cash
  per share going ex the same day (1 when there is none);
- `applied_factor` a, what Alpaca applied: the adjusted-to-raw ratio of the close the
  session before over the same ratio on the ex-date;
- `multiplier` m = f * d / a (decision 4) and `cached_step`, the adjusted ex-date close
  over the adjusted close the session before (the guard, decision 6).

The build writes all or nothing. It exits non-zero, naming every event at fault and
leaving spinoffs.csv untouched, when an event cannot be measured (no parent bar on the
ex-date or the session before) unless that event is listed with a note in
`spinoffs-reviewed.csv`; when a child has no ex-date bar and no close given by hand;
or when an event is dated in the holdout and has not been built with `--holdout`.

Run it by hand when the list changes or Alpaca's adjustment of an event changes (a
guard fired). It is not part of any run: runs read the resolved file only, and the
thresholds and lists here never enter a recipe.

`python -m harness.corporate_actions audit [--out FILE]` finds the events
that belong in the list (decision 8). Over every bar a run can read of every symbol in
every universe file and the benchmark, it lists candidates from four Alpaca sources:

- every spin-off in Alpaca's feed, after dropping warrants, rights, CVRs and
  when-issued tickers;
- every step of more than 3% in the adjusted-to-raw close ratio that is not a split in
  Alpaca's split feed;
- every open gap down of 20% or more in the adjusted bars;
- every addition to a point-in-time universe within 5 sessions of the ticker's first
  bar (spin-offs often join the index on day one).

It exits non-zero until each candidate (ticker, date) is a parent's ex-date in
spinoffs.csv (an addition: a child's, within 5 sessions), or a row in
spinoffs-reviewed.csv with a note.

The fifth source is SEC Form 10-12B filings (`harness.sec`), which find spin-offs the
other four miss when the price moves little. They are read through today, not only to
the holdout start (decision 8): an index says which companies filed, never a price, so
it is not holdout data. Each company's first filing since a year before the data start
(a Form 10 comes months before the distribution) is a candidate until it is a row of
`form10-reviewed.csv`, keyed by CIK, with one of three outcomes:

- `listed`: the spin-off of `parent` going ex on `ex_date`, which must be a parent's
  ex-date in spinoffs.csv or a row of spinoffs-reviewed.csv;
- `none`: no event for these universes (the parent is in none of them, the filing was
  withdrawn, the ex-date is before the data start); the note says which;
- `pending`: not yet distributed when reviewed, or going ex on or after the holdout
  start and not yet built with `build --holdout`. Printed on every audit as a reminder,
  and it does not fail the audit.

The SEC asks for a name and contact email in the User-Agent: the audit reads it from
`SEC_USER_AGENT` in `.env` and refuses to run without one.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, ValidationError, model_validator

from harness.config import (
    DEFAULT_CONFIG_DIR,
    SPINOFFS_FILE,
    SPINOFFS_HEADER,
    BadBars,
    ConfigError,
    SpinOffEvent,
    load_data_settings,
    load_universe,
    read_spinoffs,
    spinoff_fields,
    universe_names,
)
from harness.data import (
    PROJECT_ROOT,
    Actions,
    DataError,
    DataLoader,
    FeedSpinOff,
    HoldoutAccess,
    HoldoutLogError,
    HoldoutViolation,
    Window,
)
from harness.keys import KeysError, read_sec_user_agent
from harness.models import FrozenModel, Stock
from harness.run import _git_commit
from harness.sec import Filing, Form10Index, SecError, first_filings

EVENTS_FILE = Path("corporate-actions") / "spinoff-events.csv"
REVIEWED_FILE = Path("corporate-actions") / "spinoffs-reviewed.csv"
FORM10_REVIEWED_FILE = Path("corporate-actions") / "form10-reviewed.csv"
_FORM10_HEADER = ["cik", "company", "filed", "outcome", "parent", "ex_date", "note"]
_EVENTS_HEADER = ["parent", "ex_date", "children", "hand_closes", "note"]
_REVIEWED_HEADER = ["ticker", "date", "note"]
# Calendar days fetched before the ex-date to find the session before it. The longest
# market closure since 2015 is a three-day weekend plus one day; this leaves room. It
# finds the session before, never an earlier one (that is not widened, by decision).
_LOOKBACK = dt.timedelta(days=10)
# A significant figure count for resolved values: far below any data rounding, and the
# loader's consistency check (1e-6) cannot tell the rounded values apart.
_DIGITS = 12


class BuildError(RuntimeError):
    """The build found events it could not resolve, and wrote nothing."""


class ListedEvent(FrozenModel):
    """One row of spinoff-events.csv: an event a person reviewed and listed. Children are
    tickers with shares per parent share. `hand_closes` gives a child's raw ex-date close
    when Alpaca has no bar for it (acquired later, OTC only); the note gives the source."""

    parent: str
    ex_date: dt.date
    children: tuple[tuple[str, float], ...] = Field(min_length=1)
    hand_closes: tuple[tuple[str, float], ...] = ()
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ListedEvent:
        Stock(symbol=self.parent)
        tickers = [t for t, _ in self.children]
        for t in tickers:
            Stock(symbol=t)
        if len(set(tickers)) != len(tickers) or self.parent in tickers:
            raise ValueError("children must be distinct tickers, none of them the parent")
        if any(not r > 0 for _, r in self.children):
            raise ValueError("every child ratio must be positive")
        hand = [t for t, _ in self.hand_closes]
        if len(set(hand)) != len(hand) or not set(hand) <= set(tickers):
            raise ValueError("hand_closes must name distinct children of this event")
        if any(not c > 0 for _, c in self.hand_closes):
            raise ValueError("every hand close must be positive")
        return self


class Reviewed(FrozenModel):
    """One row of spinoffs-reviewed.csv: a (ticker, date) a person looked at and left out
    of spinoffs.csv, with the reason."""

    ticker: str
    date: dt.date
    note: str = Field(min_length=1)


Outcome = Literal["listed", "none", "pending"]


class Form10Reviewed(FrozenModel):
    """One row of form10-reviewed.csv: a company's Form 10-12B, by CIK, reviewed to the
    spin-off it registered (`listed`), to no event for these universes (`none`), or to
    one not yet distributed or not yet built (`pending`). `company` and `filed` are for
    the reader; the CIK is the key."""

    cik: int = Field(gt=0)
    company: str = Field(min_length=1)
    filed: dt.date
    outcome: Outcome
    parent: str | None = None
    ex_date: dt.date | None = None
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Form10Reviewed:
        if self.outcome == "listed":
            if self.parent is None or self.ex_date is None:
                raise ValueError("a listed filing names its parent and ex_date")
            Stock(symbol=self.parent)
        elif self.parent is not None or self.ex_date is not None:
            raise ValueError(f"a {self.outcome} filing leaves parent and ex_date empty")
        return self


class Problem(FrozenModel):
    """Why one listed event could not be resolved. `unmeasurable` problems (no parent bar
    on the ex-date or the session before) are excused by a reviewed row; others never."""

    parent: str
    ex_date: dt.date
    reason: str
    unmeasurable: bool = False


class BuildResult(FrozenModel):
    built: tuple[SpinOffEvent, ...]  # measured now
    carried: tuple[SpinOffEvent, ...]  # dated in the holdout, kept from the current file
    skipped: tuple[tuple[Problem, Reviewed], ...]  # unmeasurable, excused by a reviewed row
    removed: tuple[SpinOffEvent, ...]  # in the current file, no longer listed


# --------------------------------------------------------------------------- files


def _read_rows(path: Path, header: list[str]) -> list[tuple[int, dict[str, str]]]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] != header:
        raise ConfigError(f"{path}: the header must be {','.join(header)}")
    out: list[tuple[int, dict[str, str]]] = []
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ConfigError(f"{path} line {n}: expected {len(header)} fields, got {len(row)}")
        out.append((n, dict(zip(header, row, strict=True))))
    return out


def _number(text: str) -> float:
    """A decimal or a fraction such as `1/3`, so a ratio is written as announced."""
    return float(Fraction(text)) if "/" in text else float(text)


def _pairs(text: str) -> list[tuple[str, float]]:
    if not text:
        return []
    pairs: list[tuple[str, float]] = []
    for item in text.split(";"):
        key, sep, val = item.partition(":")
        if not sep or not key or not val:
            raise ValueError(f"{text!r} is not TICKER:VALUE pairs separated by ';'")
        pairs.append((key, _number(val)))
    return pairs


def load_events(config_dir: Path = DEFAULT_CONFIG_DIR) -> tuple[ListedEvent, ...]:
    """The rows of spinoff-events.csv, sorted by (parent, ex_date), one per pair."""
    path = config_dir / EVENTS_FILE
    events: list[ListedEvent] = []
    for n, fields in _read_rows(path, _EVENTS_HEADER):
        try:
            events.append(
                ListedEvent.model_validate(
                    {
                        "parent": fields["parent"],
                        "ex_date": dt.date.fromisoformat(fields["ex_date"]),
                        "children": _pairs(fields["children"]),
                        "hand_closes": _pairs(fields["hand_closes"]),
                        "note": fields["note"],
                    }
                )
            )
        except (ValueError, ZeroDivisionError) as exc:  # ValidationError is a ValueError
            raise ConfigError(f"{path} line {n}: {exc}") from exc
    keys = [(e.parent, e.ex_date) for e in events]
    if dupes := sorted({k for k in keys if keys.count(k) > 1}):
        raise ConfigError(f"{path}: more than one row for {dupes}; list every child in one row")
    return tuple(sorted(events, key=lambda e: (e.parent, e.ex_date)))


def load_reviewed(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[tuple[str, dt.date], Reviewed]:
    """The rows of spinoffs-reviewed.csv by (ticker, date)."""
    path = config_dir / REVIEWED_FILE
    out: dict[tuple[str, dt.date], Reviewed] = {}
    for n, fields in _read_rows(path, _REVIEWED_HEADER):
        try:
            row = Reviewed.model_validate(
                {
                    "ticker": fields["ticker"],
                    "date": dt.date.fromisoformat(fields["date"]),
                    "note": fields["note"].strip(),
                }
            )
        except ValueError as exc:
            raise ConfigError(f"{path} line {n}: {exc}") from exc
        if (row.ticker, row.date) in out:
            raise ConfigError(f"{path} line {n}: {row.ticker} {row.date} is listed twice")
        out[(row.ticker, row.date)] = row
    return out


def load_form10_reviewed(config_dir: Path = DEFAULT_CONFIG_DIR) -> dict[int, Form10Reviewed]:
    """The rows of form10-reviewed.csv by CIK."""
    path = config_dir / FORM10_REVIEWED_FILE
    out: dict[int, Form10Reviewed] = {}
    for n, fields in _read_rows(path, _FORM10_HEADER):
        try:
            row = Form10Reviewed.model_validate(
                {
                    "cik": int(fields["cik"]),
                    "company": fields["company"].strip(),
                    "filed": dt.date.fromisoformat(fields["filed"]),
                    "outcome": fields["outcome"],
                    "parent": fields["parent"] or None,
                    "ex_date": dt.date.fromisoformat(fields["ex_date"])
                    if fields["ex_date"]
                    else None,
                    "note": fields["note"].strip(),
                }
            )
        except ValueError as exc:
            raise ConfigError(f"{path} line {n}: {exc}") from exc
        if row.cik in out:
            raise ConfigError(f"{path} line {n}: CIK {row.cik} is listed twice")
        out[row.cik] = row
    return out


def write_spinoffs(path: Path, events: Sequence[SpinOffEvent]) -> None:
    """Replace `path` with `events`, sorted by (parent, ex_date). Written to a temporary
    file, read back as a run would read it, then moved into place, so a failure leaves
    the old file as it was."""
    rows = sorted(events, key=lambda e: (e.parent, e.ex_date))
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(SPINOFFS_HEADER)
            writer.writerows(spinoff_fields(e) for e in rows)
            fh.flush()
            os.fsync(fh.fileno())
        if read_spinoffs(Path(tmp)) != tuple(rows):
            raise BuildError(f"{path}: the written rows do not read back as built")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- measuring


def _g(x: float) -> float:
    return float(f"{x:.{_DIGITS}g}")


def _closes(frames: dict[str, pd.DataFrame], symbol: str) -> dict[dt.date, float]:
    frame = frames.get(symbol)
    if frame is None or frame.empty:
        return {}
    return {d: float(c) for d, c in zip(frame["date"], frame["close"], strict=True) if c > 0}


def measure(event: ListedEvent, window: Window, calendar: str) -> SpinOffEvent | Problem:
    """The resolved row for `event` from `window`, or why it cannot be resolved.
    `calendar` is a symbol that trades every session (the benchmark): its bars say which
    sessions there were."""

    def problem(reason: str, *, unmeasurable: bool = False) -> Problem:
        return Problem(
            parent=event.parent, ex_date=event.ex_date, reason=reason, unmeasurable=unmeasurable
        )

    ex = event.ex_date
    sessions = sorted(_closes(window.raw, calendar))
    if ex not in sessions:
        return problem(f"{ex} is not a session (no {calendar} bar): check the ex-date")
    earlier = [d for d in sessions if d < ex]
    if not earlier:
        return problem(f"no session found in the {_LOOKBACK.days} days before the ex-date")
    before = earlier[-1]
    raw, adj = _closes(window.raw, event.parent), _closes(window.adjusted, event.parent)
    for day, label in ((before, f"the session before ({before})"), (ex, "the ex-date")):
        if day not in raw or day not in adj:
            return problem(f"no {event.parent} bar on {label}", unmeasurable=True)
    hand = dict(event.hand_closes)
    child_closes: list[float] = []
    for ticker, _ in event.children:
        close = hand.get(ticker) or _closes(window.raw, ticker).get(ex)
        if close is None:
            return problem(
                f"no {ticker} bar on {ex}: give its raw close in hand_closes, with the "
                "source in the note"
            )
        child_closes.append(close)
    parent_close = raw[ex]
    handed_out = sum(r * c for (_, r), c in zip(event.children, child_closes, strict=True))
    factor = _g(parent_close / (parent_close + handed_out))
    cash = sum(rate for d, rate in window.dividends.get(event.parent, []) if d == ex)
    if cash >= raw[before]:
        return problem(f"a cash dividend of {cash} is not below the close before it")
    dividend_factor = _g(1 - cash / raw[before])
    applied_factor = _g((adj[before] / raw[before]) / (adj[ex] / raw[ex]))
    try:
        return SpinOffEvent(
            parent=event.parent,
            ex_date=ex,
            children=event.children,
            parent_close=parent_close,
            child_closes=tuple(child_closes),
            factor=factor,
            dividend_factor=dividend_factor,
            applied_factor=applied_factor,
            multiplier=_g(factor * dividend_factor / applied_factor),
            cached_step=_g(adj[ex] / adj[before]),
            note=event.note,
        )
    except ValidationError as exc:
        return problem(f"the resolved row is invalid: {exc}")


def _carried(event: ListedEvent, current: SpinOffEvent | None) -> SpinOffEvent | Problem:
    """The current row of a holdout-dated event, kept without fetching, if it was built
    from the same children and hand closes."""
    problem = Problem(
        parent=event.parent,
        ex_date=event.ex_date,
        reason="dated in the holdout and not built (or changed since it was built): "
        "run build --holdout",
    )
    if current is None or current.children != event.children:
        return problem
    closes = dict(zip((t for t, _ in current.children), current.child_closes, strict=True))
    if any(not math.isclose(closes[t], c, rel_tol=1e-12) for t, c in event.hand_closes):
        return problem
    return current.model_copy(update={"note": event.note})


def build(
    config_dir: Path,
    loader: DataLoader,
    calendar: str,
    *,
    holdout: HoldoutAccess | None = None,
) -> BuildResult:
    """Resolve every listed event and replace spinoffs.csv, or raise BuildError naming
    every event at fault and write nothing.

    An event dated on or after the holdout start is measured only with `holdout`, and
    each such window is logged in the holdout log. Without it, the event's current row
    is kept if it was built from the same listing; otherwise it is at fault.
    """
    events = load_events(config_dir)
    reviewed = load_reviewed(config_dir)
    path = config_dir / SPINOFFS_FILE
    current = {(e.parent, e.ex_date): e for e in read_spinoffs(path)}
    holdout_start = loader.dates.holdout_start
    built: list[SpinOffEvent] = []
    carried: list[SpinOffEvent] = []
    skipped: list[tuple[Problem, Reviewed]] = []
    faults: list[Problem] = []
    for event in events:
        key = (event.parent, event.ex_date)
        if event.ex_date >= holdout_start and holdout is None:
            result = _carried(event, current.get(key))
            if isinstance(result, Problem):
                faults.append(result)
            else:
                carried.append(result)
            continue
        access = holdout if event.ex_date >= holdout_start else None
        if event.ex_date >= loader.cutoff(access):  # past a pinned holdout_end
            faults.append(
                Problem(
                    parent=event.parent,
                    ex_date=event.ex_date,
                    reason=f"after the last loadable day ({loader.cutoff(access)} is the cutoff)",
                )
            )
            continue
        try:
            window = loader.window(
                [calendar, event.parent, *(t for t, _ in event.children)],
                event.ex_date - _LOOKBACK,
                event.ex_date,
                dividends=[event.parent],
                holdout=access,
            )
        except DataError as exc:  # a failed request, or a day not yet completed
            faults.append(Problem(parent=event.parent, ex_date=event.ex_date, reason=str(exc)))
            continue
        result = measure(event, window, calendar)
        if isinstance(result, SpinOffEvent):
            built.append(result)
        elif result.unmeasurable and key in reviewed:
            skipped.append((result, reviewed[key]))
        else:
            faults.append(result)
    if faults:
        lines = [f"  {p.parent} {p.ex_date}: {p.reason}" for p in faults]
        hint = (
            "\nAn event with no parent bar can be listed in "
            f"{REVIEWED_FILE} with a note, and is then left out."
            if any(p.unmeasurable for p in faults)
            else ""
        )
        raise BuildError(
            f"{len(faults)} event(s) could not be resolved; {path} is unchanged:\n"
            + "\n".join(lines)
            + hint
        )
    listed = {(e.parent, e.ex_date) for e in events}
    removed = [e for k, e in current.items() if k not in listed]
    write_spinoffs(path, [*built, *carried])
    return BuildResult(
        built=tuple(built), carried=tuple(carried), skipped=tuple(skipped), removed=tuple(removed)
    )


# --------------------------------------------------------------------------- audit

# What a person reviews, not what a run does: these never enter a recipe (decision 8).
STEP_THRESHOLD = 0.03  # |a - 1| above this, a not a split: a real change of adjustment
GAP_THRESHOLD = 0.20  # an open this far below the close before it
ADDITION_SESSIONS = 5  # a universe addition this close to the ticker's first bar
FILING_LEAD = dt.timedelta(days=365)  # Form 10s filed this long before the data start
# Child tickers in the feed that are not shares of a new company.
_SPLIT_TOLERANCE = 5e-4  # relative: a split feed ratio this close to a simple one is a split
_NOT_SHARES = (".WI", ".WS", ".WT", ".RT", ".U", "WI", "CVR")

Source = Literal["feed", "step", "gap", "addition"]


class Candidate(FrozenModel):
    """One lead for a person to review: `ticker` (as the universe files name it) may
    have a spin-off going ex on `date`."""

    ticker: str
    date: dt.date
    source: Source
    detail: str


class AuditResult(FrozenModel):
    start: dt.date
    end: dt.date
    candidates: tuple[Candidate, ...]
    listed: tuple[tuple[str, dt.date], ...]  # covered by a row of spinoffs.csv
    reviewed: tuple[tuple[str, dt.date], ...]  # covered by a row of spinoffs-reviewed.csv
    open: tuple[tuple[str, dt.date], ...]  # neither: the audit fails
    unused: tuple[Reviewed, ...]  # reviewed rows that match no candidate and no listed event
    filed_from: dt.date  # Form 10 filings from this day through `filed_to`
    filed_to: dt.date
    filings: tuple[Filing, ...]  # each company's first 10-12B in that range
    filings_open: tuple[Filing, ...]  # no row in form10-reviewed.csv: the audit fails
    # Listed rows whose (parent, ex_date) is neither in spinoffs.csv nor reviewed: fails.
    filings_unmatched: tuple[Form10Reviewed, ...]
    filings_pending: tuple[Form10Reviewed, ...]  # printed every run, does not fail
    filings_unused: tuple[Form10Reviewed, ...]  # rows that match no filing in the range

    @property
    def failed(self) -> bool:
        return bool(self.open or self.filings_open or self.filings_unmatched)

    def by_key(self) -> dict[tuple[str, dt.date], list[Candidate]]:
        out: dict[tuple[str, dt.date], list[Candidate]] = {}
        for c in self.candidates:
            out.setdefault((c.ticker, c.date), []).append(c)
        return out


@dataclass
class _Scope:
    """Which bars a run can read: a fixed-list symbol's (and the benchmark's) from the
    data start; a point-in-time member's from the data start to its last day in the
    index, outside its bad-bar windows. Bars after a member leaves are never read."""

    start: dt.date
    end: dt.date  # inclusive
    fixed: set[str] = field(default_factory=set[str])
    last_day: dict[str, dt.date | None] = field(default_factory=dict[str, dt.date | None])
    bad: dict[str, list[BadBars]] = field(default_factory=dict[str, list[BadBars]])
    aliases: dict[str, str] = field(default_factory=dict[str, str])  # old ticker -> new
    additions: list[tuple[str, str, dt.date]] = field(
        default_factory=list[tuple[str, str, dt.date]]
    )  # (universe, ticker, first session in)

    @property
    def symbols(self) -> list[str]:
        return sorted(self.fixed | set(self.last_day))

    def covers(self, symbol: str, day: dt.date) -> bool:
        if not self.start <= day <= self.end:
            return False
        if symbol in self.fixed:
            return True
        if symbol not in self.last_day:
            return False
        last = self.last_day[symbol]
        if last is not None and day >= last:  # `end` of an interval is the first day out
            return False
        return not any(
            (b.start is None or b.start <= day) and (b.end is None or day < b.end)
            for b in self.bad.get(symbol, [])
        )


def _scope(config_dir: Path, start: dt.date, end: dt.date, benchmark: str) -> _Scope:
    scope = _Scope(start=start, end=end, fixed={benchmark})
    for name in universe_names(config_dir):
        universe = load_universe(name, config_dir)
        if universe.membership is None:
            scope.fixed.update(universe.symbols)
            continue
        m = universe.membership
        for iv in m.intervals:
            scope.additions.append((name, iv.symbol, iv.start))
            if iv.symbol in scope.last_day and scope.last_day[iv.symbol] is None:
                continue  # already a member through today
            prev = scope.last_day.get(iv.symbol)
            scope.last_day[iv.symbol] = None if iv.end is None else max(iv.end, prev or iv.end)
        for b in m.bad_bars:
            scope.bad.setdefault(b.symbol, []).append(b)
        scope.aliases.update({r.old: r.new for r in m.renames})
    return scope


def _is_share(row: FeedSpinOff) -> bool:
    """A feed row whose child looks like shares of another company (decision 8)."""
    source, child = row.source, row.child
    if child == source or any(ch.isdigit() for ch in child):
        return False  # a CUSIP (CVRs, contingent rights) or the parent itself
    if any(child.endswith(suffix) for suffix in _NOT_SHARES):
        return False
    # A Nasdaq warrant or right: the parent's ticker plus W, R, Z or L.
    return not (child[:-1] == source and child[-1] in "WRZL")


def _is_split(ratio: float) -> bool:
    """A split feed ratio that excuses an adjustment step: a forward split of n for 1, or
    p for q with both at most 10 (3 for 2). Alpaca's split feed also holds spin-offs
    before 2021, as "splits" of the factor (DHR 2016-07-05 is 1.319, HPE 2017-04-03 is
    1.3348) or as reverse splits of the distribution ratio (WRK 2016-05-16 is 1 for 6,
    EQT 2018-11-13 is 4 for 5), so those stay candidates. Real reverse splits are rare
    in these universes, and a reviewed row settles each."""
    if ratio < 1:
        return False
    if abs(ratio - round(ratio)) <= _SPLIT_TOLERANCE * ratio:
        return True
    return any(
        abs(ratio - p / q) <= _SPLIT_TOLERANCE * ratio for p in range(1, 11) for q in range(1, 11)
    )


def candidates(
    scope: _Scope,
    actions: Actions,
    adjusted: dict[str, pd.DataFrame],
    raw: dict[str, pd.DataFrame],
    calendar: str,
) -> list[Candidate]:  # fmt: skip
    """Every lead from the four Alpaca sources, inside `scope`, sorted by (date, ticker,
    source). Pure: the frames and feed rows are given."""
    out: list[Candidate] = []
    known = set(scope.symbols)

    def ticker(symbol: str) -> str | None:
        if symbol in known:
            return symbol
        new = scope.aliases.get(symbol)
        return new if new in known else None

    for row in actions.spinoffs:
        t = ticker(row.source)
        if t is None or not _is_share(row) or not scope.covers(t, row.ex_date):
            continue
        ratio = row.new_rate / row.source_rate if row.source_rate > 0 else row.new_rate
        out.append(
            Candidate(
                ticker=t,
                date=row.ex_date,
                source="feed",
                detail=f"feed: {row.source} -> {row.child} {ratio:.6g} per share",
            )
        )
    splits: dict[tuple[str, dt.date], float] = {}
    for split in actions.splits:
        if (t := ticker(split.symbol)) is not None:
            splits[(t, split.ex_date)] = split.ratio

    bench = adjusted.get(calendar)
    bench_gap: dict[dt.date, float] = {}
    if bench is not None and not bench.empty:
        gaps = bench["open"] / bench["close"].shift(1) - 1
        bench_gap = {d: float(g) for d, g in zip(bench["date"], gaps, strict=True) if g == g}
    sessions = sorted(raw[calendar]["date"]) if calendar in raw else []

    for symbol in scope.symbols:
        adj = adjusted.get(symbol)
        if adj is None or adj.empty:
            continue
        dates = list(adj["date"])
        gaps = adj["open"] / adj["close"].shift(1) - 1
        moves = adj["close"] / adj["close"].shift(1) - 1
        for i in range(1, len(dates)):
            day, gap = dates[i], float(gaps.iloc[i])
            if (
                gap <= -GAP_THRESHOLD
                and scope.covers(symbol, day)
                and scope.covers(symbol, dates[i - 1])
            ):
                spy = bench_gap.get(day)
                market = f", {calendar} {spy:+.1%}" if spy is not None else ""
                out.append(
                    Candidate(
                        ticker=symbol,
                        date=day,
                        source="gap",
                        detail=f"gap: open {gap:+.1%} (close {float(moves.iloc[i]):+.1%}"
                        f"{market}) after {dates[i - 1]}",
                    )
                )
        rw = raw.get(symbol)
        if rw is None or rw.empty:
            continue
        both = adj[["date", "close"]].merge(rw[["date", "close"]], on="date", suffixes=("", "_raw"))
        both = both[(both["close"] > 0) & (both["close_raw"] > 0)].reset_index(drop=True)
        ratio = both["close"] / both["close_raw"]
        steps = ratio.shift(1) / ratio
        for i in range(1, len(both)):
            day, step = both["date"].iloc[i], float(steps.iloc[i])
            split = splits.get((symbol, day))
            if abs(step - 1) <= STEP_THRESHOLD:
                continue
            # A split excuses only the step it explains: a split Alpaca applied wrongly,
            # or a split and a spin-off on one day, leaves a step of another size.
            if split is not None and _is_split(split) and abs(step * split - 1) <= STEP_THRESHOLD:
                continue
            if scope.covers(symbol, day):
                listed = "" if split is None else f" (the split feed lists {split:.6g})"
                out.append(
                    Candidate(
                        ticker=symbol,
                        date=day,
                        source="step",
                        detail=f"step: Alpaca applied {step:.4f} after "
                        f"{both['date'].iloc[i - 1]}{listed}",
                    )
                )

    for universe, symbol, joined in scope.additions:
        rw = raw.get(symbol)
        if rw is None or rw.empty or not sessions or joined < sessions[0]:
            continue
        if not scope.start <= joined <= scope.end:
            continue
        first = min(rw["date"])
        if first <= sessions[0]:
            continue  # listed before the data begins: its first bar says nothing
        apart = abs(bisect.bisect_left(sessions, joined) - bisect.bisect_left(sessions, first))
        if apart <= ADDITION_SESSIONS:
            out.append(
                Candidate(
                    ticker=symbol,
                    date=joined,
                    source="addition",
                    detail=f"addition: joined {universe} {joined}, first bar {first}",
                )
            )
    return sorted(out, key=lambda c: (c.date, c.ticker, c.source))


def audit(
    config_dir: Path,
    loader: DataLoader,
    calendar: str,
    form10: Form10Index,
) -> AuditResult:
    """List every candidate in the data range, through the day before the holdout start,
    and sort them into covered by spinoffs.csv, covered by spinoffs-reviewed.csv, and
    open. The audit never reads the holdout's bars or feed rows: it is not a holdout
    route (decided by David, 2026-11-25; PR 4 may add one). Form 10 filings are read
    through today (decision 8), from `form10`."""
    start = loader.dates.data_start
    end = loader.cutoff(None) - dt.timedelta(days=1)
    scope = _scope(config_dir, start, end, calendar)
    events = read_spinoffs(config_dir / SPINOFFS_FILE)
    listed_events = {(e.parent, e.ex_date) for e in load_events(config_dir)}
    reviewed = load_reviewed(config_dir)
    actions = loader.actions(start, end)
    window = loader.window([calendar, *scope.symbols], start, end)
    found = candidates(scope, actions, window.adjusted, window.raw, calendar)

    sessions = sorted(window.raw[calendar]["date"]) if calendar in window.raw else []
    parents = {(e.parent, e.ex_date) for e in events}
    children: dict[str, list[dt.date]] = {}
    for e in events:
        for t, _ in e.children:
            children.setdefault(t, []).append(e.ex_date)

    def near(a: dt.date, b: dt.date) -> bool:
        i, j = bisect.bisect_left(sessions, a), bisect.bisect_left(sessions, b)
        return abs(i - j) <= ADDITION_SESSIONS

    keys: dict[tuple[str, dt.date], list[Candidate]] = {}
    for c in found:
        keys.setdefault((c.ticker, c.date), []).append(c)
    listed, acknowledged, open_ = [], [], []
    for key, group in keys.items():
        ticker, day = key
        as_child = all(c.source == "addition" for c in group) and any(
            near(day, ex) for ex in children.get(ticker, [])
        )
        if key in parents or as_child:
            listed.append(key)
        elif key in reviewed:
            acknowledged.append(key)
        else:
            open_.append(key)

    filed_from = start - FILING_LEAD
    filed_to = form10.today()
    filings = first_filings(form10.filings(filed_from, filed_to))
    form10_rows = load_form10_reviewed(config_dir)
    known = parents | set(reviewed)
    ciks = {f.cik for f in filings}
    filings_open = [f for f in filings if f.cik not in form10_rows]
    matched = [r for r in form10_rows.values() if r.cik in ciks]
    unmatched = [r for r in matched if r.outcome == "listed" and (r.parent, r.ex_date) not in known]
    pending = [r for r in matched if r.outcome == "pending"]
    form10_unused = [r for r in form10_rows.values() if r.cik not in ciks]
    # A reviewed row is used when a candidate, a listed event or a listed filing names it.
    named = {(r.parent, r.ex_date) for r in matched if r.outcome == "listed"}
    used = keys.keys() | listed_events | named
    unused = [r for k, r in reviewed.items() if k not in used]
    return AuditResult(
        start=start,
        end=end,
        candidates=tuple(found),
        listed=tuple(sorted(listed, key=lambda k: (k[1], k[0]))),
        reviewed=tuple(sorted(acknowledged, key=lambda k: (k[1], k[0]))),
        open=tuple(sorted(open_, key=lambda k: (k[1], k[0]))),
        unused=tuple(sorted(unused, key=lambda r: (r.date, r.ticker))),
        filed_from=filed_from,
        filed_to=filed_to,
        filings=filings,
        filings_open=tuple(filings_open),
        filings_unmatched=tuple(sorted(unmatched, key=lambda r: (r.filed, r.cik))),
        filings_pending=tuple(sorted(pending, key=lambda r: (r.filed, r.cik))),
        filings_unused=tuple(sorted(form10_unused, key=lambda r: (r.filed, r.cik))),
    )


_WORKSHEET_HEADER = ["ticker", "date", "sources", "detail"]


def write_worksheet(path: Path, result: AuditResult) -> None:
    """The open candidates as a CSV, one row per (ticker, date), for the review."""
    groups = result.by_key()
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_WORKSHEET_HEADER)
        for key in result.open:
            group = groups[key]
            writer.writerow(
                [
                    key[0],
                    key[1].isoformat(),
                    ";".join(dict.fromkeys(c.source for c in group)),
                    "; ".join(c.detail for c in group),
                ]
            )


def write_form10_worksheet(path: Path, result: AuditResult) -> None:
    """The open filings as rows of form10-reviewed.csv with the outcome left to fill."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_FORM10_HEADER)
        for f in result.filings_open:
            writer.writerow([f.cik, f.company, f.filed.isoformat(), "", "", "", f.form])


# --------------------------------------------------------------------------- CLI


def _report(result: BuildResult, previous: dict[tuple[str, dt.date], SpinOffEvent]) -> None:
    print(f"{'event':<20} {'factor':>8} {'div':>8} {'applied':>8} {'mult':>8} {'step':>8}")
    for e in result.built:
        old = previous.get((e.parent, e.ex_date))
        if old is None:
            flag = "new"
        elif old.applied() == e.applied():
            flag = ""
        else:
            flag = f"CHANGED (was mult {old.multiplier:.4f}, step {old.cached_step:.4f})"
        print(
            f"{e.parent + ' ' + str(e.ex_date):<20} {e.factor:8.4f} {e.dividend_factor:8.4f} "
            f"{e.applied_factor:8.4f} {e.multiplier:8.4f} {e.cached_step:8.4f}  {flag}".rstrip()
        )
    for e in result.carried:
        print(f"{e.parent} {e.ex_date}: dated in the holdout, kept as built")
    for p, r in result.skipped:
        print(f"{p.parent} {p.ex_date}: left out ({p.reason}); reviewed: {r.note}")
    for e in result.removed:
        print(f"{e.parent} {e.ex_date}: removed (no longer listed)")


def _report_audit(result: AuditResult) -> int:
    counts = {src: 0 for src in ("feed", "step", "gap", "addition")}
    for c in result.candidates:
        counts[c.source] += 1
    pairs = len(result.listed) + len(result.reviewed) + len(result.open)
    print(
        f"audit {result.start}..{result.end}: {len(result.candidates)} candidate(s) "
        f"({', '.join(f'{k} {v}' for k, v in counts.items())}) on {pairs} (ticker, date) pair(s)"
    )
    print(
        f"  in spinoffs.csv: {len(result.listed)}; reviewed: {len(result.reviewed)}; "
        f"open: {len(result.open)}"
    )
    for r in result.unused:
        print(f"  note: reviewed row {r.ticker} {r.date} matches no candidate")
    reviewed_ciks = {f.cik for f in result.filings} - {f.cik for f in result.filings_open}
    pending = {r.cik for r in result.filings_pending}
    print(
        f"form 10 filings {result.filed_from}..{result.filed_to}: {len(result.filings)} "
        f"compan(ies); reviewed: {len(reviewed_ciks)} ({len(pending)} pending); "
        f"open: {len(result.filings_open)}"
    )
    for r in result.filings_pending:
        print(f"  pending: {r.company} (CIK {r.cik}, filed {r.filed}): {r.note}")
    for r in result.filings_unused:
        print(f"  note: form 10 row {r.company} (CIK {r.cik}) matches no filing in the range")
    if not result.failed:
        return 0
    groups = result.by_key()
    if result.open:
        print(
            f"open: list each in {EVENTS_FILE} and build, or add it to {REVIEWED_FILE} with a note",
            file=sys.stderr,
        )
    for ticker, day in result.open:
        details = "; ".join(c.detail for c in groups[(ticker, day)])
        print(f"  {day} {ticker:<6} {details}", file=sys.stderr)
    if result.filings_open:
        print(f"open form 10 filings: add each to {FORM10_REVIEWED_FILE}", file=sys.stderr)
    for f in result.filings_open:
        print(f"  {f.filed} CIK {f.cik:<8} {f.company} ({f.form})", file=sys.stderr)
    for r in result.filings_unmatched:
        print(
            f"  {FORM10_REVIEWED_FILE}: {r.company} (CIK {r.cik}) names {r.parent} "
            f"{r.ex_date}, which is neither in {SPINOFFS_FILE} nor in {REVIEWED_FILE}",
            file=sys.stderr,
        )
    return 1


def main(
    argv: Sequence[str] | None = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    *,
    loader: DataLoader | None = None,  # tests pass one with fake fetchers
    form10: Form10Index | None = None,  # tests pass one with a fake opener
    dotenv_path: Path = PROJECT_ROOT / ".env",
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.corporate_actions",
        description="Build spinoffs.csv, or audit which events belong in it.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build", help="resolve the listed spin-offs into spinoffs.csv")
    build_cmd.add_argument(
        "--holdout",
        action="store_true",
        help="also measure events dated in the holdout (each logged in the holdout log)",
    )
    audit_cmd = commands.add_parser(
        "audit", help="list candidate spin-offs; fail until each is listed or reviewed"
    )
    audit_cmd.add_argument(
        "--out", type=Path, help="write the open candidates to this CSV, for the review"
    )
    audit_cmd.add_argument(
        "--form10-out",
        type=Path,
        help="write the open Form 10 filings to this CSV, as rows of form10-reviewed.csv",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_data_settings(config_dir)
        loader = loader or DataLoader(settings.dates, settings.data)
        holdout = None
        if args.command == "build" and args.holdout:
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            holdout = HoldoutAccess(
                run_id=f"spinoff-{args.command}-{stamp}",
                strategy=f"spinoff-{args.command}",
                git_commit=_git_commit(config_dir.resolve().parent),
            )
        if args.command == "audit":
            form10 = form10 or Form10Index(read_sec_user_agent(dotenv_path))
            found = audit(config_dir, loader, settings.benchmark, form10)
            if args.out is not None:
                write_worksheet(args.out, found)
            if args.form10_out is not None:
                write_form10_worksheet(args.form10_out, found)
            return _report_audit(found)
        previous = {(e.parent, e.ex_date): e for e in read_spinoffs(config_dir / SPINOFFS_FILE)}
        result = build(config_dir, loader, settings.benchmark, holdout=holdout)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (
        ConfigError,
        DataError,
        HoldoutLogError,
        HoldoutViolation,
        KeysError,
        SecError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _report(result, previous)
    print(f"wrote {config_dir / SPINOFFS_FILE}: {len(result.built) + len(result.carried)} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
