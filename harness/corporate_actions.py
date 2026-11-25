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

The audit (which events belong in the list) is not here yet: until it lands, coverage
is only the events listed by hand.

Run it by hand when the list changes or Alpaca's adjustment of an event changes (a
guard fired). It is not part of any run: runs read the resolved file only, and the
thresholds and lists here never enter a recipe.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import pandas as pd
from pydantic import Field, ValidationError, model_validator

from harness.config import (
    DEFAULT_CONFIG_DIR,
    SPINOFFS_FILE,
    SPINOFFS_HEADER,
    ConfigError,
    SpinOffEvent,
    load_data_settings,
    read_spinoffs,
    spinoff_fields,
)
from harness.data import (
    DataError,
    DataLoader,
    HoldoutAccess,
    HoldoutLogError,
    HoldoutViolation,
    Window,
)
from harness.models import FrozenModel, Stock
from harness.run import _git_commit

EVENTS_FILE = Path("corporate-actions") / "spinoff-events.csv"
REVIEWED_FILE = Path("corporate-actions") / "spinoffs-reviewed.csv"
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


def main(
    argv: Sequence[str] | None = None,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    *,
    loader: DataLoader | None = None,  # tests pass one with fake fetchers
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.corporate_actions", description="Build spinoffs.csv."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build_cmd = commands.add_parser("build", help="resolve the listed spin-offs into spinoffs.csv")
    build_cmd.add_argument(
        "--holdout",
        action="store_true",
        help="also measure events dated in the holdout (each logged in the holdout log)",
    )
    args = parser.parse_args(argv)
    try:
        settings = load_data_settings(config_dir)
        loader = loader or DataLoader(settings.dates, settings.data)
        holdout = None
        if args.holdout:
            stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            holdout = HoldoutAccess(
                run_id=f"spinoff-build-{stamp}",
                strategy="spinoff-build",
                git_commit=_git_commit(config_dir.resolve().parent),
            )
        previous = {(e.parent, e.ex_date): e for e in read_spinoffs(config_dir / SPINOFFS_FILE)}
        result = build(config_dir, loader, settings.benchmark, holdout=holdout)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ConfigError, DataError, HoldoutLogError, HoldoutViolation) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _report(result, previous)
    print(f"wrote {config_dir / SPINOFFS_FILE}: {len(result.built) + len(result.carried)} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
