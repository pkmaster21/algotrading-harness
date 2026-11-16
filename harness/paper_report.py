"""Paper trading's `trades.csv` and `summary.json`, built from the journal alone.

Same schema as a backtest's files (`harness/report.py`), so whatever reads one reads the
other. Two things are added, both about how live fills compare with the price a backtest
assumes (the official open):

- `trades.csv` gains six columns, three per side of the trade: the official open of the
  session the side filled on, seconds after the 09:30 open, and slippage against that
  open in basis points.
- `summary.json` gains a `paper` block with the same measures over all entry fills and
  all exit fills.

**What it covers.** The paper runner decides on session `t` the morning after, and that
is when the bar of `t` first exists. So the report made on a morning covers sessions
through `t`: the equity points of every `decision` record, and the fills made on or
before `t`. The fills of the morning itself appear in the next day's report.

**Where the official opens come from.** When it decides, the runner asks the data source
for the unadjusted open of each session that has fills (`DataLoader.raw_opens`) and
writes them into the `decision` record (`fill_opens`), with the benchmark's close of
`t` (`benchmark_close`). The report reads them from there, never from the bar cache.
The cached bars are adjusted for splits and dividends, and an adjustment reaches back
from its ex-date: the bar of `t`, read the next morning, has already moved if something
goes ex that day. A fill's price is what actually traded, so it is compared with the
open as it printed. Journaling the opens also means a rerun, or any other process,
rebuilds exactly the same files without asking for data.

Slippage is signed so that a positive number is a cost: a buy above the open, a sell
below it. Seconds after the open are what the venue's timestamp says, negative
included (an auction fill can be stamped just before 09:30:00); nothing is clamped. A
fill with no journaled open (the request failed, or the symbol had no bar) is counted
but not priced, a trade side with such a fill has its three columns empty, and the
next decision asks again.

Not modeled: operator adjustments that change a share count (a split, a delisting).
The ledger carries them; a trade list built from fills cannot, so the report refuses
(`ReportError`) and the runner leaves the previous files in place. Holding days count
sessions the runner decided on: a missed session between entry and exit is not
counted, and a trade whose entry or exit fill landed on a session that was never
decided on (the runner missed the next day) has no holding days at all. The benchmark
is the change in its journaled closes (no whole shares, no costs), not a simulated
buy-and-hold as in a backtest, and it is reported only if every session of the equity
curve has one: a benchmark over part of the run next to a return over all of it would
mislead (`paper.benchmark_missing` says how many are missing). `events` counts what the
decisions logged (signals ignored, risk shrinks), not what happened to orders at the
open. If a session has two `decision` records, the first is the one used, as for the
opens.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from harness.config import MARKET_OPEN, Recipe
from harness.engine import BacktestResult, EquityPoint
from harness.journal import JournalRecord
from harness.models import MARKET_TZ, EngineEvent, Fill
from harness.report import TRADE_COLUMNS, ReportError, summarize, trades_from_fills

_SIDE_COLUMNS = ("open", "seconds_after_open", "slippage_bps")
PAPER_TRADE_COLUMNS = [
    *TRADE_COLUMNS,
    *(f"entry_{c}" for c in _SIDE_COLUMNS),
    *(f"exit_{c}" for c in _SIDE_COLUMNS),
]

type OpenKey = tuple[dt.date, str]  # (session, symbol)


def unpriced_fills(records: Sequence[JournalRecord], through: dt.date) -> list[OpenKey]:
    """The (session, symbol) of every fill on or before `through` whose official open no
    `decision` record holds yet, in journal order, each once. What the runner looks up
    in the bars before it journals the next decision."""
    try:
        known = _journaled_opens(records)
        wanted: dict[OpenKey, None] = {}
        for fill in _fills(records, through):
            key = (fill.fill_date, fill.instrument.key)
            if key not in known:
                wanted[key] = None
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ReportError(f"the journal does not read as paper trading records: {exc!r}") from exc
    return list(wanted)


def paper_report(
    records: Sequence[JournalRecord],
    recipe: Recipe,
    *,
    profile_id: str,
    strategy: str,
    through: dt.date,
    marks: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows of `trades.csv` (`PAPER_TRADE_COLUMNS`) and the content of
    `summary.json`, for sessions through `through` (the last session decided on).

    `marks` are the closes of `through` for the symbols held then. Raises ReportError
    if the journal cannot be turned into trades; nothing is partly built.
    """
    try:
        return _report(records, recipe, profile_id, strategy, through, marks)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:  # a record of another shape
        raise ReportError(f"the journal does not read as paper trading records: {exc!r}") from exc


def _report(
    records: Sequence[JournalRecord],
    recipe: Recipe,
    profile_id: str,
    strategy: str,
    through: dt.date,
    marks: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for record in records:
        if record.kind == "adjustment" and record.payload.get("quantity_delta"):
            raise ReportError(
                f"adjustment {record.payload.get('adjustment_id')} changed a share count; "
                "trades built from fills do not model that yet"
            )
    points: dict[dt.date, EquityPoint] = {}
    benchmark: dict[dt.date, float] = {}
    events: list[EngineEvent] = []
    fingerprint = ""
    for record in records:
        if record.kind != "decision":
            continue
        point = EquityPoint.model_validate(record.payload["equity"])
        if point.date > through or point.date in points:  # the first decision of a session
            continue
        events += [EngineEvent.model_validate(e) for e in _list(record.payload["events"])]
        points[point.date] = point
        fingerprint = str(record.payload["data_fingerprint"])
        close = record.payload.get("benchmark_close")
        if isinstance(close, int | float) and close > 0:
            benchmark[point.date] = float(close)
    if not points:
        raise ReportError(f"the journal holds no decision on or before {through}")
    equity = tuple(points[d] for d in sorted(points))

    opens = _journaled_opens(records)
    fills = _fills(records, through)
    held: dict[str, int] = {}
    for fill in fills:
        sign = 1 if fill.side == "buy" else -1
        held[fill.instrument.key] = held.get(fill.instrument.key, 0) + sign * fill.quantity

    def versus_open(buys: Sequence[Fill], sells: Sequence[Fill]) -> dict[str, Any]:
        columns: dict[str, Any] = {}
        for prefix, side in (("entry", buys), ("exit", sells)):
            measured = _versus_open(side, opens)
            complete = bool(side) and measured["priced"] == len(side)
            for column in _SIDE_COLUMNS:
                columns[f"{prefix}_{column}"] = measured[column] if complete else None
        return columns

    trades = trades_from_fills(
        fills,
        [p.date for p in equity],
        {s: q for s, q in held.items() if q},
        marks,
        through,
        extra=versus_open,
    )
    starting_cash = recipe.backtest.starting_cash
    # The shapes `summarize` reads. It is given the trades, so the fills (which a
    # backtest carries as order updates) are not repeated here.
    result = BacktestResult(
        strategy=strategy,
        start=equity[0].date,
        end=through,
        warmup_sessions=0,
        starting_cash=starting_cash,
        equity=equity,
        orders={},
        updates=(),
        events=tuple(events),
        final_positions={},
        final_marks={s: marks[s] for s, q in held.items() if q},
        data_fingerprint=fingerprint,
    )
    # All of the curve or none of it: a benchmark over fewer sessions than the return
    # beside it is not a comparison.
    missing = len(points) - len(benchmark)
    days = sorted(benchmark) if not missing else []
    benchmark_result = result.model_copy(
        update={
            "strategy": "benchmark",
            "starting_cash": benchmark[days[0]] if days else starting_cash,
            "equity": tuple(
                EquityPoint(
                    date=d, cash=0.0, market_value=benchmark[d], equity=benchmark[d], n_positions=1
                )
                for d in days
            ),
            "events": (),
            "final_marks": {},
        }
    )
    summary = summarize(result, benchmark_result, recipe, holdout=False, trades=trades)
    summary["paper"] = {
        "profile_id": profile_id,
        "benchmark_missing": (
            f"no benchmark close on record for {missing} of {len(points)} sessions"
            if missing
            else None
        ),
        "entries": _side_summary([f for f in fills if f.side == "buy"], opens),
        "exits": _side_summary([f for f in fills if f.side == "sell"], opens),
    }
    return trades, summary


def _list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"expected a list, found {type(value).__name__}")
    return value


def _fills(records: Sequence[JournalRecord], through: dt.date) -> list[Fill]:
    fills = [Fill.model_validate(r.payload["fill"]) for r in records if r.kind == "fill"]
    return [f for f in fills if f.fill_date <= through]


def _journaled_opens(records: Sequence[JournalRecord]) -> dict[OpenKey, float]:
    """The official opens the decisions recorded. The first record of one wins: it is
    the open as it stood the morning after, before any later re-adjustment."""
    opens: dict[OpenKey, float] = {}
    for record in records:
        if record.kind != "decision":
            continue
        by_session = record.payload.get("fill_opens", {})
        if not isinstance(by_session, dict):
            raise TypeError("fill_opens is not a mapping")
        for session, by_symbol in by_session.items():
            if not isinstance(by_symbol, dict):
                raise TypeError("fill_opens is not a mapping of mappings")
            for symbol, value in by_symbol.items():
                if isinstance(value, int | float) and value > 0:
                    opens.setdefault((dt.date.fromisoformat(session), symbol), float(value))
    return opens


def _versus_open(fills: Sequence[Fill], opens: Mapping[OpenKey, float]) -> dict[str, Any]:
    """`fills` against the official opens of their sessions, over the fills that have
    one (`priced` of them). Averages are weighted by shares: `slippage_bps` is the
    average fill price against the average open, positive when the fills were worse
    (bought above, sold below), and `slippage_cost` is the same difference in dollars."""
    priced = 0
    units = at_open = cost = seconds = 0.0
    for fill in fills:
        official = opens.get((fill.fill_date, fill.instrument.key))
        if official is None or fill.filled_at is None:
            continue
        opened = dt.datetime.combine(fill.fill_date, MARKET_OPEN, MARKET_TZ)
        size = fill.quantity * fill.instrument.multiplier
        sign = 1.0 if fill.side == "buy" else -1.0
        priced += 1
        units += size
        at_open += official * size
        cost += sign * (fill.price - official) * size
        seconds += (fill.filled_at - opened).total_seconds() * size
    if not priced:
        return {
            "priced": 0,
            "open": None,
            "seconds_after_open": None,
            "slippage_bps": None,
            "slippage_cost": None,
        }
    return {
        "priced": priced,
        "open": at_open / units,
        "seconds_after_open": seconds / units,
        "slippage_bps": cost / at_open * 10_000,
        "slippage_cost": cost,
    }


def _side_summary(fills: Sequence[Fill], opens: Mapping[OpenKey, float]) -> dict[str, Any]:
    measured = _versus_open(fills, opens)
    return {
        "fills": len(fills),
        "priced": measured["priced"],
        "slippage_bps": measured["slippage_bps"],
        "slippage_cost": measured["slippage_cost"],
        "avg_seconds_after_open": measured["seconds_after_open"],
    }
