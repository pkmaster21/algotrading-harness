"""The look-ahead barrier: what a strategy is allowed to see.

The engine owns a `BarBuffer` and writes exactly one session into it at a time. A
strategy receives a `MarketView`, a read-only window fixed to the sessions written so far.
Future sessions are not hidden from the view; they are absent from the buffer (still NaN),
because the engine has not copied them in yet. So an oversized lookback, a bad index, or
even reaching into private attributes cannot produce a future bar. Everything the view
returns is a copy, so mutating it cannot change what later sessions see.

This stops buggy code, not deliberate introspection of the interpreter (for example walking
the call stack to the engine). See ARCHITECTURE.md, decision log.

A point-in-time universe adds a `MemberMask`: which symbols are index members on each
session. The engine uses it to say which symbols a strategy may enter (`members()`), to
force exits on removal, and to hide a symbol's bars from the strategy once it has left
for good, so a ticker reused by another security is never seen
(docs/pit-universe-design.md).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from harness.config import MembershipInterval
from harness.models import Bar

PriceField = Literal["open", "high", "low", "close", "volume"]
PRICE_FIELDS: tuple[PriceField, ...] = ("open", "high", "low", "close", "volume")


class MemberMask:
    """Index membership on a fixed list of sessions, for a fixed list of symbols.

    Built once per run from the membership intervals (start inclusive, end exclusive) and
    the run's session dates. For each session it answers three questions, each as a
    boolean row in symbol order:

    - `members`: in the index that session.
    - `leaving`: a member that session but not the next one in `dates`, so its last
      member session is today. False on the last date, whose next session is unknown.
    - `hidden`: on or after the end of the symbol's last interval. The strategy's view
      shows no bars for it from then on. Bars before a symbol first joined, and between
      two of its intervals, stay visible (that history was public).

    A symbol with no interval is never a member and never hidden.
    """

    __slots__ = ("_symbols", "_pos", "_members", "_leaving", "_hidden")

    def __init__(
        self,
        symbols: Sequence[str],
        dates: Sequence[dt.date],
        intervals: Sequence[MembershipInterval],
    ) -> None:
        self._symbols = tuple(symbols)
        self._pos = {d: i for i, d in enumerate(dates)}
        if len(self._pos) != len(dates):
            raise ValueError("member mask dates must be unique")
        column = {s: j for j, s in enumerate(self._symbols)}
        stamps = np.array(dates, dtype="datetime64[D]")
        shape = (len(dates), len(self._symbols))
        members = np.zeros(shape, dtype=bool)
        hidden = np.zeros(shape, dtype=bool)
        last_end: dict[str, dt.date] = {}
        still_member: set[str] = set()
        for iv in intervals:
            j = column.get(iv.symbol)
            if j is None:
                continue
            inside = stamps >= np.datetime64(iv.start, "D")
            if iv.end is None:
                still_member.add(iv.symbol)
            else:
                inside &= stamps < np.datetime64(iv.end, "D")
                last_end[iv.symbol] = max(iv.end, last_end.get(iv.symbol, iv.end))
            members[:, j] |= inside
        for symbol in sorted(last_end.keys() - still_member):
            hidden[:, column[symbol]] = stamps >= np.datetime64(last_end[symbol], "D")
        leaving = np.zeros(shape, dtype=bool)
        leaving[:-1] = members[:-1] & ~members[1:]
        for arr in (members, leaving, hidden):
            arr.flags.writeable = False
        self._members, self._leaving, self._hidden = members, leaving, hidden

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def _row(self, session: dt.date) -> int:
        try:
            return self._pos[session]
        except KeyError:
            raise KeyError(f"session {session} is not in the member mask") from None

    def members(self, session: dt.date) -> np.ndarray:
        return self._members[self._row(session)]

    def leaving(self, session: dt.date) -> np.ndarray:
        return self._leaving[self._row(session)]

    def hidden(self, session: dt.date) -> np.ndarray:
        return self._hidden[self._row(session)]


class BarBuffer:
    """Engine-owned, append-only bar storage backing every MarketView.

    Arrays of shape (capacity, n_symbols) per field start as NaN. `append` copies in one
    session's row; rows not yet appended stay NaN. Only the engine calls `append`. Each
    session also records which symbols a strategy may enter that day (`enterable`; all of
    them unless the engine passes a row).
    """

    __slots__ = ("_symbols", "_index", "_data", "_enterable", "_dates", "_stamps", "_length")

    def __init__(self, symbols: Sequence[str], capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._symbols = tuple(symbols)
        self._index = {s: i for i, s in enumerate(self._symbols)}
        n = len(self._symbols)
        self._data = {f: np.full((capacity, n), np.nan) for f in PRICE_FIELDS}
        for arr in self._data.values():
            arr.flags.writeable = False
        self._enterable = np.zeros((capacity, n), dtype=bool)
        self._enterable.flags.writeable = False
        self._dates: list[dt.date] = []
        self._stamps = np.full(capacity, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def append(
        self,
        session: dt.date,
        row: Mapping[str, np.ndarray],
        enterable: np.ndarray | None = None,
    ) -> None:
        """Copy in the bars for `session`, which must be later than the last appended one,
        and which symbols may be entered that session (default: all)."""
        if self._length == len(self._stamps):
            raise IndexError("BarBuffer is full")
        if self._dates and session <= self._dates[-1]:
            raise ValueError(f"session {session} is not after {self._dates[-1]}")
        i = self._length
        for f in PRICE_FIELDS:
            values = np.asarray(row[f], dtype=np.float64)
            if values.shape != (len(self._symbols),):
                raise ValueError(f"row field {f} has shape {values.shape}")
            arr = self._data[f]
            arr.flags.writeable = True
            arr[i] = values
            arr.flags.writeable = False
        allowed = np.ones(len(self._symbols), dtype=bool) if enterable is None else enterable
        if allowed.shape != (len(self._symbols),) or allowed.dtype != np.bool_:
            raise ValueError(f"enterable row has shape {allowed.shape}, dtype {allowed.dtype}")
        self._enterable.flags.writeable = True
        self._enterable[i] = allowed
        self._enterable.flags.writeable = False
        self._dates.append(session)
        self._stamps[i] = np.datetime64(session, "ns")
        self._length += 1

    def trailing_missing(self, symbol: str) -> int:
        """How many of the most recent sessions in a row have no bar for `symbol`.

        0 when the latest session has a bar; `length` when the symbol never had one.
        """
        try:
            j = self._index[symbol]
        except KeyError:
            raise KeyError(f"symbol {symbol!r} is not in the universe") from None
        close = self._data["close"]
        n = 0
        while n < self._length and np.isnan(close[self._length - 1 - n, j]):
            n += 1
        return n

    def view(self) -> MarketView:
        """A view of every session appended so far."""
        if self._length == 0:
            raise ValueError("no sessions appended yet")
        return MarketView(self, self._length)


class MarketView:
    """A strategy's read-only window onto bars up to and including `today`.

    Fixed at creation: a view kept from an earlier session keeps showing that session as
    today. All methods return copies.
    """

    __slots__ = ("_buffer", "_n")

    def __init__(self, buffer: BarBuffer, n: int) -> None:
        if not 1 <= n <= buffer.length:
            raise ValueError(f"view length {n} outside 1..{buffer.length}")
        self._buffer = buffer
        self._n = n

    @property
    def today(self) -> dt.date:
        return self._buffer._dates[self._n - 1]

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._buffer.symbols

    @property
    def n_bars(self) -> int:
        """Number of sessions visible, including today."""
        return self._n

    def _col(self, symbol: str) -> int:
        try:
            return self._buffer._index[symbol]
        except KeyError:
            raise KeyError(f"symbol {symbol!r} is not in the universe") from None

    def members(self) -> tuple[str, ...]:
        """The symbols a strategy may enter today, in universe order: every symbol for a
        fixed universe; for a point-in-time universe, today's index members minus any
        whose last member session is today (an entry would fill after it left).

        Entries for other symbols are dropped by the engine (`entry_skipped_not_member`).
        Rank among these.
        """
        row = self._buffer._enterable[self._n - 1]
        return tuple(s for s, ok in zip(self._buffer.symbols, row, strict=True) if ok)

    def bar(self, symbol: str) -> Bar | None:
        """Today's bar for `symbol`, or None if it has no bar today."""
        j = self._col(symbol)
        i = self._n - 1
        values = {f: float(self._buffer._data[f][i, j]) for f in PRICE_FIELDS}
        if np.isnan(values["close"]):
            return None
        return Bar(symbol=symbol, date=self.today, **values)

    def history(
        self, field: PriceField, lookback: int, symbols: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """The last `lookback` sessions (ending today) of one field.

        Rows are dates (oldest first), columns are symbols; NaN means no bar. Returns fewer
        rows than `lookback` early in the data, never rows after today.
        """
        if field not in PRICE_FIELDS:
            raise ValueError(f"unknown field {field!r}; use one of {PRICE_FIELDS}")
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError("lookback must be a positive int")
        cols = list(self.symbols if symbols is None else symbols)
        idx = [self._col(s) for s in cols]
        start = max(0, self._n - lookback)
        values = self._buffer._data[field][start : self._n][:, idx].copy()
        index = pd.DatetimeIndex(self._buffer._stamps[start : self._n].copy(), name="date")
        return pd.DataFrame(values, index=index, columns=cols)

    def series(self, symbol: str, field: PriceField, lookback: int) -> pd.Series:
        """`history` for one symbol, as a Series."""
        return self.history(field, lookback, [symbol]).iloc[:, 0].rename(symbol)
