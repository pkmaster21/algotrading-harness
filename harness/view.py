"""The look-ahead barrier: what a strategy is allowed to see.

The engine owns a `BarBuffer` and writes exactly one session into it at a time. A
strategy receives a `MarketView`, a read-only window fixed to the sessions written so far.
Future sessions are not hidden from the view; they are absent from the buffer (still NaN),
because the engine has not copied them in yet. So an oversized lookback, a bad index, or
even reaching into private attributes cannot produce a future bar. Everything the view
returns is a copy, so mutating it cannot change what later sessions see.

This stops buggy code, not deliberate introspection of the interpreter (for example walking
the call stack to the engine). See ARCHITECTURE.md, decision log.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from harness.models import Bar

PriceField = Literal["open", "high", "low", "close", "volume"]
PRICE_FIELDS: tuple[PriceField, ...] = ("open", "high", "low", "close", "volume")


class BarBuffer:
    """Engine-owned, append-only bar storage backing every MarketView.

    Arrays of shape (capacity, n_symbols) per field start as NaN. `append` copies in one
    session's row; rows not yet appended stay NaN. Only the engine calls `append`.
    """

    __slots__ = ("_symbols", "_index", "_data", "_dates", "_stamps", "_length")

    def __init__(self, symbols: Sequence[str], capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._symbols = tuple(symbols)
        self._index = {s: i for i, s in enumerate(self._symbols)}
        n = len(self._symbols)
        self._data = {f: np.full((capacity, n), np.nan) for f in PRICE_FIELDS}
        for arr in self._data.values():
            arr.flags.writeable = False
        self._dates: list[dt.date] = []
        self._stamps = np.full(capacity, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    def append(self, session: dt.date, row: Mapping[str, np.ndarray]) -> None:
        """Copy in the bars for `session`, which must be later than the last appended one."""
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
        self._dates.append(session)
        self._stamps[i] = np.datetime64(session, "ns")
        self._length += 1

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
