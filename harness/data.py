"""Market data for the harness.

`BarPanel` is the bulk bar container the engine runs on: many symbols on one calendar,
stored as arrays for speed. The Alpaca loader, the Parquet cache, and the holdout cutoff
are added in slice 1 PR 3; the loader is the only code that decides which dates a panel
contains, so the engine never checks dates itself.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from harness.models import Stock
from harness.view import PRICE_FIELDS, PriceField

_PRICE_TOLERANCE = 1e-6  # same relative slack as models.Bar, for vendor rounding


class PanelError(ValueError):
    """Bar data is malformed (partial bars, bad prices, unsorted dates...)."""


def _read_only(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True, order="C")
    out.flags.writeable = False
    return out


class BarPanel:
    """Immutable, validated daily bars for many symbols on one calendar.

    The calendar is the union of all symbols' session dates. Each field is a read-only
    float64 array of shape (n_dates, n_symbols). A bar is either fully present or fully
    absent: NaN in every field means the symbol has no bar that day (listed later,
    delisted, or halted). Present bars have positive prices, non-negative volume, and
    open and close within [low, high].
    """

    __slots__ = ("_dates", "_symbols", "_index", "_fields")

    def __init__(
        self,
        dates: Sequence[dt.date],
        symbols: Sequence[str],
        fields: Mapping[PriceField, np.ndarray],
    ) -> None:
        self._dates = tuple(dates)
        self._symbols = tuple(symbols)
        if not self._dates:
            raise PanelError("panel has no dates")
        if any(b <= a for a, b in zip(self._dates, self._dates[1:], strict=False)):
            raise PanelError("panel dates must be strictly increasing")
        if len(set(self._symbols)) != len(self._symbols):
            raise PanelError("panel symbols must be unique")
        for s in self._symbols:
            Stock(symbol=s)
        if set(fields) != set(PRICE_FIELDS):
            raise PanelError(f"panel needs exactly the fields {PRICE_FIELDS}")
        shape = (len(self._dates), len(self._symbols))
        self._fields = {f: _read_only(fields[f]) for f in PRICE_FIELDS}
        for f, arr in self._fields.items():
            if arr.shape != shape:
                raise PanelError(f"field {f} has shape {arr.shape}, expected {shape}")
        self._index = {s: i for i, s in enumerate(self._symbols)}
        self._validate()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> BarPanel:
        """Build a panel from one DataFrame per symbol.

        Each frame is indexed by date (or has a `date` column) and has the columns
        open, high, low, close, volume. A frame may be empty (no bars in range).
        """
        normalized: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            df = frame.set_index("date") if "date" in frame.columns else frame
            missing = [f for f in PRICE_FIELDS if f not in df.columns]
            if missing:
                raise PanelError(f"{symbol}: missing columns {missing}")
            idx = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
            if idx.has_duplicates:
                raise PanelError(f"{symbol}: duplicate dates")
            normalized[symbol] = pd.DataFrame(
                df[list(PRICE_FIELDS)].to_numpy(dtype=np.float64), index=idx, columns=PRICE_FIELDS
            )
        calendar = pd.DatetimeIndex([])
        for df in normalized.values():
            calendar = calendar.union(df.index)
        calendar = calendar.sort_values()
        symbols = list(normalized)
        arrays = {
            f: np.column_stack(
                [normalized[s][f].reindex(calendar).to_numpy() for s in symbols]
            ).reshape(len(calendar), len(symbols))
            for f in PRICE_FIELDS
        }
        return cls([ts.date() for ts in calendar], symbols, arrays)

    def select(self, symbols: Sequence[str]) -> BarPanel:
        """A panel with only `symbols`, on the same calendar."""
        cols = [self.column(s) for s in symbols]
        return BarPanel(self._dates, symbols, {f: a[:, cols] for f, a in self._fields.items()})

    # ------------------------------------------------------------------ access

    @property
    def dates(self) -> tuple[dt.date, ...]:
        return self._dates

    @property
    def symbols(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def n_dates(self) -> int:
        return len(self._dates)

    def column(self, symbol: str) -> int:
        try:
            return self._index[symbol]
        except KeyError:
            raise KeyError(f"symbol {symbol!r} is not in the panel") from None

    def field(self, name: PriceField) -> np.ndarray:
        """Read-only (n_dates, n_symbols) array."""
        return self._fields[name]

    def fingerprint(self) -> str:
        """sha256 of dates, symbols, and every value; identical data gives an identical hash."""
        h = hashlib.sha256()
        h.update(",".join(d.isoformat() for d in self._dates).encode())
        h.update(b"|" + ",".join(self._symbols).encode() + b"|")
        for f in PRICE_FIELDS:
            arr = np.where(np.isnan(self._fields[f]), np.nan, self._fields[f])  # canonical NaN
            h.update(f.encode())
            h.update(arr.tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------ validation

    def _where(self, mask: np.ndarray) -> str:
        i, j = map(int, np.argwhere(mask)[0])
        return f"{self._symbols[j]} on {self._dates[i]}"

    def _validate(self) -> None:
        o, h, lo, c, v = (self._fields[f] for f in PRICE_FIELDS)
        present = ~np.isnan(c)
        for f, arr in self._fields.items():
            if np.isinf(arr).any():
                raise PanelError(f"infinite {f} at {self._where(np.isinf(arr))}")
            partial = np.isnan(arr) != ~present
            if partial.any():
                raise PanelError(f"partial bar (some fields missing) at {self._where(partial)}")
        with np.errstate(invalid="ignore"):
            for f, arr in (("open", o), ("high", h), ("low", lo), ("close", c)):
                bad = present & ~(arr > 0)
                if bad.any():
                    raise PanelError(f"non-positive {f} at {self._where(bad)}")
            bad = present & ~(v >= 0)
            if bad.any():
                raise PanelError(f"negative volume at {self._where(bad)}")
            slack = h * _PRICE_TOLERANCE
            bad = present & (lo > h + slack)
            if bad.any():
                raise PanelError(f"low above high at {self._where(bad)}")
            for f, arr in (("open", o), ("close", c)):
                bad = present & ((arr < lo - slack) | (arr > h + slack))
                if bad.any():
                    raise PanelError(f"{f} outside [low, high] at {self._where(bad)}")
