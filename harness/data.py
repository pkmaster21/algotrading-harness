"""Market data: the bar panel, the Alpaca loader, the Parquet cache, and the holdout cutoff.

`BarPanel` is the bulk bar container the engine runs on: many symbols on one calendar,
stored as arrays for speed.

`DataLoader` is the single choke point for holdout discipline. By default it never
requests, and never returns, a bar on or after the holdout start. Only a `HoldoutAccess`
lifts the cutoff, and the loader appends a line to the holdout log before it reads any
holdout data. The engine and strategies never check dates themselves: they run on
whatever panel the loader returns.

Cache: one Parquet file per symbol in `cache_dir`, holding split- and dividend-adjusted
bars plus metadata about the range requested. Adjusted history is rescaled retroactively
after every split or dividend, so naively appending new days could mix old and new
scaling. The loader prevents that two ways:

- **Overlap check:** when extending a symbol, it re-fetches the last `overlap_bars`
  cached bars along with the new days. Any corporate action after the cached end
  rescales those bars too, so a mismatch triggers a full refetch of that symbol.
- **Scheduled refresh:** a symbol's full history is refetched at least every
  `full_refresh_days`, which also catches vendor corrections deep in history.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import math
import os
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from harness.config import DataConfig, DateConfig
from harness.keys import KeysError, read_alpaca_paper_keys
from harness.models import FrozenModel, Stock
from harness.view import PRICE_FIELDS, PriceField

MARKET_TZ = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
        fields: Mapping[str, np.ndarray],  # keys must be exactly PRICE_FIELDS (checked)
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
        dates = sorted({ts.date() for frame in normalized.values() for ts in frame.index})
        calendar = pd.DatetimeIndex(dates)
        symbols = list(normalized)
        arrays = {
            f: np.column_stack(
                [normalized[s][f].reindex(calendar).to_numpy() for s in symbols]
            ).reshape(len(calendar), len(symbols))
            for f in PRICE_FIELDS
        }
        return cls(dates, symbols, arrays)

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


# =========================================================================== loading


class HoldoutAccess(FrozenModel):
    """Proof that the caller explicitly asked for holdout data.

    Only the run entry point builds one, from `--holdout`. Passing it to
    `DataLoader.load` is the only way to lift the cutoff, and doing so logs the run.
    """

    run_id: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    git_commit: str | None = None


class HoldoutViolation(RuntimeError):
    """A request for data on or after the cutoff was about to be made. Always a bug."""


class HoldoutLogError(RuntimeError):
    """The holdout log could not be written, so no holdout data is returned."""


class DataError(RuntimeError):
    """Bars could not be fetched or credentials are missing."""


class BarFetcher(Protocol):
    """Source of adjusted daily bars. AlpacaBarFetcher in production, fakes in tests."""

    def fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        """Bars with dates in [start, end] (inclusive), one frame per requested symbol.

        Each frame has columns date (datetime.date), open, high, low, close, volume, in
        date order. A symbol with no bars in range maps to an empty frame or is omitted.
        """
        ...


_COLUMNS = ["date", *PRICE_FIELDS]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="object" if c == "date" else "float64") for c in _COLUMNS}
    )


def _clean(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_frame()
    out = frame.loc[:, _COLUMNS].copy()
    out["date"] = [d if type(d) is dt.date else pd.Timestamp(d).date() for d in out["date"]]
    for f in PRICE_FIELDS:
        out[f] = out[f].astype("float64")
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


class RateLimiter:
    """Allows at most `max_per_minute` request tokens in any 60-second window."""

    def __init__(
        self,
        max_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._stamps: list[float] = []

    def acquire(self, n: int = 1) -> None:
        for _ in range(n):
            while True:
                now = self._clock()
                self._stamps = [t for t in self._stamps if now - t < 60.0]
                if len(self._stamps) < self._max:
                    self._stamps.append(now)
                    break
                self._sleep(60.0 - (now - self._stamps[0]))


def frames_from_alpaca_df(df: pd.DataFrame, symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Convert alpaca-py's `BarSet.df` (index: symbol, timestamp in UTC) to per-symbol
    frames keyed by the session date in New York time."""
    out = {s: _empty_frame() for s in symbols}
    if df is None or df.empty:
        return out
    flat = df.reset_index()
    stamps = pd.to_datetime(flat["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    flat["date"] = [ts.date() for ts in stamps]
    for symbol, group in flat.groupby("symbol", sort=True):
        if symbol in out:
            out[str(symbol)] = _clean(group)
    return out


class AlpacaBarFetcher:
    """Adjusted daily SIP bars from Alpaca's historical data API (read-only).

    Reads ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY from `.env` into local
    variables (see `harness.keys`); they are never exported to the process environment.
    It is built lazily by the loader, only when a download is needed, so runs served
    entirely from cache need no keys. Requests are throttled by a
    RateLimiter (one token per expected page of 10,000 bars); alpaca-py also retries
    HTTP 429 responses itself.
    """

    PAGE_SIZE = 10_000

    def __init__(
        self,
        cfg: DataConfig,
        limiter: RateLimiter,
        dotenv_path: Path = PROJECT_ROOT / ".env",
    ) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        try:
            keys = read_alpaca_paper_keys(dotenv_path)
        except KeysError as exc:
            raise DataError(f"{exc} to download bars") from None
        self._cfg = cfg
        self._limiter = limiter
        self._client = StockHistoricalDataClient(api_key=keys.key_id, secret_key=keys.secret_key)

    def fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.models import BarSet
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        sessions = max(1, int((end - start).days * 252 / 365) + 1)
        self._limiter.acquire(math.ceil(len(symbols) * sessions / self.PAGE_SIZE))
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=TimeFrame(1, TimeFrameUnit("Day")),  # alpaca types .Day as str
            start=dt.datetime.combine(start, dt.time(0, 0), MARKET_TZ),
            end=dt.datetime.combine(end, dt.time(23, 59, 59), MARKET_TZ),
            adjustment=Adjustment(self._cfg.adjustment),
            feed=DataFeed(self._cfg.feed),
        )
        try:
            barset = self._client.get_stock_bars(request)
        except Exception as exc:  # network, auth, plan limits: surface with context
            raise DataError(
                f"Alpaca bars request failed for {len(symbols)} symbols: {exc}"
            ) from exc
        if not isinstance(barset, BarSet):
            raise DataError(f"unexpected response type from Alpaca: {type(barset).__name__}")
        return frames_from_alpaca_df(barset.df, symbols)


# --------------------------------------------------------------------------- cache

CACHE_SCHEMA_VERSION = 1
_META_KEY = b"harness_cache_meta"


class CacheMeta(FrozenModel):
    """What a cache file covers. Coverage is the range requested, which can be wider
    than the bars present (a symbol listed later, or delisted)."""

    coverage_start: dt.date
    coverage_end: dt.date
    full_fetched_on: dt.date
    adjustment: str
    feed: str
    schema_version: int


class _BarCache:
    """One Parquet file per symbol: `<cache_dir>/<SYMBOL>.parquet`.

    Module-private on purpose: code outside this module gets bars only through
    `DataLoader.load`, which applies the holdout cutoff. The cache itself does not know
    the cutoff, and the file may hold holdout rows after a holdout run. Python cannot make
    this airtight; the private names make any bypass obvious in review.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def path(self, symbol: str) -> Path:
        Stock(symbol=symbol)  # validated ticker, so it is a safe file name
        return self.cache_dir / f"{symbol}.parquet"

    def meta(self, symbol: str) -> CacheMeta | None:
        path = self.path(symbol)
        if not path.is_file():
            return None
        raw = (pq.read_schema(path).metadata or {}).get(_META_KEY)
        if raw is None:
            return None
        try:
            return CacheMeta.model_validate_json(raw)
        except ValueError:
            return None  # unreadable metadata: treated as no cache, so it is refetched

    def read(self, symbol: str, start: dt.date, before: dt.date) -> pd.DataFrame:
        """Rows with start <= date < before. Rows outside are filtered while reading."""
        path = self.path(symbol)
        if not path.is_file():
            return _empty_frame()
        table = pq.read_table(path, filters=[("date", ">=", start), ("date", "<", before)])
        return _clean(table.to_pandas())

    def _read_unfiltered(self, symbol: str) -> pd.DataFrame:
        """Every cached row, including any on or after the holdout start. Only for
        cache maintenance inside DataLoader; never return these rows to a caller."""
        path = self.path(symbol)
        return _clean(pq.read_table(path).to_pandas()) if path.is_file() else _empty_frame()

    def write(self, symbol: str, frame: pd.DataFrame, meta: CacheMeta) -> None:
        """Atomically replace the symbol's file (temp file, then rename).

        Each write gets its own temp file in the cache directory, so two runs writing the
        same symbol at once cannot interleave in one file; the last rename wins. A failed
        write removes its temp file and leaves the existing cache file untouched.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame = _clean(frame)
        schema = pa.schema(
            [("date", pa.date32())] + [(f, pa.float64()) for f in PRICE_FIELDS],
            metadata={_META_KEY: meta.model_dump_json().encode()},
        )
        table = pa.Table.from_pydict({c: list(frame[c]) for c in _COLUMNS}, schema=schema)
        path = self.path(symbol)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.cache_dir, prefix=f"{symbol}.", suffix=".parquet.tmp"
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            pq.write_table(table, tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


# --------------------------------------------------------------------------- loader

HOLDOUT_LOG_HEADER = ["timestamp_utc", "run_id", "strategy", "data_start", "data_end", "git_commit"]


def _last_completed_session(now: dt.datetime) -> dt.date:
    """Yesterday in New York time: today's bar may be incomplete, and the free plan
    excludes the most recent 15 minutes of SIP data."""
    return now.astimezone(MARKET_TZ).date() - dt.timedelta(days=1)


class DataLoader:
    """Loads bars for a run, from cache where possible, and enforces the holdout cutoff."""

    def __init__(
        self,
        dates: DateConfig,
        cfg: DataConfig,
        fetcher_factory: Callable[[], BarFetcher] | None = None,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        self.dates = dates
        self.cfg = cfg
        self._cache = _BarCache(cfg.cache_dir)
        self._factory = fetcher_factory or (
            lambda: AlpacaBarFetcher(cfg, RateLimiter(cfg.max_requests_per_minute))
        )
        self._fetcher: BarFetcher | None = None
        self._now = now

    # ------------------------------------------------------------------ cutoff

    def cutoff(self, holdout: HoldoutAccess | None) -> dt.date:
        """Exclusive end date. Without holdout access this is always the holdout start.

        This is the permission boundary. `load` additionally never goes past the last
        completed session, whichever comes first.
        """
        if holdout is None:
            return self.dates.holdout_start
        last = _last_completed_session(self._now())
        if self.dates.holdout_end is not None:
            last = min(last, self.dates.holdout_end)
        return last + dt.timedelta(days=1)

    def load(self, symbols: Sequence[str], *, holdout: HoldoutAccess | None = None) -> BarPanel:
        """Bars from `dates.data_start` through the last completed session before the
        cutoff, as a BarPanel.

        The end is `min(cutoff - 1 day, last completed session)` in every mode, and that
        date is what the cache records as covered. Without the clamp, a cutoff later than
        today (an `in_sample_end` in the future) would mark sessions that have not
        happened yet as cached, and they would stay missing until the next full refresh.

        With `holdout`, the holdout log line is appended first; if that fails, nothing
        is loaded.
        """
        symbols = list(dict.fromkeys(symbols))  # dedupe, keep order
        start = self.dates.data_start
        cutoff = self.cutoff(holdout)
        end = min(cutoff - dt.timedelta(days=1), _last_completed_session(self._now()))
        if end < start:
            raise DataError(
                f"no completed session to load: data_start is {start}, "
                f"and the last loadable day is {end}"
            )
        if holdout is not None:
            self._log_holdout(holdout, start, end)
        self._refresh(symbols, start, end, cutoff)
        # Read through `end`, not the cutoff: never past the holdout boundary, and never a
        # row later than the range this load asked for.
        before = end + dt.timedelta(days=1)
        frames = {s: self._cache.read(s, start, before) for s in symbols}
        return BarPanel.from_frames(frames)

    def _log_holdout(self, access: HoldoutAccess, start: dt.date, end: dt.date) -> None:
        path = self.cfg.holdout_log
        row = [
            self._now().astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            access.run_id,
            access.strategy,
            start.isoformat(),
            end.isoformat(),
            access.git_commit or "",
        ]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            new = not path.exists()
            with path.open("a", newline="") as fh:
                writer = csv.writer(fh)
                if new:
                    writer.writerow(HOLDOUT_LOG_HEADER)
                writer.writerow(row)
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise HoldoutLogError(f"could not append to holdout log {path}: {exc}") from exc

    # ------------------------------------------------------------------ cache refresh

    def _fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date, cutoff: dt.date
    ) -> dict[str, pd.DataFrame]:
        if end >= cutoff:
            raise HoldoutViolation(f"refusing to request bars through {end} (cutoff {cutoff})")
        if self._fetcher is None:
            self._fetcher = self._factory()
        result: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), self.cfg.batch_size):
            batch = list(symbols[i : i + self.cfg.batch_size])
            fetched = self._fetcher.fetch(batch, start, end)
            for s in batch:
                frame = _clean(fetched.get(s))
                dates = list(frame["date"])
                if dates and (min(dates) < start or max(dates) > end):
                    raise DataError(f"{s}: fetcher returned bars outside {start}..{end}")
                result[s] = frame
        return result

    def _plan(self, symbol: str, start: dt.date, end: dt.date) -> tuple[str, dt.date]:
        """("none" | "full" | "extend", fetch start)."""
        meta = self._cache.meta(symbol)
        today = self._now().astimezone(MARKET_TZ).date()
        if (
            meta is None
            or meta.schema_version != CACHE_SCHEMA_VERSION
            or meta.feed != self.cfg.feed
            or meta.adjustment != self.cfg.adjustment
            or start < meta.coverage_start
            or (today - meta.full_fetched_on).days >= self.cfg.full_refresh_days
        ):
            return "full", start
        if meta.coverage_end >= end:
            return "none", start
        cached = self._cache._read_unfiltered(symbol)
        if cached.empty:
            return "extend", meta.coverage_end + dt.timedelta(days=1)
        tail = cached["date"].iloc[-self.cfg.overlap_bars :]
        return "extend", tail.iloc[0]

    def _refresh(
        self, symbols: Sequence[str], start: dt.date, end: dt.date, cutoff: dt.date
    ) -> None:
        today = self._now().astimezone(MARKET_TZ).date()
        full: list[str] = []
        extend: dict[dt.date, list[str]] = defaultdict(list)
        for s in symbols:
            action, fetch_start = self._plan(s, start, end)
            if action == "full":
                full.append(s)
            elif action == "extend":
                extend[fetch_start].append(s)

        for fetch_start in sorted(extend):
            group = extend[fetch_start]
            fetched = self._fetch(group, fetch_start, end, cutoff)
            for s in group:
                cached = self._cache._read_unfiltered(s)
                new = fetched[s]
                overlap = cached.loc[cached["date"] >= fetch_start]
                if not _overlap_matches(overlap, new):
                    full.append(s)  # adjustment basis changed: refetch everything
                    continue
                meta = self._cache.meta(s)
                assert meta is not None  # extend is only planned for cached symbols
                kept = cached.loc[cached["date"] < fetch_start]
                merged = pd.concat([kept, new], ignore_index=True)
                self._cache.write(s, merged, meta.model_copy(update={"coverage_end": end}))

        if full:
            fetched = self._fetch(full, start, end, cutoff)
            for s in full:
                meta = CacheMeta(
                    coverage_start=start,
                    coverage_end=end,
                    full_fetched_on=today,
                    adjustment=self.cfg.adjustment,
                    feed=self.cfg.feed,
                    schema_version=CACHE_SCHEMA_VERSION,
                )
                self._cache.write(s, fetched[s], meta)


_OVERLAP_TOLERANCE = 1e-6


def _overlap_matches(cached: pd.DataFrame, fetched: pd.DataFrame) -> bool:
    """True if every cached overlap bar is present in `fetched` with the same OHLC."""
    if cached.empty:
        return True
    merged = cached.merge(fetched, on="date", how="left", suffixes=("_c", "_f"))
    for f in ("open", "high", "low", "close"):
        c, n = merged[f"{f}_c"].to_numpy(), merged[f"{f}_f"].to_numpy()
        if np.isnan(n).any():
            return False
        if (np.abs(c - n) > _OVERLAP_TOLERANCE * np.abs(c)).any():
            return False
    return True


def read_holdout_log(path: Path) -> list[dict[str, Any]]:
    """Rows of the holdout log (empty if it does not exist)."""
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))
