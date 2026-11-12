"""Synthetic bar data for tests. No network, no API keys.

Within a symbol, every price is unique per (session, field), so a test can tell exactly
which bar a fill price came from:

    open  = base + 1.00 * t        high = close + 0.25
    close = base + 1.00 * t + 0.50 low  = open  - 0.25

with base = 50 + 10 * symbol_index (or a custom per-symbol path).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from harness.config import ConstructionConfig, CostConfig
from harness.data import BarPanel
from harness.view import PRICE_FIELDS

COSTS = CostConfig(slippage_bps=5.0, commission_per_order=0.0)


def construction(max_positions: int = 5, max_missing_sessions: int = 5) -> ConstructionConfig:
    return ConstructionConfig(
        max_positions=max_positions, max_missing_sessions=max_missing_sessions
    )


def sessions(n: int, start: dt.date = dt.date(2024, 1, 2)) -> list[dt.date]:
    """`n` weekdays starting at `start`."""
    out: list[dt.date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += dt.timedelta(days=1)
    return out


def make_panel(
    symbols: Sequence[str],
    n: int,
    *,
    opens: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    closes: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    missing: dict[str, Sequence[int]] | None = None,
    start: dt.date = dt.date(2024, 1, 2),
) -> BarPanel:
    """Build a panel. `opens`/`closes` override a symbol's path; `missing` lists session
    indexes where a symbol has no bar."""
    dates = sessions(n, start)
    t = np.arange(n, dtype=np.float64)
    cols: dict[str, list[np.ndarray]] = {f: [] for f in PRICE_FIELDS}
    for k, sym in enumerate(symbols):
        base = 50.0 + 10.0 * k
        o = np.asarray(opens[sym], dtype=float) if opens and sym in opens else base + t
        c = np.asarray(closes[sym], dtype=float) if closes and sym in closes else base + t + 0.5
        h = np.maximum(o, c) + 0.25
        lo = np.minimum(o, c) - 0.25
        v = np.full(n, 1_000_000.0)
        values = {"open": o, "high": h, "low": lo, "close": c, "volume": v}
        gone = list(missing.get(sym, [])) if missing else []
        for f in PRICE_FIELDS:
            col = values[f].copy()
            col[gone] = np.nan
            cols[f].append(col)
    return BarPanel(dates, list(symbols), {f: np.column_stack(cols[f]) for f in PRICE_FIELDS})


def price_lookup(panel: BarPanel) -> Callable[[str, dt.date, str], float]:
    """(symbol, date, field) -> price."""
    pos = {d: i for i, d in enumerate(panel.dates)}

    def get(symbol: str, date: dt.date, field: str) -> float:
        return float(panel.field(field)[pos[date], panel.column(symbol)])  # type: ignore[arg-type]

    return get
