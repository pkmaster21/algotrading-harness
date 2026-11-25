"""Spin-off adjustment of cached bars (docs/spinoff-design.md, decisions 4, 6 and 7).

Alpaca's adjusted bars get most spin-offs before 2021 wrong: not applied, or applied as a
reverse split. Each reviewed event in a recipe carries one multiplier, measured against
Alpaca's own adjustment, that scales the open, high, low and close of every bar of the
parent before the ex-date. Volume is not changed.

`apply_spinoffs` is pure: it reads nothing but its arguments (no API, keys or files).
Every place that loads bars for a recipe calls it straight after `DataLoader.load`, so
backtest, paper and the restart check run on the same bars.

The guard: a multiplier is right only while Alpaca adjusts the event the way it did when
the row was built. The ratio of the ex-date close to the close the session before does
not change under any later split or dividend (both bars are rescaled alike), so each
event records it as `cached_step`, and every load compares it with the bars as served,
before any multiplier. A difference over `GUARD_TOLERANCE` fails the load, so a fixed
vendor adjustment is never applied twice.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from harness.config import SpinOff
from harness.data import BarPanel, DataError
from harness.view import PRICE_FIELDS

# Relative difference between the served and the recorded step that fails a load: far
# above vendor rounding (four decimals on low adjusted prices), well below any real
# change of adjustment (3% or more). Decision 6.
GUARD_TOLERANCE = 0.005
_ADJUSTED_FIELDS = ("open", "high", "low", "close")  # not volume (open question 1)


def apply_spinoffs(panel: BarPanel, spinoffs: Sequence[SpinOff]) -> BarPanel:
    """`panel` with each event's multiplier applied to the symbol's prices before its
    ex-date. Several events on one symbol compound.

    An event is skipped when its symbol is not in the panel, or when the symbol has no
    bar before the ex-date or none on or after it in this panel: scaling a symbol's whole
    history in the panel changes no return. Raises DataError when an event inside the
    panel cannot be checked (no bar on the ex-date or the session before) or when its
    guard fires. Returns `panel` itself when nothing applies.
    """
    close = panel.field("close")
    scale: dict[int, np.ndarray] = {}  # column -> per-row multiplier
    for event in spinoffs:
        if event.symbol not in panel.symbols:
            continue
        j = panel.column(event.symbol)
        present = np.flatnonzero(~np.isnan(close[:, j]))
        if present.size == 0:
            continue
        first, last = panel.dates[int(present[0])], panel.dates[int(present[-1])]
        if not first < event.ex_date <= last:
            continue
        name = f"spin-off {event.symbol} {event.ex_date}"
        if event.ex_date not in panel.dates:
            raise DataError(f"{name}: the panel has no session on the ex-date")
        i = panel.dates.index(event.ex_date)
        before, on = close[i - 1, j], close[i, j]
        if np.isnan(before) or np.isnan(on):
            raise DataError(
                f"{name}: no {event.symbol} bar on the ex-date or the session before "
                f"({panel.dates[i - 1]}), so the adjustment cannot be checked"
            )
        step = float(on / before)
        if abs(step / event.cached_step - 1) > GUARD_TOLERANCE:
            raise DataError(
                f"{name}: the cached close steps by {step:.6g} on the ex-date, but the "
                f"event was measured at {event.cached_step:.6g}. Alpaca's adjustment of "
                "it has changed, so its multiplier no longer holds: rebuild the row "
                "(docs/spinoff-design.md, decision 6)"
            )
        rows = scale.setdefault(j, np.ones(panel.n_dates))
        rows[:i] *= event.multiplier
    if not scale:
        return panel
    fields = {f: panel.field(f).copy() for f in PRICE_FIELDS}
    for j, rows in scale.items():
        for f in _ADJUSTED_FIELDS:
            fields[f][:, j] *= rows
    return BarPanel(panel.dates, panel.symbols, fields)
