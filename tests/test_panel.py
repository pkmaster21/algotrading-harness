import datetime as dt

import numpy as np
import pandas as pd
import pytest

from harness.data import BarPanel, PanelError
from harness.view import PRICE_FIELDS
from tests.synthetic import make_panel


def _frame(dates, closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 100.0},
        index=pd.to_datetime(dates),
    )


def test_from_frames_builds_union_calendar_with_gaps_as_nan() -> None:
    early = _frame(["2024-01-02", "2024-01-03", "2024-01-04"], [10, 11, 12])
    late = _frame(["2024-01-04", "2024-01-05"], [50, 51])  # listed later
    panel = BarPanel.from_frames({"OLD": early, "NEW": late})
    assert panel.dates == tuple(dt.date(2024, 1, d) for d in (2, 3, 4, 5))
    close = panel.field("close")
    np.testing.assert_array_equal(close[:, 0], [10, 11, 12, np.nan])
    np.testing.assert_array_equal(close[:, 1], [np.nan, np.nan, 50, 51])


def test_from_frames_accepts_date_column_and_empty_frames() -> None:
    frame = _frame(["2024-01-02"], [10]).rename_axis("date").reset_index()
    empty = _frame([], [])
    panel = BarPanel.from_frames({"A": frame, "B": empty})
    assert panel.symbols == ("A", "B")
    assert np.isnan(panel.field("close")[0, 1])


def test_arrays_are_read_only_copies() -> None:
    panel = make_panel(["SPY"], 3)
    with pytest.raises(ValueError):
        panel.field("close")[0, 0] = 1.0


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda f: f["volume"].__setitem__((1, 0), np.nan), "partial bar"),
        (lambda f: f["close"].__setitem__((1, 0), -5.0), "non-positive close|outside"),
        (lambda f: f["low"].__setitem__((1, 0), 10_000.0), "low above high"),
        (lambda f: f["open"].__setitem__((1, 0), np.inf), "infinite"),
        (lambda f: f["volume"].__setitem__((1, 0), -1.0), "negative volume"),
    ],
)
def test_rejects_malformed_bars(mutate, match) -> None:
    good = make_panel(["SPY"], 3)
    fields = {f: good.field(f).copy() for f in PRICE_FIELDS}
    mutate(fields)
    with pytest.raises(PanelError, match=match):
        BarPanel(good.dates, good.symbols, fields)


def test_rejects_unsorted_dates_and_duplicate_symbols() -> None:
    good = make_panel(["SPY", "QQQ"], 3)
    fields = {f: good.field(f) for f in PRICE_FIELDS}
    with pytest.raises(PanelError, match="increasing"):
        BarPanel(tuple(reversed(good.dates)), good.symbols, fields)
    with pytest.raises(PanelError, match="unique"):
        BarPanel(good.dates, ["SPY", "SPY"], fields)


def test_fingerprint_changes_with_any_value() -> None:
    a = make_panel(["SPY"], 5)
    fields = {f: a.field(f).copy() for f in PRICE_FIELDS}
    assert BarPanel(a.dates, a.symbols, fields).fingerprint() == a.fingerprint()
    fields["volume"][2, 0] += 1
    assert BarPanel(a.dates, a.symbols, fields).fingerprint() != a.fingerprint()


def test_select_keeps_calendar() -> None:
    panel = make_panel(["SPY", "AAPL"], 4, missing={"SPY": [0]})
    spy = panel.select(["SPY"])
    assert spy.dates == panel.dates
    assert spy.symbols == ("SPY",)
    assert np.isnan(spy.field("close")[0, 0])
