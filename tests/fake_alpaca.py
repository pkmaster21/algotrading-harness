"""A fake BarFetcher and recipe helpers for loader, holdout, and run tests. No network."""

from __future__ import annotations

import datetime as dt
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from harness.config import SPINOFFS_FILE, Recipe, load_recipe

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"
NOW = dt.datetime(2026, 7, 1, 15, 0, tzinfo=dt.UTC)  # last completed session: 2026-06-30


class FakeFetcher:
    """Weekday bars for any symbol, deterministic per (symbol, date).

    `listed`/`delisted` bound a symbol's history. `scale_before` simulates a corporate
    action: bars before that date are multiplied by `scale`, as a vendor's adjusted
    history would be after a split or dividend. `fetch_raw` ignores it: unadjusted
    prices never change. Every request is recorded.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[tuple[str, ...], dt.date, dt.date]] = []
        self.raw_requests: list[tuple[tuple[str, ...], dt.date, dt.date]] = []
        self.listed: dict[str, dt.date] = {}
        self.delisted: dict[str, dt.date] = {}
        self.scale_before: dict[str, tuple[dt.date, float]] = {}

    def price(self, symbol: str, day: dt.date, *, raw: bool = False) -> float:
        base = 20.0 + sum(ord(c) for c in symbol) % 50
        value = base + (day.toordinal() % 97) * 0.5 + (day.toordinal() - 738000) * 0.01
        if not raw and symbol in self.scale_before and day < self.scale_before[symbol][0]:
            value *= self.scale_before[symbol][1]
        return value

    def fetch(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        self.requests.append((tuple(symbols), start, end))
        return self._bars(symbols, start, end, raw=False)

    def fetch_raw(
        self, symbols: Sequence[str], start: dt.date, end: dt.date
    ) -> dict[str, pd.DataFrame]:
        self.raw_requests.append((tuple(symbols), start, end))
        return self._bars(symbols, start, end, raw=True)

    def _bars(
        self, symbols: Sequence[str], start: dt.date, end: dt.date, *, raw: bool
    ) -> dict[str, pd.DataFrame]:
        out = {}
        for s in symbols:
            rows = []
            day = start
            while day <= end:
                live = day.weekday() < 5
                live &= s not in self.listed or day >= self.listed[s]
                live &= s not in self.delisted or day < self.delisted[s]
                if live:
                    c = self.price(s, day, raw=raw)
                    rows.append({"date": day, "open": c - 0.1, "high": c + 0.4,
                                 "low": c - 0.4, "close": c, "volume": 1e6})  # fmt: skip
                day += dt.timedelta(days=1)
            out[s] = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        return out

    @property
    def latest_end(self) -> dt.date | None:
        return max((end for _, _, end in self.requests), default=None)


def tmp_config(
    tmp_path: Path,
    *,
    data_start: str = "2025-10-01",
    holdout_end: str | None = None,
    symbols: Sequence[str] = ("SPY", "AAPL", "MSFT"),
    spinoffs: Sequence[str] = (),
) -> Path:
    """A copy of the repo config inside tmp_path, so data and run outputs land there.
    `spinoffs` are rows added to corporate-actions/spinoffs.csv (see `spinoff_row`)."""
    config = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, config)
    dates = (config / "dates.yaml").read_text().replace("2015-01-01", data_start)
    if holdout_end is not None:
        dates = dates.replace("holdout_end: null", f"holdout_end: {holdout_end}")
    (config / "dates.yaml").write_text(dates)
    (config / "universe" / "test-3.yaml").write_text(f"symbols: [{', '.join(symbols)}]\n")
    if spinoffs:
        with (config / SPINOFFS_FILE).open("a") as fh:
            fh.writelines(f"{row}\n" for row in spinoffs)
    return config


def spinoff_row(
    parent: str,
    ex_date: dt.date,
    *,
    cached_step: float,
    applied: float = 1.0,
    dividend: float = 1.0,
    child: str = "SPUN",
) -> str:
    """A consistent spinoffs.csv row: parent closes at 60 on the ex-date and one child
    share per parent share closes at 40, so the true factor is 0.6."""
    factor = 60 / (60 + 40)
    multiplier = factor * dividend / applied
    return (
        f"{parent},{ex_date},{child}:1,60,{child}:40,{factor!r},{dividend!r},{applied!r},"
        f"{multiplier!r},{cached_step!r},synthetic"
    )


def recipe(config: Path, strategy: str = "example_buy_and_hold") -> Recipe:
    return load_recipe(strategy, "test-3", config)
