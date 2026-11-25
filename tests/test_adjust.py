"""Spin-off adjustment (docs/spinoff-design.md): the three ways Alpaca serves a spin-off,
a same-day dividend, several events on one symbol, the guard, skipped events, loading
spinoffs.csv into the recipe, and the wiring in run, paper and the restart check."""

from __future__ import annotations

import datetime as dt
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.adjust import GUARD_TOLERANCE, apply_spinoffs
from harness.config import SPINOFFS_FILE, ConfigError, Recipe, SpinOff, load_recipe, load_spinoffs
from harness.data import BarPanel, DataError, DataLoader
from harness.profiles import FrozenProfile, check_restart_equivalence
from harness.run import run_backtest
from tests.fake_alpaca import NOW, REPO_CONFIG, FakeFetcher, recipe, spinoff_row, tmp_config

DAYS = [dt.date(2020, 3, 30) + dt.timedelta(days=i) for i in range(5)]  # Mon..Fri
EX = DAYS[2]  # 2020-04-01
F = 0.6  # true factor: parent 60 on the ex-date, child worth 40


def _frame(closes: list[float], volume: float = 1000.0) -> pd.DataFrame:
    rows = [
        {
            "date": d,
            "open": c * 0.99,
            "high": c * 1.01,
            "low": c * 0.98,
            "close": c,
            "volume": volume,
        }
        for d, c in zip(DAYS, closes, strict=True)
    ]
    return pd.DataFrame(rows)


def _panel(**closes: list[float]) -> BarPanel:
    return BarPanel.from_frames({s: _frame(c) for s, c in closes.items()})


def _close(panel: BarPanel, symbol: str) -> list[float]:
    return [round(float(x), 9) for x in panel.field("close")[:, panel.column(symbol)]]


def _event(
    applied: float, *, dividend: float = 1.0, step: float, ex: dt.date = EX, symbol: str = "P"
) -> SpinOff:
    return SpinOff(symbol=symbol, ex_date=ex, multiplier=F * dividend / applied, cached_step=step)


# The parent's raw closes: 100 before the ex-date, 60 from it (40 handed out per share).
RAW = [100.0, 100.0, 60.0, 60.0, 60.0]
TRUE = [60.0, 60.0, 60.0, 60.0, 60.0]  # correctly adjusted: no move on the ex-date


@pytest.mark.parametrize(
    ("served", "applied"),
    [
        pytest.param([60.0, 60.0, 60.0, 60.0, 60.0], F, id="applied correctly (a = f)"),
        pytest.param(RAW, 1.0, id="not applied (a = 1)"),
        pytest.param([200.0, 200.0, 60.0, 60.0, 60.0], 2.0, id="reverse split (a = 2)"),
    ],
)
def test_each_way_alpaca_serves_a_spinoff_ends_continuous(
    served: list[float], applied: float
) -> None:
    panel = _panel(P=served, SPY=[300.0] * 5)
    step = served[2] / served[1]
    out = apply_spinoffs(panel, [_event(applied, step=step)])
    assert _close(out, "P") == TRUE
    j = out.column("P")
    for f in ("open", "high", "low"):
        np.testing.assert_allclose(
            out.field(f)[:, j], panel.field(f)[:, j] * [*[F / applied] * 2, 1, 1, 1]
        )
    np.testing.assert_array_equal(out.field("volume"), panel.field("volume"))  # untouched
    assert _close(out, "SPY") == [300.0] * 5


def test_a_same_day_dividend_alpaca_applied_is_kept() -> None:
    # A 1% cash dividend goes ex with the spin-off. Alpaca applied only the dividend
    # (a = d), so the cached series is raw * 0.99 before the ex-date.
    d = 0.99
    served = [99.0, 99.0, 60.0, 60.0, 60.0]
    out = apply_spinoffs(_panel(P=served), [_event(d, dividend=d, step=60 / 99)])
    assert _close(out, "P") == [59.4, 59.4, 60.0, 60.0, 60.0]  # 100 * f * d


def test_two_events_on_one_symbol_compound() -> None:
    # Not applied twice: 100 -> 60 on DAYS[2], then 60 -> 36 on DAYS[4].
    served = [100.0, 100.0, 60.0, 60.0, 36.0]
    first = _event(1.0, step=0.6)
    second = _event(1.0, step=0.6, ex=DAYS[4])
    out = apply_spinoffs(_panel(P=served), [first, second])
    assert _close(out, "P") == [36.0] * 5


def test_the_guard_fails_when_alpaca_has_changed_its_adjustment() -> None:
    # Measured when Alpaca did not apply the event (step 0.6); it now serves it correctly.
    event = _event(1.0, step=0.6)
    with pytest.raises(DataError, match=r"spin-off P 2020-04-01: .*has changed"):
        apply_spinoffs(_panel(P=TRUE), [event])


def test_the_guard_allows_vendor_rounding_but_not_more() -> None:
    near = 0.6 * (1 + GUARD_TOLERANCE * 0.8)
    far = 0.6 * (1 + GUARD_TOLERANCE * 1.2)
    apply_spinoffs(_panel(P=RAW), [_event(1.0, step=near)])
    with pytest.raises(DataError, match="has changed"):
        apply_spinoffs(_panel(P=RAW), [_event(1.0, step=far)])


def test_one_firing_guard_fails_the_load_whatever_the_order() -> None:
    served = [100.0, 100.0, 60.0, 60.0, 36.0]
    good, bad = _event(1.0, step=0.6), _event(1.0, step=0.9, ex=DAYS[4])
    for events in ([good, bad], [bad, good]):
        with pytest.raises(DataError, match="2020-04-03"):
            apply_spinoffs(_panel(P=served), events)


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(_event(1.0, step=0.6, symbol="Q"), id="symbol not in the panel"),
        pytest.param(
            _event(1.0, step=0.6, ex=DAYS[4] + dt.timedelta(days=3)), id="after the panel"
        ),
        pytest.param(_event(1.0, step=0.6, ex=DAYS[0]), id="on the first bar"),
        pytest.param(
            _event(1.0, step=0.6, ex=DAYS[0] - dt.timedelta(days=30)), id="before the panel"
        ),
    ],
)
def test_an_event_outside_the_panel_is_skipped(event: SpinOff) -> None:
    panel = _panel(P=RAW)
    assert apply_spinoffs(panel, [event]) is panel


def test_an_event_whose_bars_are_missing_inside_the_panel_fails() -> None:
    frames = {"P": _frame(RAW).drop(index=1), "SPY": _frame([300.0] * 5)}  # no bar before EX
    panel = BarPanel.from_frames(frames)
    with pytest.raises(DataError, match="cannot be checked"):
        apply_spinoffs(panel, [_event(1.0, step=0.6)])


def test_a_symbol_whose_bars_start_after_the_ex_date_is_skipped() -> None:
    frames = {"P": _frame(RAW).iloc[3:], "SPY": _frame([300.0] * 5)}
    panel = BarPanel.from_frames(frames)
    assert apply_spinoffs(panel, [_event(1.0, step=0.6)]) is panel


# --------------------------------------------------------------------------- config

AAPL_EX = dt.date(2026, 3, 2)  # in-sample in the test config
FETCHER = FakeFetcher()
AAPL_STEP = FETCHER.price("AAPL", AAPL_EX) / FETCHER.price("AAPL", dt.date(2026, 2, 27))


def test_the_shipped_file_loads_with_the_events_it_was_built_for() -> None:
    # The rows themselves are checked against their listing in test_corporate_actions.py.
    assert {e.parent for e in load_spinoffs(REPO_CONFIG)} >= {"RTX", "TGNA", "APTV"}


def test_the_recipe_carries_the_events_of_its_own_symbols_only(tmp_path: Path) -> None:
    rows = [
        spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP),
        spinoff_row("SPY", dt.date(2026, 1, 5), cached_step=1.0, applied=0.6),  # benchmark
        spinoff_row("XOM", dt.date(2026, 1, 5), cached_step=0.6),  # not in test-3
    ]
    config = tmp_config(tmp_path, spinoffs=rows)
    r = recipe(config)
    assert [(e.symbol, e.ex_date) for e in r.spinoffs] == [
        ("AAPL", AAPL_EX),
        ("SPY", dt.date(2026, 1, 5)),
    ]
    assert r.spinoffs[0].multiplier == pytest.approx(0.6)
    assert r.spinoffs[1].multiplier == pytest.approx(1.0)


def test_an_edit_to_another_symbols_row_leaves_the_recipe_unchanged(tmp_path: Path) -> None:
    config = tmp_config(tmp_path, spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)])
    before = recipe(config)
    with (config / SPINOFFS_FILE).open("a") as fh:
        fh.write(spinoff_row("XOM", AAPL_EX, cached_step=0.6) + "\n")
    assert recipe(config) == before


@pytest.mark.parametrize(
    ("row", "error"),
    [
        (spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace(",0.6,", ",0.5,", 1), "factor"),
        (
            spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace(",1.0,1.0,0.6,", ",1.0,1.0,0.7,"),
            "multiplier",
        ),
        (
            spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace("SPUN:40", "OTHER:40"),
            "same tickers",
        ),
        (spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace("SPUN:1", "SPUN"), "TICKER:VALUE"),
        (
            spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace("SPUN", "AAPL"),
            "none of them the parent",
        ),
        (spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace("synthetic", ""), "note"),
        (spinoff_row("AAPL", AAPL_EX, cached_step=0.0), "cached_step"),
        (spinoff_row("AAPL", AAPL_EX, cached_step=1.0).replace("2026-03-02", "2026-3-2"), "date"),
    ],
)
def test_an_inconsistent_row_is_refused(tmp_path: Path, row: str, error: str) -> None:
    config = tmp_config(tmp_path, spinoffs=[row])
    line = len(load_spinoffs(REPO_CONFIG)) + 2  # after the header and the shipped rows
    with pytest.raises(ConfigError, match=rf"spinoffs.csv line {line}: (?s:.*){error}"):
        recipe(config)


def test_two_rows_for_one_day_are_refused(tmp_path: Path) -> None:
    row = spinoff_row("AAPL", AAPL_EX, cached_step=1.0)
    config = tmp_config(tmp_path, spinoffs=[row, row.replace("SPUN", "OTHR")])
    with pytest.raises(ConfigError, match="more than one row"):
        recipe(config)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    config = tmp_config(tmp_path)
    (config / SPINOFFS_FILE).unlink()
    with pytest.raises(ConfigError, match="missing config file"):
        recipe(config)


def test_the_recipe_refuses_unsorted_or_stray_events(tmp_path: Path) -> None:
    r = recipe(tmp_config(tmp_path))
    late = SpinOff(symbol="AAPL", ex_date=AAPL_EX, multiplier=0.6, cached_step=0.6)
    early = late.model_copy(update={"ex_date": dt.date(2026, 1, 5)})
    dump = r.model_dump()
    with pytest.raises(ValueError, match="sorted"):
        Recipe.model_validate(dump | {"spinoffs": [late, early]})
    with pytest.raises(ValueError, match="sorted"):
        Recipe.model_validate(dump | {"spinoffs": [late, late]})
    with pytest.raises(ValueError, match="does not use"):
        Recipe.model_validate(dump | {"spinoffs": [late.model_copy(update={"symbol": "XOM"})]})
    assert Recipe.model_validate(dump | {"spinoffs": [early, late]}).spinoffs == (early, late)


def test_a_recipe_saved_before_the_field_loads_with_none(tmp_path: Path) -> None:
    dump = recipe(tmp_config(tmp_path)).model_dump(mode="json")
    del dump["spinoffs"]
    assert Recipe.model_validate(dump).spinoffs == ()


# --------------------------------------------------------------------------- wiring


def _loader(config: Path, fetcher: FakeFetcher) -> DataLoader:
    r = recipe(config)
    return DataLoader(r.dates, r.data, fetcher_factory=lambda: fetcher, now=lambda: NOW)


def _equity(summary: dict) -> str:  # type: ignore[type-arg]
    return (Path(summary["run_dir"]) / "equity.csv").read_text()


def test_a_backtest_applies_the_recipes_events(tmp_path: Path) -> None:
    # The example strategy holds SPY, so an event on SPY changes what the run books.
    spy_step = FETCHER.price("SPY", AAPL_EX) / FETCHER.price("SPY", dt.date(2026, 2, 27))
    plain = tmp_config(tmp_path / "plain")
    adjusted = tmp_config(
        tmp_path / "adjusted", spinoffs=[spinoff_row("SPY", AAPL_EX, cached_step=spy_step)]
    )
    a = run_backtest(recipe(plain), loader=_loader(plain, FakeFetcher()), now=lambda: NOW)
    b = run_backtest(recipe(adjusted), loader=_loader(adjusted, FakeFetcher()), now=lambda: NOW)
    assert b["total_return"] != a["total_return"]  # SPY bought at 0.6 of the price
    rerun = run_backtest(recipe(adjusted), loader=_loader(adjusted, FakeFetcher()), now=lambda: NOW)
    assert _equity(rerun) == _equity(b)


def test_a_backtest_fails_when_the_guard_fires(tmp_path: Path) -> None:
    config = tmp_config(tmp_path, spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)])
    fetcher = FakeFetcher()
    fetcher.scale_before["AAPL"] = (AAPL_EX, 2.0)  # Alpaca now applies it as a reverse split
    with pytest.raises(DataError, match="spin-off AAPL 2026-03-02"):
        run_backtest(recipe(config), loader=_loader(config, fetcher), now=lambda: NOW)


def test_the_restart_check_runs_on_adjusted_bars_and_refuses_a_firing_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_config(tmp_path, spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)])
    r = load_recipe("example_buy_and_hold", "test-3", config, risk="balanced")
    import harness.profiles as profiles

    seen: list[tuple[BarPanel, BarPanel]] = []
    real = profiles.apply_spinoffs

    def spy(panel: BarPanel, events: Sequence[SpinOff]) -> BarPanel:
        seen.append((panel, out := real(panel, events)))
        return out

    monkeypatch.setattr(profiles, "apply_spinoffs", spy)
    check_restart_equivalence(r, _loader(config, FakeFetcher()), sessions=2)
    ((served, adjusted),) = seen
    assert adjusted.fingerprint() != served.fingerprint()

    shutil.rmtree(r.data.cache_dir)
    fetcher = FakeFetcher()
    fetcher.scale_before["AAPL"] = (AAPL_EX, 2.0)
    with pytest.raises(DataError, match="spin-off AAPL"):
        check_restart_equivalence(r, _loader(config, fetcher), sessions=2)


def test_a_profile_hashes_its_events(tmp_path: Path) -> None:
    from tests.test_profiles import Rig

    rig = Rig(
        tmp_path, holdout=False, spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)]
    )
    profile = rig.promote("plumbing", plumbing=True)
    assert [e.symbol for e in profile.recipe.spinoffs] == ["AAPL"]
    data = profile.model_dump(mode="json")
    data["recipe"]["spinoffs"][0]["multiplier"] = 0.5
    with pytest.raises(ValueError, match="content hash does not match"):
        FrozenProfile.model_validate(data)


def test_promotion_is_refused_when_the_guard_fires(tmp_path: Path) -> None:
    from tests.test_profiles import Rig

    rig = Rig(
        tmp_path, holdout=False, spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)]
    )
    shutil.rmtree(rig.recipe.data.cache_dir)
    rig.fetcher.scale_before["AAPL"] = (AAPL_EX, 2.0)  # Alpaca changed it after the run
    with pytest.raises(DataError, match="spin-off AAPL"):
        rig.promote("plumbing", plumbing=True)
    assert not any(rig.execution.profiles_dir.glob("*/profile.json"))


def test_paper_stops_the_day_when_the_guard_fires(tmp_path: Path) -> None:
    from tests.test_paper import Rig as PaperRig

    rig = PaperRig(tmp_path, spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)])
    shutil.rmtree(rig.profile.recipe.data.cache_dir)
    rig.base.fetcher.scale_before["AAPL"] = (AAPL_EX, 2.0)
    report = rig.run()
    assert report.outcome == "failed"
    assert [(a.severity, a.kind) for a in report.alerts] == [("critical", "data")]
    assert "spin-off AAPL 2026-03-02" in report.alerts[0].message
    assert rig.venue.place_attempts == []


def test_paper_trades_on_adjusted_bars(tmp_path: Path) -> None:
    from tests.test_paper import Rig as PaperRig

    def decided_on(rig: PaperRig) -> str:
        assert rig.run().outcome == "traded"
        (decision,) = [r for r in rig.records() if r.kind == "decision"]
        return str(decision.payload["data_fingerprint"])

    plain = PaperRig(tmp_path / "plain")
    rig = PaperRig(
        tmp_path / "adjusted", spinoffs=[spinoff_row("AAPL", AAPL_EX, cached_step=AAPL_STEP)]
    )
    assert [e.symbol for e in rig.profile.recipe.spinoffs] == ["AAPL"]
    assert decided_on(rig) != decided_on(plain)  # the decision read the adjusted bars
