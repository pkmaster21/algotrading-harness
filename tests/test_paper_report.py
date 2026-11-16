"""Paper trading's trades.csv and summary.json: built from the journal alone, the same
schema as a backtest's, with each fill measured against the official open."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

import harness.paper as paper
from harness.data import DataError
from harness.journal import JournalKind, JournalRecord
from harness.models import MARKET_TZ, Fill, Side, Stock
from harness.paper_report import PAPER_TRADE_COLUMNS, paper_report, unpriced_fills
from harness.report import TRADE_COLUMNS, ReportError
from tests.fixture_strategies import Churn
from tests.test_paper import D1, D2, D3, Rig, T, crash_run, use_strategy

# --------------------------------------------------------------------------- from records

DAYS = [dt.date(2024, 1, d) for d in (2, 3, 4, 5)]  # sessions; the first decision is on DAYS[0]
AT = dt.datetime(2024, 1, 2, 13, 0, tzinfo=dt.UTC)


def _record(seq: int, kind: JournalKind, payload: dict[str, Any]) -> JournalRecord:
    return JournalRecord(seq=seq, at=AT, kind=kind, payload=payload)


def _fill(
    order_id: str, side: Side, quantity: int, price: float, day: int, second: int = 60
) -> dict[str, Any]:
    """A `fill` record's payload: a venue fill on DAYS[day], `second` seconds after 09:30."""
    date = DAYS[day]
    fill = Fill(
        order_id=order_id,
        kind="venue",
        signal_date=DAYS[day - 1],
        fill_date=date,
        price_date=date,
        instrument=Stock(symbol="AAPL"),
        side=side,
        quantity=quantity,
        reference_price=price,
        price=price,
        slippage_cost=0.0,
        commission=0.0,
        filled_at=dt.datetime.combine(date, dt.time(9, 30), MARKET_TZ)
        + dt.timedelta(seconds=second),
    )
    return {"fill": fill.model_dump(mode="json"), "cumulative_quantity": quantity}


def _decision(
    day: int,
    equity: float,
    fill_opens: dict[str, dict[str, float]] | None = None,
    benchmark_close: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session": DAYS[day + 1].isoformat() if day + 1 < len(DAYS) else "2024-01-08",
        "decided_on": DAYS[day].isoformat(),
        "orders": [],
        "equity": {"date": DAYS[day].isoformat(), "cash": equity, "market_value": 0.0,
                   "equity": equity, "n_positions": 0},
        "events": [{"date": DAYS[day].isoformat(), "kind": "signal_duplicate", "symbol": "AAPL",
                    "detail": ""}],
        "data_fingerprint": f"fp{day}",
    }  # fmt: skip
    if fill_opens is not None:
        payload["fill_opens"] = fill_opens
    if benchmark_close is not None:
        payload["benchmark_close"] = benchmark_close
    return payload


def _journal(*entries: tuple[JournalKind, dict[str, Any]]) -> list[JournalRecord]:
    return [_record(i, kind, payload) for i, (kind, payload) in enumerate(entries, start=1)]


def _report(
    rig: Rig, records: list[JournalRecord], through: dt.date, marks: dict[str, float] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return paper_report(
        records,
        rig.profile.recipe,
        profile_id="p-12345678",
        strategy="s",
        through=through,
        marks=marks or {},
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Rig:
    """Only its recipe is used here (starting cash 10,000, benchmark SPY)."""
    return Rig(tmp_path_factory.mktemp("recipe"))


def test_each_side_of_a_trade_is_measured_against_the_official_open(rig: Rig) -> None:
    # Bought in two pieces on DAYS[1] (open 100), sold on DAYS[2] (open 110).
    records = _journal(
        ("decision", _decision(0, 10_000.0, {}, 400.0)),
        ("fill", _fill("e1", "buy", 10, 101.0, 1, second=60)),
        ("fill", {**_fill("e1", "buy", 30, 100.5, 1, second=180), "cumulative_quantity": 40}),
        ("decision", _decision(1, 10_020.0, {DAYS[1].isoformat(): {"AAPL": 100.0}}, 404.0)),
        ("fill", _fill("x1", "sell", 40, 109.45, 2, second=5)),
        ("decision", _decision(2, 10_353.0, {DAYS[2].isoformat(): {"AAPL": 110.0}}, 396.0)),
    )
    trades, summary = _report(rig, records, DAYS[2])
    (trade,) = trades
    assert list(trade) == PAPER_TRADE_COLUMNS
    assert (trade["entry_date"], trade["exit_date"]) == ("2024-01-03", "2024-01-04")
    assert (trade["quantity"], trade["is_open"], trade["holding_days"]) == (40, False, 1)
    assert trade["entry_price"] == pytest.approx(100.625)
    assert trade["pnl"] == pytest.approx(40 * (109.45 - 100.625))
    # Weighted by shares: (10 * 60 + 30 * 180) / 40 seconds, and 100.625 against 100.
    assert trade["entry_open"] == 100.0
    assert trade["entry_seconds_after_open"] == pytest.approx(150.0)
    assert trade["entry_slippage_bps"] == pytest.approx(62.5)
    # A sell below the open is a cost, so it is positive too.
    assert trade["exit_open"] == 110.0
    assert trade["exit_seconds_after_open"] == pytest.approx(5.0)
    assert trade["exit_slippage_bps"] == pytest.approx(50.0)

    assert summary["paper"] == {
        "profile_id": "p-12345678",
        "benchmark_missing": None,
        "entries": {
            "fills": 2, "priced": 2, "slippage_bps": pytest.approx(62.5),
            "slippage_cost": pytest.approx(25.0), "avg_seconds_after_open": pytest.approx(150.0),
        },
        "exits": {
            "fills": 1, "priced": 1, "slippage_bps": pytest.approx(50.0),
            "slippage_cost": pytest.approx(22.0), "avg_seconds_after_open": pytest.approx(5.0),
        },
    }  # fmt: skip
    assert (summary["strategy"], summary["holdout"]) == ("s", False)
    assert (summary["start"], summary["end"], summary["sessions"]) == (
        "2024-01-02", "2024-01-04", 3,
    )  # fmt: skip
    assert summary["final_equity"] == 10_353.0
    assert summary["total_return"] == pytest.approx(0.0353)
    assert (summary["trade_count"], summary["closed_trades"], summary["win_rate"]) == (1, 1, 1.0)
    assert summary["events"] == {"signal_duplicate": 3}
    assert summary["data_fingerprint"] == "fp2"
    assert summary["benchmark"]["symbol"] == "SPY"
    assert summary["benchmark"]["total_return"] == pytest.approx(396.0 / 400.0 - 1)
    assert summary["benchmark"]["max_drawdown"] == pytest.approx(396.0 / 404.0 - 1)


def test_only_sessions_whose_bar_exists_are_reported(rig: Rig) -> None:
    # The morning of DAYS[2]: the decision is on DAYS[1], and today's fill has no bar yet.
    records = _journal(
        ("decision", _decision(0, 10_000.0, {})),
        ("fill", _fill("e1", "buy", 10, 101.0, 1)),
        ("decision", _decision(1, 10_010.0, {DAYS[1].isoformat(): {"AAPL": 100.0}})),
        ("fill", _fill("x1", "sell", 10, 103.0, 2)),
    )
    assert unpriced_fills(records, DAYS[1]) == []
    assert unpriced_fills(records, DAYS[2]) == [(DAYS[2], "AAPL")]
    trades, summary = _report(rig, records, DAYS[1], {"AAPL": 102.0})
    (trade,) = trades
    assert (trade["is_open"], trade["exit_date"], trade["exit_price"]) == (
        True,
        "2024-01-03",
        102.0,
    )
    assert (trade["exit_open"], trade["exit_slippage_bps"]) == (None, None)
    assert summary["paper"]["exits"]["fills"] == 0 and summary["open_trades"] == 1
    assert summary["benchmark"]["total_return"] is None  # no benchmark close on record
    assert summary["paper"]["benchmark_missing"] == (
        "no benchmark close on record for 2 of 2 sessions"
    )


def test_a_benchmark_over_part_of_the_run_is_not_reported(rig: Rig) -> None:
    # A journal begun before decisions carried the benchmark's close.
    records = _journal(
        ("decision", _decision(0, 10_000.0)),
        ("decision", _decision(1, 10_100.0, {}, 400.0)),
        ("decision", _decision(2, 10_200.0, {}, 440.0)),
    )
    _, summary = _report(rig, records, DAYS[2])
    assert summary["total_return"] == pytest.approx(0.02)  # the whole run
    assert summary["benchmark"] == {
        "symbol": "SPY", "total_return": None, "cagr": None, "max_drawdown": None, "sharpe": None,
    }  # fmt: skip
    assert summary["paper"]["benchmark_missing"] == (
        "no benchmark close on record for 1 of 3 sessions"
    )


def test_a_fill_without_a_journaled_open_is_counted_but_not_priced(rig: Rig) -> None:
    records = _journal(
        ("decision", _decision(0, 10_000.0, {})),
        ("fill", _fill("e1", "buy", 10, 101.0, 1)),
        ("decision", _decision(1, 10_010.0, {})),  # AAPL had no bar on DAYS[1]
    )
    assert unpriced_fills(records, DAYS[1]) == [(DAYS[1], "AAPL")]  # asked for again
    trades, summary = _report(rig, records, DAYS[1], {"AAPL": 102.0})
    assert [trades[0][c] for c in PAPER_TRADE_COLUMNS[len(TRADE_COLUMNS) :]] == [None] * 6
    assert summary["paper"]["entries"] == {
        "fills": 1, "priced": 0, "slippage_bps": None, "slippage_cost": None,
        "avg_seconds_after_open": None,
    }  # fmt: skip


def test_the_first_record_of_an_open_or_a_session_is_the_one_used(rig: Rig) -> None:
    records = _journal(
        ("decision", _decision(0, 10_000.0, {})),
        ("fill", _fill("e1", "buy", 10, 101.0, 1)),
        ("decision", _decision(1, 10_010.0, {DAYS[1].isoformat(): {"AAPL": 100.0}})),
        ("decision", _decision(2, 10_020.0, {DAYS[1].isoformat(): {"AAPL": 50.0}})),
        ("decision", {**_decision(2, 9_000.0, {}), "data_fingerprint": "second"}),
    )
    trades, summary = _report(rig, records, DAYS[2], {"AAPL": 102.0})
    assert trades[0]["entry_open"] == 100.0
    assert (summary["final_equity"], summary["data_fingerprint"]) == (10_020.0, "fp2")
    assert summary["events"] == {"signal_duplicate": 3} and summary["sessions"] == 3


def test_holding_days_need_both_fills_on_sessions_that_were_decided_on(rig: Rig) -> None:
    # Bought on DAYS[1], sold on DAYS[3]. The runner missed DAYS[2], so the decision
    # after the buy was on DAYS[2]: no decision is on DAYS[1], the session of the buy.
    records = _journal(
        ("decision", _decision(0, 10_000.0, {})),
        ("fill", _fill("e1", "buy", 10, 101.0, 1)),
        ("decision", _decision(2, 10_010.0, {})),
        ("fill", _fill("x1", "sell", 10, 103.0, 3)),
        ("decision", _decision(3, 10_020.0, {})),
    )
    (trade,) = _report(rig, records, DAYS[3])[0]
    assert (trade["is_open"], trade["holding_days"]) == (False, None)


def test_what_the_report_cannot_build_is_a_report_error(rig: Rig) -> None:
    decision: tuple[JournalKind, dict[str, Any]] = ("decision", _decision(0, 10_000.0, {}))
    with pytest.raises(ReportError, match="no decision"):
        _report(rig, [], DAYS[0])
    split = {"adjustment_id": "2024-01-03-AAPL-split", "quantity_delta": 10}
    with pytest.raises(ReportError, match="2024-01-03-AAPL-split changed a share count"):
        _report(rig, _journal(decision, ("adjustment", split)), DAYS[0])
    dividend = {"adjustment_id": "2024-01-03-AAPL-dividend", "cash_delta": 4.0}
    assert _report(rig, _journal(decision, ("adjustment", dividend)), DAYS[0])[0] == []
    held = _journal(decision, ("fill", _fill("e1", "buy", 10, 101.0, 1)))
    with pytest.raises(ReportError, match="AAPL is still held but has no close"):
        _report(rig, held, DAYS[1])
    with pytest.raises(ReportError, match="does not read as paper trading records"):
        _report(rig, _journal(("decision", {"decided_on": "2024-01-02"})), DAYS[0])
    with pytest.raises(ReportError, match="does not read as paper trading records"):
        unpriced_fills(_journal(("fill", {"cumulative_quantity": 1})), DAYS[1])


# --------------------------------------------------------------------------- from the runner


def _trades(rig: Rig) -> list[dict[str, str]]:
    with (rig.state / "trades.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == PAPER_TRADE_COLUMNS
        return list(reader)


def _summary(rig: Rig) -> dict[str, Any]:
    return json.loads((rig.state / "summary.json").read_text())


def _decisions(rig: Rig) -> list[dict[str, Any]]:
    return [r.payload for r in rig.records() if r.kind == "decision"]


def test_the_first_day_reports_the_session_decided_on_and_no_trades(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    assert _trades(rig) == []  # today's fill has no bar yet
    summary = _summary(rig)
    backtest = json.loads((rig.base.run_dir / "summary.json").read_text())
    assert list(summary) == [*backtest, "paper"]  # a backtest's schema, plus one block
    assert set(summary["benchmark"]) == set(backtest["benchmark"])
    assert (summary["start"], summary["end"], summary["sessions"]) == (
        T.isoformat(), T.isoformat(), 1,
    )  # fmt: skip
    assert (summary["final_equity"], summary["trade_count"]) == (10_000.0, 0)
    assert (summary["strategy"], summary["risk_preset"]) == (backtest["strategy"], "balanced")
    assert summary["paper"]["profile_id"] == rig.profile_id
    assert summary["paper"]["entries"]["fills"] == 0
    (decision,) = _decisions(rig)
    assert decision["fill_opens"] == {}
    assert decision["benchmark_close"] == rig.close("SPY", T)
    assert rig.alerts.alerts == []


def test_the_next_day_measures_yesterdays_fill_against_the_official_open(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    quantity = rig.run().orders[0].quantity
    rig.morning(D2)
    rig.run()
    official = rig.close("SPY", D1) - 0.1  # the fake data source's open
    price = rig.close("SPY", T)  # the fake venue filled at the last close
    assert _decisions(rig)[1]["fill_opens"] == {D1.isoformat(): {"SPY": official}}
    assert rig.base.fetcher.raw_requests == [(("SPY",), D1, D1)]  # asked for once, unadjusted
    (trade,) = _trades(rig)
    assert (trade["symbol"], trade["entry_date"], trade["exit_date"]) == (
        "SPY", D1.isoformat(), D1.isoformat(),
    )  # fmt: skip
    assert (trade["quantity"], trade["is_open"], trade["exit_reason"]) == (
        str(quantity), "True", "open",
    )  # fmt: skip
    assert float(trade["entry_price"]) == price
    assert float(trade["exit_price"]) == rig.close("SPY", D1)  # marked at the close of t
    assert float(trade["entry_open"]) == official
    assert 0 <= float(trade["entry_seconds_after_open"]) < 60
    bps = (price / official - 1) * 10_000
    assert float(trade["entry_slippage_bps"]) == pytest.approx(bps)
    assert (trade["exit_open"], trade["exit_seconds_after_open"], trade["exit_slippage_bps"]) == (
        "", "", "",
    )  # fmt: skip

    summary = _summary(rig)
    assert (summary["sessions"], summary["open_trades"], summary["closed_trades"]) == (2, 1, 0)
    assert summary["final_equity"] == float(rig.equity_rows()[-1]["equity"])
    assert summary["final_equity"] == pytest.approx(10_000.0 + float(trade["pnl"]))
    assert summary["benchmark"]["total_return"] == pytest.approx(
        rig.close("SPY", D1) / rig.close("SPY", T) - 1
    )
    entries = summary["paper"]["entries"]
    assert (entries["fills"], entries["priced"]) == (1, 1)
    assert entries["slippage_bps"] == pytest.approx(bps)
    assert entries["slippage_cost"] == pytest.approx(quantity * (price - official))
    assert summary["paper"]["exits"]["fills"] == 0
    assert rig.alerts.alerts == []


def test_trades_over_several_days_add_up_to_the_equity_curve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, Churn)
    day = D1
    for _ in range(12):
        rig.morning(day)
        assert rig.run().outcome == "traded"
        day += dt.timedelta(days=3 if day.weekday() == 4 else 1)
    trades, summary = _trades(rig), _summary(rig)
    closed = [t for t in trades if t["is_open"] == "False"]
    assert len(closed) >= 2 and len(trades) > len(closed)
    for trade in closed:
        assert (
            trade["exit_open"] and trade["exit_seconds_after_open"] and trade["exit_slippage_bps"]
        )
        assert int(trade["holding_days"]) >= 1
    assert all(t["entry_open"] and t["entry_slippage_bps"] for t in trades)
    # Realized plus unrealized P&L is the change in equity: nothing is lost between the
    # journal's fills and its equity points.
    total = sum(float(t["pnl"]) for t in trades)
    assert total == pytest.approx(summary["final_equity"] - 10_000.0)
    assert summary["strategy"] == "test_churn" and summary["sessions"] == 12
    paper_block = summary["paper"]
    assert paper_block["exits"]["fills"] == paper_block["exits"]["priced"] == len(closed)
    assert paper_block["entries"]["fills"] == paper_block["entries"]["priced"] == len(trades)
    assert [a for a in rig.alerts.alerts if a.severity != "info"] == []


def test_a_resumed_day_writes_the_same_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    rig.morning(D2)
    rig.run()
    expected = [(rig.state / name).read_bytes() for name in ("trades.csv", "summary.json")]
    decided = [i for i, k in enumerate(rig.kinds(), start=1) if k == "decision"][1]

    rig.activate()
    rig.run()
    rig.morning(D2)
    crash_run(rig, monkeypatch, decided + 1, after=False)  # decided, then died
    fetcher = rig.base.fetcher
    requests = len(fetcher.requests), len(fetcher.raw_requests)
    report = rig.run()
    assert (report.outcome, report.resumed) == ("traded", True)
    # Built without loading bars or asking for the opens again.
    assert (len(fetcher.requests), len(fetcher.raw_requests)) == requests
    assert [(rig.state / n).read_bytes() for n in ("trades.csv", "summary.json")] == expected


def test_a_fill_is_measured_against_the_open_as_it_printed_not_an_adjusted_one(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    quantity = rig.run().orders[0].quantity
    # SPY goes ex-dividend on D2. By the morning of D2 the vendor has already adjusted
    # every earlier bar, the one of D1 (yesterday's fill) included, by about 1%.
    unadjusted = rig.close("SPY", D1) - 0.1
    rig.base.fetcher.scale_before["SPY"] = (D2, 0.99)
    rig.morning(D2)
    assert rig.run().outcome == "traded"
    adjusted = rig.base.fetcher.price("SPY", D1) - 0.1
    assert adjusted == pytest.approx(unadjusted - 0.01 * (unadjusted + 0.1))  # what the cache holds
    (trade,) = _trades(rig)
    assert float(trade["exit_price"]) == rig.base.fetcher.price("SPY", D1)  # marks stay adjusted
    assert _decisions(rig)[1]["fill_opens"] == {D1.isoformat(): {"SPY": unadjusted}}
    assert float(trade["entry_open"]) == unadjusted
    price = float(trade["entry_price"])
    assert float(trade["entry_slippage_bps"]) == pytest.approx((price / unadjusted - 1) * 10_000)
    cost = _summary(rig)["paper"]["entries"]["slippage_cost"]
    assert cost == pytest.approx(quantity * (price - unadjusted))


def test_opens_that_cannot_be_fetched_are_a_warning_and_asked_for_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    fetcher = rig.base.fetcher
    with monkeypatch.context() as patched:

        def down(*args: object) -> None:
            raise DataError("Alpaca bars request failed")

        patched.setattr(fetcher, "fetch_raw", down)
        rig.morning(D2)
        report = rig.run()
    assert report.outcome == "traded" and not rig.kill_switch().engaged()
    assert [(a.severity, a.kind) for a in report.alerts] == [("warn", "report")]
    assert "official opens" in report.alerts[0].message
    assert _decisions(rig)[1]["fill_opens"] == {}
    (trade,) = _trades(rig)
    assert trade["entry_open"] == "" and _summary(rig)["paper"]["entries"]["priced"] == 0
    # The next decision asks again. Unadjusted opens do not change, so late is as good.
    rig.alerts.alerts.clear()
    rig.morning(D3)
    assert rig.run().outcome == "traded" and rig.alerts.alerts == []
    assert fetcher.raw_requests == [(("SPY",), D1, D1)]
    assert _decisions(rig)[2]["fill_opens"] == {D1.isoformat(): {"SPY": rig.close("SPY", D1) - 0.1}}
    assert float(_trades(rig)[0]["entry_open"]) == rig.close("SPY", D1) - 0.1


def test_a_later_readjustment_of_the_bars_does_not_change_a_measured_fill(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    rig.morning(D2)
    rig.run()
    (before,) = _trades(rig)
    # The vendor re-adjusts SPY's history: every bar before D2 is halved, D1's open included.
    (rig.profile.recipe.data.cache_dir / "SPY.parquet").unlink()
    rig.base.fetcher.scale_before["SPY"] = (D2, 0.5)
    rig.morning(D3)
    assert rig.run().outcome == "traded"
    (after,) = _trades(rig)
    for column in ("entry_open", "entry_seconds_after_open", "entry_slippage_bps"):
        assert after[column] == before[column]
    assert "fill_opens" in _decisions(rig)[2] and _decisions(rig)[2]["fill_opens"] == {}


def test_a_journal_the_report_cannot_be_built_from_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    before = (rig.state / "summary.json").read_bytes()

    def refuse(*args: object, **kw: object) -> None:
        raise ReportError("adjustment 2026-07-02-SPY-split changed a share count")

    monkeypatch.setattr(paper, "paper_report", refuse)
    rig.morning(D2)
    report = rig.run()
    assert report.outcome == "traded" and rig.kinds()[-1] == "day_complete"
    assert [(a.severity, a.kind) for a in report.alerts] == [("warn", "report")]
    assert "were not updated" in report.alerts[0].message
    assert (rig.state / "summary.json").read_bytes() == before  # yesterday's file stays
    assert not rig.kill_switch().engaged()
