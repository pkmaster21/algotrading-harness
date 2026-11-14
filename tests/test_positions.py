"""Ledger (the harness's book) and reconciliation against the venue."""

from __future__ import annotations

import datetime as dt
import itertools
import os
import random
from pathlib import Path
from typing import Any

import pytest

from harness.broker.venue import (
    VenueAccount,
    VenueOrder,
    VenueOrderState,
    VenuePosition,
    VenueUnavailable,
)
from harness.journal import Journal, JournalError, read_journal
from harness.models import MARKET_TZ, AccountState, Fill, OptionContract, Side, Stock
from harness.positions import (
    Ledger,
    LedgerAdjustment,
    LedgerError,
    LedgerSnapshot,
    reconcile,
    reconcile_venue,
)
from tests.fake_venue import FakeVenue, at

DAYS = [dt.date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
NOW = dt.datetime(2024, 1, 3, 14, 50, tzinfo=dt.UTC)


def fill(
    order_id: str,
    side: Side,
    quantity: int,
    price: float,
    day: int = 1,
    *,
    symbol: str = "AAPL",
    commission: float = 0.0,
) -> Fill:
    return Fill(
        order_id=order_id,
        kind="venue",
        signal_date=DAYS[day - 1],
        fill_date=DAYS[day],
        price_date=DAYS[day],
        instrument=Stock(symbol=symbol),
        side=side,
        quantity=quantity,
        reference_price=price,
        price=price,
        slippage_cost=0.0,
        commission=commission,
        filled_at=dt.datetime.combine(DAYS[day], dt.time(9, 31), MARKET_TZ),
    )


_ids = itertools.count(1)


def adjust(**kw: Any) -> LedgerAdjustment:
    defaults = {
        "adjustment_id": f"adj-{next(_ids)}",
        "as_of": DAYS[2],
        "operator": "david",
        "note": "test",
    }
    return LedgerAdjustment(**{**defaults, **kw})


def check_identity(ledger: Ledger, marks: dict[str, float]) -> None:
    """Realized plus unrealized P&L always equals the change in equity."""
    pnl = ledger.pnl(marks, DAYS[-1])
    assert pnl.realized + pnl.unrealized == pytest.approx(pnl.equity - 10_000.0)
    assert pnl.equity == pytest.approx(pnl.cash + pnl.market_value)


# --------------------------------------------------------------------------- fills


def test_round_trip_with_commissions() -> None:
    ledger = Ledger(10_000.0)
    assert ledger.apply_fill(fill("e1", "buy", 10, 100.0, commission=1.0), 10)
    account = ledger.account()
    assert account.cash == pytest.approx(10_000 - 1_000 - 1.0)
    position = account.positions["AAPL"]
    assert (position.quantity, position.avg_price) == (10, 100.0)
    assert (position.entry_date, position.entry_order_id) == (DAYS[1], "e1")
    assert ledger.realized == 0.0
    pnl = ledger.pnl({"AAPL": 105.0}, DAYS[1], prior_equity=10_000.0)
    assert pnl.unrealized == pytest.approx(1_050 - 1_001)  # the buy commission is a cost
    assert pnl.day_change == pytest.approx(49.0)

    assert ledger.apply_fill(fill("x1", "sell", 10, 110.0, day=2, commission=2.0), 10)
    assert ledger.account().positions == {}
    assert ledger.realized == pytest.approx(1_100 - 2.0 - 1_001)
    assert ledger.account().cash == pytest.approx(10_000 + 97.0)
    check_identity(ledger, {})


def test_entry_filled_in_pieces_keeps_the_first_date_and_averages_the_price() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 3, 100.0), 3)
    ledger.apply_fill(fill("e1", "buy", 7, 101.0), 10)
    position = ledger.account().positions["AAPL"]
    assert position.quantity == 10
    assert position.avg_price == pytest.approx(100.7)
    assert (position.entry_date, position.entry_order_id) == (DAYS[1], "e1")
    check_identity(ledger, {"AAPL": 99.0})


def test_partial_exits_realize_their_share_of_the_cost() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0, commission=5.0), 10)  # cost 1,005
    ledger.apply_fill(fill("x1", "sell", 4, 110.0, day=2), 4)
    assert ledger.realized == pytest.approx(440 - 402.0)  # 4/10 of the cost
    assert ledger.account().positions["AAPL"].quantity == 6
    assert ledger.account().positions["AAPL"].avg_price == 100.0  # unchanged by a sell
    pnl = ledger.pnl({"AAPL": 120.0}, DAYS[2])
    assert pnl.unrealized == pytest.approx(720 - 603.0)
    check_identity(ledger, {"AAPL": 120.0})
    ledger.apply_fill(fill("x2", "sell", 6, 105.0, day=3, commission=1.0), 6)
    assert ledger.realized == pytest.approx(38.0 + 630 - 1.0 - 603.0)
    check_identity(ledger, {})


def test_the_same_fill_seen_twice_is_applied_once() -> None:
    ledger = Ledger(10_000.0)
    first = fill("e1", "buy", 4, 100.0)
    second = fill("e1", "buy", 6, 101.0)
    assert ledger.apply_fill(first, 4)
    assert ledger.apply_fill(second, 10)
    before = ledger.snapshot()
    assert not ledger.apply_fill(first, 4)  # an old poll result
    assert not ledger.apply_fill(second, 10)  # the same one again after a restart
    assert ledger.snapshot() == before


@pytest.mark.parametrize(
    ("bad", "cumulative", "match"),
    [
        (fill("e1", "buy", 4, 100.0), 9, "does not follow the 3 already applied"),  # skipped 2
        (fill("e1", "buy", 4, 100.0), 5, "does not follow"),  # overlaps
        (fill("x1", "sell", 4, 100.0, day=2), 4, "exceeds the 3 held"),
        (fill("x1", "sell", 1, 100.0, day=2, symbol="MSFT"), 1, "exceeds the 0 held"),
        (fill("e2", "buy", 4, 100.0, day=2), 4, "opened by e1"),
    ],
)
def test_fill_that_does_not_fit_is_refused_and_changes_nothing(
    tmp_path: Path, bad: Fill, cumulative: int, match: str
) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    ledger = Ledger.open(journal, 10_000.0)
    ledger.apply_fill(fill("e1", "buy", 3, 100.0), 3)
    before, seq = ledger.snapshot(), journal.last_seq
    with pytest.raises(LedgerError, match=match):
        ledger.apply_fill(bad, cumulative)
    assert ledger.snapshot() == before
    assert journal.last_seq == seq  # a refused fill never reaches the journal


def test_only_stock_fills() -> None:
    option = fill("e1", "buy", 1, 5.0).model_copy(
        update={
            "instrument": OptionContract(
                underlying="AAPL", expiry=DAYS[4], strike=100, right="call"
            )
        }
    )
    with pytest.raises(LedgerError, match="stock positions only"):
        Ledger(10_000.0).apply_fill(option, 1)


# --------------------------------------------------------------------------- adjustments


def test_dividend_and_fee_are_realized() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    ledger.apply_adjustment(adjust(kind="dividend", symbol="AAPL", cash_delta=2.4))
    ledger.apply_adjustment(adjust(kind="fee", cash_delta=-0.4))
    assert ledger.realized == pytest.approx(2.0)
    assert ledger.account().cash == pytest.approx(9_002.0)
    assert ledger.account().positions["AAPL"].quantity == 10
    check_identity(ledger, {"AAPL": 100.0})


def test_split_adds_shares_at_no_cost() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    ledger.apply_adjustment(adjust(kind="split", symbol="AAPL", quantity_delta=10))
    position = ledger.account().positions["AAPL"]
    assert (position.quantity, position.avg_price) == (20, 50.0)
    assert (position.entry_date, position.entry_order_id) == (DAYS[1], "e1")
    assert ledger.realized == 0.0
    check_identity(ledger, {"AAPL": 50.0})


def test_reverse_split_removes_shares_and_keeps_the_cost() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 100, 10.0, commission=5.0), 100)  # cost 1,005
    ledger.apply_adjustment(adjust(kind="split", symbol="AAPL", quantity_delta=-90))
    position = ledger.account().positions["AAPL"]
    assert (position.quantity, position.avg_price) == (10, 100.0)
    assert (position.entry_date, position.entry_order_id) == (DAYS[1], "e1")
    assert ledger.realized == 0.0  # nothing was sold
    assert ledger.snapshot().cost_basis == {"AAPL": 1_005.0}
    assert ledger.pnl({"AAPL": 100.0}, DAYS[2]).unrealized == pytest.approx(-5.0)
    check_identity(ledger, {"AAPL": 100.0})
    # Selling afterwards realizes against the full original cost.
    ledger.apply_fill(fill("x1", "sell", 10, 100.0, day=3), 10)
    assert ledger.realized == pytest.approx(-5.0)
    check_identity(ledger, {})


def test_forward_split_keeps_the_cost() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0, commission=5.0), 10)
    ledger.apply_adjustment(adjust(kind="split", symbol="AAPL", quantity_delta=30))  # 4 for 1
    assert ledger.account().positions["AAPL"].quantity == 40
    assert ledger.account().positions["AAPL"].avg_price == 25.0
    assert ledger.realized == 0.0
    assert ledger.snapshot().cost_basis == {"AAPL": 1_005.0}
    check_identity(ledger, {"AAPL": 25.0})


def test_split_cash_for_a_fractional_share_is_realized() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 15, 10.0), 15)
    # 1 for 10: 1.5 shares, the half share is paid out in cash.
    ledger.apply_adjustment(
        adjust(kind="split", symbol="AAPL", quantity_delta=-14, cash_delta=50.0)
    )
    assert ledger.account().positions["AAPL"].quantity == 1
    assert ledger.account().cash == pytest.approx(10_000 - 150 + 50)
    assert ledger.realized == pytest.approx(50.0)
    assert ledger.snapshot().cost_basis == {"AAPL": 150.0}
    check_identity(ledger, {"AAPL": 100.0})


def test_split_needs_a_position_before_and_after() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    before = ledger.snapshot()
    with pytest.raises(LedgerError, match="needs a position before and after"):
        ledger.apply_adjustment(adjust(kind="split", symbol="MSFT", quantity_delta=5))
    with pytest.raises(LedgerError, match="needs a position before and after"):
        ledger.apply_adjustment(adjust(kind="split", symbol="AAPL", quantity_delta=-10))
    assert ledger.snapshot() == before
    with pytest.raises(ValueError, match="a split changes a quantity"):
        adjust(kind="split", symbol="AAPL", cash_delta=5.0)
    with pytest.raises(ValueError, match="a split does not change the cost basis"):
        adjust(kind="split", symbol="AAPL", quantity_delta=10, cost_basis_delta=100.0)


def test_the_same_adjustment_entered_twice_is_applied_once(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    ledger = Ledger.open(journal, 10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    dividend = adjust(adjustment_id="2024-01-04-AAPL-div", kind="dividend", cash_delta=2.4)
    assert ledger.apply_adjustment(dividend)
    before, seq = ledger.snapshot(), journal.last_seq
    assert not ledger.apply_adjustment(dividend)
    assert not ledger.apply_adjustment(dividend.model_copy())
    assert ledger.snapshot() == before and journal.last_seq == seq
    assert ledger.realized == pytest.approx(2.4)
    # The same id for something else is a mistake, not a repeat.
    with pytest.raises(LedgerError, match="already used for a different adjustment"):
        ledger.apply_adjustment(dividend.model_copy(update={"cash_delta": 24.0}))
    assert ledger.snapshot() == before and journal.last_seq == seq
    # Still a repeat after a restart.
    journal.close()
    replayed = Ledger.open(Journal(path), 10_000.0)
    assert not replayed.apply_adjustment(dividend)
    assert replayed.snapshot() == before
    with pytest.raises(ValueError):
        adjust(adjustment_id="", kind="fee", cash_delta=-1.0)


def test_adjustment_whose_write_failed_but_reached_the_disk_is_not_doubled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "j.jsonl"
    journal = Journal(path)
    ledger = Ledger.open(journal, 10_000.0)
    dividend = adjust(kind="dividend", cash_delta=2.4)

    def disk_error(fd: int) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", disk_error)
        with pytest.raises(JournalError, match="write failed"):
            ledger.apply_adjustment(dividend)
    assert ledger.realized == 0.0  # not applied in memory
    journal.close()
    # The operator reopens and enters it again. The bytes had reached the file.
    reopened = Ledger.open(Journal(path), 10_000.0)
    assert reopened.realized == pytest.approx(2.4)
    assert not reopened.apply_adjustment(dividend)
    assert reopened.realized == pytest.approx(2.4)


def test_journal_with_a_repeated_adjustment_does_not_replay(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.jsonl")
    Ledger.open(journal, 10_000.0)
    payload = adjust(kind="fee", cash_delta=-1.0).model_dump(mode="json")
    journal.append("adjustment", payload)
    journal.append("adjustment", payload)
    journal.close()
    with pytest.raises(LedgerError, match="record 3 does not replay: adjustment was already"):
        Ledger.open(Journal(journal.path), 10_000.0)


def test_delisting_cash_out_realizes_the_difference_to_cost() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    ledger.apply_adjustment(
        adjust(kind="delisting", symbol="AAPL", quantity_delta=-10, cash_delta=800.0)
    )
    assert ledger.account().positions == {}
    assert ledger.realized == pytest.approx(-200.0)
    assert ledger.account().cash == pytest.approx(9_800.0)
    check_identity(ledger, {})


def test_manual_adjustment_can_open_a_position_with_a_cost() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_adjustment(
        adjust(
            kind="manual", symbol="MSFT", quantity_delta=5, cash_delta=-1_500.0,
            cost_basis_delta=1_500.0,
        )
    )  # fmt: skip
    position = ledger.account().positions["MSFT"]
    assert (position.quantity, position.avg_price, position.entry_date) == (5, 300.0, DAYS[2])
    assert ledger.realized == 0.0
    check_identity(ledger, {"MSFT": 310.0})
    ledger.apply_adjustment(adjust(kind="manual", symbol="MSFT", quantity_delta=-2, cash_delta=640))
    assert ledger.account().positions["MSFT"].quantity == 3
    assert ledger.realized == pytest.approx(640 - 600)
    check_identity(ledger, {"MSFT": 310.0})


def test_adjustments_that_do_not_fit_are_refused() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    with pytest.raises(LedgerError, match="only 10 are held"):
        ledger.apply_adjustment(adjust(kind="bust", symbol="AAPL", quantity_delta=-11))
    with pytest.raises(LedgerError, match="only 0 are held"):
        ledger.apply_adjustment(adjust(kind="bust", symbol="MSFT", quantity_delta=-1))
    with pytest.raises(LedgerError, match="positive cost_basis_delta"):
        ledger.apply_adjustment(adjust(kind="manual", symbol="MSFT", quantity_delta=1))
    for bad in (
        {"kind": "manual"},  # changes nothing
        {"kind": "manual", "quantity_delta": 1},  # no symbol
        {"kind": "manual", "symbol": "AAPL", "quantity_delta": -1, "cost_basis_delta": 5.0},
        {"kind": "manual", "cash_delta": 1.0, "note": ""},
        {"kind": "manual", "cash_delta": 1.0, "operator": ""},
        {"kind": "gift", "cash_delta": 1.0},
    ):
        with pytest.raises(ValueError):
            adjust(**bad)


def test_pnl_needs_a_mark_for_every_position() -> None:
    ledger = Ledger(10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0), 10)
    with pytest.raises(LedgerError, match="no mark for held symbols: AAPL"):
        ledger.pnl({}, DAYS[1])
    assert ledger.pnl({"AAPL": 100.0}, DAYS[1]).day_change is None


# --------------------------------------------------------------------------- journal and replay


def _random_history(ledger: Ledger, rng: random.Random) -> None:
    symbols = ["AAPL", "MSFT", "NVDA"]
    orders = 0
    for _ in range(60):
        symbol = rng.choice(symbols)
        held = ledger.account().positions.get(symbol)
        roll = rng.random()
        if held is None:
            orders += 1
            total = rng.randint(1, 30)
            first = rng.randint(1, total)
            price = round(rng.uniform(20, 400), 2)
            ledger.apply_fill(fill(f"e{orders}", "buy", first, price, symbol=symbol), first)
            if first < total:
                rest = fill(f"e{orders}", "buy", total - first, price + 0.5, symbol=symbol)
                ledger.apply_fill(rest, total)
        elif roll < 0.6:
            orders += 1
            quantity = rng.randint(1, held.quantity)
            price = round(rng.uniform(20, 400), 2)
            sale = fill(f"x{orders}", "sell", quantity, price, day=2, symbol=symbol, commission=0.3)
            ledger.apply_fill(sale, quantity)
        elif roll < 0.8:
            ledger.apply_adjustment(adjust(kind="dividend", symbol=symbol, cash_delta=1.25))
        elif roll < 0.9 or held.quantity < 2:
            ledger.apply_adjustment(
                adjust(kind="split", symbol=symbol, quantity_delta=held.quantity)
            )
        else:  # reverse split
            ledger.apply_adjustment(
                adjust(kind="split", symbol=symbol, quantity_delta=-(held.quantity // 2))
            )


def test_replaying_the_journal_rebuilds_the_same_ledger(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    with Journal(path) as journal:
        ledger = Ledger.open(journal, 10_000.0)
        _random_history(ledger, random.Random(11))
    assert len(read_journal(path)) > 60
    replayed = Ledger.open(Journal(path), 10_000.0)
    assert replayed.snapshot() == ledger.snapshot()  # exact, not approximate
    marks = {s: 100.0 for s in ("AAPL", "MSFT", "NVDA")}
    assert replayed.pnl(marks, DAYS[4]) == ledger.pnl(marks, DAYS[4])
    check_identity(replayed, marks)
    # A fill the first ledger already saw is still a duplicate for the replayed one.
    last = next(r for r in reversed(read_journal(path)) if r.kind == "fill")
    again = Fill.model_validate(last.payload["fill"])
    assert not replayed.apply_fill(again, last.payload["cumulative_quantity"])  # type: ignore[arg-type]
    # So is an adjustment.
    last = next(r for r in reversed(read_journal(path)) if r.kind == "adjustment")
    assert not replayed.apply_adjustment(LedgerAdjustment.model_validate(last.payload))
    assert replayed.snapshot() == ledger.snapshot()


def test_fill_journaled_just_before_a_crash_is_in_the_replayed_ledger(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    Ledger.open(journal, 10_000.0)
    # The process died after the journal write and before the in-memory update.
    journal.append(
        "fill",
        {"fill": fill("e1", "buy", 5, 100.0).model_dump(mode="json"), "cumulative_quantity": 5},
    )
    journal.close()
    assert Ledger.open(Journal(path), 10_000.0).account().positions["AAPL"].quantity == 5


def test_journal_is_opened_once_with_its_starting_cash(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    Ledger.open(Journal(path), 10_000.0)
    Ledger.open(Journal(path), 10_000.0)
    assert [r.kind for r in read_journal(path)] == ["ledger_open"]
    with pytest.raises(LedgerError, match="starting cash 10000.0, not 25000.0"):
        Ledger.open(Journal(path), 25_000.0)


def test_journal_that_does_not_replay_is_an_error(tmp_path: Path) -> None:
    orphan = Journal(tmp_path / "orphan.jsonl")
    orphan.append("adjustment", adjust(kind="fee", cash_delta=-1.0).model_dump(mode="json"))
    with pytest.raises(LedgerError, match="no ledger_open"):
        Ledger.open(orphan, 10_000.0)

    journal = Journal(tmp_path / "j.jsonl")
    Ledger.open(journal, 10_000.0)
    oversell = fill("x1", "sell", 5, 100.0, day=2).model_dump(mode="json")
    journal.append("fill", {"fill": oversell, "cumulative_quantity": 5})
    journal.close()
    with pytest.raises(LedgerError, match="record 2 does not replay: sell of 5 AAPL exceeds"):
        Ledger.open(Journal(journal.path), 10_000.0)

    garbled = Journal(tmp_path / "g.jsonl")
    Ledger.open(garbled, 10_000.0)
    garbled.append("fill", {"fill": {"order_id": "e1"}, "cumulative_quantity": 1})
    garbled.close()
    with pytest.raises(LedgerError, match="record 2 does not replay"):
        Ledger.open(Journal(garbled.path), 10_000.0)


def test_snapshot_file_is_written_whole(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "journal.jsonl")
    ledger = Ledger.open(journal, 10_000.0)
    ledger.apply_fill(fill("e1", "buy", 10, 100.0, commission=1.0), 10)
    path = tmp_path / "state" / "ledger.json"
    ledger.write_snapshot(path)
    ledger.apply_fill(fill("x1", "sell", 4, 110.0, day=2), 4)
    ledger.write_snapshot(path)  # replaces the old one
    loaded = LedgerSnapshot.model_validate_json(path.read_text())
    assert loaded == ledger.snapshot()
    assert (loaded.journal_seq, loaded.positions["AAPL"].quantity) == (3, 6)
    assert loaded.cost_basis == {"AAPL": pytest.approx(600.6)}
    assert [p.name for p in path.parent.iterdir()] == ["ledger.json"]  # no temp file left


# --------------------------------------------------------------------------- reconcile


def _book(cash: float = 9_000.0, **holdings: int) -> AccountState:
    """The ledger's side of a comparison: cash and {symbol: quantity}."""
    probe = Ledger(1e9)
    for n, (symbol, quantity) in enumerate(holdings.items()):
        probe.apply_fill(fill(f"e{n}", "buy", quantity, 100.0, symbol=symbol), quantity)
    return AccountState(cash=cash, positions=probe.account().positions)


def _account(cash: float = 9_000.0) -> VenueAccount:
    return VenueAccount(
        account_id="PA1", cash=cash, buying_power=cash, equity=cash + 1_000,
        trading_blocked=False, shorting_enabled=False, margin_multiplier=1.0,
    )  # fmt: skip


def _held(symbol: str, quantity: int) -> VenuePosition:
    return VenuePosition(symbol=symbol, quantity=quantity, avg_entry_price=100.0)


def _open(client_order_id: str) -> VenueOrderState:
    return VenueOrderState(
        client_order_id=client_order_id, venue_order_id="v", status="pending", quantity=1,
        filled_quantity=0, filled_avg_price=None, updated_at=NOW,
    )  # fmt: skip


def _reconcile(**kw: Any):  # type: ignore[no-untyped-def]
    args: dict[str, Any] = {
        "ledger": _book(AAPL=10),
        "expected_open": [],
        "account": _account(),
        "positions": [_held("AAPL", 10)],
        "open_orders": [],
        "cash_offset": 0.0,
        "cash_tolerance": 1.0,
        "at": NOW,
    }
    return reconcile(**{**args, **kw})


def test_matching_books_are_ok() -> None:
    report = _reconcile()
    assert (report.severity, report.reasons, report.position_diffs) == ("ok", (), ())
    assert report.cash_diff == 0.0 and report.at == NOW
    report = _reconcile(expected_open=["a", "b"], open_orders=[_open("b"), _open("a")])
    assert report.severity == "ok"


@pytest.mark.parametrize(
    ("changes", "severity", "reason"),
    [
        # cash
        ({"account": _account(9_000.004)}, "ok", None),  # under half a cent: float noise
        ({"account": _account(9_000.60)}, "warn", "cash differs by +0.60, within tolerance"),
        ({"account": _account(8_999.00)}, "warn", "cash differs by -1.00, within tolerance"),
        ({"account": _account(9_001.01)}, "stop", "cash differs by +1.01 (tolerance 1.00)"),
        ({"account": _account(8_000.0)}, "stop", "cash differs by -1000.00"),
        ({"account": _account(59_000.0), "cash_offset": 50_000.0}, "ok", None),
        ({"account": _account(59_000.0), "cash_offset": 49_000.0}, "stop", "cash differs"),
        # positions
        ({"positions": [_held("AAPL", 9)]}, "stop", "AAPL: ledger holds 10, venue holds 9"),
        ({"positions": []}, "stop", "AAPL: ledger holds 10, venue holds 0"),
        (
            {"positions": [_held("AAPL", 10), _held("MSFT", 3)]},
            "stop",
            "MSFT: ledger holds 0, venue holds 3",
        ),
        ({"positions": [_held("AAPL", -10)]}, "stop", "AAPL: the venue reports a short of 10"),
        # orders
        ({"open_orders": [_open("stray")]}, "stop", "order stray is open at the venue"),
        ({"expected_open": ["mine"]}, "stop", "order mine should be open"),
    ],
)
def test_each_difference_has_its_severity(
    changes: dict[str, Any], severity: str, reason: str | None
) -> None:
    report = _reconcile(**changes)
    assert report.severity == severity
    if reason is None:
        assert report.reasons == ()
    else:
        assert any(reason in r for r in report.reasons), report.reasons


def test_report_lists_every_difference() -> None:
    report = _reconcile(
        account=_account(9_000.50),
        positions=[_held("MSFT", -2)],
        open_orders=[_open("stray")],
        expected_open=["mine"],
    )
    assert report.severity == "stop"  # the small cash difference does not soften it
    assert [(d.symbol, d.ledger_quantity, d.venue_quantity) for d in report.position_diffs] == [
        ("AAPL", 10, 0),
        ("MSFT", 0, -2),
    ]
    assert (report.unexpected_orders, report.missing_orders) == (("stray",), ("mine",))
    assert report.cash_diff == pytest.approx(0.5)
    assert len(report.reasons) == 6


# --------------------------------------------------------------------------- against a venue


def test_ledger_fed_from_venue_fills_reconciles(tmp_path: Path) -> None:
    venue = FakeVenue(clock=at(DAYS[1], 9, 45), cash=10_000.0)
    venue.set_price("AAPL", 100.0)
    ledger = Ledger.open(Journal(tmp_path / "journal.jsonl"), 10_000.0)
    assert reconcile_venue(ledger.account(), [], venue, 0.0, 1.0).severity == "ok"

    order = VenueOrder(
        client_order_id="p-20240102-AAPL-e-a1", symbol="AAPL", side="buy", quantity=10,
        type="limit", limit_price=102.0, time_in_force="day",
    )  # fmt: skip
    state = venue.place(order)
    assert state.status == "filled" and state.filled_avg_price == 100.0
    # Before the fill reaches the ledger the two books differ.
    report = reconcile_venue(ledger.account(), [], venue, 0.0, 1.0)
    assert report.severity == "stop" and report.cash_diff == pytest.approx(-1_000.0)

    ledger.apply_fill(fill("p-20240102-AAPL-e", "buy", 10, 100.0), 10)
    report = reconcile_venue(ledger.account(), [], venue, 0.0, 1.0)
    assert (report.severity, report.reasons) == ("ok", ())
    assert report.at == venue.clock


def test_bad_venue_data_stops_with_a_plain_reason_not_a_traceback() -> None:
    class BadVenue(FakeVenue):
        def positions(self) -> list[VenuePosition]:
            return [VenuePosition(symbol="AAPL", quantity=10, avg_entry_price=0.0)]

    venue = BadVenue(clock=at(DAYS[1], 9, 45))
    report = reconcile_venue(Ledger(10_000.0).account(), [], venue, 0.0, 1.0)
    assert report.severity == "stop"
    (reason,) = report.reasons
    assert reason.startswith("venue data failed validation, stopping: VenuePosition")
    assert "avg_entry_price" in reason


def test_unreachable_venue_is_not_reported_as_a_reconciliation_result() -> None:
    venue = FakeVenue(clock=at(DAYS[1], 9, 45))
    venue.fail_next("positions")
    with pytest.raises(VenueUnavailable):
        reconcile_venue(Ledger(10_000.0).account(), [], venue, 0.0, 1.0)
