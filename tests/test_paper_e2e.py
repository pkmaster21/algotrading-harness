"""End to end: the frozen example strategy (buy and hold) run day after day against the
fake venue, the way the operator runs it against the paper account. One account, one
journal and one clock carry through seven calendar days, with a fault on every one of
them, and the operator acts only through the command line (`harness.cli`).

It follows the checklist for the real paper account (docs/slice2-design.md, PR 22) where
the fake venue can stand in for it: approve the entry, run a day twice, a day whose bar
is late, a trade made by hand, a process that dies mid-day and mid-open, a missed
session, flatten, and the way back in. `docs/paper-runbook.md` describes the same
situations for the operator.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import JsonValue

import harness.paper as paper
import harness.service as service
from harness.journal import Journal, JournalKind
from harness.paper import DayReport
from tests.fake_venue import OrderScript, at
from tests.test_cli import op
from tests.test_cli import run as cli
from tests.test_control import ME, _stamp, controls
from tests.test_orders import Crash
from tests.test_paper import D1, D2, D3, SATURDAY, Rig, T

MON, TUE, WED, THU = (dt.date(2026, 7, day) for day in (6, 7, 8, 9))
CASH_OFFSET = 2_000.0  # the account holds 12,000; the profile trades 10,000 of it

Capture = pytest.CaptureFixture[str]


class DiesAfter(Journal):
    """The runner's journal, set to kill the process once a record of one kind is
    written: the crash comes when the runner tries to write the record after it, so
    everything between the two (a request to the venue, for one) has happened and is not
    on record. Configured on the class, because the runner builds its own journal."""

    after_kind: ClassVar[str | None] = None
    _armed: ClassVar[bool] = False

    def append(self, kind: JournalKind, payload: Mapping[str, JsonValue]) -> int:
        if DiesAfter._armed:
            DiesAfter._armed, DiesAfter.after_kind = False, None
            raise Crash(f"before the {kind} record")
        seq = super().append(kind, payload)
        if kind == DiesAfter.after_kind:
            DiesAfter._armed = True
        return seq


def approving(rig: Rig) -> Callable[[float], None]:
    """A `sleep` during which the operator approves, from a second terminal and once,
    exactly what `pending` shows."""
    approved: list[str] = []

    def sleep(seconds: float) -> None:
        if not approved:
            for request in service.list_pending_approvals(
                execution=rig.execution, now=lambda: rig.venue.clock
            ):
                quoted = f"{request.order_id}={request.content_hash}"
                assert cli(rig, *op("approve", quoted)) == 0
                approved.append(request.order_id)
            for path in (rig.state / "inbox").glob("*.json"):
                _stamp(path, rig.venue.clock)
        rig.venue.advance(seconds)

    return sleep


def alerts(report: DayReport) -> list[tuple[str, str]]:
    return [(a.severity, a.kind) for a in report.alerts]


def orders(report: DayReport) -> list[tuple[str, str, str, int]]:
    return [(o.order.symbol, o.order.action, o.phase, o.filled_quantity) for o in report.orders]


def books_agree(rig: Rig) -> None:
    """The ledger (replayed from the journal) is the account: shares and cash."""
    view = service.get_positions(execution=rig.execution)
    assert {p.symbol: p.quantity for p in view.positions} == rig.held()
    assert view.cash == pytest.approx(rig.venue.cash - CASH_OFFSET)
    assert rig.venue.open_orders() == []


def test_seven_days_of_buy_and_hold_with_a_fault_on_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Capture
) -> None:
    rig = Rig(tmp_path, approval="entries")
    monkeypatch.setattr(paper, "Journal", DiesAfter)

    # ---- Wednesday: the entry waits for an approval. The venue does not answer the
    # first two account requests, and it accepts the order but the reply is lost.
    rig.venue.fail_next("account", times=2)
    rig.venue.script("SPY", OrderScript(timeout_after_accept=True))
    rig.sleep = approving(rig)
    report = rig.run()
    assert (report.outcome, report.decided_on, alerts(report)) == ("traded", T, [])
    (entry,) = report.orders
    quantity = entry.quantity
    assert orders(report) == [("SPY", "enter", "filled", quantity)]
    assert entry.approval == "approved"
    first_id = f"{entry.order_id}-a1"
    # The lost reply was answered by a lookup, not by sending again.
    assert rig.accepted() == [first_id] and len(rig.venue.place_attempts) == 1
    assert ("get_order", first_id) in rig.venue.calls
    books_agree(rig)
    # The same day again: nothing.
    assert rig.run().outcome == "already_done"
    assert len(rig.venue.place_attempts) == 1

    # ---- Thursday: yesterday's SPY bar has not arrived. No decision, however often
    # the day is run, and the day stays open. (The source has no bar on either run, so
    # this says nothing about a bar that arrives between two runs of one day.)
    rig.morning(D2)
    rig.base.fetcher.delisted["SPY"] = D1
    for _ in range(2):
        report = rig.run()
        assert (report.outcome, alerts(report)) == ("skipped_stale_data", [("warn", "stale_data")])
    assert rig.kinds().count("day_complete") == 1

    # ---- Friday: the bar is there. Someone bought a share of AAPL in the account by
    # hand, so reconciliation stops the day before anything is decided.
    rig.morning(D3)
    del rig.base.fetcher.delisted["SPY"]
    rig.venue.hold("AAPL", 1, 50.0)
    rig.venue.cash -= 50.0  # and paid for it: `accept` has cash to settle too
    report = rig.run()
    assert report.outcome == "killed"
    assert alerts(report) == [("warn", "missed_sessions"), ("critical", "kill_switch")]
    capsys.readouterr()
    assert cli(rig, "status") == 0
    out = capsys.readouterr().out
    assert "kill switch: ENGAGED" in out and "by auto:runner" in out
    assert "AAPL: ledger holds 0, venue holds 1" in out and "outcome killed" in out
    assert rig.run().outcome == "killed"  # refused until an operator resets it
    assert cli(rig, *op("reset-kill", "--confirm", rig.profile_id)) == 2  # not while they differ
    assert cli(rig, *op("accept", "--note", "bought by hand in the dashboard")) == 0
    assert cli(rig, *op("reset-kill", "--confirm", rig.profile_id)) == 0
    report = rig.run()
    assert (report.outcome, report.decided_on, report.orders) == ("traded", D2, ())
    # The accepted share has no fill behind it, so the trade report stops updating (the
    # books do not: see `books_agree`). The runner says so every day from here on.
    assert alerts(report) == [("warn", "missed_sessions"), ("warn", "report")]
    assert "changed a share count" in report.alerts[1].message
    books_agree(rig)

    # ---- Saturday: not a session.
    rig.venue.set_now(at(SATURDAY, 9, 15))
    report = rig.run()
    assert (report.outcome, alerts(report)) == ("no_session", [("info", "no_session")])

    # ---- Monday: the machine was off. Tuesday: the process dies right after the
    # decision is on record, and is started again.
    rig.morning(TUE)
    DiesAfter.after_kind = "decision"
    with pytest.raises(Crash):
        rig.run()
    assert rig.kinds()[-1] == "decision"
    report = rig.run()
    assert (report.outcome, report.resumed, report.decided_on) == ("traded", True, MON)
    assert report.orders == ()
    assert alerts(report) == [("warn", "missed_sessions"), ("warn", "report")]
    assert f"since {D3}: {MON}" in report.alerts[0].message
    assert rig.kinds().count("decision") == 3  # not decided a second time

    # ---- Wednesday: the operator flattens before the open instead of running the day.
    rig.morning(WED)
    capsys.readouterr()
    assert cli(rig, *op("flatten", "--confirm", rig.profile_id)) == 0
    out = capsys.readouterr().out
    assert f"sell SPY  {quantity}/{quantity}  filled" in out and "sell AAPL  1/1  filled" in out
    assert "The ledger holds nothing." in out
    sells = rig.venue.place_attempts[1:]
    assert sorted((o.side, o.symbol, o.quantity, o.time_in_force) for o in sells) == [
        ("sell", "AAPL", 1, "opg"),
        ("sell", "SPY", quantity, "opg"),
    ]
    assert rig.held() == {} and rig.kill_switch().engaged()
    report = rig.run()  # a runner started now would buy straight back: it is refused
    assert (report.outcome, alerts(report)) == ("killed", [("warn", "kill_switch")])
    assert cli(rig, *op("reset-kill", "--confirm", rig.profile_id)) == 0
    books_agree(rig)

    # ---- Thursday: flat, so buy and hold enters again. The process dies during the
    # open, after the venue accepted the order and before that is on record.
    rig.morning(THU)
    rig.sleep = approving(rig)
    DiesAfter.after_kind = "order_intent"
    with pytest.raises(Crash):
        rig.run()
    assert rig.kinds()[-1] == "order_intent" and len(rig.accepted()) == 4
    rig.sleep = None
    report = rig.run()
    assert (report.outcome, report.resumed, report.decided_on) == ("traded", True, WED)
    (again,) = report.orders
    assert orders(report) == [("SPY", "enter", "filled", again.quantity)]
    assert again.approval == "approved"  # the approval given before the crash stands
    assert alerts(report) == [("warn", "missed_sessions"), ("warn", "report")]

    # ---- The whole week: one venue order per decision or flatten sell, never two.
    placed = [o.client_order_id for o in rig.venue.place_attempts]
    assert placed == rig.accepted() and len(set(placed)) == 4
    assert rig.held() == {"SPY": again.quantity}
    books_agree(rig)
    assert report.reconcile_after is not None and report.reconcile_after.severity == "ok"
    assert not rig.kill_switch().engaged()
    # One equity point per close a decision was made on; days that never decided
    # (the late bar, the machine off, the flatten) have none.
    assert [row["date"] for row in rig.equity_rows()] == [d.isoformat() for d in (T, D2, MON, WED)]
    assert float(rig.equity_rows()[-1]["market_value"]) == 0.0  # flat at Wednesday's close
    assert rig.kinds().count("day_complete") == 4 and rig.kinds().count("decision") == 4
    # Everything the operator did is on record under their name.
    done = controls(rig)
    assert [c["command"] for c in done] == [
        "accept_broker_state",
        "reset_kill_switch",
        "flatten",
        "reset_kill_switch",
    ]
    assert {c["operator"] for c in done} == {ME}
    capsys.readouterr()
    assert cli(rig, "status") == 0
    out = capsys.readouterr().out
    assert "kill switch: not engaged" in out and f"session {THU}: complete" in out
    assert "last reconcile (after" in out and "problem:" not in out
