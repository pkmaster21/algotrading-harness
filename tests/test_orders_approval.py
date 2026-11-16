"""Manual approval: entries wait for the operator, exits never do."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from harness.inbox import ApproveOrders, Inbox, RejectOrders
from harness.journal import Journal, read_journal
from harness.orders import ApprovalRequest, OrderManager, OrderManagerError, approval_hash
from harness.positions import Ledger
from tests.fake_venue import at
from tests.test_orders import (
    CFG,
    D0,
    D1,
    E_AMD,
    E_MSFT,
    E_NVDA,
    PROFILE,
    X_AAPL,
    Crash,
    Rig,
    order,
    outcome,
)
from tests.test_orders_open import placed, run

DECIDED = at(D1, 7, 45)  # when the operator answers, before the runner starts
APPROVAL_DEADLINE = at(D1, 9, 20)


def gated(tmp_path: Path, **kw) -> Rig:  # type: ignore[no-untyped-def]
    return Rig(tmp_path, approval="entries", **kw)


def hash_of(rig: Rig, order_id: str) -> str:
    return next(r.content_hash for r in rig.om.pending_approvals(D1) if r.order_id == order_id)


def approve(rig: Rig, *order_ids: str, when: dt.datetime = DECIDED) -> Path:
    approvals = {order_id: hash_of(rig, order_id) for order_id in order_ids}
    return rig.inbox.put(ApproveOrders(approvals=approvals, operator="david", at=when))


def reject(rig: Rig, *order_ids: str, reason: str = "not today") -> Path:
    return rig.inbox.put(
        RejectOrders(order_ids=order_ids, operator="david", reason=reason, at=DECIDED)
    )


# --------------------------------------------------------------------------- requests


def test_pending_entries_are_listed_with_what_the_operator_must_see(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    msft, nvda = rig.om.pending_approvals(D1)  # the exit is not listed
    assert msft == ApprovalRequest(
        order_id=E_MSFT,
        content_hash=approval_hash(E_MSFT, "MSFT", "buy", 3, 306.0),
        symbol="MSFT",
        side="buy",
        quantity=3,
        limit_price=306.0,
        est_notional=918.0,
        expires_at=APPROVAL_DEADLINE,
    )
    assert (nvda.order_id, nvda.limit_price, nvda.est_notional) == (E_NVDA, 51.0, 204.0)
    assert [o.approval for o in rig.om.orders()] == ["none", "pending", "pending"]


def test_approval_hash_changes_with_every_term() -> None:
    base = approval_hash(E_MSFT, "MSFT", "buy", 3, 306.0)
    assert base == approval_hash(E_MSFT, "MSFT", "buy", 3, 306.0)
    others = {
        approval_hash(E_NVDA, "MSFT", "buy", 3, 306.0),
        approval_hash(E_MSFT, "MSFX", "buy", 3, 306.0),
        approval_hash(E_MSFT, "MSFT", "sell", 3, 306.0),
        approval_hash(E_MSFT, "MSFT", "buy", 4, 306.0),
        approval_hash(E_MSFT, "MSFT", "buy", 3, 306.01),
    }
    assert base not in others and len(others) == 5


def test_approval_off_asks_for_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    assert rig.om.pending_approvals(D1) == []
    assert run(rig) == [(E_MSFT, "filled", "", 3)]


def test_approval_of_entries_needs_an_inbox(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.journal is not None
    with pytest.raises(OrderManagerError, match="needs an inbox"):
        OrderManager(
            rig.venue, rig.ledger, rig.journal, rig.kill, None, CFG, PROFILE, rig.risk, 10,
            "entries",
        )  # fmt: skip


# --------------------------------------------------------------------------- the gate


def test_approved_entry_is_sent_and_unapproved_one_times_out(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    approve(rig, E_MSFT)
    assert run(rig) == [
        (E_NVDA, "canceled", "approval_timeout", 0),
        (X_AAPL, "filled", "", 10),
        (E_MSFT, "filled", "", 3),
    ]
    assert [p[0] for p in placed(rig)] == [X_AAPL + "-a1", E_MSFT + "-a1"]
    assert rig.inbox.read() == []  # handled commands are removed
    record = next(r for r in read_journal(rig.path) if r.kind == "approval")
    assert record.payload == {
        "order_id": E_MSFT,
        "decision": "approved",
        "operator": "david",
        "content_hash": approval_hash(E_MSFT, "MSFT", "buy", 3, 306.0),
        "at": DECIDED.isoformat(),
        "written_at": DECIDED.astimezone(dt.UTC).isoformat(),
    }
    (alert,) = rig.alerts.alerts
    assert (alert.severity, alert.kind) == ("warn", "approval_timeout")
    assert E_NVDA in alert.message
    rig.reconciles()


def test_exits_never_wait_for_approval(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert run(rig) == [(E_MSFT, "canceled", "approval_timeout", 0), (X_AAPL, "filled", "", 10)]
    # The exit went into the auction at the start, long before the approval deadline.
    assert placed(rig) == [(X_AAPL + "-a1", "market", "opg", 10)]
    assert rig.venue.calls[1] == ("place", X_AAPL + "-a1")


def test_approval_that_arrives_while_the_manager_waits_is_picked_up(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    request = hash_of(rig, E_MSFT)
    real = rig.venue.advance
    sent = False

    def advance(seconds: float) -> None:
        nonlocal sent
        real(seconds)
        if not sent and rig.venue.clock >= at(D1, 9, 10):  # the operator answers at 09:10
            rig.inbox.put(
                ApproveOrders(approvals={E_MSFT: request}, operator="david", at=rig.venue.clock)
            )
            sent = True

    rig.om._sleep = advance
    assert run(rig) == [(E_MSFT, "filled", "", 3)]


def test_entry_still_waiting_at_the_open_is_not_sent(tmp_path: Path) -> None:
    # The settings allow an approval deadline after the open. An entry that is still
    # waiting then must not go out with the others.
    late = CFG.model_copy(update={"approval_deadline": dt.time(9, 35)})
    rig = gated(tmp_path, cfg=late)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    approve(rig, E_NVDA)
    request = hash_of(rig, E_MSFT)
    real = rig.venue.advance
    sent = False

    def advance(seconds: float) -> None:
        nonlocal sent
        real(seconds)
        if not sent and rig.venue.clock >= at(D1, 9, 33):
            assert rig.phase(E_NVDA) == "filled" and rig.phase(E_MSFT) == "staged"
            assert [p[0] for p in placed(rig)] == [E_NVDA + "-a1"]  # MSFT was held back
            rig.inbox.put(
                ApproveOrders(approvals={E_MSFT: request}, operator="david", at=rig.venue.clock)
            )
            sent = True

    rig.om._sleep = advance
    assert run(rig) == [(E_NVDA, "filled", "", 4), (E_MSFT, "filled", "", 3)]
    assert sent


def test_rejected_entry_is_canceled_at_once(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    approve(rig, E_NVDA)
    reject(rig, E_MSFT, reason="earnings tomorrow")
    assert run(rig) == [(E_MSFT, "canceled", "approval_denied", 0), (E_NVDA, "filled", "", 4)]
    assert [p[0] for p in placed(rig)] == [E_NVDA + "-a1"]
    denial = [r for r in read_journal(rig.path) if r.kind == "approval"][1]
    assert (denial.payload["decision"], denial.payload["reason"]) == ("denied", "earnings tomorrow")
    assert rig.alerts.alerts == []


def test_approved_entry_can_still_be_rejected_until_it_is_sent(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    approve(rig, E_MSFT)
    real = rig.venue.advance
    sent = False

    def advance(seconds: float) -> None:
        nonlocal sent
        real(seconds)
        if not sent and rig.venue.clock >= at(D1, 9, 25):  # after the approval deadline
            reject(rig, E_MSFT, reason="changed my mind")
            sent = True

    rig.om._sleep = advance
    assert run(rig) == [(E_MSFT, "canceled", "approval_denied", 0)]
    assert rig.venue.place_attempts == []


def test_entry_can_be_rejected_when_approval_is_off(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    reject(rig, E_MSFT)
    assert run(rig) == [(E_MSFT, "canceled", "approval_denied", 0), (E_NVDA, "filled", "", 4)]


def test_approval_holds_when_the_entry_is_cut_at_release(tmp_path: Path) -> None:
    rig = gated(tmp_path, holdings={"AAPL": 90})  # $1,000 in cash: 15 shares do not fit
    rig.om.submit([order("MSFT", "buy", 15)])
    approve(rig, E_MSFT)
    # Approved for up to 15 shares at up to $306. Sending fewer is inside the approval.
    assert run(rig) == [(E_MSFT, "filled", "reduced_for_cash", 3)]


# --------------------------------------------------------------------------- refused commands


def refusals(rig: Rig) -> list[str]:
    return [str(r.payload["why"]) for r in read_journal(rig.path) if r.kind == "command_refused"]


def test_approval_with_the_wrong_hash_is_refused(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    stale = approval_hash(E_MSFT, "MSFT", "buy", 2, 306.0)  # shown a different quantity
    rig.inbox.put(
        ApproveOrders(
            approvals={E_MSFT: stale, E_NVDA: hash_of(rig, E_NVDA)}, operator="david", at=DECIDED
        )
    )
    assert run(rig) == [(E_MSFT, "canceled", "approval_timeout", 0), (E_NVDA, "filled", "", 4)]
    assert refusals(rig) == [
        f"approve {E_MSFT}: the order is not the one that was shown (hash mismatch)"
    ]
    assert rig.alerts.kinds() == ["command_refused", "approval_timeout"]
    assert rig.inbox.read() == []  # a refused command is not retried


def test_approval_decided_after_the_deadline_is_refused(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    late = approve(rig, E_MSFT, when=APPROVAL_DEADLINE + dt.timedelta(seconds=1))
    early = at(D1, 9, 0).timestamp()
    os.utime(late, (early, early))  # the file is older, but the command itself says "late"
    approve(rig, E_NVDA, when=APPROVAL_DEADLINE)  # exactly at the deadline still counts
    rig.to(9, 25)  # the runner comes up late and finds both files waiting
    assert run(rig) == [(E_MSFT, "canceled", "approval_timeout", 0), (E_NVDA, "filled", "", 4)]
    assert "after the deadline" in refusals(rig)[0]


def test_approval_written_late_is_refused_whatever_time_it_claims(tmp_path: Path) -> None:
    # The command says 09:10, but its file was written at 09:21: a stale draft, a bug in
    # whatever wrote it, or a lie. The file's own time decides.
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    backdated = approve(rig, E_MSFT, when=at(D1, 9, 10))
    written = at(D1, 9, 21).timestamp()
    os.utime(backdated, (written, written))
    approve(rig, E_NVDA, when=at(D1, 9, 10))  # written when it says: still honored
    rig.to(9, 25)  # the runner restarts after the deadline and finds both
    assert run(rig) == [(E_MSFT, "canceled", "approval_timeout", 0), (E_NVDA, "filled", "", 4)]
    (refusal,) = refusals(rig)
    assert "decided after the deadline" in refusal and "09:21" in refusal
    assert rig.venue.place_attempts[0].symbol == "NVDA" and len(rig.venue.place_attempts) == 1


def test_approval_dated_in_the_future_is_refused(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    ahead = approve(rig, E_MSFT, when=at(D1, 9, 0))  # the runner's clock says 08:00
    early = at(D1, 7, 50).timestamp()
    os.utime(ahead, (early, early))
    approve(rig, E_NVDA, when=rig.venue.clock + dt.timedelta(seconds=20))  # within the skew
    rig.om.ingest_inbox(APPROVAL_DEADLINE)
    assert "in the future" in refusals(rig)[0] and len(refusals(rig)) == 1
    assert [o.approval for o in rig.om.orders()] == ["pending", "approved"]


def test_approval_given_the_evening_before_is_honored(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    approve(rig, E_MSFT, when=at(D0, 18, 30))
    assert run(rig) == [(E_MSFT, "filled", "", 3)]


def test_approval_whose_file_time_cannot_be_read_is_refused(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    approve(rig, E_MSFT)
    real = rig.inbox.read
    rig.inbox.read = lambda: [i.model_copy(update={"written_at": None}) for i in real()]  # type: ignore[method-assign]
    rig.om.ingest_inbox(APPROVAL_DEADLINE)
    assert "time cannot be read" in refusals(rig)[0]
    assert rig.om.orders()[0].approval == "pending"


@pytest.mark.parametrize(
    ("target", "why"),
    [
        ("0a1b2c3d-20240102-ZZZ-e", "unknown order"),
        (X_AAPL, "exits are never gated"),
    ],
)
def test_approval_or_rejection_of_the_wrong_thing_is_refused(
    tmp_path: Path, target: str, why: str
) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.inbox.put(ApproveOrders(approvals={target: "h"}, operator="david", at=DECIDED))
    reject(rig, target)
    assert run(rig) == [(X_AAPL, "filled", "", 10)]  # the exit is untouched
    assert len(refusals(rig)) == 2 and all(why in r for r in refusals(rig))


def test_commands_for_an_order_that_is_already_sent_or_closed_are_refused(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    msft, nvda = hash_of(rig, E_MSFT), hash_of(rig, E_NVDA)
    approve(rig, E_MSFT)
    run(rig)  # MSFT filled, NVDA timed out
    reject(rig, E_MSFT)
    rig.inbox.put(ApproveOrders(approvals={E_NVDA: nvda}, operator="david", at=DECIDED))
    rig.inbox.put(ApproveOrders(approvals={E_MSFT: msft}, operator="david", at=DECIDED))  # again
    rig.om.ingest_inbox()
    assert refusals(rig) == [
        f"reject {E_MSFT}: the order is already filled",
        f"approve {E_NVDA}: the order is not waiting for approval (canceled, pending)",
    ]  # the repeated approval of MSFT is not an error: it changes nothing
    assert (rig.phase(E_MSFT), rig.phase(E_NVDA)) == ("filled", "canceled")


def test_unreadable_command_is_moved_aside_and_the_rest_still_apply(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.inbox.directory.mkdir(parents=True, exist_ok=True)
    (rig.inbox.directory / "00000000-junk.json").write_text("{not json")
    approve(rig, E_MSFT)
    assert run(rig) == [(E_MSFT, "filled", "", 3)]
    assert "cannot be read" in refusals(rig)[0]
    assert [p.name for p in rig.inbox.directory.iterdir()] == ["00000000-junk.bad"]


def test_same_command_twice_changes_nothing_the_second_time(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    command = ApproveOrders(approvals={E_MSFT: hash_of(rig, E_MSFT)}, operator="david", at=DECIDED)
    for _ in range(2):
        rig.inbox.put(command)
        reject(rig, E_NVDA)
        rig.om.ingest_inbox()
    assert [r.payload["decision"] for r in read_journal(rig.path) if r.kind == "approval"] == [
        "approved",
        "denied",
    ]
    assert refusals(rig) == [] and rig.alerts.alerts == []


def test_ingest_without_an_inbox_does_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.journal is not None
    rig.journal.close()
    journal = Journal(rig.path)
    manager = OrderManager(
        rig.venue, Ledger.open(journal, 10_000.0), journal, rig.kill, None, CFG, PROFILE,
        rig.risk, 10, "off",
    )  # fmt: skip
    manager.recover()
    manager.ingest_inbox()


# --------------------------------------------------------------------------- restarts


def test_approval_survives_a_restart(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4), order("AMD", "buy", 2)])
    approve(rig, E_MSFT)
    reject(rig, E_NVDA)
    rig.om.ingest_inbox()
    rig.restart()
    assert [(o.order_id, o.approval, o.phase) for o in rig.om.orders()] == [
        (E_MSFT, "approved", "staged"),
        (E_NVDA, "denied", "canceled"),
        (E_AMD, "pending", "staged"),
    ]
    assert [r.order_id for r in rig.om.pending_approvals(D1)] == [E_AMD]
    assert run(rig) == [(E_AMD, "canceled", "approval_timeout", 0), (E_MSFT, "filled", "", 3)]


def test_staged_orders_keep_the_mode_they_were_staged_under(tmp_path: Path) -> None:
    rig = gated(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.approval = "off"  # the profile is re-activated without approval
    rig.restart()
    rig.om.submit([order("NVDA", "buy", 4)])
    assert [o.approval for o in rig.om.orders()] == ["pending", "none"]


def gated_morning_rig(tmp_path: Path) -> Rig:
    rig = gated(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4), order("AMD", "buy", 2)])
    approve(rig, E_MSFT)
    reject(rig, E_NVDA)  # AMD is left to time out
    return rig


def gated_morning(rig: Rig) -> None:
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4), order("AMD", "buy", 2)])
    rig.om.on_session_open(D1, {})


def test_crash_at_every_record_of_a_gated_morning_ends_in_the_same_place(tmp_path: Path) -> None:
    baseline = gated_morning_rig(tmp_path / "baseline")
    gated_morning(baseline)
    expected = outcome(baseline)
    assert expected["phases"] == [
        (X_AAPL, "filled", 10, ""),
        (E_MSFT, "filled", 3, ""),
        (E_NVDA, "canceled", 0, "approval_denied"),
        (E_AMD, "canceled", 0, "approval_timeout"),
    ]
    kinds = baseline.kinds()
    first = len(kinds) - kinds[::-1].index("order_staged") + 1  # after the staging
    assert len(kinds) - first > 8

    for seq in range(first, len(kinds) + 1):
        for after_write in (False, True):
            rig = gated_morning_rig(tmp_path / f"crash-{seq}-{int(after_write)}")
            assert rig.journal is not None
            rig.journal.crash_at = seq
            rig.journal.crash_after_write = after_write
            with pytest.raises(Crash):
                gated_morning(rig)
            rig.restart()
            gated_morning(rig)
            where = f"crash at record {seq}, {'after' if after_write else 'before'} the write"
            assert outcome(rig) == expected, where
            assert Inbox(rig.inbox.directory).read() == [], where
            decisions = [r for r in read_journal(rig.path) if r.kind == "approval"]
            assert len(decisions) == 2, where  # each decision is on the record once


def test_orders_here_are_decided_on_d0() -> None:
    assert order("AAPL", "sell", 1).created_on == D0
