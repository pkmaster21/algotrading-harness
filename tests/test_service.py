"""The read-only views (`harness.service`): what they show at each point of a profile's
life, and that reading changes nothing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import harness.control as control
import harness.service as service
from harness.journal import JournalError
from harness.profiles import save_risk_state
from harness.risk import RiskState
from tests.fake_venue import at
from tests.test_paper import D1, D2, Rig, T


def read(rig: Rig) -> dict[str, Any]:
    return {"execution": rig.execution}


def clocked(rig: Rig) -> dict[str, Any]:
    return {"execution": rig.execution, "now": lambda: rig.venue.clock}


def test_before_the_first_run_there_is_a_profile_and_nothing_else(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    status = service.get_status(**clocked(rig))
    assert service.trading_mode() == status.mode == "paper"
    assert status.active is not None and status.active.profile_id == rig.profile_id
    assert (status.stop_level, status.day_stage, status.session) == ("none", "none", None)
    assert status.last_run is None and status.last_reconcile is None and status.problems == ()
    assert status.risk is not None and status.risk.peak_equity == 10_000.0
    assert status.as_of == rig.venue.clock
    positions = service.get_positions(**read(rig))
    assert positions.positions == () and positions.as_of is None
    assert service.list_orders(**read(rig)) == []
    assert service.list_pending_approvals(**clocked(rig)) == []
    assert service.list_alerts(**read(rig)) == []
    assert [p.profile_id for p in service.list_profiles(**read(rig))] == [rig.profile_id]


def test_after_a_day_the_views_show_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    report = rig.run()
    (order,) = report.orders
    status = service.get_status(**clocked(rig))
    assert (status.session, status.day_stage, status.stop_level) == (D1, "complete", "none")
    assert status.last_run is not None and status.last_run.report == report
    assert status.last_run.finished_at == rig.venue.clock
    assert status.last_reconcile == report.reconcile_after
    assert status.last_reconcile_stage == "after"
    assert status.risk is not None and status.risk.marked_on == T

    view = service.get_positions(**read(rig))
    (spy,) = view.positions
    close = rig.close("SPY", T)
    assert (spy.symbol, spy.quantity, spy.entry_order_id) == ("SPY", order.quantity, order.order_id)
    assert (spy.mark, view.marks_as_of) == (close, T)
    assert spy.market_value == order.quantity * close
    assert view.pnl is not None and view.pnl.market_value == spy.market_value
    assert view.pnl.equity == pytest.approx(view.cash + order.quantity * close)
    assert view.journal_seq == len(rig.records()) and view.as_of == rig.records()[-1].at
    assert view.cash == rig.ledger()["cash"]  # agrees with the runner's own snapshot

    assert service.list_orders(**read(rig)) == [order]
    assert service.list_orders(T, **read(rig)) == [order]
    assert service.list_orders(D1, **read(rig)) == []


def test_a_day_stopped_part_way_shows_how_far_it_got(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    calls = 0

    def sleep(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            rig.kill_switch().engage("operator said stop", "operator:cli")
        rig.venue.advance(seconds)

    rig.sleep = sleep
    rig.run()
    status = service.get_status(**clocked(rig))
    assert (status.session, status.day_stage) == (D1, "decided")
    assert status.stop_level == "killed"
    assert status.stop_reason == "operator:cli: operator said stop"
    assert status.kill.engaged and status.last_reconcile_stage == "before"
    assert status.last_run is not None and status.last_run.report.outcome == "killed"
    # The orders snapshot is written however the run ended.
    assert [(o.phase, o.reason) for o in service.list_orders(**read(rig))] == [
        ("canceled", "kill_switch")
    ]


def test_stop_levels_weakest_first(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    blocked = RiskState(
        peak_equity=10_000.0, prior_equity=9_500.0, marked_on=T, entries_blocked_for=T
    )
    save_risk_state(blocked, rig.state_dir, rig.profile_id)
    status = service.get_status(**read(rig))
    assert status.stop_level == "entries_blocked" and str(T) in status.stop_reason
    halted = blocked.model_copy(update={"halted": True, "halt_reason": "25% below the peak"})
    save_risk_state(halted, rig.state_dir, rig.profile_id)
    status = service.get_status(**read(rig))
    assert (status.stop_level, status.stop_reason) == ("entries_halted", "25% below the peak")
    rig.kill_switch().engage("stop", "operator:test")
    assert service.get_status(**read(rig)).stop_level == "killed"
    # A cap tripped on an earlier close no longer blocks anything.
    rig.kill_switch().reset("david", rig.profile_id, rig.profile_id)
    old = blocked.model_copy(update={"marked_on": D1})
    save_risk_state(old, rig.state_dir, rig.profile_id)
    assert service.get_status(**read(rig)).stop_level == "none"


def test_pending_approvals_are_shown_until_their_deadline(tmp_path: Path) -> None:
    rig = Rig(tmp_path, approval="entries")
    seen: list[Any] = []

    def sleep(seconds: float) -> None:
        if not seen:
            seen.extend(service.list_pending_approvals(**clocked(rig)))
            rig.kill_switch().engage("stop", "operator:test")  # the file is left as it is
        rig.venue.advance(seconds)

    rig.sleep = sleep
    assert rig.run().outcome == "killed"
    (request,) = seen
    assert request.symbol == "SPY" and request.expires_at == at(D1, 9, 20)
    rig.venue.set_now(at(D1, 9, 19))
    assert service.list_pending_approvals(**clocked(rig)) == [request]
    rig.venue.set_now(at(D1, 9, 20))
    assert service.list_pending_approvals(**clocked(rig)) == []  # too late to approve


def test_positions_are_current_to_the_last_fill_not_the_last_snapshot(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    quantity = rig.run().orders[0].quantity
    snapshot = (rig.state / "ledger.json").read_bytes()
    control.flatten(
        "david", rig.profile_id, venue=rig.venue, execution=rig.execution, alerts=rig.alerts,
        sleep=rig.venue.advance, now=lambda: rig.venue.clock,
    )  # fmt: skip
    (rig.state / "ledger.json").write_bytes(snapshot)  # as if the snapshot were stale
    view = service.get_positions(**read(rig))
    assert view.positions == () and view.pnl is not None
    assert view.pnl.market_value == 0.0 and quantity > 0


def test_reading_changes_nothing_and_asks_the_venue_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    rig.morning(D2)

    def tree() -> dict[str, tuple[int, bytes]]:
        roots = (rig.state_dir, rig.execution.profiles_dir)
        found = [p for root in roots for p in root.rglob("*") if p.is_file()]
        return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in found}

    before, calls = tree(), len(rig.venue.calls)
    service.get_status(**clocked(rig))
    service.get_positions(**read(rig))
    service.list_orders(**read(rig))
    service.list_pending_approvals(**clocked(rig))
    service.list_alerts(**read(rig))
    service.list_profiles(**read(rig))
    assert tree() == before and len(rig.venue.calls) == calls
    assert rig.run().outcome == "traded"  # and the runner is not in anyone's way


def test_the_read_surface_has_no_way_to_act() -> None:
    assert set(service.__all__) == {
        "PositionView", "PositionsView", "TradingStatus", "get_positions", "get_status",
        "list_alerts", "list_orders", "list_pending_approvals", "list_profiles", "trading_mode",
    }  # fmt: skip
    acting = {name for name in vars(control) if not name.startswith("_")} - set(service.__all__)
    for name in ("engage_kill_switch", "reset_kill_switch", "flatten", "approve_orders",
                 "reject_orders", "reset_risk_halt", "accept_broker_state", "record_adjustment",
                 "activate_profile", "promote_run"):  # fmt: skip
        assert name in acting and not hasattr(service, name)
    assert not hasattr(service, "control") and not hasattr(service, "Inbox")


def test_state_that_cannot_be_read_is_reported_not_hidden(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    (rig.state / "last_run.json").write_text("{not json")
    (rig.state / "orders.json").write_text("{not json")
    (rig.state / "risk_state.json").unlink()
    status = service.get_status(**read(rig))
    assert status.last_run is None and status.risk is None
    assert len(status.problems) == 2 and "last_run.json" in " ".join(status.problems)
    assert status.day_stage == "complete"  # what can be read is still shown
    assert service.list_orders(**read(rig)) == []

    journal = rig.state / "journal.jsonl"
    journal.write_bytes(journal.read_bytes().replace(b'"kind":"fill"', b'"kind":"fell"', 1))
    assert "journal cannot be read" in " ".join(service.get_status(**read(rig)).problems)
    with pytest.raises(JournalError, match="not a valid record"):
        service.get_positions(**read(rig))  # never an empty book for a broken one
