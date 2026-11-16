"""Operator commands (`harness.control`): each against a profile that has traded on the
fake venue, and each refusal."""

from __future__ import annotations

import datetime as dt
import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import harness.control as control
import harness.service as service
from harness.broker.venue import KillSwitchEngaged, VenueOrder
from harness.control import ControlError
from harness.killswitch import KillSwitch, kill
from harness.orders import OrderManagerError, SessionIncomplete
from harness.positions import LedgerAdjustment
from harness.profiles import load_risk_state, save_risk_state
from harness.risk import RiskState
from tests.fake_venue import OrderScript, at
from tests.test_orders import D0 as OD0
from tests.test_orders import D1 as OD1
from tests.test_orders import E_MSFT, X_AAPL, order
from tests.test_orders import Rig as OrdersRig
from tests.test_paper import D1, D2, SATURDAY, Rig, T

ME = "david"


def kw(rig: Rig, **extra: Any) -> dict[str, Any]:
    """What every command that talks to the venue is given in these tests."""
    return {
        "venue": rig.venue,
        "execution": rig.execution,
        "alerts": rig.alerts,
        "sleep": rig.venue.advance,
        "now": lambda: rig.venue.clock,
        **extra,
    }


def files(rig: Rig, **extra: Any) -> dict[str, Any]:
    """For the commands that only touch files."""
    return {"execution": rig.execution, "now": lambda: rig.venue.clock, **extra}


def controls(rig: Rig) -> list[dict[str, Any]]:
    return [dict(r.payload) for r in rig.records() if r.kind == "control"]


@contextmanager
def runner_running(rig: Rig) -> Iterator[None]:
    fd = os.open(rig.state_dir / "runner.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def holding_spy(tmp_path: Path) -> tuple[Rig, int]:
    """A profile after its first day: it holds SPY, and it is 09:30 on D1."""
    rig = Rig(tmp_path)
    quantity = rig.run().orders[0].quantity
    assert rig.held() == {"SPY": quantity}
    return rig, quantity


# --------------------------------------------------------------------------- flatten


def test_flatten_sells_exactly_what_is_held_and_works_while_killed(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    killed = control.engage_kill_switch("testing", ME, **kw(rig))
    before = len(rig.venue.place_attempts)
    report = control.flatten(ME, rig.profile_id, **kw(rig))
    sent = rig.venue.place_attempts[before:]
    assert [(o.side, o.symbol, o.quantity, o.type, o.time_in_force) for o in sent] == [
        ("sell", "SPY", quantity, "market", "day")
    ]
    assert sent[0].client_order_id.endswith(f"-{T:%Y%m%d}-SPY-f1-a1")
    assert report.remaining == {} and rig.held() == {}
    assert [(o.phase, o.filled_quantity) for o in report.orders] == [("filled", quantity)]
    assert report.reconcile.severity == "ok"
    assert report.differences == ()
    state = rig.kill_switch().state()  # the latch that was already set stays, as set
    assert (state.reason, state.at, state.cancel_confirmed) == ("testing", killed.at, True)
    assert [c["command"] for c in controls(rig)] == ["flatten"]
    assert controls(rig)[0]["operator"] == ME
    assert service.get_positions(execution=rig.execution).positions == ()
    assert rig.ledger()["positions"] == {}  # the snapshot readers use is refreshed


def test_flatten_before_the_open_goes_into_the_auction(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    rig.morning(D2)
    before = len(rig.venue.place_attempts)
    report = control.flatten(ME, rig.profile_id, **kw(rig))
    (sent,) = rig.venue.place_attempts[before:]
    assert (sent.side, sent.quantity, sent.time_in_force) == ("sell", quantity, "opg")
    assert sent.client_order_id.endswith(f"-{D1:%Y%m%d}-SPY-f1-a1")
    assert report.remaining == {} and at(D2, 9, 30) <= rig.venue.clock < at(D2, 9, 31)
    # Flatten engaged the switch itself, before anything was sent, and leaves it set.
    state = rig.kill_switch().state()
    assert (state.source, state.reason, state.at) == (f"operator:{ME}", "flatten", at(D2, 9, 15))
    assert state.cancel_confirmed  # nothing of the flatten is left at the venue
    assert "reset it to trade again" in rig.alerts.alerts[-1].message


def test_flatten_cancels_what_is_open_and_never_buys(tmp_path: Path) -> None:
    rig = OrdersRig(tmp_path)  # holds AAPL 10 and TSLA 5; it is 08:00
    rig.om.submit([order("MSFT", "buy", 3), order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)  # the strategy's own exit is already in the auction
    rig.kill.engage("stop", "operator:test")
    before = len(rig.venue.place_attempts)
    result = rig.om.flatten(OD1, OD0, at(OD1, 9, 50))
    ids = list(result.order_ids)
    assert result.differences == () and ids == [
        f"0a1b2c3d-{OD0:%Y%m%d}-AAPL-f1",
        f"0a1b2c3d-{OD0:%Y%m%d}-TSLA-f1",
    ]
    orders = {o.order_id: o for o in rig.om.orders()}
    assert (orders[E_MSFT].phase, orders[E_MSFT].reason) == ("canceled", "flatten")
    assert (orders[X_AAPL].phase, orders[X_AAPL].reason) == ("canceled", "flatten")
    sent = rig.venue.place_attempts[before:]
    assert [(o.side, o.symbol, o.quantity, o.time_in_force) for o in sent] == [
        ("sell", "AAPL", 10, "opg"),
        ("sell", "TSLA", 5, "opg"),
    ]  # the whole of each position, once, and no buy
    assert rig.venue.positions() == [] and rig.kill.engaged()
    rig.reconciles()
    with pytest.raises(KillSwitchEngaged):  # the pass ended with the flatten
        rig.om.submit([order("TSLA", "sell", 1, created_on=OD0)])
        rig.send_exit(f"0a1b2c3d-{OD0:%Y%m%d}-TSLA-x")

    rig.restart()  # the flatten orders replay under the ids they were staged with
    assert [o.phase for o in rig.om.orders() if o.order_id in ids] == ["filled", "filled"]


def test_a_kill_from_elsewhere_stops_a_running_flatten(tmp_path: Path) -> None:
    rig = OrdersRig(tmp_path)  # holds AAPL 10 and TSLA 5; it is 08:00
    rig.kill.engage("flatten", "operator:david")
    reports: list[Any] = []

    def sleep(seconds: float) -> None:
        if not reports and len(rig.venue.place_attempts) == 2:  # both sells are out
            other = KillSwitch(rig.kill.path)  # another terminal, another process
            reports.append(
                kill(other, "stop everything", "operator:other", venue=rig.venue,
                     sleep=rig.venue.advance, now=lambda: rig.venue.clock)
            )  # fmt: skip
        rig.venue.advance(seconds)

    rig.om._sleep = sleep
    with pytest.raises(SessionIncomplete, match="flatten stopped"):
        rig.om.flatten(OD1, OD0, at(OD1, 9, 50))
    assert reports[0].cancel_confirmed  # the kill said "nothing is open", and it stays true
    assert rig.venue.open_orders() == [] and len(rig.venue.place_attempts) == 2
    assert rig.venue.clock < at(OD1, 8, 1)  # it did not carry on to the open
    assert {p.symbol: p.quantity for p in rig.venue.positions()} == {"AAPL": 10, "TSLA": 5}
    flattens = [o for o in rig.om.orders() if o.order_id.endswith("-f1")]
    assert [(o.phase, o.reason) for o in flattens] == [("canceled", "kill_switch")] * 2
    assert "flatten_stopped" in rig.alerts.kinds() and rig.kill.engaged()


def test_flatten_sells_no_more_than_the_venue_holds(tmp_path: Path) -> None:
    rig = OrdersRig(tmp_path)  # the ledger holds AAPL 10 and TSLA 5
    rig.venue.hold("AAPL", 6, 100.0)  # the venue holds 6: a sell for 10 would be rejected
    rig.kill.engage("flatten", "operator:david")
    result = rig.om.flatten(OD1, OD0, at(OD1, 9, 50))
    sent = [(o.symbol, o.quantity) for o in rig.venue.place_attempts]
    assert sent == [("AAPL", 6), ("TSLA", 5)] and rig.venue.positions() == []
    assert result.differences == ("AAPL: ledger holds 10, venue holds 6; selling 6",)
    assert {s: p.quantity for s, p in rig.ledger.account().positions.items()} == {"AAPL": 4}
    (alert,) = [a for a in rig.alerts.alerts if a.kind == "flatten_mismatch"]
    assert alert.severity == "critical" and "accept_broker_state" in alert.message
    assert "AAPL: ledger holds 10, venue holds 6" in alert.message


def test_flatten_sends_nothing_if_the_latch_note_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = OrdersRig(tmp_path)
    rig.kill.engage("stop", "operator:david")
    rig.kill.confirm_cancel()  # as a kill leaves it: "no open orders remain"

    def fails(*args: Any) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr("harness.killswitch.os.replace", fails)
    with pytest.raises(OrderManagerError, match="could not be marked"):
        rig.om.flatten(OD1, OD0, at(OD1, 9, 50))
    monkeypatch.undo()
    assert rig.venue.place_attempts == [] and "order_staged" not in rig.kinds()
    assert rig.kill.on_disk().cancel_confirmed  # the file is as the kill left it

    rig.kill.path.write_text("garbled")  # a latch that cannot be read cannot carry the note
    with pytest.raises(OrderManagerError, match="could not be marked"):
        rig.om.flatten(OD1, OD0, at(OD1, 9, 50))
    assert rig.venue.place_attempts == []


def test_flatten_needs_the_switch_engaged_first(tmp_path: Path) -> None:
    rig = OrdersRig(tmp_path)
    with pytest.raises(OrderManagerError, match="kill switch engaged"):
        rig.om.flatten(OD1, OD0, at(OD1, 9, 50))
    assert rig.venue.place_attempts == []


def test_the_flatten_report_names_what_the_venue_did_not_hold(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    rig.venue.hold("SPY", quantity - 1, rig.close("SPY", T))  # one share short at the venue
    report = control.flatten(ME, rig.profile_id, **kw(rig))
    assert report.differences == (
        f"SPY: ledger holds {quantity}, venue holds {quantity - 1}; selling {quantity - 1}",
    )
    assert report.remaining == {"SPY": 1} and rig.held() == {}
    assert report.reconcile.severity == "stop"  # for accept_broker_state to settle
    assert rig.alerts.alerts[-1].severity == "critical"


def test_a_second_flatten_tries_again_under_a_new_id(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    rig.venue.script("SPY", OrderScript(reject="halted"))
    first = control.flatten(ME, rig.profile_id, **kw(rig))
    assert first.remaining == {"SPY": quantity}
    assert [o.phase for o in first.orders] == ["rejected"]
    assert ("critical", "flatten_unfilled") in [(a.severity, a.kind) for a in rig.alerts.alerts]
    rig.venue.script("SPY", OrderScript())
    second = control.flatten(ME, rig.profile_id, **kw(rig))
    assert second.remaining == {} and rig.held() == {}
    assert second.orders[0].order_id.endswith("-SPY-f2")


class _CtrlC(BaseException):
    """What Ctrl-C is to the command: not an Exception, so nothing cleans up."""


def test_a_sell_an_interrupted_flatten_left_behind_is_never_sent_by_the_runner(
    tmp_path: Path,
) -> None:
    rig, quantity = holding_spy(tmp_path)
    rig.morning(D2)

    def interrupted(seconds: float) -> None:
        raise _CtrlC

    with pytest.raises(_CtrlC):
        control.flatten(ME, rig.profile_id, **kw(rig, sleep=interrupted))
    (left,) = rig.venue.open_orders()  # the auction sell is at the venue, not final
    assert left.client_order_id.endswith("-SPY-f1-a1")

    # The operator decides not to sell after all: cancel it, reset, trade on.
    assert control.engage_kill_switch("not selling after all", ME, **kw(rig)).cancel_confirmed
    control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    places = len(rig.venue.place_attempts)
    report = rig.run()

    assert report.outcome == "traded" and rig.held() == {"SPY": quantity}
    assert len(rig.venue.place_attempts) == places  # the sell was not sent again
    (sell,) = [o for o in service.list_orders(execution=rig.execution) if "-f1" in o.order_id]
    assert (sell.phase, sell.reason, sell.filled_quantity) == ("canceled", "flatten", 0)
    assert [(a.severity, a.kind) for a in report.alerts] == [("critical", "flatten_unfinished")]
    assert f"{quantity} of {quantity} shares unsold" in report.alerts[0].message
    # A flatten after that sells under a new id.
    again = control.flatten(ME, rig.profile_id, **kw(rig))
    assert again.remaining == {} and again.orders[0].order_id.endswith("-SPY-f2")


def test_after_a_flatten_nothing_trades_until_the_switch_is_reset(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    control.flatten(ME, rig.profile_id, **kw(rig))
    rig.morning(D2)
    assert rig.run().outcome == "killed" and rig.held() == {}  # no buying back
    control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    report = rig.run()  # the deliberate way back in: the strategy buys again
    assert report.outcome == "traded" and set(rig.held()) == {"SPY"}
    assert report.reconcile_after is not None and report.reconcile_after.severity == "ok"


def test_flatten_is_refused_without_the_typed_profile_id(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    places = len(rig.venue.place_attempts)
    with pytest.raises(ControlError, match="confirmation"):
        control.flatten(ME, "yes", **kw(rig))
    assert len(rig.venue.place_attempts) == places and rig.held() == {"SPY": quantity}
    assert not rig.kill_switch().engaged()  # nothing at all happens without it


def test_flatten_is_refused_when_the_market_is_closed(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    places = len(rig.venue.place_attempts)
    for moment in (at(D1, 16, 0), at(SATURDAY, 10, 0)):
        rig.venue.set_now(moment)
        with pytest.raises(ControlError, match="market is closed"):
            control.flatten(ME, rig.profile_id, **kw(rig))
    assert len(rig.venue.place_attempts) == places and rig.held() == {"SPY": quantity}
    assert controls(rig) == []
    assert rig.kill_switch().state().reason == "flatten"  # refused, but trading is stopped


def test_flatten_is_refused_for_another_account(tmp_path: Path, monkeypatch: Any) -> None:
    rig, _ = holding_spy(tmp_path)
    places = len(rig.venue.place_attempts)
    account = rig.venue.account().model_copy(update={"account_id": "OTHER"})
    monkeypatch.setattr(rig.venue, "verify_account", lambda: account)
    with pytest.raises(ControlError, match="keys are for account OTHER"):
        control.flatten(ME, rig.profile_id, **kw(rig))
    assert len(rig.venue.place_attempts) == places


# --------------------------------------------------------------------------- the lock


def test_commands_that_write_the_books_refuse_while_a_runner_is_running(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    rig.kill_switch().engage("stop", "operator:test")
    records, places = len(rig.records()), len(rig.venue.place_attempts)
    dividend = LedgerAdjustment(
        adjustment_id="d", as_of=D1, kind="dividend", symbol="SPY", cash_delta=1.0,
        operator=ME, note="n",
    )  # fmt: skip
    with runner_running(rig):
        for command in (
            lambda: control.flatten(ME, rig.profile_id, **kw(rig)),
            lambda: control.reset_kill_switch(ME, rig.profile_id, **kw(rig)),
            lambda: control.accept_broker_state(ME, "note", **kw(rig)),
            lambda: control.reset_risk_halt(ME, "why", **files(rig)),
            lambda: control.record_adjustment(dividend, **files(rig)),
            lambda: control.activate_profile(
                rig.profile_id, "off", ME, venue=rig.venue, execution=rig.execution
            ),
        ):
            with pytest.raises(ControlError, match="is running"):
                command()
    assert len(rig.records()) == records and len(rig.venue.place_attempts) == places
    assert rig.kill_switch().engaged()


def test_the_kill_switch_does_not_wait_for_the_lock(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    with runner_running(rig):
        state = control.engage_kill_switch("the runner is hung", ME, **kw(rig))
    assert state.engaged and state.source == f"operator:{ME}" and state.cancel_confirmed


def test_every_command_needs_an_operator(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    for command in (
        lambda: control.engage_kill_switch("r", " ", **kw(rig)),
        lambda: control.reset_kill_switch("", rig.profile_id, **kw(rig)),
        lambda: control.flatten("", rig.profile_id, **kw(rig)),
        lambda: control.accept_broker_state("", "note", **kw(rig)),
        lambda: control.reset_risk_halt("", "why", **files(rig)),
        lambda: control.approve_orders({"a": "b"}, "", **files(rig)),
        lambda: control.reject_orders(["a"], "", "why", **files(rig)),
    ):
        with pytest.raises(ControlError, match="operator"):
            command()
    assert not rig.kill_switch().engaged()


# --------------------------------------------------------------------------- kill switch


def test_engaging_cancels_open_orders_and_sells_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path, approval="entries")
    engaged: list[Any] = []

    def sleep(seconds: float) -> None:
        if not engaged:  # from "another terminal", while the runner waits for approval
            engaged.append(control.engage_kill_switch("stop now", ME, **kw(rig, alerts=None)))
        rig.venue.advance(seconds)

    rig.sleep = sleep
    assert rig.run().outcome == "killed"
    state = engaged[0]
    assert state.engaged and state.cancel_confirmed and state.reason == "stop now"
    assert rig.venue.place_attempts == [] and rig.venue.open_orders() == []
    kinds = [a.kind for a in service.list_alerts(execution=rig.execution)]
    assert "kill_switch" in kinds  # the default sinks: alerts.jsonl


def test_engaging_with_keys_for_another_account_sets_the_latch_and_cancels_nothing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    rig, _ = holding_spy(tmp_path)
    account = rig.venue.account().model_copy(update={"account_id": "OTHER"})
    monkeypatch.setattr(rig.venue, "verify_account", lambda: account)
    calls = len(rig.venue.calls)
    state = control.engage_kill_switch("stop", ME, **kw(rig))
    assert state.engaged and not state.cancel_confirmed
    assert rig.venue.calls[calls:] == []  # asked who it is (patched above), nothing else
    messages = " ".join(a.message for a in rig.alerts.alerts)
    assert "OTHER" in messages and "NOT canceled" in messages


def test_reset_needs_a_clean_reconciliation(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.hold("AAPL", 1, 50.0)  # a trade made by hand in the account
    assert rig.run().outcome == "killed"
    with pytest.raises(ControlError, match="AAPL: ledger holds 0, venue holds 1"):
        control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    assert rig.kill_switch().engaged() and controls(rig) == []

    after = control.accept_broker_state(ME, "bought by hand to test", **kw(rig))
    assert after.severity == "ok"
    view = service.get_positions(execution=rig.execution)
    assert [(p.symbol, p.quantity, p.avg_price) for p in view.positions] == [("AAPL", 1, 50.0)]
    assert view.cash == 10_000.0  # the shares came in at their cost; cash did not differ

    reset = control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    assert not rig.kill_switch().engaged()
    assert reset.cleared.source == "auto:runner" and "Start the runner again" in reset.message
    assert reset.reconcile is not None and reset.reconcile.severity == "ok"
    record = controls(rig)[-1]
    assert (record["command"], record["operator"]) == ("reset_kill_switch", ME)
    assert record["cleared"]["source"] == "auto:runner"  # what was cleared, on the record
    accepted = controls(rig)[0]
    assert (accepted["command"], accepted["note"]) == (
        "accept_broker_state",
        "bought by hand to test",
    )
    assert rig.run().outcome == "traded"  # the runner starts again from the accepted book


def test_reset_is_refused_while_an_order_is_open_at_the_venue(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    rig.kill_switch().engage("stop", "operator:test")
    rig.venue.script("stray", OrderScript(fills=()))
    rig.venue.place(
        VenueOrder(
            client_order_id="stray",
            symbol="SPY",
            side="sell",
            quantity=1,
            type="market",
            time_in_force="day",
        )  # fmt: skip
    )
    with pytest.raises(ControlError, match="stray is open at the venue"):
        control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    assert rig.kill_switch().engaged()
    with pytest.raises(ControlError, match="open orders"):
        control.accept_broker_state(ME, "note", **kw(rig))  # an order cannot be accepted
    assert "adjustment" not in rig.kinds()


def test_reset_refusals(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    with pytest.raises(ControlError, match="not engaged"):
        control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    rig.kill_switch().engage("stop", "operator:test")
    with pytest.raises(ControlError, match="confirmation"):
        control.reset_kill_switch(ME, "yes", **kw(rig))
    assert rig.kill_switch().engaged()


def test_a_latch_set_with_no_profile_active_can_be_reset(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    (rig.state_dir / "active.json").unlink()
    rig.kill_switch().engage("before anything was activated", "operator:test")
    with pytest.raises(ControlError, match="confirmation"):
        control.reset_kill_switch(ME, rig.profile_id, **kw(rig))
    reset = control.reset_kill_switch(ME, control.NO_PROFILE, **kw(rig))
    assert reset.reconcile is None and not rig.kill_switch().engaged()


# --------------------------------------------------------------------------- the books


def test_accept_refuses_what_the_ledger_cannot_hold(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    rig.venue.hold("NVDA", 3, 50.0)  # not in the profile's universe
    with pytest.raises(ControlError, match="NVDA is not in the profile's universe"):
        control.accept_broker_state(ME, "note", **kw(rig))
    assert "adjustment" not in rig.kinds() and controls(rig) == []
    with pytest.raises(ControlError, match="say why"):
        control.accept_broker_state(ME, " ", **kw(rig))


def test_accept_records_a_sale_made_at_the_broker_and_its_cash(tmp_path: Path) -> None:
    rig, quantity = holding_spy(tmp_path)
    holding = rig.venue._holdings.pop("SPY")  # sold by hand, at a small profit
    rig.venue.cash += holding.cost + 12.5
    after = control.accept_broker_state(ME, "sold in the dashboard", **kw(rig))
    assert after.severity == "ok"
    view = service.get_positions(execution=rig.execution)
    assert view.positions == () and view.cash == pytest.approx(10_000.0 + 12.5)
    assert view.realized == pytest.approx(12.5)
    adjustments = [r.payload for r in rig.records() if r.kind == "adjustment"]
    assert [(a["symbol"], a["quantity_delta"], a["operator"]) for a in adjustments] == [
        ("SPY", -quantity, ME),
        (None, 0, ME),
    ]
    # Nothing left to accept: a second call changes nothing.
    records = len(rig.records())
    control.accept_broker_state(ME, "again", **kw(rig))
    assert [r.kind for r in rig.records()[records:]] == ["reconcile"]


def test_an_adjustment_entered_twice_is_applied_once(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    cash = service.get_positions(execution=rig.execution).cash
    dividend = LedgerAdjustment(
        adjustment_id="2026-07-01-SPY-dividend", as_of=D1, kind="dividend", symbol="SPY",
        cash_delta=4.2, operator=ME, note="quarterly dividend",
    )  # fmt: skip
    assert control.record_adjustment(dividend, **files(rig)) is True
    assert control.record_adjustment(dividend, **files(rig)) is False
    assert service.get_positions(execution=rig.execution).cash == pytest.approx(cash + 4.2)
    assert rig.ledger()["cash"] == pytest.approx(cash + 4.2)
    changed = dividend.model_copy(update={"cash_delta": 5.0})
    with pytest.raises(ControlError, match="already used for a different adjustment"):
        control.record_adjustment(changed, **files(rig))


# --------------------------------------------------------------------------- risk halt


def test_resetting_a_halt_journals_who_why_and_the_old_peak(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    with pytest.raises(ControlError, match="no drawdown halt"):
        control.reset_risk_halt(ME, "why", **files(rig))
    halted = RiskState(
        peak_equity=12_000.0, prior_equity=9_000.0, marked_on=T, halted=True,
        halt_reason="equity 9000.00 is 25.00% below its peak",
    )  # fmt: skip
    save_risk_state(halted, rig.state_dir, rig.profile_id)
    assert service.get_status(execution=rig.execution).stop_level == "entries_halted"
    with pytest.raises(ControlError, match="say why"):
        control.reset_risk_halt(ME, "", **files(rig))
    state = control.reset_risk_halt(ME, "reviewed the drawdown", **files(rig))
    assert (state.halted, state.peak_equity) == (False, 9_000.0)
    assert load_risk_state(rig.state_dir, rig.profile_id) == state
    (record,) = controls(rig)
    assert record == {
        "command": "reset_risk_halt",
        "operator": ME,
        "reason": "reviewed the drawdown",
        "halt_reason": halted.halt_reason,
        "old_peak": 12_000.0,
        "new_peak": 9_000.0,
    }
    assert service.get_status(execution=rig.execution).stop_level == "none"


# --------------------------------------------------------------------------- approvals


def _stamp(path: Path, when: dt.datetime) -> None:
    """Date a command file by the fake clock, as `ClockedInbox` does."""
    os.utime(path, (when.timestamp(), when.timestamp()))


def test_an_approval_quotes_the_hash_that_was_shown(tmp_path: Path) -> None:
    rig = Rig(tmp_path, approval="entries")
    shown: list[Any] = []

    def sleep(seconds: float) -> None:
        if not shown:
            shown.extend(service.list_pending_approvals(**files(rig)))
            (request,) = shown
            with pytest.raises(ControlError, match="not the order that was shown"):
                control.approve_orders({request.order_id: "0" * 64}, ME, **files(rig))
            with pytest.raises(ControlError, match="not waiting for approval"):
                control.approve_orders({"nope": request.content_hash}, ME, **files(rig))
            assert list((rig.state / "inbox").glob("*.json")) == []  # refused: nothing left
            path = control.approve_orders(
                {request.order_id: request.content_hash}, ME, **files(rig)
            )
            _stamp(path, rig.venue.clock)
        rig.venue.advance(seconds)

    rig.sleep = sleep
    report = rig.run()
    (order,) = report.orders
    assert (order.phase, order.approval) == ("filled", "approved")
    approval = next(r.payload for r in rig.records() if r.kind == "approval")
    assert (approval["operator"], approval["decision"]) == (ME, "approved")
    assert service.list_pending_approvals(**files(rig)) == []


def test_a_rejection_cancels_the_entry(tmp_path: Path) -> None:
    rig = Rig(tmp_path, approval="entries")
    done: list[Path] = []

    def sleep(seconds: float) -> None:
        if not done:
            (request,) = service.list_pending_approvals(**files(rig))
            done.append(control.reject_orders([request.order_id], ME, "too big", **files(rig)))
            _stamp(done[0], rig.venue.clock)
        rig.venue.advance(seconds)

    rig.sleep = sleep
    report = rig.run()
    assert [(o.phase, o.reason) for o in report.orders] == [("canceled", "approval_denied")]
    assert rig.venue.place_attempts == []
    with pytest.raises(ControlError, match="not a usable rejection"):
        control.reject_orders([], ME, "why", **files(rig))


# --------------------------------------------------------------------------- activation


def test_activation_goes_through_the_same_checks(tmp_path: Path) -> None:
    rig, _ = holding_spy(tmp_path)
    with pytest.raises(ControlError, match="has traded before"):
        control.activate_profile(rig.profile_id, "off", ME, **files(rig, venue=rig.venue))


def test_the_sell_pass_is_for_sells_inside_the_block_only(tmp_path: Path) -> None:
    switch = KillSwitch(tmp_path / "KILL")
    switch.engage("stop", "operator:test")
    with switch.flattening():
        switch.check(selling=True)
        with pytest.raises(KillSwitchEngaged):
            switch.check()
        with pytest.raises(KillSwitchEngaged):
            KillSwitch(tmp_path / "KILL").check(selling=True)  # another object has no pass
    with pytest.raises(KillSwitchEngaged):
        switch.check(selling=True)
