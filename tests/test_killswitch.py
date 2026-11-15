"""Kill switch: the latch fails closed, and `kill` latches before it cancels."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from harness.alerts import Alert
from harness.broker.venue import KillSwitchEngaged, VenueOrder, VenueOrderState
from harness.killswitch import KillState, KillSwitch, KillSwitchError, kill
from tests.fake_venue import FakeVenue, at

DAY = dt.date(2024, 1, 3)
T0 = dt.datetime(2024, 1, 3, 13, 0, tzinfo=dt.UTC)


def switch(path: Path) -> KillSwitch:
    return KillSwitch(path, now=lambda: T0)


class Recorder:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


def venue_with_orders(cancel_latency: float = 1.0) -> FakeVenue:
    """Pre-open, holding AAPL, with one exit and one entry working."""
    venue = FakeVenue(clock=at(DAY, 8, 0), cancel_latency=cancel_latency)
    venue.set_price("AAPL", 100.0)
    venue.set_price("MSFT", 300.0)
    venue.hold("AAPL", 10, 90.0)
    venue.place(
        VenueOrder(
            client_order_id="AAPL-x-a1", symbol="AAPL", side="sell", quantity=10,
            type="market", time_in_force="opg",
        )
    )  # fmt: skip
    venue.place(
        VenueOrder(
            client_order_id="MSFT-e-a1", symbol="MSFT", side="buy", quantity=3,
            type="limit", limit_price=306.0, time_in_force="day",
        )
    )  # fmt: skip
    assert len(venue.open_orders()) == 2
    return venue


# --------------------------------------------------------------------------- the latch


def test_latch_starts_clear_and_engages_for_every_process(tmp_path: Path) -> None:
    path = tmp_path / "state" / "KILL"  # the directory is created
    latch = switch(path)
    assert not latch.engaged() and latch.state() == KillState(engaged=False)
    latch.check()

    state = latch.engage("reconciliation stop", "auto:reconcile")
    assert state == KillState(
        engaged=True, at=T0, reason="reconciliation stop", source="auto:reconcile"
    )
    other = KillSwitch(path)  # as another process would see it
    assert other.engaged() and other.state() == state
    with pytest.raises(KillSwitchEngaged):
        other.check()
    assert json.loads(path.read_text())["reason"] == "reconciliation stop"
    assert os.listdir(path.parent) == ["KILL"]  # no temp file left behind


def test_engaging_twice_keeps_the_first_reason(tmp_path: Path) -> None:
    path = tmp_path / "KILL"
    first = switch(path).engage("first", "operator:cli")
    later = KillSwitch(path, now=lambda: T0 + dt.timedelta(hours=1))
    assert later.engage("second", "auto:reconcile") == first
    assert later.state() == first and switch(path).state() == first


@pytest.mark.parametrize("damage", ["empty", "garbage", "wrong_shape", "directory", "symlink"])
def test_latch_that_cannot_be_read_counts_as_engaged(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "KILL"
    if damage == "empty":
        path.write_bytes(b"")
    elif damage == "garbage":
        path.write_bytes(b"\x00\xff not json")
    elif damage == "wrong_shape":
        path.write_text('{"engaged": false, "surprise": 1}')
    elif damage == "directory":
        path.mkdir()
    else:
        path.symlink_to(tmp_path / "nowhere")  # dangling
    latch = switch(path)
    assert latch.engaged()
    with pytest.raises(KillSwitchEngaged):
        latch.check()
    state = latch.state()
    assert state.engaged and "cannot be read" in state.reason and state.source == "unknown"


def test_latch_file_saying_not_engaged_is_still_engaged(tmp_path: Path) -> None:
    path = tmp_path / "KILL"
    path.write_text(KillState(engaged=False).model_dump_json())
    latch = switch(path)
    assert latch.engaged()  # the file existing is what counts
    assert latch.state().engaged and "says not engaged" in latch.state().reason


def test_latch_whose_presence_cannot_be_checked_counts_as_engaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latch = switch(tmp_path / "KILL")

    def denied(path: object) -> None:
        raise PermissionError("no access")

    monkeypatch.setattr(os, "lstat", denied)
    assert latch.engaged()
    with pytest.raises(KillSwitchEngaged):
        latch.check()


def test_engage_that_cannot_write_still_latches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "KILL"
    latch = switch(path)

    def disk_full(*args: object, **kw: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(tempfile, "mkstemp", disk_full)
        with pytest.raises(KillSwitchError, match="could not write the kill latch"):
            latch.engage("clock skew", "auto:runner")
    assert latch.engaged() and latch.state().reason == "clock skew"
    # The fallback left a bare file, which other processes read as engaged.
    assert KillSwitch(path).engaged()


def test_engage_latches_this_process_even_if_no_file_can_be_made(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    latch = switch(blocker / "KILL")  # the parent is a file: nothing can be created
    with pytest.raises(KillSwitchError):
        latch.engage("stop", "operator:cli")
    assert latch.engaged()
    assert latch.state() == KillState(engaged=True, at=T0, reason="stop", source="operator:cli")


def test_engage_latches_this_process_when_nothing_reaches_the_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "KILL"
    latch = switch(path)

    def disk_full(*args: object, **kw: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(tempfile, "mkstemp", disk_full)
    monkeypatch.setattr(KillSwitch, "_engage_bare", lambda self: None)
    with pytest.raises(KillSwitchError):
        latch.engage("stop", "operator:cli")
    assert not path.exists()
    assert latch.engaged() and latch.state().reason == "stop"
    with pytest.raises(KillSwitchEngaged):
        latch.check()
    latch.reset("david", "p1", "p1")  # an operator can still clear it
    assert not latch.engaged()


def test_engage_works_with_a_broken_clock(tmp_path: Path) -> None:
    def broken() -> dt.datetime:
        raise RuntimeError("no clock")

    naive = KillSwitch(tmp_path / "A", now=lambda: dt.datetime(2024, 1, 3, 13, 0))
    assert naive.engage("stop", "operator:cli").engaged and naive.engaged()
    raising = KillSwitch(tmp_path / "B", now=broken)
    assert raising.engage("stop", "operator:cli").engaged and KillSwitch(tmp_path / "B").engaged()


def test_reset_needs_the_profile_id_typed_back(tmp_path: Path) -> None:
    path = tmp_path / "KILL"
    latch = switch(path)
    with pytest.raises(KillSwitchError, match="not engaged"):
        latch.reset("david", "p1", "p1")
    engaged = latch.engage("stop", "operator:cli")
    for operator, confirm, profile in (
        ("david", "p2", "p1"),
        ("david", "", "p1"),
        ("david", "", ""),
        ("", "p1", "p1"),
        ("  ", "p1", "p1"),
    ):
        with pytest.raises(KillSwitchError):
            latch.reset(operator, confirm, profile)
        assert latch.engaged() and KillSwitch(path).engaged()
    assert latch.reset("david", "p1", "p1") == engaged  # what was cleared, for the audit trail
    assert not latch.engaged() and not path.exists()
    latch.check()
    assert latch.engage("again", "operator:cli").reason == "again"


def test_reset_clears_an_unreadable_latch_but_not_one_it_cannot_remove(tmp_path: Path) -> None:
    path = tmp_path / "KILL"
    path.write_bytes(b"garbage")
    cleared = switch(path).reset("david", "p1", "p1")
    assert "cannot be read" in cleared.reason and not path.exists()
    path.mkdir()
    with pytest.raises(KillSwitchError, match="could not remove"):
        switch(path).reset("david", "p1", "p1")
    assert switch(path).engaged()


def test_kill_from_another_process_stops_a_running_one(tmp_path: Path) -> None:
    path = tmp_path / "KILL"
    runner = switch(path)  # the runner built its switch before the kill
    runner.check()
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from harness.killswitch import KillSwitch, kill\n"
        "state = kill(KillSwitch(Path(sys.argv[1])), 'operator stop', 'operator:cli')\n"
        "sys.exit(0 if state.engaged else 1)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    with pytest.raises(KillSwitchEngaged):
        runner.check()  # the next check between steps
    assert runner.state().reason == "operator stop" and runner.state().source == "operator:cli"


# --------------------------------------------------------------------------- kill


def test_kill_latches_then_cancels_and_confirms(tmp_path: Path) -> None:
    latch = switch(tmp_path / "KILL")
    venue = venue_with_orders(cancel_latency=1.0)
    alerts = Recorder()
    venue.calls.clear()
    state = kill(
        latch, "operator stop", "operator:cli",
        venue=venue, alerts=alerts, sleep=venue.advance, now=lambda: T0,
    )  # fmt: skip
    assert state == KillState(
        engaged=True, at=T0, reason="operator stop", source="operator:cli", cancel_confirmed=True
    )
    assert KillSwitch(latch.path).state() == state  # recorded in the latch file
    assert venue.open_orders() == []
    # Cancels only: nothing was placed and the position is untouched.
    assert {method for method, _ in venue.calls} == {"cancel_all", "open_orders"}
    assert [(p.symbol, p.quantity) for p in venue.positions()] == [("AAPL", 10)]
    (alert,) = alerts.alerts
    assert (alert.severity, alert.kind, alert.at) == ("critical", "kill_switch", T0)
    assert alert.message == (
        "kill switch engaged by operator:cli: operator stop. "
        "No open orders remain at the venue. Positions were not touched."
    )


def test_latch_is_set_before_the_first_venue_call(tmp_path: Path) -> None:
    latch = switch(tmp_path / "KILL")
    seen: list[bool] = []

    class Spy(FakeVenue):
        def cancel_all(self) -> None:
            seen.append(KillSwitch(latch.path).engaged())
            super().cancel_all()

    kill(latch, "stop", "operator:cli", venue=Spy(clock=at(DAY, 8, 0)), sleep=lambda s: None)
    assert seen == [True]


def test_kill_retries_a_venue_that_fails_once(tmp_path: Path) -> None:
    latch = switch(tmp_path / "KILL")
    venue = venue_with_orders()
    venue.fail_next("cancel_all")
    alerts = Recorder()
    state = kill(latch, "stop", "operator:cli", venue=venue, alerts=alerts, sleep=venue.advance)
    assert state.cancel_confirmed and venue.open_orders() == []
    assert "PROBLEM" not in alerts.alerts[0].message


def test_cancel_failure_leaves_the_latch_set_and_says_so(tmp_path: Path) -> None:
    latch = switch(tmp_path / "KILL")
    venue = venue_with_orders()
    venue.fail_next("cancel_all", times=99)
    alerts = Recorder()
    slept: list[float] = []
    state = kill(
        latch, "stop", "operator:cli",
        venue=venue, alerts=alerts, attempts=3, wait_seconds=2.0, sleep=slept.append,
    )  # fmt: skip
    assert state.engaged and not state.cancel_confirmed
    assert latch.engaged() and not KillSwitch(latch.path).state().cancel_confirmed
    assert slept == [2.0, 2.0]  # three attempts, a wait between them
    (alert,) = alerts.alerts
    assert alert.severity == "critical"
    assert "PROBLEM: open orders are NOT confirmed canceled: VenueUnavailable" in alert.message


def test_orders_still_open_after_every_attempt_are_reported(tmp_path: Path) -> None:
    latch = switch(tmp_path / "KILL")
    venue = venue_with_orders(cancel_latency=3_600.0)  # the venue is slow to cancel
    alerts = Recorder()
    state = kill(latch, "stop", "operator:cli", venue=venue, alerts=alerts, sleep=venue.advance)
    assert state.engaged and not state.cancel_confirmed
    assert "NOT confirmed canceled: 2 order(s) still open at the venue" in alerts.alerts[0].message


def test_bad_venue_data_counts_as_not_confirmed(tmp_path: Path) -> None:
    class Garbled(FakeVenue):
        def open_orders(self) -> list[VenueOrderState]:
            raise ValueError("unparseable response")

    alerts = Recorder()
    state = kill(
        switch(tmp_path / "KILL"), "stop", "operator:cli",
        venue=Garbled(clock=at(DAY, 8, 0)), alerts=alerts, sleep=lambda s: None,
    )  # fmt: skip
    assert state.engaged and not state.cancel_confirmed
    assert "ValueError: unparseable response" in alerts.alerts[0].message


def test_kill_without_a_venue_says_orders_were_not_canceled(tmp_path: Path) -> None:
    latch = switch(tmp_path / "KILL")
    alerts = Recorder()
    state = kill(latch, "stop", "operator:cli", alerts=alerts)
    assert state.engaged and not state.cancel_confirmed and latch.engaged()
    assert "no venue was given, so open orders were NOT canceled" in alerts.alerts[0].message


def test_alert_sink_that_raises_does_not_break_kill(tmp_path: Path) -> None:
    class Broken:
        def send(self, alert: Alert) -> None:
            raise RuntimeError("webhook down")

    latch = switch(tmp_path / "KILL")
    venue = venue_with_orders()
    state = kill(latch, "stop", "operator:cli", venue=venue, alerts=Broken(), sleep=venue.advance)
    assert state.cancel_confirmed and latch.engaged() and venue.open_orders() == []


def test_kill_still_cancels_when_the_latch_cannot_be_written(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    latch = switch(blocker / "KILL")
    venue = venue_with_orders()
    alerts = Recorder()
    state = kill(latch, "stop", "operator:cli", venue=venue, alerts=alerts, sleep=venue.advance)
    assert state.engaged and state.cancel_confirmed and latch.engaged()
    assert venue.open_orders() == []
    message = alerts.alerts[0].message
    assert "PROBLEM: could not write the kill latch" in message
    assert "other processes may not be" in message


def test_kill_when_already_engaged_names_the_first_reason_and_still_cancels(
    tmp_path: Path,
) -> None:
    latch = switch(tmp_path / "KILL")
    latch.engage("reconciliation stop", "auto:reconcile")
    venue = venue_with_orders()
    alerts = Recorder()
    state = kill(latch, "operator stop", "operator:cli", venue=venue, alerts=alerts,
                 sleep=venue.advance)  # fmt: skip
    assert (state.reason, state.source, state.cancel_confirmed) == (
        "reconciliation stop", "auto:reconcile", True,
    )  # fmt: skip
    assert venue.open_orders() == []
    assert (
        "kill switch engaged by operator:cli: operator stop. "
        "It was already engaged by auto:reconcile: reconciliation stop."
    ) in alerts.alerts[0].message


def test_kill_survives_a_sleep_or_clock_that_raises(tmp_path: Path) -> None:
    def broken_clock() -> dt.datetime:
        raise RuntimeError("broken")

    def broken_sleep(seconds: float) -> None:
        raise RuntimeError("broken")

    latch = switch(tmp_path / "KILL")
    venue = venue_with_orders(cancel_latency=3_600.0)
    alerts = Recorder()
    state = kill(
        latch,
        "stop",
        "operator:cli",
        venue=venue,
        alerts=alerts,
        sleep=broken_sleep,
        now=broken_clock,
    )
    assert state.engaged and not state.cancel_confirmed
    assert alerts.alerts[0].severity == "critical"  # still sent, stamped with the real clock
