"""The operator's command line (`harness.cli`): each command reaches the function it
names, prints what the operator needs, and exits with the right status."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import harness.cli as cli
import harness.service as service
from harness.profiles import save_risk_state
from harness.risk import RiskState
from tests.test_control import (
    ME,
    _stamp,
    controls,
    holding_spy,
    latch_unwritable,
    runner_running,
)
from tests.test_paper import D1, Rig, T

Capture = pytest.CaptureFixture[str]


def run(rig: Rig, *argv: str, to_file: bool = False) -> int:
    """The command line, against the rig's venue and clock. `to_file`: alerts go to the
    default sinks (`alerts.jsonl`), as they do from a real command line."""
    return cli.main(
        argv,
        venue=rig.venue,
        execution=rig.execution,
        alerts=None if to_file else rig.alerts,
        sleep=rig.venue.advance,
        now=lambda: rig.venue.clock,
    )


def op(*argv: str) -> tuple[str, ...]:
    return (*argv, "--operator", ME)


# --------------------------------------------------------------------------- views


def test_the_views_print_what_the_service_reads(tmp_path: Path, capsys: Capture) -> None:
    rig, quantity = holding_spy(tmp_path)
    assert run(rig, "status") == 0
    out = capsys.readouterr().out
    assert f"profile {rig.profile_id} on account" in out
    assert "kill switch: not engaged" in out and "entries: none" in out
    assert f"session {D1}: complete" in out and "outcome traded" in out

    assert run(rig, "positions") == 0
    out = capsys.readouterr().out
    assert f"  SPY  {quantity} @ " in out and "equity " in out

    assert run(rig, "orders", "--decided-on", T.isoformat()) == 0
    assert f"buy SPY  {quantity}/{quantity}  filled" in capsys.readouterr().out
    assert run(rig, "orders", "--decided-on", D1.isoformat()) == 0
    assert capsys.readouterr().out == "no orders\n"

    assert run(rig, "pending") == 0
    assert capsys.readouterr().out == "nothing is waiting for approval\n"
    assert run(rig, "alerts") == 0
    assert capsys.readouterr().out == "no alerts\n"
    assert run(rig, "profiles") == 0
    assert f"{rig.profile_id}  example_buy_and_hold  plumbing" in capsys.readouterr().out
    assert controls(rig) == []  # reading changed nothing


def test_alerts_can_be_cut_by_time_and_count(tmp_path: Path, capsys: Capture) -> None:
    rig = Rig(tmp_path)
    assert run(rig, *op("kill", "--reason", "first"), to_file=True) == 0
    rig.venue.advance(3600)
    assert run(rig, *op("kill", "--reason", "second"), to_file=True) == 0
    capsys.readouterr()
    assert run(rig, "alerts", "--last", "0") == 0
    assert len(capsys.readouterr().out.splitlines()) == len(
        service.list_alerts(execution=rig.execution)
    )
    assert run(rig, "alerts", "--last", "1") == 0
    assert len(capsys.readouterr().out.splitlines()) == 1
    with pytest.raises(SystemExit) as stopped:
        run(rig, "alerts", "--last", "-3")
    assert stopped.value.code == 2 and "must be 0 or more" in capsys.readouterr().err
    # A time without an offset is New York time: 10:00 is between the two kills.
    assert run(rig, "alerts", "--since", f"{D1}T10:00") == 0
    (line,) = capsys.readouterr().out.splitlines()
    assert line.startswith(f"{D1} 10:15:00 New York  [critical] kill_switch:")


def test_a_damaged_journal_is_an_error_and_a_crash_is_not_exit_1(
    tmp_path: Path, capsys: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig, _ = holding_spy(tmp_path)
    journal = rig.state / "journal.jsonl"
    journal.write_bytes(journal.read_bytes().replace(b'"kind":"fill"', b'"kind":"fell"', 1))
    assert run(rig, "positions") == 2
    err = capsys.readouterr().err
    assert "error: " in err and "not a valid record" in err and "Traceback" not in err
    assert run(rig, "status") == 0  # status lists it as a problem instead
    assert "problem: the journal cannot be read" in capsys.readouterr().out

    def broken(**kw: Any) -> None:
        raise RuntimeError("a bug")

    # Anything uncaught: never 1, which means "it ran, something is left for you".
    monkeypatch.setattr(service, "list_profiles", broken)
    assert run(rig, "profiles") == 3
    err = capsys.readouterr().err
    assert "Traceback" in err and "RuntimeError: a bug" in err and "crashed" in err


# --------------------------------------------------------------------------- kill switch


def test_kill_from_a_second_terminal_then_reset(tmp_path: Path, capsys: Capture) -> None:
    rig = Rig(tmp_path, approval="entries")
    codes: list[int] = []

    def sleep(seconds: float) -> None:
        if not codes:  # while the runner waits for an approval
            codes.append(run(rig, *op("kill", "--reason", "stop now")))
        rig.venue.advance(seconds)

    rig.sleep = sleep
    assert rig.run().outcome == "killed" and codes == [0]
    out = capsys.readouterr().out
    assert f"kill switch engaged by operator:{ME}: stop now" in out
    assert "every open order at the venue is canceled" in out
    assert rig.venue.place_attempts == []

    assert run(rig, "status") == 0
    out = capsys.readouterr().out
    assert "kill switch: ENGAGED" in out and "entries: killed" in out
    assert rig.run().outcome == "killed"  # the next run refuses

    for wrong in ("wrong-0a1b2c3d", "none"):  # "none" is for a latch with no profile
        assert run(rig, *op("reset-kill", "--confirm", wrong)) == 2
        assert "does not match the profile id" in capsys.readouterr().err
    assert rig.kill_switch().engaged()
    assert run(rig, *op("reset-kill", "--confirm", rig.profile_id)) == 0
    out = capsys.readouterr().out
    assert "reconcile: ok" in out and "Start the runner again" in out
    assert not rig.kill_switch().engaged()
    assert controls(rig)[-1]["operator"] == ME


def test_a_kill_that_cannot_confirm_the_cancel_exits_1(tmp_path: Path, capsys: Capture) -> None:
    rig = Rig(tmp_path)
    (rig.state_dir / "active.json").unlink()  # no profile: no venue to cancel at
    assert run(rig, *op("kill", "--reason", "testing")) == 1
    assert "no profile is active, so no venue was asked to cancel" in capsys.readouterr().out
    assert rig.kill_switch().engaged()
    assert run(rig, *op("reset-kill", "--confirm", "none")) == 0


def test_a_kill_whose_latch_was_not_written_exits_1_and_says_so(
    tmp_path: Path, capsys: Capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig, _ = holding_spy(tmp_path)
    latch_unwritable(monkeypatch)
    assert run(rig, *op("kill", "--reason", "testing")) == 1  # although the cancel is confirmed
    out = capsys.readouterr().out
    assert "THE LATCH FILE WAS NOT WRITTEN, so nothing stops the next run" in out
    assert "Open orders were canceled" in out and "Do not start the runner" in out
    assert not rig.kill_switch().engaged()  # the truth `status` shows


# --------------------------------------------------------------------------- approvals


def test_approve_takes_what_pending_printed(tmp_path: Path, capsys: Capture) -> None:
    rig = Rig(tmp_path, approval="entries")
    done: list[str] = []

    def sleep(seconds: float) -> None:
        if not done:
            capsys.readouterr()
            assert run(rig, "pending") == 0
            shown = capsys.readouterr().out
            assert "buy SPY  at most " in shown
            quoted = shown.splitlines()[-1].strip()  # ORDER_ID=HASH
            order_id, content_hash = quoted.split("=")
            assert run(rig, *op("approve", order_id)) == 2
            assert "is not ORDER_ID=HASH" in capsys.readouterr().err
            assert run(rig, *op("approve", f"{order_id}={'0' * 64}")) == 2
            assert "not the order that was shown" in capsys.readouterr().err
            assert run(rig, *op("approve", quoted, f"{order_id}={'0' * 64}")) == 2
            assert "given twice" in capsys.readouterr().err
            assert list((rig.state / "inbox").glob("*.json")) == []
            assert run(rig, *op("approve", quoted)) == 0
            assert f"approval of {order_id} left for the runner" in capsys.readouterr().out
            (path,) = (rig.state / "inbox").glob("*.json")
            _stamp(path, rig.venue.clock)
            done.append(content_hash)
        rig.venue.advance(seconds)

    rig.sleep = sleep
    (order,) = rig.run().orders
    assert (order.phase, order.approval) == ("filled", "approved")
    approval = next(r.payload for r in rig.records() if r.kind == "approval")
    assert (approval["operator"], approval["decision"]) == (ME, "approved")


def test_reject_cancels_the_entry(tmp_path: Path, capsys: Capture) -> None:
    rig = Rig(tmp_path, approval="entries")
    done: list[bool] = []

    def sleep(seconds: float) -> None:
        if not done:
            (request,) = service.list_pending_approvals(
                execution=rig.execution, now=lambda: rig.venue.clock
            )
            assert run(rig, *op("reject", request.order_id, "--reason", "too big")) == 0
            assert "a refusal is an alert" in capsys.readouterr().out
            (path,) = (rig.state / "inbox").glob("*.json")
            _stamp(path, rig.venue.clock)
            done.append(True)
        rig.venue.advance(seconds)

    rig.sleep = sleep
    assert [(o.phase, o.reason) for o in rig.run().orders] == [("canceled", "approval_denied")]
    assert rig.venue.place_attempts == []


# --------------------------------------------------------------------------- the books


def test_flatten_sells_everything_and_leaves_the_switch_engaged(
    tmp_path: Path, capsys: Capture
) -> None:
    rig, quantity = holding_spy(tmp_path)
    assert run(rig, *op("flatten", "--confirm", "wrong-0a1b2c3d")) == 2
    assert "does not match the profile id" in capsys.readouterr().err
    assert rig.held() == {"SPY": quantity} and not rig.kill_switch().engaged()
    with runner_running(rig):
        assert run(rig, *op("flatten", "--confirm", rig.profile_id)) == 2
        assert "the runner (or another command) is running" in capsys.readouterr().err
    assert run(rig, *op("flatten", "--confirm", rig.profile_id)) == 0
    out = capsys.readouterr().out
    assert f"sell SPY  {quantity}/{quantity}  filled" in out
    assert "reconcile: ok" in out and "The ledger holds nothing." in out
    assert rig.held() == {} and rig.kill_switch().engaged()


def test_a_flatten_that_leaves_a_position_exits_1(tmp_path: Path, capsys: Capture) -> None:
    rig, quantity = holding_spy(tmp_path)
    rig.venue._holdings.pop("SPY")  # sold by hand: the venue has nothing to sell
    assert run(rig, *op("flatten", "--confirm", rig.profile_id)) == 1
    out = capsys.readouterr().out
    assert "difference: " in out and f"STILL HELD: SPY {quantity}" in out


def test_accept_then_reset_after_a_trade_made_by_hand(tmp_path: Path, capsys: Capture) -> None:
    rig = Rig(tmp_path)
    rig.venue.hold("AAPL", 1, 50.0)
    assert rig.run().outcome == "killed"
    assert run(rig, *op("reset-kill", "--confirm", rig.profile_id)) == 2
    assert "AAPL: ledger holds 0, venue holds 1" in capsys.readouterr().err
    assert run(rig, *op("accept", "--note", "bought by hand")) == 0
    assert "reconcile after: ok" in capsys.readouterr().out
    assert run(rig, *op("reset-kill", "--confirm", rig.profile_id)) == 0
    assert rig.run().outcome == "traded"


def test_adjust_records_an_event_once(tmp_path: Path, capsys: Capture) -> None:
    rig, _ = holding_spy(tmp_path)
    cash = service.get_positions(execution=rig.execution).cash
    argv = op(
        "adjust", "--id", "2026-07-01-SPY-dividend", "--as-of", D1.isoformat(),
        "--kind", "dividend", "--symbol", "SPY", "--cash", "4.2", "--note", "quarterly",
    )  # fmt: skip
    assert run(rig, *argv) == 0
    assert "recorded" in capsys.readouterr().out
    assert run(rig, *argv) == 0
    assert "already recorded; nothing changed" in capsys.readouterr().out
    assert service.get_positions(execution=rig.execution).cash == pytest.approx(cash + 4.2)
    (record,) = [r.payload for r in rig.records() if r.kind == "adjustment"]
    assert (record["operator"], record["note"]) == (ME, "quarterly")
    # Neither cash nor shares: the model refuses, and the refusal reads as an answer.
    empty = op("adjust", "--id", "x", "--as-of", D1.isoformat(), "--kind", "fee", "--note", "n")
    assert run(rig, *empty) == 2
    assert "not a usable adjustment, nothing was recorded" in capsys.readouterr().err


def test_reset_halt(tmp_path: Path, capsys: Capture) -> None:
    rig, _ = holding_spy(tmp_path)
    assert run(rig, *op("reset-halt", "--reason", "why")) == 2
    assert "no drawdown halt" in capsys.readouterr().err
    halted = RiskState(
        peak_equity=12_000.0, prior_equity=9_000.0, marked_on=T, halted=True,
        halt_reason="equity 9000.00 is 25.00% below its peak",
    )  # fmt: skip
    save_risk_state(halted, rig.state_dir, rig.profile_id)
    assert run(rig, "status") == 0
    assert "HALTED: equity 9000.00" in capsys.readouterr().out
    assert run(rig, *op("reset-halt", "--reason", "reviewed the drawdown")) == 0
    assert "the peak is now 9000.00" in capsys.readouterr().out
    assert controls(rig)[-1]["reason"] == "reviewed the drawdown"


# --------------------------------------------------------------------------- the surface


@pytest.mark.parametrize(
    "argv",
    [
        ("kill", "--reason", "r"),
        ("reset-kill", "--confirm", "none"),
        ("approve", "a=b"),
        ("reject", "a", "--reason", "r"),
        ("reset-halt", "--reason", "r"),
        ("flatten", "--confirm", "x"),
        ("accept", "--note", "n"),
        ("adjust", "--id", "x", "--as-of", "2026-07-01", "--kind", "fee", "--note", "n"),
    ],
)
def test_every_command_that_acts_needs_an_operator(
    argv: tuple[str, ...], tmp_path: Path, capsys: Capture
) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        run(rig, *argv)
    assert stopped.value.code == 2 and "--operator" in capsys.readouterr().err
    assert not rig.kill_switch().engaged() and controls(rig) == []
    assert run(rig, *argv, "--operator", " ") == 2  # a blank name is refused by the command


def test_the_command_line_covers_the_control_surface() -> None:
    """A command added to `harness.control` needs a line here: either the CLI command
    that reaches it, or the reason it has none."""
    reached: dict[str, Any] = {
        "engage_kill_switch": "kill",
        "reset_kill_switch": "reset-kill",
        "approve_orders": "approve",
        "reject_orders": "reject",
        "reset_risk_halt": "reset-halt",
        "flatten": "flatten",
        "accept_broker_state": "accept",
        "record_adjustment": "adjust",
        "promote_run": None,  # python -m harness.paper promote
        "activate_profile": None,  # python -m harness.paper activate
    }
    import harness.control as control

    public = {
        name
        for name, value in vars(control).items()
        if callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", "") == control.__name__
        and not isinstance(value, type)
    }
    assert public == set(reached)
    assert {c for c in reached.values() if c} <= set(cli._HANDLERS)
