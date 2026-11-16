"""The open workflow: `OrderManager.on_session_open` from the first exit to the deadline."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

import tests.fake_venue as fake_venue_module
from harness.broker.base import LookAheadError
from harness.broker.venue import KillSwitchEngaged
from harness.journal import read_journal
from harness.orders import OrderManagerError, SessionIncomplete, entry_limit_price
from harness.risk import RiskState
from tests.fake_venue import FakeVenue, FillStep, OrderScript, at
from tests.test_orders import (
    CFG,
    D0,
    D1,
    E_AMD,
    E_MSFT,
    E_NVDA,
    LOOSE,
    X_AAPL,
    X_TSLA,
    Crash,
    Rig,
    order,
    outcome,
    risk_manager,
)

OPEN = at(D1, 9, 30)
EXIT_DEADLINE = at(D1, 9, 40)
DEADLINE = at(D1, 9, 50)


def run(rig: Rig) -> list[tuple[str, str, str, int]]:
    """Run the open and return (order id, status, reason, shares) per update."""
    rig.om.on_session_open(D1, {})
    return [
        (u.order_id, u.status, u.reason, sum(f.quantity for f in u.fills))
        for u in rig.om.poll_updates()
    ]


def placed(rig: Rig) -> list[tuple[str, str, str, int]]:
    """(client id, type, time in force, quantity) of every order the venue accepted."""
    seen: dict[str, tuple[str, str, str, int]] = {}
    for o in rig.venue.place_attempts:
        if o.client_order_id in rig.venue._orders:
            seen[o.client_order_id] = (o.client_order_id, o.type, o.time_in_force, o.quantity)
    return list(seen.values())


def placed_at(rig: Rig, client_order_id: str) -> dt.datetime:
    return rig.venue._orders[client_order_id].eligible_at


# --------------------------------------------------------------------------- the normal morning


def test_exits_go_into_the_auction_and_entries_follow_the_open(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3), order("AAPL", "sell", 10)])  # entry staged first
    rig.venue.calls.clear()
    assert run(rig) == [(X_AAPL, "filled", "", 10), (E_MSFT, "filled", "", 3)]
    assert placed(rig) == [
        (X_AAPL + "-a1", "market", "opg", 10),
        (E_MSFT + "-a1", "limit", "day", 3),
    ]
    assert rig.venue.place_attempts[1].limit_price == 306.0  # 300 * 1.02
    assert rig.venue._orders[E_MSFT + "-a1"].updated_at >= OPEN  # never before the open
    assert rig.venue.clock < at(D1, 9, 31)  # done: it does not sit until the deadline
    assert sorted(rig.om.account().positions) == ["MSFT", "TSLA"]
    assert rig.venue.open_orders() == []
    rig.reconciles()
    # Nothing is polled before the open: an hour and a half of waiting costs no requests.
    assert sum(1 for method, _ in rig.venue.calls if method == "get_order") <= 4
    assert rig.alerts.alerts == []


def test_nothing_to_do_returns_at_once(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.calls.clear()
    start = rig.venue.clock
    rig.om.on_session_open(D1, {})
    assert rig.venue.calls == [] and rig.venue.clock == start
    rig.om.submit([order("AAPL", "sell", 10)])
    run(rig)
    rig.venue.calls.clear()
    rig.om.on_session_open(D1, {})  # everything is final: nothing to do again
    assert rig.venue.calls == [] and rig.om.poll_updates() == []


@pytest.mark.parametrize(
    ("start", "tif"),
    [((9, 24, 59), "opg"), ((9, 25, 0), "day"), ((9, 29, 0), "day"), ((9, 35, 0), "day")],
)
def test_exit_after_the_auction_cutoff_goes_as_a_market_order(
    tmp_path: Path, start: tuple[int, int, int], tif: str
) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.to(*start)
    assert run(rig) == [(X_AAPL, "filled", "", 10)]
    assert placed(rig) == [(X_AAPL + "-a1", "market", tif, 10)]
    rig.reconciles()


def test_what_the_auction_leaves_goes_out_again_as_a_market_order(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(fills=(FillStep(0, 4, 101.0),)))
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert run(rig) == [
        (X_AAPL, "partially_filled", "", 4),
        (X_AAPL, "filled", "", 6),
        (E_MSFT, "filled", "", 3),
    ]
    assert placed(rig) == [
        (X_AAPL + "-a1", "market", "opg", 10),
        (X_AAPL + "-a2", "market", "day", 6),
        (E_MSFT + "-a1", "limit", "day", 3),
    ]
    # The entry waited for the exit to be final, not just for the open.
    assert placed_at(rig, E_MSFT + "-a1") >= placed_at(rig, X_AAPL + "-a2") > OPEN
    assert rig.venue.clock < EXIT_DEADLINE
    rig.reconciles()


def test_auction_order_still_working_at_the_exit_deadline_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fake_venue_module, "OPG_GRACE_SECONDS", 3_600.0)  # the venue is slow
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(fills=()))
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert run(rig) == [(X_AAPL, "filled", "", 10), (E_MSFT, "filled", "", 3)]
    assert placed(rig)[:2] == [
        (X_AAPL + "-a1", "market", "opg", 10),
        (X_AAPL + "-a2", "market", "day", 10),
    ]
    assert rig.venue._orders[X_AAPL + "-a1"].status == "canceled"
    assert placed_at(rig, X_AAPL + "-a2") >= EXIT_DEADLINE
    rig.reconciles()


# --------------------------------------------------------------------------- a rejected auction


def test_exit_the_auction_rejects_goes_out_as_a_day_order_at_once(tmp_path: Path) -> None:
    # Issue #38. The day order goes out under a new client id: the venue may remember
    # the rejected request, and would then refuse its id a second time.
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(reject="opg not accepted"))
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert run(rig) == [(X_AAPL, "filled", "", 10), (E_MSFT, "filled", "", 3)]
    assert [(o.client_order_id, o.time_in_force) for o in rig.venue.place_attempts] == [
        (X_AAPL + "-a1", "opg"),
        (X_AAPL + "-a2", "day"),
        (E_MSFT + "-a1", "day"),
    ]
    # Not held back until the cutoff: the day order was at the venue before the open.
    assert rig.venue._orders[X_AAPL + "-a2"].updated_at < at(D1, 9, 31)
    assert placed_at(rig, E_MSFT + "-a1") < EXIT_DEADLINE
    assert [(a.severity, a.kind) for a in rig.alerts.alerts] == [("critical", "order_rejected")]
    rig.reconciles()


def test_restart_after_a_rejected_auction_order_does_not_try_the_auction_again(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(reject="opg not accepted"))
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "staged"
    rig.restart()
    assert rig.phase(X_AAPL) == "staged"
    assert run(rig) == [(X_AAPL, "filled", "", 10)]  # still before the auction cutoff
    assert [(o.client_order_id, o.time_in_force) for o in rig.venue.place_attempts] == [
        (X_AAPL + "-a1", "opg"),
        (X_AAPL + "-a2", "day"),  # the restarted manager does not reuse the id either
    ]
    assert len(rig.alerts.alerts) == 1
    rig.reconciles()


def test_exit_rejected_by_the_auction_and_then_as_a_day_order_ends_rejected(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.venue.trading_blocked = True  # every order is refused
    rig.om.submit([order("AAPL", "sell", 10)])
    assert run(rig) == [(X_AAPL, "rejected", "broker_rejected", 0)]
    assert [o.time_in_force for o in rig.venue.place_attempts] == ["opg", "day"]  # no loop
    assert rig.om.account().positions["AAPL"].quantity == 10
    assert [a.kind for a in rig.alerts.alerts] == ["order_rejected", "order_rejected"]


@pytest.mark.parametrize("after_write", [False, True])
def test_crash_around_the_rejected_auction_order_still_sells(
    tmp_path: Path, after_write: bool
) -> None:
    baseline = Rig(tmp_path / "baseline")
    baseline.venue.script("AAPL", OrderScript(reject="opg not accepted"))
    baseline.om.submit([order("AAPL", "sell", 10)])
    baseline.om.on_session_open(D1, {})
    kinds = baseline.kinds()
    first = kinds.index("order_staged") + 1
    for seq in range(first, len(kinds) + 1):
        rig = Rig(tmp_path / f"crash-{seq}")
        rig.venue.script("AAPL", OrderScript(reject="opg not accepted"))
        assert rig.journal is not None
        rig.journal.crash_at, rig.journal.crash_after_write = seq, after_write
        with pytest.raises(Crash):
            rig.om.submit([order("AAPL", "sell", 10)])
            rig.om.on_session_open(D1, {})
        rig.restart()
        rig.om.submit([order("AAPL", "sell", 10)])
        rig.om.on_session_open(D1, {})
        assert rig.phase(X_AAPL) == "filled", f"crash at record {seq}"
        assert rig.venue.open_orders() == [] and not rig.kill.engaged()
        rig.reconciles()


# --------------------------------------------------------------------------- exits that do not fill


def test_exit_that_cannot_fill_is_canceled_at_the_deadline_with_an_alert(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.set_price("AAPL", None)  # halted
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert run(rig) == [(E_MSFT, "filled", "", 3), (X_AAPL, "canceled", "session_deadline", 0)]
    assert placed(rig) == [
        (X_AAPL + "-a1", "market", "opg", 10),
        (X_AAPL + "-a2", "market", "day", 10),
        (E_MSFT + "-a1", "limit", "day", 3),
    ]
    # The entry was held back until the exit deadline, then released against actual cash.
    assert placed_at(rig, E_MSFT + "-a1") >= EXIT_DEADLINE
    assert DEADLINE <= rig.venue.clock < DEADLINE + dt.timedelta(seconds=30)
    assert rig.venue.open_orders() == []
    assert rig.om.account().positions["AAPL"].quantity == 10
    (alert,) = rig.alerts.alerts
    assert (alert.severity, alert.kind) == ("critical", "exit_unfilled")
    assert "10 of 10 shares unsold" in alert.message
    rig.reconciles()


def test_unfilled_exit_keeps_its_slot_so_the_position_cap_holds(tmp_path: Path) -> None:
    rig = Rig(tmp_path, max_positions=2)  # AAPL and TSLA are held
    rig.venue.set_price("AAPL", None)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert run(rig) == [
        (E_MSFT, "rejected", "risk_rejected", 0),
        (X_AAPL, "canceled", "session_deadline", 0),
    ]
    assert E_MSFT + "-a1" not in rig.venue._orders  # never sent
    assert sorted(rig.om.account().positions) == ["AAPL", "TSLA"]
    assert [a.kind for a in rig.alerts.alerts] == ["entry_refused", "exit_unfilled"]
    assert "max_positions" in rig.alerts.alerts[0].message


def test_exit_whose_sends_keep_failing_is_bounded(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeouts_before_accept=10_000))
    rig.om.submit([order("AAPL", "sell", 10)])
    with pytest.raises(SessionIncomplete, match="still does not show it"):
        run(rig)
    intents = rig.kinds().count("order_intent")
    assert intents == 1 + CFG.max_submit_attempts  # tried again, but not forever
    assert {o.client_order_id for o in rig.venue.place_attempts} == {X_AAPL + "-a1"}
    # No send was ever answered, so at the deadline it may be at the venue: the day stops.
    assert rig.kill.engaged() and rig.phase(X_AAPL) == "unknown"
    assert rig.alerts.kinds()[-1] == "kill_switch"


# --------------------------------------------------------------------------- entries at release


def test_entry_is_funded_by_the_exit_that_filled_before_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdings={"AAPL": 90})  # $1,000 in cash
    rig.om.submit([order("MSFT", "buy", 15), order("AAPL", "sell", 90)])
    assert run(rig) == [(X_AAPL, "filled", "", 90), (E_MSFT, "filled", "", 15)]
    assert rig.om.account().cash == pytest.approx(10_000 - 4_500)
    rig.reconciles()


def test_entry_is_cut_to_the_cash_that_is_really_there(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdings={"AAPL": 90})
    rig.venue.set_price("AAPL", None)  # the exit that should have funded it does not fill
    rig.om.submit([order("MSFT", "buy", 20), order("AAPL", "sell", 90)])
    updates = run(rig)
    assert updates[0] == (E_MSFT, "filled", "reduced_for_cash", 3)  # 3 * 306 fits in 1,000
    assert placed(rig)[-1] == (E_MSFT + "-a1", "limit", "day", 3)
    working = next(o for o in rig.om.orders() if o.order_id == E_MSFT)
    assert (working.quantity, working.order.quantity, working.filled_quantity) == (3, 20, 3)
    reduced = next(r for r in read_journal(rig.path) if r.kind == "order_reduced")
    assert (reduced.payload["quantity"], reduced.payload["rule"]) == (3, "no_leverage")
    after = read_journal(rig.path)[rig.kinds().index("order_reduced") + 1]
    assert (after.kind, after.payload["order_id"]) == ("order_intent", E_MSFT)  # cut, then sent
    assert rig.om.account().cash >= 0
    rig.reconciles()
    # The cut survives a restart, and the same decision is still the same order.
    rig.restart()
    assert next(o for o in rig.om.orders() if o.order_id == E_MSFT).quantity == 3
    assert rig.om.submit([order("MSFT", "buy", 20)]) == [E_MSFT]


def test_entry_with_no_cash_at_all_is_canceled(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdings={"AAPL": 99})  # $100 in cash
    rig.om.submit([order("MSFT", "buy", 2)])
    assert run(rig) == [(E_MSFT, "canceled", "insufficient_cash", 0)]
    assert rig.venue.place_attempts == [] and rig.alerts.alerts == []


def test_cash_reserved_by_an_earlier_entry_is_not_spent_twice(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdings={"AAPL": 90})  # $1,000 in cash
    rig.venue.set_price("MSFT", 400.0)  # above its cap: the order rests and reserves 612
    rig.om.submit([order("MSFT", "buy", 2), order("NVDA", "buy", 10)])
    assert run(rig) == [
        (E_NVDA, "filled", "reduced_for_cash", 7),  # (1,000 - 612) / 51
        (E_MSFT, "canceled", "limit_not_reached", 0),
    ]
    assert rig.om.account().cash >= 0
    rig.reconciles()


def test_entry_is_never_sent_before_the_open(tmp_path: Path) -> None:
    sent_at: list[dt.datetime] = []

    class Timed(FakeVenue):
        def place(self, order):  # type: ignore[no-untyped-def]
            sent_at.append(self.clock)
            return super().place(order)

    rig = Rig(tmp_path, venue_class=Timed)
    rig.om.submit([order("MSFT", "buy", 3)])  # no exits to wait for
    assert run(rig) == [(E_MSFT, "filled", "", 3)]
    assert sent_at == [OPEN]


def test_entry_released_but_not_filled_yet_counts_toward_the_position_cap(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, max_positions=3)  # AAPL and TSLA are held: one slot is free
    rig.venue.set_price("MSFT", 400.0)  # the first entry rests, holding the slot
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    assert run(rig) == [
        (E_NVDA, "rejected", "risk_rejected", 0),
        (E_MSFT, "canceled", "limit_not_reached", 0),
    ]
    assert [p[0] for p in placed(rig)] == [E_MSFT + "-a1"]


def test_entries_are_released_in_the_order_they_were_staged(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("NVDA", "buy", 4), order("AMD", "buy", 2), order("MSFT", "buy", 3)])
    run(rig)
    assert [p[0] for p in placed(rig)] == [E_NVDA + "-a1", E_AMD + "-a1", E_MSFT + "-a1"]


def test_entry_above_its_cap_is_not_chased(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.set_price("MSFT", 306.01)  # one cent past the cap
    rig.om.submit([order("MSFT", "buy", 3)])
    assert run(rig) == [(E_MSFT, "canceled", "limit_not_reached", 0)]
    assert DEADLINE <= rig.venue.clock < DEADLINE + dt.timedelta(seconds=30)
    assert rig.venue.open_orders() == [] and "MSFT" not in rig.om.account().positions
    assert rig.alerts.alerts == []  # not an error: it is logged for the comparison


def test_partly_filled_entry_is_canceled_at_the_deadline(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("NVDA", OrderScript(fills=(FillStep(0, 2, 50.0), FillStep(90, 5, 50.5))))
    rig.om.submit([order("NVDA", "buy", 10)])
    assert run(rig) == [
        (E_NVDA, "partially_filled", "", 2),
        (E_NVDA, "partially_filled", "", 5),
        (E_NVDA, "canceled", "limit_not_reached", 0),
    ]
    assert rig.om.account().positions["NVDA"].quantity == 7
    rig.reconciles()


def test_entry_the_venue_ended_early_is_not_sent_again(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.set_price("MSFT", 400.0)
    rig.om.submit([order("MSFT", "buy", 3)])
    real = rig.venue.advance
    canceled = False

    def advance(seconds: float) -> None:
        nonlocal canceled
        real(seconds)
        if not canceled and E_MSFT + "-a1" in rig.venue._orders:
            rig.venue.cancel(E_MSFT + "-a1")  # the venue (or the operator) cancels it
            canceled = True

    rig.om._sleep = advance
    assert run(rig) == [(E_MSFT, "canceled", "limit_not_reached", 0)]
    assert len(rig.venue.place_attempts) == 1
    assert rig.venue.clock < DEADLINE


def test_unknown_entry_stops_the_entries_behind_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("MSFT", OrderScript(timeouts_before_accept=99))
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4), order("AAPL", "sell", 10)])
    with pytest.raises(SessionIncomplete, match="still does not show it"):
        run(rig)
    updates = [(u.order_id, u.status, u.reason) for u in rig.om.poll_updates()]
    assert updates == [(X_AAPL, "filled", ""), (E_NVDA, "canceled", "unknown_outcome")]
    assert rig.phase(E_MSFT) == "unknown"  # still being looked for at the deadline: kills
    assert [p[0] for p in placed(rig)] == [X_AAPL + "-a1"]
    assert rig.alerts.kinds() == ["order_unknown", "kill_switch"]
    rig.reconciles()


def test_halted_risk_state_refuses_entries_and_lets_exits_through(tmp_path: Path) -> None:
    halted = RiskState(peak_equity=10_000.0, halted=True, halt_reason="drawdown 16%")
    rig = Rig(tmp_path, risk=risk_manager(LOOSE, halted))
    rig.om.submit([order("MSFT", "buy", 3), order("AAPL", "sell", 10)])
    assert run(rig) == [(X_AAPL, "filled", "", 10), (E_MSFT, "rejected", "risk_rejected", 0)]
    assert [p[0] for p in placed(rig)] == [X_AAPL + "-a1"]
    assert "drawdown_halt" in rig.alerts.alerts[0].message


def test_position_limit_cuts_an_entry_without_the_cash_reason(tmp_path: Path) -> None:
    tight = LOOSE.model_copy(update={"max_position_pct": 0.1})  # $1,000 of $10,000
    rig = Rig(tmp_path, risk=risk_manager(tight))
    rig.om.submit([order("MSFT", "buy", 5)])
    assert run(rig) == [(E_MSFT, "filled", "", 3)]  # 3 * 306 <= 1,000; cash was not the limit
    reduced = next(r for r in read_journal(rig.path) if r.kind == "order_reduced")
    assert reduced.payload["rule"] == "max_position"


@pytest.mark.parametrize(
    ("close", "limit"),
    [(100.0, 102.0), (300.0, 306.0), (33.33, 33.99), (10.01, 10.21), (0.5, 0.51), (0.004, None)],
)
def test_entry_limit_is_the_close_plus_the_cap_rounded_down_to_a_cent(
    close: float, limit: float | None
) -> None:
    assert entry_limit_price(close, 0.02) == limit


# --------------------------------------------------------------------------- stops


def test_kill_switch_engaged_before_the_open_sends_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    rig.kill.engage("operator stop", "operator:cli")
    with pytest.raises(KillSwitchEngaged):
        rig.om.on_session_open(D1, {})
    assert rig.venue.place_attempts == []
    assert [(u.status, u.reason) for u in rig.om.poll_updates()] == [
        ("canceled", "kill_switch")
    ] * 2


def test_kill_switch_engaged_mid_session_cancels_and_stops(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.set_price("MSFT", 400.0)  # the entry rests
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    real = rig.venue.advance

    def advance(seconds: float) -> None:
        real(seconds)
        if rig.venue.clock >= at(D1, 9, 35) and not rig.kill.engaged():
            rig.kill.engage("operator stop", "operator:cli")

    rig.om._sleep = advance
    with pytest.raises(KillSwitchEngaged):
        rig.om.on_session_open(D1, {})
    assert rig.venue.open_orders() == []  # canceled and confirmed before it raised
    assert rig.phase(E_MSFT) == "canceled" and rig.phase(X_AAPL) == "filled"
    assert rig.om.poll_updates()[-1].reason == "kill_switch"
    assert rig.venue.clock < at(D1, 9, 36)
    # The kill switch never sells: positions bought before it stay.
    assert rig.om.account().positions["NVDA"].quantity == 4
    rig.reconciles()


def test_order_decided_today_is_refused_outright(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("TSLA", "sell", 5, created_on=D1)])
    with pytest.raises(LookAheadError):
        rig.om.on_session_open(D1, {})
    assert rig.venue.place_attempts == [] and rig.om.poll_updates() == []


def test_session_that_is_not_today_is_refused(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    with pytest.raises(OrderManagerError, match="is not today"):
        rig.om.on_session_open(dt.date(2024, 1, 4), {})
    assert rig.venue.place_attempts == []


def test_call_after_the_session_deadline_sends_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    rig.to(9, 50)
    assert run(rig) == [
        (X_AAPL, "canceled", "session_deadline", 0),
        (E_MSFT, "canceled", "session_deadline", 0),
    ]
    assert rig.venue.place_attempts == []
    assert rig.alerts.kinds() == ["exit_unfilled"]


def test_deadlines_follow_the_venue_clock_not_the_local_one(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om._now = lambda: rig.venue.clock - dt.timedelta(seconds=20)  # local clock is slow
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.to(9, 25, 10)  # past the cutoff by the venue's clock, before it by the local one
    run(rig)
    assert placed(rig) == [(X_AAPL + "-a1", "market", "day", 10)]


def test_clock_too_far_from_the_venue_kills(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om._now = lambda: rig.venue.clock + dt.timedelta(seconds=45)
    rig.om.submit([order("AAPL", "sell", 10)])
    with pytest.raises(SessionIncomplete, match="clock skew"):
        rig.om.on_session_open(D1, {})
    assert rig.kill.engaged() and rig.venue.place_attempts == []


def test_unreadable_venue_clock_falls_back_to_the_local_clock(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.fail_next("now", times=CFG.max_submit_attempts)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert run(rig) == [(X_AAPL, "filled", "", 10)]
    assert rig.alerts.kinds() == ["clock"] and rig.alerts.alerts[0].severity == "warn"


def test_a_flaky_venue_does_not_stop_the_morning(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    rig.venue.fail_next("get_order", times=4)
    rig.venue.fail_next("place", times=1)
    assert sorted(run(rig)) == [(X_AAPL, "filled", "", 10), (E_MSFT, "filled", "", 3)]
    assert not rig.kill.engaged()
    rig.reconciles()


# --------------------------------------------------------------------------- crash and resume


def morning_rig(tmp_path: Path, reject_duplicate_ids: bool = False) -> Rig:
    rig = Rig(tmp_path, reject_duplicate_ids=reject_duplicate_ids)
    venue: FakeVenue = rig.venue
    venue.script("AAPL", OrderScript(fills=(FillStep(0, 4, 101.0),)))  # rest goes out again
    venue.script("TSLA", OrderScript(timeout_after_accept=True))
    venue.script("MSFT", OrderScript(timeouts_before_accept=1))
    venue.script("NVDA", OrderScript(fills=(FillStep(0, 2, 50.0), FillStep(90, 5, 50.5))))
    venue.script("AMD", OrderScript(fills=(), fill_on_cancel=1))
    return rig


def morning(rig: Rig) -> None:
    """What the runner does: stage the day's orders (idempotent) and run the open."""
    rig.om.submit([order("AAPL", "sell", 10), order("TSLA", "sell", 5)])
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 10), order("AMD", "buy", 6)])
    rig.om.on_session_open(D1, {})


def test_morning_without_a_crash(tmp_path: Path) -> None:
    rig = morning_rig(tmp_path)
    morning(rig)
    assert outcome(rig)["phases"] == [
        (X_AAPL, "filled", 10, ""),
        (X_TSLA, "filled", 5, ""),
        (E_MSFT, "filled", 3, ""),
        (E_NVDA, "canceled", 7, "limit_not_reached"),
        (E_AMD, "canceled", 1, "limit_not_reached"),
    ]
    assert rig.venue.open_orders() == [] and rig.alerts.alerts == []
    rig.reconciles()


@pytest.mark.parametrize("reject_duplicate_ids", [False, True])
def test_crash_at_every_record_of_the_open_ends_in_the_same_place(
    tmp_path: Path, reject_duplicate_ids: bool
) -> None:
    baseline = morning_rig(tmp_path / "baseline", reject_duplicate_ids)
    morning(baseline)
    expected = outcome(baseline)
    records = len(read_journal(baseline.path))
    first = baseline.kinds().index("order_staged") + 1
    assert records - first > 30

    for seq in range(first, records + 1):
        for after_write in (False, True):
            rig = morning_rig(tmp_path / f"crash-{seq}-{int(after_write)}", reject_duplicate_ids)
            assert rig.journal is not None
            rig.journal.crash_at = seq
            rig.journal.crash_after_write = after_write
            with pytest.raises(Crash):
                morning(rig)
            rig.restart()  # a new process: recover, then run the morning again
            morning(rig)
            where = f"crash at record {seq}, {'after' if after_write else 'before'} the write"
            assert outcome(rig) == expected, where
            assert rig.venue.open_orders() == [] and not rig.kill.engaged(), where
            rig.reconciles()


def test_entries_are_still_released_after_a_restart(tmp_path: Path) -> None:
    # The restarted manager is not handed the closes again: it values the positions it
    # holds from its journal. Without that, every entry would be refused.
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    rig.to(9, 20)
    rig.restart()
    assert run(rig) == [(X_AAPL, "filled", "", 10), (E_MSFT, "filled", "", 3)]
    assert rig.alerts.alerts == []


def test_orders_decided_on_d0_only(tmp_path: Path) -> None:
    # Guard for the constants this module leans on.
    assert order("AAPL", "sell", 1).created_on == D0 < D1
