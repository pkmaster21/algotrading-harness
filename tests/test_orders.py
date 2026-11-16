"""Order manager core: staging, ids, the submission protocol, cancel, and recovery."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Literal

import pytest

from harness.alerts import Alert
from harness.broker.venue import KillSwitchEngaged, VenueOrderState, VenueUnavailable
from harness.config import ExecutionConfig, RiskConfig
from harness.inbox import Command, Inbox
from harness.journal import Journal, JournalKind, read_journal
from harness.killswitch import KillSwitch
from harness.models import Fill, Leg, OptionContract, Order, Side, Stock
from harness.orders import (
    ApprovalMode,
    OrderManager,
    OrderManagerError,
    OrderPhase,
    SessionIncomplete,
    order_id_for,
)
from harness.positions import Ledger, reconcile_venue
from harness.risk import RiskManager, RiskState
from tests.fake_venue import FakeVenue, FillStep, OrderScript, at

PROFILE = "plumbing-0a1b2c3d"
DM1 = dt.date(2023, 12, 29)  # the session before the signal
D0 = dt.date(2024, 1, 2)  # the signal date
D1 = dt.date(2024, 1, 3)  # the session orders are sent in
CLOSES = {"AAPL": 100.0, "TSLA": 200.0, "MSFT": 300.0, "NVDA": 50.0, "AMD": 150.0}
CFG = ExecutionConfig.model_validate(
    {
        "entry_limit_pct": 0.02,
        "approval_deadline": "09:20",
        "opg_cutoff": "09:25",
        "exit_deadline": "09:40",
        "session_deadline": "09:50",
        "poll_seconds": 5.0,
        "max_submit_attempts": 3,
        "cancel_confirm_seconds": 30.0,
        "max_requests_per_minute": 150,
        "cash_tolerance": 1.0,
        "max_clock_skew_seconds": 30.0,
        "limits": {"max_order_notional": 5000.0, "max_orders_per_session": 25},
    }
)


LOOSE = RiskConfig(
    name="loose",
    max_position_pct=1.0,
    max_exposure_pct=1.0,
    daily_loss_cap_pct=0.5,
    max_drawdown_halt_pct=0.5,
)


def risk_manager(cfg: RiskConfig = LOOSE, state: RiskState | None = None) -> RiskManager:
    return RiskManager(cfg, CFG.limits, CFG.entry_limit_pct, state or RiskState.initial(10_000.0))


class Crash(BaseException):
    """The process died. Not an Exception, so no handler in the harness can swallow it."""


class CrashingJournal(Journal):
    """A journal that kills the process at one chosen record, before or after writing it."""

    crash_at: int | None = None
    crash_after_write = False

    def append(self, kind: JournalKind, payload) -> int:  # type: ignore[no-untyped-def]
        if self.crash_at is not None and self.last_seq + 1 == self.crash_at:
            if self.crash_after_write:
                super().append(kind, payload)
            raise Crash(f"at record {self.crash_at} ({kind})")
        return super().append(kind, payload)


class Tampering(FakeVenue):
    """Base for venues that misreport an order: the list of open orders goes through
    `get_order`, so a subclass that overrides `get_order` misreports it everywhere."""

    def open_orders(self) -> list[VenueOrderState]:
        states = [self.get_order(s.client_order_id) for s in super().open_orders()]
        return [s for s in states if s is not None and not s.is_terminal]


class ClockedInbox(Inbox):
    """An inbox whose files are stamped with the fake clock instead of the real one: a
    command's file is dated when the command says it was decided. A test that needs
    the two to differ sets the file's time itself."""

    def put(self, command: Command) -> Path:
        path = super().put(command)
        os.utime(path, (command.at.timestamp(), command.at.timestamp()))
        return path


class Recorder:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)

    def kinds(self) -> list[str]:
        return [a.kind for a in self.alerts]


def order(symbol: str, side: Side, quantity: int, created_on: dt.date = D0) -> Order:
    return Order(
        created_on=created_on,
        action="enter" if side == "buy" else "exit",
        legs=(Leg(instrument=Stock(symbol=symbol), side=side),),
        quantity=quantity,
        strength=1.0 if side == "buy" else None,
    )


class Rig:
    """A fake venue, a journal on disk, and an order manager that can be restarted."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        holdings: dict[str, int] | None = None,
        cfg: ExecutionConfig = CFG,
        venue_class: type[FakeVenue] = FakeVenue,
        cancel_latency: float = 1.0,
        reject_duplicate_ids: bool = False,
        risk: RiskManager | None = None,
        max_positions: int = 10,
        approval: ApprovalMode = "off",
    ) -> None:
        self.holdings = {"AAPL": 10, "TSLA": 5} if holdings is None else holdings
        self.cfg = cfg
        self.risk = risk or risk_manager()
        self.max_positions = max_positions
        self.approval: ApprovalMode = approval
        self.inbox = ClockedInbox(tmp_path / "inbox")
        spent = sum(q * CLOSES[s] for s, q in self.holdings.items())
        self.venue = venue_class(
            clock=at(D1, 8, 0),
            cash=10_000.0 - spent,
            cancel_latency=cancel_latency,
            reject_duplicate_ids=reject_duplicate_ids,
        )
        for symbol, close in CLOSES.items():
            self.venue.set_price(symbol, close)
        for symbol, quantity in self.holdings.items():
            self.venue.hold(symbol, quantity, CLOSES[symbol])
        self.path = tmp_path / "journal.jsonl"
        self.kill = KillSwitch(tmp_path / "KILL")
        self.alerts = Recorder()
        self.journal: CrashingJournal | None = None
        self.start()

    def start(
        self, recover: bool = True, crash_at: int | None = None, crash_after_write: bool = False
    ) -> None:
        self.journal = CrashingJournal(self.path)
        self.journal.crash_at = crash_at
        self.journal.crash_after_write = crash_after_write
        fresh = self.journal.last_seq == 0
        self.ledger = Ledger.open(self.journal, 10_000.0)
        if fresh:  # positions bought in an earlier session
            for symbol, quantity in self.holdings.items():
                seed = Fill(
                    order_id=f"seed-{symbol}", kind="venue", signal_date=DM1, fill_date=D0,
                    price_date=D0, instrument=Stock(symbol=symbol), side="buy",
                    quantity=quantity, reference_price=CLOSES[symbol], price=CLOSES[symbol],
                    slippage_cost=0.0, commission=0.0, filled_at=at(D0, 9, 31),
                )  # fmt: skip
                self.ledger.apply_fill(seed, quantity)
        self.om = OrderManager(
            self.venue, self.ledger, self.journal, self.kill, self.alerts, self.cfg, PROFILE,
            self.risk, self.max_positions, self.approval, self.inbox,
            sleep=self.venue.advance, now=lambda: self.venue.clock,
        )  # fmt: skip
        if recover:
            self.om.recover()
            if fresh:  # a restarted manager has the closes from its journal
                self.om.set_closes(CLOSES)

    def restart(
        self, recover: bool = True, crash_at: int | None = None, crash_after_write: bool = False
    ) -> None:
        assert self.journal is not None
        self.journal.close()
        self.start(recover, crash_at, crash_after_write)

    def phase(self, order_id: str) -> OrderPhase:
        return next(o.phase for o in self.om.orders() if o.order_id == order_id)

    def kinds(self) -> list[str]:
        return [r.kind for r in read_journal(self.path)]

    def places(self, client_order_id: str) -> int:
        return sum(o.client_order_id == client_order_id for o in self.venue.place_attempts)

    def to(self, hour: int, minute: int, second: int = 0) -> None:
        """Move the clock forward to this time on D1 (never back)."""
        self.venue.set_now(max(self.venue.clock, at(D1, hour, minute, second)))

    def send_exit(self, order_id: str, tif: Literal["opg", "day"] = "opg") -> OrderPhase:
        return self.om.send(order_id, D1, type="market", time_in_force=tif)

    def send_entry(self, order_id: str, limit: float) -> OrderPhase:
        return self.om.send(order_id, D1, type="limit", time_in_force="day", limit_price=limit)

    def reconciles(self) -> None:
        report = reconcile_venue(self.ledger.account(), [], self.venue, 0.0, 0.01)
        assert report.severity == "ok", report.reasons


X_AAPL = "0a1b2c3d-20240102-AAPL-x"
X_TSLA = "0a1b2c3d-20240102-TSLA-x"
E_MSFT = "0a1b2c3d-20240102-MSFT-e"
E_NVDA = "0a1b2c3d-20240102-NVDA-e"
E_AMD = "0a1b2c3d-20240102-AMD-e"


# --------------------------------------------------------------------------- ids and staging


def test_order_id_is_a_function_of_the_decision() -> None:
    assert order_id_for(PROFILE, order("AAPL", "sell", 10)) == X_AAPL
    assert order_id_for(PROFILE, order("AAPL", "sell", 3)) == X_AAPL  # quantity is not in it
    assert order_id_for(PROFILE, order("AAPL", "buy", 10)) == "0a1b2c3d-20240102-AAPL-e"
    assert order_id_for(PROFILE, order("AAPL", "sell", 10, D1)) == "0a1b2c3d-20240103-AAPL-x"
    assert order_id_for("other-99999999", order("AAPL", "sell", 10)) != X_AAPL
    longest = order_id_for(PROFILE, order("ABCDEFGHIJKLMNOP", "buy", 1)) + "-a99"
    assert len(longest) <= 48  # the venue's client id limit, with room for 99 attempts
    for bad in ("short", "name-0a1b-2c3", "name-0a1b2c3!"):
        with pytest.raises(OrderManagerError, match="8 letters or digits"):
            order_id_for(bad, order("AAPL", "sell", 10))


def test_submit_journals_and_sends_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.calls.clear()
    ids = rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    assert ids == [X_AAPL, E_MSFT]
    assert rig.venue.calls == [] and rig.om.poll_updates() == []
    assert [(o.order_id, o.phase, o.reference_price) for o in rig.om.orders()] == [
        (X_AAPL, "staged", 100.0),
        (E_MSFT, "staged", 300.0),
    ]
    assert rig.kinds()[-2:] == ["order_staged", "order_staged"]
    # The same decision again is the same order, not a second one.
    assert rig.om.submit([order("AAPL", "sell", 10)]) == [X_AAPL]
    assert rig.kinds().count("order_staged") == 2
    with pytest.raises(OrderManagerError, match="different content"):
        rig.om.submit([order("AAPL", "sell", 9)])


def test_submit_refuses_the_whole_batch_if_one_order_is_bad(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    before = rig.kinds()
    option = Leg(
        instrument=OptionContract(underlying="AAPL", expiry=D1, strike=100, right="call"),
        side="buy",
    )
    stock = Leg(instrument=Stock(symbol="MSFT"), side="buy")
    bad_orders = [
        order("ZZZ", "buy", 1),  # no close set
        Order(created_on=D0, action="enter", legs=(option,), quantity=1, strength=1.0),
        Order(created_on=D0, action="enter", legs=(stock, stock), quantity=1, strength=1.0),
        Order(created_on=D0, action="exit", legs=(stock,), quantity=1),  # an exit that buys
        Order(
            created_on=D0, action="enter", quantity=1, strength=1.0,
            legs=(Leg(instrument=Stock(symbol="MSFT"), side="buy", ratio=2),),
        ),
    ]  # fmt: skip
    for bad in bad_orders:
        with pytest.raises(OrderManagerError):
            rig.om.submit([order("AAPL", "sell", 10), bad])
    with pytest.raises(OrderManagerError, match="twice in one submit"):
        rig.om.submit([order("MSFT", "buy", 3), order("MSFT", "buy", 3)])
    assert rig.kinds() == before and rig.om.orders() == []
    with pytest.raises(OrderManagerError, match="positive number"):
        rig.om.set_closes({"AAPL": 0.0})


def test_closes_are_journaled_and_survive_a_restart(tmp_path: Path) -> None:
    rig = Rig(tmp_path)  # the rig sets the closes once, on its first start
    assert rig.kinds().count("closes") == 1
    rig.om.set_closes(CLOSES)  # the same closes again: nothing new to record
    assert rig.kinds().count("closes") == 1
    rig.restart()  # a real restart: nobody hands the closes over again
    assert rig.om.submit([order("MSFT", "buy", 3)]) == [E_MSFT]
    assert rig.om.orders()[0].reference_price == 300.0
    rig.om.set_closes({**CLOSES, "MSFT": 310.0})
    assert rig.kinds().count("closes") == 2
    rig.restart()
    rig.om.submit([order("NVDA", "buy", 1), order("AAPL", "sell", 10)])
    assert [o.reference_price for o in rig.om.orders()] == [300.0, 50.0, 100.0]


def test_closes_must_cover_every_held_symbol(tmp_path: Path) -> None:
    rig = Rig(tmp_path)  # holds AAPL and TSLA
    before = rig.kinds()
    with pytest.raises(OrderManagerError, match="no close for held symbols: AAPL, TSLA"):
        rig.om.set_closes({"MSFT": 300.0})
    with pytest.raises(OrderManagerError, match="no close for held symbols: TSLA"):
        rig.om.set_closes({"MSFT": 300.0, "AAPL": 100.0})
    assert rig.kinds() == before
    assert rig.om.submit([order("MSFT", "buy", 3)]) == [E_MSFT]  # the earlier closes stand


def test_quiet_poll_costs_one_request_however_many_orders_are_working(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    for symbol in ("MSFT", "NVDA", "AMD"):
        rig.venue.set_price(symbol, 1_000.0)  # above every cap: the entries rest
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4), order("AMD", "buy", 2)])
    rig.to(9, 31)
    for order_id, limit in ((E_MSFT, 306.0), (E_NVDA, 51.0), (E_AMD, 153.0)):
        rig.send_entry(order_id, limit)
    rig.venue.calls.clear()
    rig.om.refresh()
    assert rig.venue.calls == [("open_orders", "")]
    # An order that left the list is asked about by itself, and only that one.
    rig.venue.set_price("AMD", 150.0)
    rig.venue.calls.clear()
    rig.om.refresh()
    assert rig.venue.calls == [("open_orders", ""), ("get_order", E_AMD + "-a1")]
    assert rig.phase(E_AMD) == "filled"
    # If the list cannot be read, each order is asked about instead.
    rig.venue.fail_next("open_orders")
    rig.venue.calls.clear()
    rig.om.refresh()
    assert [m for m, _ in rig.venue.calls] == ["open_orders", "get_order", "get_order"]
    rig.venue.calls.clear()
    rig.om.cancel_all("session_deadline")
    rig.om.refresh()  # nothing is working: no request at all
    assert "open_orders" not in [m for m, _ in rig.venue.calls]


def test_nothing_works_before_recover(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.restart(recover=False)
    for call in (
        lambda: rig.om.set_closes(CLOSES),
        lambda: rig.om.submit([order("AAPL", "sell", 10)]),
        lambda: rig.send_exit(X_AAPL),
        lambda: rig.om.refresh(),
        lambda: rig.om.cancel_all("session_deadline"),
        lambda: rig.om.close(X_AAPL, "canceled", "session_deadline"),
    ):
        with pytest.raises(OrderManagerError, match="recover"):
            call()
    rig.om.recover()
    with pytest.raises(OrderManagerError, match="already called"):
        rig.om.recover()


# --------------------------------------------------------------------------- fills


def test_exit_in_the_auction_fills_and_reaches_the_ledger(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "working"
    assert rig.om.poll_updates() == []
    (sent,) = rig.venue.place_attempts
    assert (sent.client_order_id, sent.type, sent.time_in_force) == (
        X_AAPL + "-a1",
        "market",
        "opg",
    )
    rig.venue.set_price("AAPL", 104.0)
    rig.to(9, 30, 5)
    rig.om.refresh()
    (update,) = rig.om.poll_updates()
    assert (update.order_id, update.status, update.reason) == (X_AAPL, "filled", "")
    (fill,) = update.fills
    assert (fill.kind, fill.side, fill.quantity, fill.price) == ("venue", "sell", 10, 104.0)
    assert (fill.signal_date, fill.fill_date, fill.price_date) == (D0, D1, D1)
    assert fill.reference_price == 100.0 and fill.filled_at == at(D1, 9, 30)
    assert (fill.slippage_cost, fill.commission) == (0.0, 0.0)
    assert rig.phase(X_AAPL) == "filled"
    assert "AAPL" not in rig.om.account().positions
    assert rig.om.account().cash == pytest.approx(10_000 - 1_000 - 1_000 + 1_040)
    rig.om.refresh()  # a final order is not asked about again
    assert rig.om.poll_updates() == []
    rig.reconciles()


def test_entry_fills_and_the_position_is_tagged_with_the_order(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.to(9, 31)
    rig.venue.set_price("MSFT", 301.0)
    assert rig.send_entry(E_MSFT, 306.0) == "filled"  # marketable: filled in the place reply
    (update,) = rig.om.poll_updates()
    assert update.status == "filled" and update.fills[0].price == 301.0
    position = rig.om.account().positions["MSFT"]
    assert (position.quantity, position.entry_order_id, position.entry_date) == (3, E_MSFT, D1)
    rig.reconciles()


def test_fills_in_pieces_are_reported_incrementally_at_their_own_prices(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdings={})
    rig.venue.script(
        "NVDA",
        OrderScript(fills=(FillStep(0, 2, 50.0), FillStep(60, 3, 50.5), FillStep(120, None, 51.0))),
    )
    rig.om.submit([order("NVDA", "buy", 10)])
    rig.to(9, 31)
    assert rig.send_entry(E_NVDA, 51.0) == "working"
    rig.to(9, 32, 30)
    rig.om.refresh()
    rig.om.refresh()  # nothing new: no second report
    rig.to(9, 34)
    rig.om.refresh()
    updates = rig.om.poll_updates()
    assert [u.status for u in updates] == ["partially_filled", "partially_filled", "filled"]
    fills = [f for u in updates for f in u.fills]
    assert [f.quantity for f in fills] == [2, 3, 5]
    assert [f.price for f in fills] == pytest.approx([50.0, 50.5, 51.0])
    assert rig.om.orders()[0].filled_quantity == 10
    assert rig.om.account().positions["NVDA"].quantity == 10
    rig.reconciles()


# --------------------------------------------------------------------------- the protocol


def test_intent_is_journaled_before_the_venue_is_called(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    class Spy(FakeVenue):
        def place(self, order):  # type: ignore[no-untyped-def]
            seen.append([r.kind for r in read_journal(tmp_path / "journal.jsonl")])
            return super().place(order)

    rig = Rig(tmp_path, venue_class=Spy)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    assert seen[0][-1] == "order_intent"
    record = read_journal(rig.path)[len(seen[0]) - 1]
    assert record.payload["venue_order"]["client_order_id"] == X_AAPL + "-a1"  # type: ignore[index,call-overload]
    assert rig.kinds()[-1] == "venue_state"


def test_rejected_order_is_final_and_never_retried(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("MSFT", OrderScript(reject="insufficient buying power"))
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.to(9, 31)
    assert rig.send_entry(E_MSFT, 306.0) == "rejected"
    (update,) = rig.om.poll_updates()
    assert (update.status, update.reason, update.fills) == ("rejected", "broker_rejected", ())
    assert rig.places(E_MSFT + "-a1") == 1
    assert rig.alerts.kinds() == ["order_rejected"] and rig.alerts.alerts[0].severity == "warn"
    with pytest.raises(OrderManagerError, match="is rejected"):
        rig.send_entry(E_MSFT, 306.0)
    assert not rig.om.entries_blocked(D0)  # a rejection is a known outcome


def test_rejected_exit_is_a_critical_alert(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(reject="halted"))
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "rejected"
    assert rig.alerts.alerts[0].severity == "critical"


def test_timeout_after_the_venue_accepted_adopts_the_order(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "working"
    assert rig.places(X_AAPL + "-a1") == 1  # looked up, not sent again
    assert len(rig.venue.open_orders()) == 1
    assert rig.alerts.alerts == []
    rig.to(9, 30, 5)
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "filled"
    rig.reconciles()


def test_timeout_before_the_venue_accepted_resends_under_the_same_id(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeouts_before_accept=2))
    rig.om.submit([order("AAPL", "sell", 10)])
    start = rig.venue.clock
    assert rig.send_exit(X_AAPL) == "working"
    assert [o.client_order_id for o in rig.venue.place_attempts] == [X_AAPL + "-a1"] * 3
    assert len(rig.venue.open_orders()) == 1
    assert rig.venue.clock - start == dt.timedelta(seconds=5 + 10)  # backoff between tries
    # A lookup came before each resend.
    calls = [c for c in rig.venue.calls if c[1] == X_AAPL + "-a1"]
    assert [m for m, _ in calls] == ["place", "get_order", "place", "get_order", "place"]
    assert rig.kinds().count("order_intent") == 1


@pytest.mark.parametrize("reject_duplicate_ids", [False, True])
def test_order_accepted_but_not_visible_yet_is_not_lost(
    tmp_path: Path, reject_duplicate_ids: bool
) -> None:
    # The first send times out after the venue accepted it, and the lookup that follows
    # says "never seen" because the order is not visible yet. The resend then either
    # returns the existing order or, as on Alpaca, is rejected as a repeated id.
    rig = Rig(tmp_path, reject_duplicate_ids=reject_duplicate_ids)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.lag_lookups(1)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "working"  # adopted, not closed as rejected
    assert rig.places(X_AAPL + "-a1") == 2 and len(rig.venue.open_orders()) == 1
    assert "attempt_void" not in rig.kinds() and "order_closed" not in rig.kinds()
    assert rig.alerts.alerts == [] and rig.om.poll_updates() == []
    rig.to(9, 30, 5)
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "filled" and "AAPL" not in rig.om.account().positions
    rig.reconciles()


def test_rejected_resend_that_still_cannot_be_found_stays_unknown(tmp_path: Path) -> None:
    rig = Rig(tmp_path, reject_duplicate_ids=True)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.lag_lookups(2)  # hidden from the lookup after the rejection too
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"  # not rejected: the order is live
    assert rig.places(X_AAPL + "-a1") == 2
    assert "attempt_void" not in rig.kinds() and "order_closed" not in rig.kinds()
    (alert,) = rig.alerts.alerts
    assert alert.kind == "order_unknown" and "the resend was rejected" in alert.message
    assert rig.om.poll_updates() == []
    # Its fill is still picked up once the venue shows the order.
    rig.to(9, 30, 5)
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "filled" and "AAPL" not in rig.om.account().positions
    rig.reconciles()


def test_rejected_resend_with_a_failed_lookup_stays_unknown(tmp_path: Path) -> None:
    rig = Rig(tmp_path, reject_duplicate_ids=True)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.lag_lookups(1)
    rig.om.submit([order("AAPL", "sell", 10)])

    real = rig.venue.get_order
    calls = 0

    def second_lookup_times_out(client_order_id: str) -> VenueOrderState | None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise VenueUnavailable("timeout")
        return real(client_order_id)

    rig.venue.get_order = second_lookup_times_out  # type: ignore[method-assign]
    assert rig.send_exit(X_AAPL) == "unknown"
    assert "order_closed" not in rig.kinds() and rig.kinds()[-1] == "order_unknown"
    assert rig.alerts.kinds() == ["order_unknown"]
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "working"


def test_first_send_rejected_is_still_final(tmp_path: Path) -> None:
    # No send of this id went unanswered, so the rejection is believed without a lookup.
    rig = Rig(tmp_path, reject_duplicate_ids=True)
    rig.venue.script("AAPL", OrderScript(reject="halted"))
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.venue.calls.clear()
    assert rig.send_exit(X_AAPL) == "rejected"
    assert [m for m, _ in rig.venue.calls] == ["place"]


def test_resend_of_the_same_order_adds_no_note(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.lag_lookups(99)  # the order is there but never visible to a lookup
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "working"  # the resend's reply showed the order
    (intent,) = [r for r in read_journal(rig.path) if r.kind == "order_intent"]
    assert "may_be_at_venue_as" not in intent.payload


def test_unknown_auction_exit_resent_as_a_day_order_notes_the_earlier_spec(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.fail_next("get_order", times=2)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"  # the auction order is at the venue
    rig.venue.lag_lookups(1)
    assert rig.send_exit(X_AAPL, "day") == "working"  # the venue returned the auction order
    intents = [r for r in read_journal(rig.path) if r.kind == "order_intent"]
    assert len(intents) == 2
    assert intents[1].payload["venue_order"]["time_in_force"] == "day"  # type: ignore[index,call-overload]
    assert intents[1].payload["may_be_at_venue_as"]["time_in_force"] == "opg"  # type: ignore[index,call-overload]
    assert "may_be_at_venue_as" not in intents[0].payload
    assert len(rig.venue.open_orders()) == 1


def test_lookup_that_fails_is_not_followed_by_a_blind_resend(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.fail_next("get_order", times=5)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"
    assert rig.places(X_AAPL + "-a1") == 1
    assert rig.alerts.kinds() == ["order_unknown"]
    assert rig.kinds()[-1] == "order_unknown"
    assert not rig.om.entries_blocked(D0)  # an exit does not block entries
    # Once the venue answers, the order turns out to be there and is adopted.
    rig.venue._failures.clear()  # the venue answers again
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "working" and rig.places(X_AAPL + "-a1") == 1


def test_unknown_entry_blocks_further_entries_but_not_exits(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("MSFT", OrderScript(timeouts_before_accept=99))
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    rig.to(9, 31)
    assert rig.send_entry(E_MSFT, 306.0) == "unknown"
    assert rig.places(E_MSFT + "-a1") == CFG.max_submit_attempts
    assert rig.om.entries_blocked(D0) and not rig.om.entries_blocked(D1)
    (alert,) = rig.alerts.alerts
    assert alert.severity == "critical" and "Entries are blocked" in alert.message
    with pytest.raises(OrderManagerError, match="entries are blocked"):
        rig.send_entry(E_NVDA, 51.0)
    with pytest.raises(OrderManagerError, match="entries are blocked"):
        rig.send_entry(E_MSFT, 306.0)
    assert rig.send_exit(X_AAPL, "day") == "filled"  # exits keep going
    assert rig.om.poll_updates()[-1].order_id == X_AAPL
    rig.restart()
    assert rig.om.entries_blocked(D0)  # still blocked after a restart


def test_unknown_exit_can_be_tried_again_under_the_same_id(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeouts_before_accept=3))
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"
    rig.to(9, 31)
    assert rig.send_exit(X_AAPL, "day") == "filled"  # too late for the auction now
    ids = {o.client_order_id for o in rig.venue.place_attempts}
    assert ids == {X_AAPL + "-a1"}  # never a second id while the first is unresolved
    assert rig.venue.place_attempts[-1].time_in_force == "day"
    rig.reconciles()


def test_unknown_exit_found_at_the_venue_is_adopted_not_resent(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.fail_next("get_order", times=2)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"
    assert rig.send_exit(X_AAPL) == "working"
    assert rig.places(X_AAPL + "-a1") == 1 and rig.kinds().count("order_intent") == 1


def test_nothing_is_sent_while_the_kill_switch_is_engaged(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.kill.engage("test", "operator:cli")
    before = rig.kinds()
    rig.venue.calls.clear()
    with pytest.raises(KillSwitchEngaged):
        rig.send_exit(X_AAPL)
    assert rig.venue.calls == [] and rig.kinds() == before
    assert rig.phase(X_AAPL) == "staged"


def test_kill_during_the_retry_backoff_stops_the_resend(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeouts_before_accept=1))

    def sleep_and_kill(seconds: float) -> None:
        rig.venue.advance(seconds)
        rig.kill.engage("operator stop", "operator:cli")

    rig.om._sleep = sleep_and_kill  # the operator hits the switch while the manager waits
    rig.om.submit([order("AAPL", "sell", 10)])
    with pytest.raises(KillSwitchEngaged):
        rig.send_exit(X_AAPL)
    assert rig.places(X_AAPL + "-a1") == 1 and rig.venue.open_orders() == []
    assert rig.om.cancel_all("kill_switch") == [X_AAPL]  # cleanup still works while killed
    assert rig.om.poll_updates()[-1].reason == "kill_switch"


def test_an_order_is_never_sent_in_its_own_signal_session(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    for session in (D0, DM1):
        with pytest.raises(OrderManagerError, match="look-ahead"):
            rig.om.send(X_AAPL, session, type="market", time_in_force="opg")
    assert rig.venue.place_attempts == []


def test_only_one_attempt_is_live_at_a_time(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    with pytest.raises(OrderManagerError, match="is working"):
        rig.send_exit(X_AAPL, "day")
    with pytest.raises(OrderManagerError, match="is working"):
        rig.om.close(X_AAPL, "canceled", "session_deadline")
    with pytest.raises(OrderManagerError, match="unknown order"):
        rig.send_exit("nope")
    assert len(rig.venue.place_attempts) == 1


def test_bad_order_shape_is_refused_before_anything_is_journaled(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("MSFT", "buy", 3)])
    before = rig.kinds()
    with pytest.raises(OrderManagerError, match="cannot be sent"):
        rig.om.send(E_MSFT, D1, type="market", time_in_force="day")  # a buy must be a limit
    assert rig.kinds() == before and rig.phase(E_MSFT) == "staged"


def test_exit_larger_than_the_ledger_position_is_not_sent(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 11), order("MSFT", "sell", 1)])
    assert rig.send_exit(X_AAPL) == "rejected"
    assert rig.send_exit("0a1b2c3d-20240102-MSFT-x") == "rejected"
    assert rig.venue.place_attempts == []
    assert [u.reason for u in rig.om.poll_updates()] == ["not_held", "not_held"]
    assert [a.severity for a in rig.alerts.alerts] == ["critical", "critical"]


@pytest.mark.parametrize("ending", ["expired", "canceled"])
def test_remainder_of_an_ended_attempt_goes_out_as_a_new_attempt(
    tmp_path: Path, ending: str
) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(fills=(FillStep(0, 4, 101.0),)))
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    rig.to(9, 30, 5)
    if ending == "canceled":  # the venue (or anyone) cancels the rest
        rig.venue.cancel(X_AAPL + "-a1")
    rig.to(9, 32)  # past the auction's grace period
    rig.om.refresh()
    assert rig.venue.get_order(X_AAPL + "-a1").status == ending  # type: ignore[union-attr]
    # Either way the order is idle with 6 shares left: the exact status does not matter.
    assert rig.phase(X_AAPL) == "idle"
    (partial,) = rig.om.poll_updates()
    assert (partial.status, partial.fills[0].quantity) == ("partially_filled", 4)

    assert rig.send_exit(X_AAPL, "day") == "filled"
    second = rig.venue.place_attempts[-1]
    assert (second.client_order_id, second.quantity) == (X_AAPL + "-a2", 6)
    (final,) = rig.om.poll_updates()
    assert (final.status, final.fills[0].quantity, final.fills[0].price) == ("filled", 6, 100.0)
    assert [o.client_order_id for o in rig.om.orders()[0].venue_orders] == [
        X_AAPL + "-a1",
        X_AAPL + "-a2",
    ]
    assert "AAPL" not in rig.om.account().positions
    rig.reconciles()


def test_idle_order_can_be_closed_instead_of_resent(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(fills=()))
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3)])
    rig.send_exit(X_AAPL)
    rig.to(9, 32)
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "idle" and rig.om.poll_updates() == []
    rig.om.close(X_AAPL, "expired", "session_deadline")
    rig.om.close(E_MSFT, "canceled", "approval_timeout")  # never sent
    assert [(u.order_id, u.status, u.reason) for u in rig.om.poll_updates()] == [
        (X_AAPL, "expired", "session_deadline"),
        (E_MSFT, "canceled", "approval_timeout"),
    ]
    assert len(rig.venue.place_attempts) == 1
    with pytest.raises(OrderManagerError, match="is expired"):
        rig.om.close(X_AAPL, "canceled", "session_deadline")


def test_venue_rejecting_after_accepting_ends_the_order(tmp_path: Path) -> None:
    class RejectsLater(Tampering):
        def get_order(self, client_order_id: str) -> VenueOrderState | None:
            state = super().get_order(client_order_id)
            assert state is not None
            return state.model_copy(update={"status": "rejected"})

    rig = Rig(tmp_path, venue_class=RejectsLater)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    rig.om.refresh()
    assert rig.phase(X_AAPL) == "rejected"
    assert rig.om.poll_updates()[-1].reason == "broker_rejected"
    assert rig.alerts.alerts[-1].severity == "critical"


# --------------------------------------------------------------------------- cancel


def test_cancel_all_confirms_with_the_venue_before_returning(tmp_path: Path) -> None:
    rig = Rig(tmp_path, cancel_latency=7.0)
    rig.venue.set_price("MSFT", 400.0)  # above the limit: the entry rests
    rig.om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 4), order("AAPL", "sell", 10)])
    rig.to(9, 31)
    assert rig.send_entry(E_MSFT, 306.0) == "working"
    rig.send_exit(X_AAPL, "day")
    rig.om.poll_updates()
    start = rig.venue.clock
    assert rig.om.cancel_all("session_deadline") == [E_NVDA, E_MSFT]
    assert rig.venue.open_orders() == []  # confirmed, not assumed
    assert rig.venue.clock - start == dt.timedelta(seconds=10)  # two polls for a 7 second cancel
    assert [(u.order_id, u.status, u.reason) for u in rig.om.poll_updates()] == [
        (E_NVDA, "canceled", "session_deadline"),  # never sent: no venue call needed
        (E_MSFT, "canceled", "session_deadline"),
    ]
    assert rig.phase(X_AAPL) == "filled"  # final orders are left alone
    assert rig.om.cancel_all("session_deadline") == []
    rig.reconciles()


def test_fill_that_arrives_during_the_cancel_is_recorded(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdings={})
    rig.venue.script("MSFT", OrderScript(fills=(), fill_on_cancel=2))
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.to(9, 31)
    rig.send_entry(E_MSFT, 306.0)
    assert rig.om.cancel_all("session_deadline") == [E_MSFT]
    partial, final = rig.om.poll_updates()
    assert (partial.status, partial.fills[0].quantity) == ("partially_filled", 2)
    assert (final.status, final.reason, final.fills) == ("canceled", "session_deadline", ())
    assert rig.om.account().positions["MSFT"].quantity == 2
    assert rig.om.orders()[0].filled_quantity == 2
    rig.reconciles()


def test_order_that_fills_completely_during_the_cancel_is_filled_not_canceled(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, holdings={})
    rig.venue.script("MSFT", OrderScript(fills=(), fill_on_cancel=3))
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.to(9, 31)
    rig.send_entry(E_MSFT, 306.0)
    assert rig.om.cancel_all("session_deadline") == []
    (update,) = rig.om.poll_updates()
    assert update.status == "filled" and rig.phase(E_MSFT) == "filled"


def test_cancel_that_is_not_confirmed_kills(tmp_path: Path) -> None:
    rig = Rig(tmp_path, cancel_latency=3_600.0)
    rig.venue.set_price("MSFT", 400.0)
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.to(9, 31)
    rig.send_entry(E_MSFT, 306.0)
    start = rig.venue.clock
    with pytest.raises(SessionIncomplete, match="cancel not confirmed"):
        rig.om.cancel_all("session_deadline")
    assert rig.kill.engaged() and rig.kill.state().source == "auto:orders"
    assert rig.alerts.alerts[-1].kind == "kill_switch"
    assert rig.venue.clock - start >= dt.timedelta(seconds=CFG.cancel_confirm_seconds)
    assert rig.phase(E_MSFT) == "working"  # not reported as canceled


def test_cancel_survives_a_venue_that_fails_now_and_then(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.set_price("MSFT", 400.0)
    rig.om.submit([order("MSFT", "buy", 3)])
    rig.to(9, 31)
    rig.send_entry(E_MSFT, 306.0)
    rig.venue.fail_next("get_order")
    rig.venue.fail_next("cancel")
    assert rig.om.cancel_all("session_deadline") == [E_MSFT]
    assert rig.venue.open_orders() == []


def test_cancel_of_an_unknown_order_that_never_reached_the_venue(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeouts_before_accept=99))
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"
    rig.om.poll_updates()
    start = rig.venue.clock
    assert rig.om.cancel_all("session_deadline") == [X_AAPL]
    # A send went unanswered, so "not found" is only believed after the whole window.
    assert rig.venue.clock - start == dt.timedelta(seconds=CFG.cancel_confirm_seconds)
    assert rig.kinds()[-2:] == ["attempt_void", "order_closed"]
    assert rig.om.poll_updates()[0].status == "canceled"


def test_cancel_finds_an_unknown_order_that_was_not_visible_at_first(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.fail_next("get_order", times=2)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"
    rig.venue.lag_lookups(3)  # still hidden when the cancel starts
    assert rig.om.cancel_all("session_deadline") == [X_AAPL]
    assert rig.venue.open_orders() == []  # it was found and canceled, not written off
    assert "attempt_void" not in rig.kinds()
    assert rig.venue.get_order(X_AAPL + "-a1").status == "canceled"  # type: ignore[union-attr]


def test_cancel_of_an_order_that_was_never_sent_does_not_wait(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.venue.killed = True  # the adapter refuses: nothing reaches the venue
    with pytest.raises(KillSwitchEngaged):
        rig.send_exit(X_AAPL)
    assert rig.phase(X_AAPL) == "submitting"
    start = rig.venue.clock
    assert rig.om.cancel_all("kill_switch") == [X_AAPL]
    assert rig.venue.clock == start


def test_cancel_of_an_unknown_order_that_is_at_the_venue(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("AAPL", OrderScript(timeout_after_accept=True))
    rig.venue.fail_next("get_order", times=2)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.send_exit(X_AAPL) == "unknown"
    assert len(rig.venue.open_orders()) == 1
    assert rig.om.cancel_all("session_deadline") == [X_AAPL]
    assert rig.venue.open_orders() == []


# --------------------------------------------------------------------------- invariants


def test_venue_state_that_contradicts_the_order_kills(tmp_path: Path) -> None:
    class WrongQuantity(Tampering):
        def get_order(self, client_order_id: str) -> VenueOrderState | None:
            state = super().get_order(client_order_id)
            return state.model_copy(update={"quantity": 11}) if state else None

    rig = Rig(tmp_path, venue_class=WrongQuantity)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    with pytest.raises(SessionIncomplete, match="11 shares"):
        rig.om.refresh()
    assert rig.kill.engaged() and rig.venue.open_orders() == []  # the kill canceled it


def test_filled_quantity_that_goes_down_kills(tmp_path: Path) -> None:
    class Shrinks(Tampering):
        shrink = False

        def get_order(self, client_order_id: str) -> VenueOrderState | None:
            state = super().get_order(client_order_id)
            assert state is not None
            if not self.shrink:
                return state
            return state.model_copy(update={"status": "partially_filled", "filled_quantity": 1})

    rig = Rig(tmp_path, holdings={}, venue_class=Shrinks)
    rig.venue.script("NVDA", OrderScript(fills=(FillStep(0, 4, 50.0),)))
    rig.om.submit([order("NVDA", "buy", 10)])
    rig.to(9, 31)
    rig.send_entry(E_NVDA, 51.0)
    assert rig.om.orders()[0].filled_quantity == 4
    rig.venue.shrink = True  # type: ignore[attr-defined]
    with pytest.raises(SessionIncomplete, match="went from 4 down to 1"):
        rig.om.refresh()
    assert rig.kill.engaged() and rig.om.account().positions["NVDA"].quantity == 4


def test_venue_forgetting_an_order_it_reported_kills(tmp_path: Path) -> None:
    class Forgets(Tampering):
        forget = False

        def get_order(self, client_order_id: str) -> VenueOrderState | None:
            return None if self.forget else super().get_order(client_order_id)

    rig = Rig(tmp_path, venue_class=Forgets)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    rig.venue.forget = True  # type: ignore[attr-defined]
    with pytest.raises(SessionIncomplete, match="no longer knows"):
        rig.om.refresh()
    assert rig.kill.engaged()


def test_fill_dated_on_the_signal_session_kills(tmp_path: Path) -> None:
    class Backdated(Tampering):
        def get_order(self, client_order_id: str) -> VenueOrderState | None:
            state = super().get_order(client_order_id)
            assert state is not None
            return state.model_copy(update={"updated_at": at(D0, 15, 0)})

    rig = Rig(tmp_path, venue_class=Backdated)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    rig.to(9, 31)
    with pytest.raises(SessionIncomplete, match="does not fit the book"):
        rig.om.refresh()
    assert rig.kill.engaged()
    assert rig.om.account().positions["AAPL"].quantity == 10  # the fill was not booked


def test_session_order_cap_kills(tmp_path: Path) -> None:
    limits = CFG.limits.model_copy(update={"max_orders_per_session": 2})
    rig = Rig(tmp_path, cfg=CFG.model_copy(update={"limits": limits}))
    rig.om.submit([order("AAPL", "sell", 10), order("TSLA", "sell", 5), order("MSFT", "buy", 1)])
    rig.send_exit(X_AAPL)
    rig.send_exit(X_TSLA)
    rig.to(9, 31)
    with pytest.raises(SessionIncomplete, match="already has 2 venue orders"):
        rig.send_entry(E_MSFT, 306.0)
    assert rig.alerts.alerts[-1].message.startswith(
        "kill switch engaged by auto:orders: order cap reached"
    )
    assert rig.kill.engaged() and len(rig.venue.place_attempts) == 2
    # The count survives a restart: it comes from the journal.
    rig.kill.reset("david", PROFILE, PROFILE)
    rig.restart()
    with pytest.raises(SessionIncomplete):
        rig.send_entry(E_MSFT, 306.0)


def test_force_close_never_invents_a_fill(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    before = rig.om.account()
    assert rig.om.force_close("AAPL", 100.0, DM1, D1) is None
    assert rig.om.account() == before == rig.ledger.account()
    assert rig.om.poll_updates() == [] and rig.venue.place_attempts == []
    assert rig.alerts.kinds() == ["position_stale_no_data"]


# --------------------------------------------------------------------------- recovery


def crash(rig: Rig, kind: JournalKind, after_write: bool, action) -> None:  # type: ignore[no-untyped-def]
    """Run `action` and kill the process at the next journal record of `kind`."""
    assert rig.journal is not None
    journal = rig.journal
    real = Journal.append

    def append(kind_now: JournalKind, payload) -> int:  # type: ignore[no-untyped-def]
        if kind_now == kind:
            if after_write:
                real(journal, kind_now, payload)
            raise Crash(kind)
        return real(journal, kind_now, payload)

    journal.append = append  # type: ignore[method-assign]
    with pytest.raises(Crash):
        action()


def test_crash_after_the_intent_and_before_the_send(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    crash(rig, "order_intent", True, lambda: rig.send_exit(X_AAPL))
    assert rig.venue.place_attempts == []
    rig.restart()
    assert rig.phase(X_AAPL) == "staged"  # looked up, not at the venue
    assert rig.kinds()[-1] == "attempt_void"
    assert rig.send_exit(X_AAPL) == "working"
    assert [o.client_order_id for o in rig.venue.place_attempts] == [X_AAPL + "-a1"]


def test_crash_after_the_send_and_before_the_reply_is_journaled(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    crash(rig, "venue_state", False, lambda: rig.send_exit(X_AAPL))
    assert len(rig.venue.open_orders()) == 1
    rig.restart()
    assert rig.phase(X_AAPL) == "working"  # adopted by its client id
    assert rig.places(X_AAPL + "-a1") == 1
    with pytest.raises(OrderManagerError, match="is working"):
        rig.send_exit(X_AAPL)


def test_fills_while_the_process_was_down_reach_the_ledger_without_updates(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10), order("TSLA", "sell", 5)])
    rig.venue.script("TSLA", OrderScript(fills=(FillStep(0, 2),)))
    rig.send_exit(X_AAPL)
    rig.send_exit(X_TSLA)
    rig.to(9, 32)  # both auctions are over while nothing is running
    rig.restart()
    assert (rig.phase(X_AAPL), rig.phase(X_TSLA)) == ("filled", "idle")
    assert rig.om.poll_updates() == []  # whoever submitted these is gone
    assert "AAPL" not in rig.om.account().positions
    assert rig.om.account().positions["TSLA"].quantity == 3
    assert rig.kinds().count("order_closed") == 1
    assert rig.send_exit(X_TSLA, "day") == "filled"
    assert rig.venue.place_attempts[-1].client_order_id == X_TSLA + "-a2"
    assert rig.venue.place_attempts[-1].quantity == 3
    rig.reconciles()
    rig.restart()  # and again: nothing changes
    assert (rig.phase(X_AAPL), rig.phase(X_TSLA)) == ("filled", "filled")
    assert len(rig.venue.place_attempts) == 3


def test_crash_between_the_ledger_fill_and_the_order_record(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    rig.to(9, 31)
    crash(rig, "venue_state", False, rig.om.refresh)  # the fill record is already written
    rig.restart()
    assert rig.phase(X_AAPL) == "filled"
    assert rig.kinds().count("fill") == 2 + 1  # the two seed fills and this one, once
    rig.reconciles()


def test_closed_orders_stay_closed_after_a_restart(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("MSFT", OrderScript(reject="no"))
    rig.om.submit([order("AAPL", "sell", 10), order("MSFT", "buy", 3), order("NVDA", "buy", 4)])
    rig.send_exit(X_AAPL)
    rig.to(9, 31)
    rig.send_entry(E_MSFT, 306.0)
    rig.om.refresh()
    rig.om.close(E_NVDA, "canceled", "approval_timeout")
    before = rig.om.orders()
    rig.venue.calls.clear()
    rig.restart()
    assert rig.om.orders() == before
    assert rig.venue.calls == []  # nothing to ask the venue about
    assert rig.om.submit([order("NVDA", "buy", 4)]) == [E_NVDA]  # still the same, closed order
    assert rig.phase(E_NVDA) == "canceled"


def test_recovery_that_cannot_reach_the_venue_kills(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    rig.send_exit(X_AAPL)
    rig.venue.fail_next("get_order", times=99)
    with pytest.raises(SessionIncomplete, match="cannot recover"):
        rig.restart()
    assert rig.kill.engaged()


def test_journal_from_another_profile_or_out_of_step_with_the_ledger_is_refused(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.om.submit([order("AAPL", "sell", 10)])
    assert rig.journal is not None
    other = OrderManager(
        rig.venue,
        rig.ledger,
        rig.journal,
        rig.kill,
        None,
        CFG,
        "other-99999999",
        rig.risk,
        10,
        "off",
    )
    with pytest.raises(OrderManagerError, match="does not replay"):
        other.recover()
    # A fill for a known order that the manager's attempts cannot account for.
    stray = Fill(
        order_id=X_AAPL, kind="venue", signal_date=D0, fill_date=D1, price_date=D1,
        instrument=Stock(symbol="AAPL"), side="sell", quantity=1, reference_price=100.0,
        price=100.0, slippage_cost=0.0, commission=0.0, filled_at=at(D1, 9, 31),
    )  # fmt: skip
    rig.ledger.apply_fill(stray, 1)
    with pytest.raises(OrderManagerError, match="does not replay"):
        rig.restart()


# --------------------------------------------------------------------------- the crash test


def scripted_day(rig: Rig) -> None:
    """One morning, written so it can be run again from the top after a crash: every
    step looks at the order's phase first, as the open workflow will."""
    om = rig.om
    exits = om.submit([order("AAPL", "sell", 10), order("TSLA", "sell", 5)])
    entries = om.submit([order("MSFT", "buy", 3), order("NVDA", "buy", 10), order("AMD", "buy", 6)])
    limits = {E_MSFT: 306.0, E_NVDA: 51.0, E_AMD: 153.0}

    for order_id in exits:
        if rig.phase(order_id) in ("staged", "unknown") and rig.venue.clock < at(D1, 9, 30):
            rig.send_exit(order_id)
    rig.to(9, 30, 30)
    om.refresh()
    rig.to(9, 32)
    om.refresh()
    for order_id in exits:
        if rig.phase(order_id) in ("staged", "idle", "unknown"):
            rig.send_exit(order_id, "day")
    rig.to(9, 33)
    om.refresh()
    for order_id in entries:
        if rig.phase(order_id) in ("staged", "unknown") and not om.entries_blocked(D0):
            rig.send_entry(order_id, limits[order_id])
    for minute in (34, 36, 38):
        rig.to(9, minute)
        om.refresh()
    om.cancel_all("session_deadline")


def scripted_rig(tmp_path: Path, reject_duplicate_ids: bool = False) -> Rig:
    rig = Rig(tmp_path, reject_duplicate_ids=reject_duplicate_ids)
    venue = rig.venue
    venue.script("AAPL", OrderScript(fills=(FillStep(0, 4, 101.0),)))  # rest goes out again
    venue.script("TSLA", OrderScript(timeout_after_accept=True))
    venue.script("MSFT", OrderScript(timeouts_before_accept=1))
    venue.script("NVDA", OrderScript(fills=(FillStep(0, 2, 50.0), FillStep(90, 5, 50.5))))
    venue.script("AMD", OrderScript(fills=(), fill_on_cancel=1))
    return rig


def outcome(rig: Rig) -> dict[str, object]:
    account = rig.ledger.account()
    return {
        "venue_orders": {
            cid: (o.status, o.filled, round(o.filled_value, 6))
            for cid, o in rig.venue._orders.items()
        },
        "positions": account.positions,
        "cash": round(account.cash, 6),
        "realized": round(rig.ledger.realized, 6),
        "phases": [(o.order_id, o.phase, o.filled_quantity, o.reason) for o in rig.om.orders()],
    }


def test_scripted_day_without_a_crash(tmp_path: Path) -> None:
    rig = scripted_rig(tmp_path)
    scripted_day(rig)
    assert outcome(rig)["phases"] == [
        (X_AAPL, "filled", 10, ""),
        (X_TSLA, "filled", 5, ""),
        (E_MSFT, "filled", 3, ""),
        (E_NVDA, "canceled", 7, "session_deadline"),
        (E_AMD, "canceled", 1, "session_deadline"),
    ]
    assert sorted(rig.venue._orders) == sorted(
        [X_AAPL + "-a1", X_AAPL + "-a2", X_TSLA + "-a1", E_MSFT + "-a1", E_NVDA + "-a1",
         E_AMD + "-a1"]
    )  # fmt: skip
    assert rig.venue.open_orders() == []
    rig.reconciles()
    assert rig.alerts.alerts == []


@pytest.mark.parametrize("reject_duplicate_ids", [False, True])
def test_crash_at_every_journal_record_ends_in_the_same_place(
    tmp_path: Path, reject_duplicate_ids: bool
) -> None:
    baseline = scripted_rig(tmp_path / "baseline", reject_duplicate_ids)
    scripted_day(baseline)
    expected = outcome(baseline)
    records = len(read_journal(baseline.path))
    first = baseline.kinds().index("order_staged") + 1
    assert records - first > 30  # the day has many steps to crash at

    for seq in range(first, records + 1):
        for after_write in (False, True):
            rig = scripted_rig(tmp_path / f"crash-{seq}-{int(after_write)}", reject_duplicate_ids)
            assert rig.journal is not None
            rig.journal.crash_at = seq
            rig.journal.crash_after_write = after_write
            with pytest.raises(Crash):
                scripted_day(rig)
            rig.restart()  # a new process: recover, then run the morning again
            scripted_day(rig)
            where = f"crash at record {seq}, {'after' if after_write else 'before'} the write"
            assert outcome(rig) == expected, where
            assert rig.venue.open_orders() == [], where
            assert not rig.kill.engaged(), where
            rig.reconciles()
            # Recovering once more changes nothing.
            rig.restart()
            assert outcome(rig) == expected, where


def test_crash_during_recovery_ends_in_the_same_place(tmp_path: Path) -> None:
    baseline = scripted_rig(tmp_path / "baseline")
    scripted_day(baseline)
    expected = outcome(baseline)
    records = len(read_journal(baseline.path))
    first = baseline.kinds().index("order_staged") + 1
    second_crashes = 0

    for seq in range(first, records + 1):
        for nth in (1, 2, 3):  # crash again at the nth record recovery writes
            for after_write in (False, True):
                rig = scripted_rig(tmp_path / f"crash-{seq}-{nth}-{int(after_write)}")
                assert rig.journal is not None
                rig.journal.crash_at = seq
                rig.journal.crash_after_write = True
                with pytest.raises(Crash):
                    scripted_day(rig)

                target = len(read_journal(rig.path)) + nth
                try:
                    rig.restart(crash_at=target, crash_after_write=after_write)
                except Crash:
                    second_crashes += 1
                    rig.restart()  # the third process recovers cleanly
                else:
                    rig.journal.crash_at = None  # recovery wrote fewer than nth records
                scripted_day(rig)
                where = f"crash at {seq}, then at recovery record {nth} ({after_write})"
                assert outcome(rig) == expected, where
                assert rig.venue.open_orders() == [] and not rig.kill.engaged(), where
                rig.reconciles()
    assert second_crashes > 20  # recovery really was interrupted, many times
