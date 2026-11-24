"""The paper runner: full days against the fake venue with a fake clock, each way a day
can end, and a crash at every journal record followed by a rerun."""

from __future__ import annotations

import csv
import datetime as dt
import fcntl
import json
import os
import shutil
import stat
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import yaml

import harness.engine as engine_module
import harness.paper as paper
from harness.broker.sim import SimBroker
from harness.broker.venue import VenueUnavailable
from harness.data import BarPanel, DataLoader, read_paper_access_log
from harness.engine import Engine
from harness.inbox import ApproveOrders
from harness.journal import Journal, JournalRecord, read_journal
from harness.killswitch import KillSwitch
from harness.models import AccountState, Order, PortfolioSnapshot, Signal
from harness.paper import DayReport, PaperError, run_trading_day
from harness.positions import Ledger, reconcile_venue
from harness.profiles import load_risk_state
from harness.risk import RiskManager, RiskState
from harness.view import PRICE_FIELDS, MarketView
from strategies.base import Strategy
from tests.fake_venue import FakeVenue, OrderScript, at
from tests.fixture_strategies import Churn
from tests.test_orders import ClockedInbox, Crash, CrashingJournal, Recorder
from tests.test_profiles import COMMIT
from tests.test_profiles import Rig as ProfileRig

T = dt.date(2026, 6, 30)  # the close the first day decides on (a Tuesday)
D1 = dt.date(2026, 7, 1)  # the first session traded
D2 = dt.date(2026, 7, 2)
D3 = dt.date(2026, 7, 3)
SATURDAY = dt.date(2026, 7, 4)
SYMBOLS = ("SPY", "AAPL", "MSFT")


class Rig:
    """A plumbing profile of the example strategy, activated on a fake venue whose clock
    drives everything: the runner, the order manager, and the data loader."""

    def __init__(self, tmp_path: Path, *, approval: str = "off") -> None:
        self.base = ProfileRig(tmp_path, holdout=False)
        self.profile = self.base.promote("plumbing", plumbing=True)
        self.profile_id = self.profile.profile_id
        self.execution = self.base.execution
        self.state_dir = self.execution.state_dir
        self.state = self.state_dir / self.profile_id
        self.approval = approval
        self.sleep: Callable[[float], None] | None = None
        self.activate()

    def activate(self) -> None:
        """A fresh account and a fresh activation (also used to start over in a loop)."""
        shutil.rmtree(self.state_dir, ignore_errors=True)
        self.venue = FakeVenue(clock=at(D1, 9, 15), cash=12_000.0)
        self.base.activate(self.profile_id, self.venue, approval=self.approval)
        self.alerts = Recorder()
        self.morning(D1)

    def morning(self, day: dt.date, hour: int = 9, minute: int = 15) -> None:
        """Move to `day` before the open. Every symbol trades at its last close."""
        self.venue.set_now(at(day, hour, minute))
        previous = day - dt.timedelta(days=1)
        while previous.weekday() >= 5:
            previous -= dt.timedelta(days=1)
        for symbol in SYMBOLS:
            self.venue.set_price(symbol, self.base.fetcher.price(symbol, previous))

    def loader(self) -> DataLoader:
        recipe = self.profile.recipe
        return DataLoader(
            recipe.dates,
            recipe.data,
            fetcher_factory=lambda: self.base.fetcher,
            now=lambda: self.venue.clock,
        )

    def run(self, **kw) -> DayReport:  # type: ignore[no-untyped-def]
        kw.setdefault("now", lambda: self.venue.clock)
        return run_trading_day(
            self.profile_id,
            venue=self.venue,
            loader=self.loader(),
            sleep=self.sleep or self.venue.advance,
            execution=self.execution,
            alerts=self.alerts,
            git_commit=lambda: COMMIT,
            **kw,
        )

    def records(self) -> list[JournalRecord]:
        return read_journal(self.state / "journal.jsonl")

    def kinds(self) -> list[str]:
        return [r.kind for r in self.records()]

    def kill_switch(self) -> KillSwitch:
        return KillSwitch(self.state_dir / "KILL")

    def accepted(self) -> list[str]:
        """Client ids of the orders the venue holds, in placement order."""
        return list(self.venue._orders)

    def held(self) -> dict[str, int]:
        return {p.symbol: p.quantity for p in self.venue.positions()}

    def ledger(self) -> dict:  # type: ignore[type-arg]
        return json.loads((self.state / "ledger.json").read_text())

    def equity_rows(self) -> list[dict[str, str]]:
        with (self.state / "equity.csv").open(newline="") as fh:
            return list(csv.DictReader(fh))

    def close(self, symbol: str, day: dt.date) -> float:
        return self.base.fetcher.price(symbol, day)


def use_strategy(monkeypatch: pytest.MonkeyPatch, strategy: type[Strategy]) -> None:
    """Trade `strategy` instead of the profile's frozen copy (the profile is unchanged)."""
    monkeypatch.setattr(paper, "build_profile_strategy", lambda profile, **kw: strategy())


# --------------------------------------------------------------------------- a normal day


def test_first_day_decides_on_yesterdays_close_and_trades_the_open(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    report = rig.run()
    assert (report.outcome, report.session, report.decided_on) == ("traded", D1, T)
    assert not report.resumed and rig.alerts.alerts == [] and report.alerts == ()
    (order,) = report.orders
    assert (order.order.symbol, order.order.action, order.phase) == ("SPY", "enter", "filled")
    assert order.order.created_on == T  # decided on t, sent on t+1: never on the signal day
    quantity = order.quantity
    assert quantity == int(2_000 / (rig.close("SPY", T) * 1.0005))  # equity / max_positions
    assert rig.held() == {"SPY": quantity}
    assert rig.accepted() == [order.order_id + "-a1"]
    sent = rig.venue.place_attempts[0]
    assert (sent.type, sent.time_in_force) == ("limit", "day")
    assert rig.venue._orders[sent.client_order_id].updated_at >= at(D1, 9, 30)
    assert rig.venue.clock < at(D1, 9, 31)  # done once the order is final
    assert report.reconcile_before is not None and report.reconcile_before.severity == "ok"
    assert report.reconcile_after is not None and report.reconcile_after.severity == "ok"

    # Staged before anything is sent, and the decision is on record before the first send.
    kinds = rig.kinds()
    assert kinds[0] == "ledger_open" and kinds[-1] == "day_complete"
    order_of = [kinds.index(k) for k in ("day_start", "reconcile", "closes", "order_staged",
                                         "decision", "order_intent", "fill")]  # fmt: skip
    assert order_of == sorted(order_of)
    assert kinds.count("reconcile") == 2

    # Derived files: the equity point is the close of t, before today's fill.
    assert rig.equity_rows() == [
        {"date": T.isoformat(), "cash": "10000.0", "market_value": "0.0", "equity": "10000.0",
         "n_positions": "0"}
    ]  # fmt: skip
    assert rig.ledger()["positions"]["SPY"]["quantity"] == quantity
    assert json.loads((rig.state / "pending_approvals.json").read_text()) == []
    assert report.pnl is not None and report.pnl.as_of == T
    assert report.pnl.market_value == quantity * rig.close("SPY", T)
    # The risk state the next day starts from, and one line in the paper access log.
    state = load_risk_state(rig.state_dir, rig.profile_id)
    assert (state.marked_on, state.prior_equity) == (T, 10_000.0)
    (line,) = read_paper_access_log(rig.profile.recipe.data.paper_access_log)
    assert (line["profile_id"], line["data_end"]) == (rig.profile_id, T.isoformat())


def test_the_days_snapshot_records_every_execution_setting(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    snapshot = yaml.safe_load((rig.state / "config_snapshot.yaml").read_text())
    assert snapshot["execution"] == rig.execution.model_dump(mode="json")  # not only `limits`
    assert snapshot["recipe"] == rig.profile.recipe.model_dump(mode="json")
    assert snapshot["run"]["profile_id"] == rig.profile_id
    assert snapshot["run"]["harness_git_commit"] == COMMIT


def test_running_the_day_twice_does_nothing_the_second_time(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    records, places = len(rig.records()), len(rig.venue.place_attempts)
    again = rig.run()
    assert (again.outcome, again.session) == ("already_done", D1)
    assert len(rig.records()) == records and len(rig.venue.place_attempts) == places


def test_second_day_holds_and_adds_an_equity_point(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    quantity = rig.run().orders[0].quantity
    rig.morning(D2)
    report = rig.run()
    assert (report.outcome, report.decided_on, report.orders) == ("traded", D1, ())
    assert len(rig.venue.place_attempts) == 1
    rows = rig.equity_rows()
    assert [r["date"] for r in rows] == [T.isoformat(), D1.isoformat()]
    cash = 10_000.0 - quantity * rig.close("SPY", T)  # filled at the venue's price
    assert float(rows[1]["equity"]) == pytest.approx(cash + quantity * rig.close("SPY", D1))
    assert rig.alerts.alerts == []


def test_a_harness_that_changed_since_the_profile_was_frozen_is_a_warning(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    report = run_trading_day(
        rig.profile_id, venue=rig.venue, loader=rig.loader(), sleep=rig.venue.advance,
        now=lambda: rig.venue.clock, execution=rig.execution, alerts=rig.alerts,
        git_commit=lambda: "f" * 40,
    )  # fmt: skip
    assert report.outcome == "traded"
    assert [(a.severity, a.kind) for a in report.alerts] == [("warn", "harness_changed")]


def test_the_decision_is_the_one_a_backtest_makes_from_the_same_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, Churn)
    rig.run()
    staged = [
        Order.model_validate(r.payload["order"]) for r in rig.records() if r.kind == "order_staged"
    ]
    # The same engine over a simulated broker, fed the same bars through t.
    recipe = rig.profile.recipe
    assert recipe.risk is not None
    frames = rig.base.fetcher.fetch(SYMBOLS, recipe.dates.data_start, T)
    panel = BarPanel.from_frames(frames)
    risk = RiskManager(
        recipe.risk, rig.execution.limits, rig.execution.entry_limit_pct,
        RiskState.initial(10_000.0),
    )  # fmt: skip
    engine = Engine(
        Churn(), SimBroker(10_000.0, recipe.costs), recipe.universe.symbols,
        recipe.construction, recipe.costs, panel.n_dates, risk,
    )  # fmt: skip
    rows = [{f: panel.field(f)[i] for f in PRICE_FIELDS} for i in range(panel.n_dates)]
    for i, session in enumerate(panel.dates[:-1]):
        engine.warm_up(session, rows[i])
    engine.on_session_close(T, rows[-1])
    assert staged and staged == list(engine.orders.values())


def test_exits_and_entries_over_several_days_stay_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, Churn)
    exits = entries = 0
    day = D1
    for _ in range(12):
        rig.morning(day)
        report = rig.run()
        assert report.outcome == "traded", [a.message for a in report.alerts]
        assert all(o.order.created_on < day for o in report.orders)
        for order in report.orders:
            assert order.phase == "filled", order
            exits += order.order.action == "exit"
            entries += order.order.action == "enter"
        assert rig.held() == {s: p["quantity"] for s, p in rig.ledger()["positions"].items()}
        assert rig.venue.open_orders() == []
        day += dt.timedelta(days=3 if day.weekday() == 4 else 1)
    assert exits >= 2 and entries >= 3
    assert [a for a in rig.alerts.alerts if a.severity != "info"] == []
    assert len(rig.equity_rows()) == 12


# --------------------------------------------------------------------------- days that do not trade


def test_a_closed_market_is_not_a_trading_day(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.set_now(at(SATURDAY, 9, 15))
    report = rig.run()
    assert (report.outcome, report.session, report.decided_on) == ("no_session", SATURDAY, None)
    rig.venue.holidays.add(dt.date(2026, 7, 6))
    rig.venue.set_now(at(dt.date(2026, 7, 6), 9, 15))
    assert rig.run().outcome == "no_session"
    assert not (rig.state / "journal.jsonl").exists() and rig.venue.place_attempts == []


def test_no_decision_on_stale_data(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.base.fetcher.delisted["SPY"] = T  # the benchmark's bar for t never arrives
    report = rig.run()
    assert report.outcome == "skipped_stale_data"
    assert [(a.severity, a.kind) for a in report.alerts] == [("warn", "stale_data")]
    assert "decision" not in rig.kinds() and "order_staged" not in rig.kinds()
    assert rig.venue.place_attempts == []
    assert rig.run().outcome == "skipped_stale_data"  # not done: a rerun tries again


def test_a_start_after_the_session_deadline_decides_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.morning(D1, 9, 50)
    report = rig.run()
    assert report.outcome == "failed" and not rig.kill_switch().engaged()
    assert [(a.severity, a.kind) for a in report.alerts] == [("critical", "late_start")]
    assert "09:50" in report.alerts[0].message
    assert rig.kinds() == [] and rig.venue.place_attempts == []
    # Nothing was decided, so the close of t is not spent: the next morning trades.
    rig.morning(D2)
    assert rig.run().outcome == "traded"


def test_a_start_after_the_auction_cutoff_sends_exits_as_market_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, Churn)
    exits = []
    day = D1
    for _ in range(8):
        rig.morning(day, 9, 35)  # after the open: too late for the auction
        report = rig.run()
        assert report.outcome == "traded", [a.message for a in report.alerts]
        assert all(o.phase == "filled" for o in report.orders)
        exits += [o for o in report.orders if o.order.action == "exit"]
        day += dt.timedelta(days=3 if day.weekday() == 4 else 1)
    assert len(exits) >= 2
    for order in exits:
        assert [(v.type, v.time_in_force) for v in order.venue_orders] == [("market", "day")]
    assert rig.alerts.alerts == []


def test_a_calendar_that_fails_for_the_missed_session_check_does_not_cost_the_day(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    rig.morning(D3)
    real, calls = rig.venue.sessions, 0

    def flaky(start: dt.date, end: dt.date):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:  # the first read finds today's session; the second is the check
            raise VenueUnavailable("timeout in sessions")
        return real(start, end)

    rig.venue.sessions = flaky  # type: ignore[method-assign]
    report = rig.run()
    assert report.outcome == "traded"
    (alert,) = report.alerts
    assert (alert.severity, alert.kind) == ("warn", "missed_sessions")
    assert alert.message.startswith("could not check")


def test_a_missed_session_is_alerted_and_the_day_carries_on(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.run()
    rig.morning(D3)  # nothing ran on D2
    report = rig.run()
    assert (report.outcome, report.decided_on) == ("traded", D2)
    (alert,) = report.alerts
    assert (alert.severity, alert.kind) == ("warn", "missed_sessions")
    assert alert.message.startswith(f"the runner did not complete 1 session(s) since {D1}: {D2}.")


# --------------------------------------------------------------------------- refusals


def test_only_the_active_profile_runs_and_only_one_runner(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(PaperError, match="no profile"):
        run_trading_day("nothing-0a1b2c3d", venue=rig.venue, execution=rig.execution)
    other = rig.base.promote("second", plumbing=True)
    with pytest.raises(PaperError, match="not the active profile"):
        run_trading_day(other.profile_id, venue=rig.venue, execution=rig.execution)
    fd = os.open(rig.state_dir / "runner.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with pytest.raises(PaperError, match="another runner"):
            rig.run()
    finally:
        os.close(fd)
    assert rig.venue.calls == [("verify_account", ""), ("positions", ""), ("open_orders", "")]
    assert rig.run().outcome == "traded"  # the lock is free again


def test_an_engaged_kill_switch_refuses_the_day_without_touching_the_venue(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.kill_switch().engage("testing", "operator:test")
    calls = len(rig.venue.calls)
    report = rig.run()
    assert report.outcome == "killed"
    assert [(a.severity, a.kind) for a in report.alerts] == [("warn", "kill_switch")]
    assert "start the runner again" in report.alerts[0].message
    assert len(rig.venue.calls) == calls and not (rig.state / "journal.jsonl").exists()


def test_a_reconciliation_difference_kills_before_anything_is_decided(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.hold("AAPL", 1, 50.0)  # a trade made by hand in the account
    report = rig.run()
    assert report.outcome == "killed"
    assert report.reconcile_before is not None and report.reconcile_before.severity == "stop"
    assert rig.kill_switch().engaged() and rig.kill_switch().state().source == "auto:runner"
    assert "decision" not in rig.kinds() and rig.venue.place_attempts == []
    assert rig.alerts.kinds() == ["kill_switch"]
    assert rig.run().outcome == "killed"  # and stays refused until an operator resets it


def test_a_clock_too_far_from_the_venues_kills(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    report = rig.run(now=lambda: rig.venue.clock + dt.timedelta(seconds=31))
    assert report.outcome == "killed" and rig.kill_switch().engaged()
    assert "clock skew" in rig.kill_switch().state().reason
    assert not (rig.state / "journal.jsonl").exists()


def test_a_tampered_profile_kills(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    source = rig.execution.profiles_dir / rig.profile_id / "strategy.py"
    os.chmod(source, stat.S_IRUSR | stat.S_IWUSR)
    source.write_text(source.read_text() + "\n# edited\n")
    calls = len(rig.venue.calls)
    report = rig.run()
    assert report.outcome == "killed" and rig.kill_switch().engaged()
    assert "does not verify" in rig.kill_switch().state().reason
    # The latch only. No account has been verified at this point, and the keys could be
    # for another one, so nothing is canceled (or even asked) at the venue.
    assert len(rig.venue.calls) == calls
    assert "open orders were NOT canceled" in report.alerts[0].message


def test_a_point_in_time_profile_is_not_traded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Promotion refuses these; a profile that has one anyway stops before the venue.
    rig = Rig(tmp_path)
    universe = rig.profile.recipe.universe.model_copy(update={"membership": object()})
    recipe = rig.profile.recipe.model_copy(update={"universe": universe})
    pit = rig.profile.model_copy(update={"recipe": recipe})
    monkeypatch.setattr(paper, "load_profile", lambda *a, **kw: pit)  # type: ignore[misc]
    calls = len(rig.venue.calls)
    with pytest.raises(PaperError, match="point-in-time"):
        rig.run()
    assert len(rig.venue.calls) == calls


def test_an_error_before_the_account_is_verified_latches_without_canceling(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)

    def broken() -> None:
        raise RuntimeError("a bug while starting up")

    rig.venue.verify_account = broken  # type: ignore[method-assign,assignment]
    calls = len(rig.venue.calls)
    with pytest.raises(RuntimeError, match="starting up"):
        rig.run()
    assert rig.kill_switch().engaged() and len(rig.venue.calls) == calls


def test_keys_for_another_account_fail_the_day_without_killing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    active = rig.state_dir / "active.json"
    active.write_text(active.read_text().replace("FAKE-PAPER", "OTHER-ACCOUNT"))
    report = rig.run()
    assert report.outcome == "failed" and not rig.kill_switch().engaged()
    assert [(a.severity, a.kind) for a in report.alerts] == [("critical", "wrong_account")]
    # Nothing but the check itself reached the venue: no cancel on someone else's account.
    assert rig.venue.calls[-1] == ("verify_account", "") and rig.venue.place_attempts == []


def test_an_account_that_fails_the_setup_check_fails_the_day(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.no_shorting = False
    report = rig.run()
    assert report.outcome == "failed" and rig.alerts.kinds() == ["account_setup"]
    assert not rig.kill_switch().engaged() and not (rig.state / "journal.jsonl").exists()


def test_a_venue_that_does_not_answer_fails_the_day_and_a_rerun_works(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.fail_next("account", times=3)
    report = rig.run()
    assert report.outcome == "failed" and rig.alerts.kinds() == ["venue_unavailable"]
    assert not rig.kill_switch().engaged()
    rig.venue.fail_next("account", times=2)  # a short outage is ridden out
    assert rig.run().outcome == "traded"


def test_a_paper_access_log_that_cannot_be_used_means_no_bars_and_no_orders(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    requests = len(rig.base.fetcher.requests)
    log = rig.profile.recipe.data.paper_access_log
    log.mkdir(parents=True)
    report = rig.run()
    assert report.outcome == "failed" and rig.alerts.kinds() == ["paper_access_log"]
    assert len(rig.base.fetcher.requests) == requests
    assert "order_staged" not in rig.kinds() and rig.venue.place_attempts == []
    log.rmdir()
    assert rig.run().outcome == "traded"


def test_a_log_line_without_a_real_profile_is_alerted(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    log = rig.profile.recipe.data.paper_access_log
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "timestamp_utc,day,profile_id,data_start,data_end\n"
        "2026-06-01T12:00:00Z,2026-06-01,ghost-0a1b2c3d,2025-10-01,2026-05-29\n"
    )
    report = rig.run()
    assert report.outcome == "traded"
    (alert,) = report.alerts
    assert (alert.severity, alert.kind) == ("critical", "paper_access_log")
    assert "ghost-0a1b2c3d" in alert.message


class Boom(Strategy):
    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        raise RuntimeError("strategy bug")


def test_a_strategy_that_raises_means_no_orders_and_no_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, Boom)
    report = rig.run()
    assert report.outcome == "failed" and not rig.kill_switch().engaged()
    (alert,) = report.alerts
    assert (alert.severity, alert.kind) == ("critical", "decision_failed")
    assert "Boom.on_bar raised RuntimeError('strategy bug')" in alert.message
    assert "positions are untouched" in alert.message
    assert "decision" not in rig.kinds() and rig.venue.place_attempts == []


class BoomOnStart(Boom):
    def on_start(self, symbols: tuple[str, ...]) -> None:
        raise RuntimeError("bad start")


def test_a_strategy_that_raises_in_on_start_is_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, BoomOnStart)
    report = rig.run()
    assert report.outcome == "failed" and not rig.kill_switch().engaged()
    assert rig.alerts.kinds() == ["decision_failed"]
    assert "BoomOnStart.on_start raised" in report.alerts[0].message


def test_a_harness_error_during_the_decision_kills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Not the strategy's failure: construction, risk and the order manager are the
    # harness's own code, and an error there is not a reason to carry on tomorrow.
    rig = Rig(tmp_path)

    def broken(*args: object, **kw: object) -> None:
        raise ZeroDivisionError("a bug in construction")

    monkeypatch.setattr(engine_module, "build_orders", broken)
    with pytest.raises(ZeroDivisionError):
        rig.run()
    assert rig.kill_switch().engaged() and rig.alerts.kinds() == ["kill_switch"]
    assert ("cancel_all", "") in rig.venue.calls  # the account was verified by then
    assert "decision" not in rig.kinds() and rig.venue.place_attempts == []


def test_an_error_the_runner_does_not_know_kills_and_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)

    def broken(*args: object, **kw: object) -> None:
        raise ZeroDivisionError("a bug in the runner")

    monkeypatch.setattr(paper._Day, "_finish", broken)
    with pytest.raises(ZeroDivisionError):
        rig.run()
    assert rig.kill_switch().engaged() and rig.alerts.kinds() == ["kill_switch"]
    assert "day_complete" not in rig.kinds()


# --------------------------------------------------------------------------- crash and resume


class Dying(CrashingJournal):
    """The runner's journal, set to die at one record. Configured on the class, because
    the runner builds its own journal."""


def crash_run(rig: Rig, monkeypatch: pytest.MonkeyPatch, at_record: int, after: bool) -> None:
    Dying.crash_at, Dying.crash_after_write = at_record, after
    monkeypatch.setattr(paper, "Journal", Dying)
    try:
        with pytest.raises(Crash):
            rig.run()
    finally:
        Dying.crash_at = None


def test_a_crash_at_any_record_resumes_to_the_same_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    use_strategy(monkeypatch, Churn)  # several entries on the first day
    baseline = rig.run()
    expected_orders = sorted((o.order_id, o.phase, o.filled_quantity) for o in baseline.orders)
    expected_accepted, expected_held = sorted(rig.accepted()), rig.held()
    count = len(rig.records())
    assert len(expected_orders) >= 2 and "fill" in rig.kinds()

    for at_record in range(1, count + 1):
        for after in (False, True):
            where = f"record {at_record} ({'after' if after else 'before'} the write)"
            rig.activate()
            crash_run(rig, monkeypatch, at_record, after)
            report = rig.run()
            if at_record == count and after:  # died with the day already on record as done
                assert report.outcome == "already_done", where
            else:
                assert report.outcome == "traded", (where, [a.message for a in report.alerts])
                got = sorted((o.order_id, o.phase, o.filled_quantity) for o in report.orders)
                assert got == expected_orders, where
            # Never a second order for the same decision, at any crash point.
            assert sorted(rig.accepted()) == expected_accepted, where
            assert rig.held() == expected_held, where
            assert rig.kinds()[-1] == "day_complete" and rig.kinds().count("decision") == 1, where
            assert not rig.kill_switch().engaged(), where
            assert rig.run().outcome == "already_done", where


def test_after_the_decision_is_journaled_the_strategy_is_not_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    decided = rig.kinds().index("decision") + 1
    rig.activate()
    crash_run(rig, monkeypatch, decided + 1, after=False)  # dies before the first send
    assert rig.venue.place_attempts == []
    use_strategy(monkeypatch, Boom)  # would fail the day if it were called
    requests = len(rig.base.fetcher.requests)
    report = rig.run()
    assert (report.outcome, report.resumed) == ("traded", True)
    assert [o.phase for o in report.orders] == ["filled"]
    assert len(rig.base.fetcher.requests) == requests  # and no bars are loaded again
    assert report.pnl is not None and report.pnl.market_value > 0


def test_a_rerun_that_finds_a_different_account_does_not_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    decided = rig.kinds().index("decision") + 1
    rig.activate()
    crash_run(rig, monkeypatch, decided, after=False)  # the mark is saved, the decision is not
    state = rig.state / "risk_state.json"
    saved = json.loads(state.read_text())
    assert saved["marked_on"] == T.isoformat()
    saved["prior_equity"] += 5.0  # the first attempt marked a different equity
    state.write_text(json.dumps(saved))
    report = rig.run()
    assert report.outcome == "failed" and not rig.kill_switch().engaged()
    (alert,) = report.alerts
    assert alert.kind == "account_changed"
    assert "account changed since the first attempt, reconcile first" in alert.message
    assert rig.venue.place_attempts == []


def test_a_rerun_that_would_stage_a_different_order_does_not_trade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    staged = rig.kinds().index("order_staged") + 1
    rig.activate()
    crash_run(rig, monkeypatch, staged, after=True)  # staged, the decision not on record
    # The vendor re-adjusts SPY's history before the rerun: same equity (the account is
    # flat), a different close, so a different quantity under the same order id.
    (rig.profile.recipe.data.cache_dir / "SPY.parquet").unlink()
    rig.base.fetcher.scale_before["SPY"] = (D1, 0.5)
    report = rig.run()
    assert report.outcome == "failed" and not rig.kill_switch().engaged()
    (alert,) = report.alerts
    assert alert.kind == "decision_changed" and "different content" in alert.message
    assert "decision" not in rig.kinds() and rig.venue.place_attempts == []


def test_orders_left_by_an_unfinished_day_are_closed_not_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path)
    rig.run()
    decided = rig.kinds().index("decision") + 1
    rig.activate()
    crash_run(rig, monkeypatch, decided + 1, after=False)  # staged and decided, never sent
    (stale,) = [str(r.payload["order_id"]) for r in rig.records() if r.kind == "order_staged"]
    rig.morning(D2)  # nobody reran the first day
    report = rig.run()
    assert (report.outcome, report.decided_on, report.resumed) == ("traded", D1, False)
    assert [(a.severity, a.kind) for a in report.alerts] == [("warn", "entry_dropped")]
    closed = [r.payload for r in rig.records() if r.kind == "order_closed"]
    assert closed[0] == {"order_id": stale, "status": "canceled", "reason": "session_deadline"}
    # Today's decision starts from the flat account and is the only thing sent.
    (order,) = report.orders
    assert order.order.created_on == D1 and order.phase == "filled"
    assert rig.accepted() == [order.order_id + "-a1"]
    assert reconcile_venue(
        _ledger_account(rig), [], rig.venue, 2_000.0, 0.01
    ).severity == "ok"  # fmt: skip


def _ledger_account(rig: Rig) -> AccountState:
    with Journal(rig.state / "journal.jsonl") as journal:
        return Ledger.open(journal, 10_000.0).account()


# --------------------------------------------------------------------------- kill and approval


def test_a_kill_from_another_process_during_the_open_stops_the_day(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    calls = 0

    def sleep(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            rig.kill_switch().engage("operator said stop", "operator:cli")
        rig.venue.advance(seconds)

    rig.sleep = sleep
    report = rig.run()
    assert report.outcome == "killed" and rig.venue.open_orders() == []
    assert [(o.phase, o.reason) for o in report.orders] == [("canceled", "kill_switch")]
    assert rig.venue.place_attempts == [] and "day_complete" not in rig.kinds()
    assert [(a.severity, a.kind) for a in report.alerts] == [("critical", "kill_switch")]
    assert "start the runner again" in report.alerts[0].message

    # After a reset the runner is started again and finishes the day; nothing is re-sent.
    rig.kill_switch().reset("david", rig.profile_id, rig.profile_id)
    rig.sleep = None
    again = rig.run()
    assert (again.outcome, again.resumed) == ("traded", True)
    assert rig.venue.place_attempts == [] and rig.kinds()[-1] == "day_complete"


def test_an_entry_no_send_was_answered_for_kills_the_day_at_the_deadline(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.venue.script("SPY", OrderScript(timeouts_before_accept=99))  # no send is answered
    report = rig.run()
    # Still not at the venue when the deadline cancel has looked for it long enough: it
    # may be there all the same, so the day stops instead of writing the order off.
    assert report.outcome == "killed" and "day_complete" not in rig.kinds()
    assert [(o.phase, o.reason) for o in report.orders] == [("unknown", "")]
    assert [(a.severity, a.kind) for a in report.alerts] == [
        ("critical", "order_unknown"),
        ("critical", "kill_switch"),
    ]
    assert rig.kill_switch().state().source == "auto:orders"
    assert "attempt_void" not in rig.kinds()
    places = len(rig.venue.place_attempts)

    # After a reset the runner looks the order up again. It never reached the venue, so
    # it is voided then, and the day is finished without sending it again.
    rig.kill_switch().reset("david", rig.profile_id, rig.profile_id)
    again = rig.run()
    assert (again.outcome, again.resumed) == ("traded", True)
    assert [(o.phase, o.reason) for o in again.orders] == [("canceled", "session_deadline")]
    assert "attempt_void" in rig.kinds() and rig.kinds()[-1] == "day_complete"
    assert len(rig.venue.place_attempts) == places and rig.held() == {}


def test_entries_wait_for_an_approval_read_from_the_pending_file(tmp_path: Path) -> None:
    rig = Rig(tmp_path, approval="entries")
    shown: list[dict] = []  # type: ignore[type-arg]

    def sleep(seconds: float) -> None:
        if not shown:  # the operator reads what is waiting and approves exactly that
            shown.extend(json.loads((rig.state / "pending_approvals.json").read_text()))
            ClockedInbox(rig.state / "inbox").put(
                ApproveOrders(
                    approvals={r["order_id"]: r["content_hash"] for r in shown},
                    operator="david",
                    at=rig.venue.clock,
                )
            )
        rig.venue.advance(seconds)

    rig.sleep = sleep
    report = rig.run()
    assert report.outcome == "traded"
    (order,) = report.orders
    assert (order.phase, order.approval) == ("filled", "approved")
    assert [r["symbol"] for r in shown] == ["SPY"] and shown[0]["quantity"] == order.quantity
    assert "approval" in rig.kinds()
    assert json.loads((rig.state / "pending_approvals.json").read_text()) == []


def test_an_unapproved_entry_times_out_and_the_day_still_completes(tmp_path: Path) -> None:
    rig = Rig(tmp_path, approval="entries")
    report = rig.run()
    assert report.outcome == "traded" and rig.venue.place_attempts == []
    assert [(o.phase, o.reason) for o in report.orders] == [("canceled", "approval_timeout")]
    assert rig.kinds()[-1] == "day_complete"


# --------------------------------------------------------------------------- CLI


def test_cli_run_reports_a_refusal_and_a_killed_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path)
    monkeypatch.setattr(paper, "load_execution", lambda: rig.execution)
    monkeypatch.setattr(paper, "_log_to_file", lambda state_dir: None)
    monkeypatch.setattr(paper, "_alpaca_venue", lambda cfg, kill_switch: rig.venue)
    assert paper.main(["run", "--profile", "nothing-0a1b2c3d"]) == 2
    assert "no profile nothing-0a1b2c3d" in capsys.readouterr().err
    rig.kill_switch().engage("testing", "operator:test")
    assert paper.main(["run"]) == 1  # the active profile, refused by the latch
    out = capsys.readouterr().out
    assert f"profile {rig.profile_id}" in out and "outcome killed" in out
    (rig.state_dir / "active.json").unlink()
    assert paper.main(["run"]) == 2
    assert "no profile is active" in capsys.readouterr().err


def test_cli_activate_binds_the_account_through_the_venue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rig = Rig(tmp_path)
    shutil.rmtree(rig.state_dir)
    venue = FakeVenue(clock=at(D1, 8, 0), cash=10_500.0)
    monkeypatch.setattr(paper, "load_execution", lambda: rig.execution)
    monkeypatch.setattr(paper, "_alpaca_venue", lambda cfg, kill_switch: venue)
    argv = ["activate", rig.profile_id, "--approval", "entries", "--operator", "david"]
    assert paper.main(argv) == 0
    assert "cash offset 500.00" in capsys.readouterr().out
    active = json.loads((rig.state_dir / "active.json").read_text())
    assert (active["approval"], active["operator"]) == ("entries", "david")
    venue.hold("SPY", 1, 50.0)
    assert paper.main([*argv, "--replace"]) == 2
    assert "not flat" in capsys.readouterr().err
    # Not while a runner is working: activation takes the runner's lock.
    fd = os.open(rig.state_dir / "runner.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        calls = len(venue.calls)
        assert paper.main(argv) == 2
        assert "another runner" in capsys.readouterr().err and len(venue.calls) == calls
    finally:
        os.close(fd)
