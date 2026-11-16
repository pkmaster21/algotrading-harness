"""The paper runner: one trading day for one frozen profile, driven by the clock.

    uv run python -m harness.paper run                      # trade today (the active profile)
    uv run python -m harness.paper promote <run_dir> <name> [--plumbing]
    uv run python -m harness.paper activate <profile_id> --approval entries --operator <name>

`run_trading_day` is started once a morning by the OS scheduler or by hand (about 08:00
New York). It decides on the last completed session `t`, trades the open of today's
session, reconciles, and exits. The engine never reads a clock: the runner does, and
calls the same two engine methods a backtest calls, in the same order
(`on_session_close(t)`, then `on_session_open(today)`).

The steps, with the kill latch checked between them:

1. Take `runner.lock`. Stop if the latch is set. Load and verify the profile.
2. Verify the account, compare the local clock with the venue's, and ask the venue's
   calendar whether today is a session and which session came before it (`t`).
3. Open the journal. A `day_complete` record for today ends the run (`already_done`).
   Rebuild the ledger, the risk state and the order manager, and `recover()`.
4. Reconcile the ledger with the venue. Any difference stops trading (kill).
5. Decide, unless the journal already holds the decision for `t`: load bars through
   `t`, replay them into a fresh engine, and process the close of `t`. Orders are
   staged in the journal; nothing is sent. The risk state is saved, then the decision
   is journaled.
6. `engine.on_session_open(today)`: the order manager works the staged orders.
7. Reconcile again, write the derived files (`ledger.json`, `equity.csv`, `trades.csv`,
   `summary.json`), and journal `day_complete`.

`trades.csv` and `summary.json` (`harness/paper_report.py`) cover sessions through `t`:
they need each session's bar, which first exists the morning after, so the fills of
today's open appear in tomorrow's files. They are built from the journal alone. A
journal they cannot be built from is a warning, the previous files stay, and the day
still completes.

**Safe to run twice.** The journal's `decision` record is the line between deciding and
trading. A run that died before it decides again: the same bars and the same ledger give
the same orders, the order manager returns the ids it already staged, and anything that
no longer matches is refused. A run that died after it never calls the strategy again:
it recovers the orders and carries on with the open.

**How a day ends** (`DayReport.outcome`):

- `traded`: the day ran to the end (possibly with no orders).
- `no_session`: the market is closed today.
- `already_done`: today was completed by an earlier run.
- `skipped_stale_data`: the latest bar is not `t`. No decision is made on old data.
- `failed`: the day did not finish, and nothing is sent after the failure. The latch is
  not set. A rerun tries again, or carries on from the journal if orders were already
  worked.
- `killed`: the kill latch is set (it was already, or this run set it). Open orders
  were canceled. After the latch is reset, start the runner again: a process that
  engaged the latch itself keeps it in memory, and this one exits anyway.

What kills: a reconciliation difference, a clock too far from the venue's, a profile or
strategy file that no longer matches its hash, an order from another day that is still
at the venue, anything the order manager cannot vouch for (`SessionIncomplete`), and
any error this module did not expect, the harness's own code failing during the
decision included. What only fails the day: the venue or the data source not answering,
an account that does not pass the setup check, the paper access log not writable, the
strategy raising, a decision that differs from an earlier attempt's, a start after the
session deadline, a saved state that does not load.

A kill cancels open orders only once the venue has confirmed the account is the one the
profile was activated on. Before that (a tampered profile, an error while starting up)
it sets the latch and cancels nothing: the keys may be for another account.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
import yaml
from pydantic import AwareDatetime, JsonValue

from harness.alerts import Alert, Alerter, Alerters, JsonlAlerter, LogAlerter, Severity, send_alert
from harness.broker.alpaca import AlpacaBroker
from harness.broker.venue import KillSwitchEngaged, Venue, VenueDataError, VenueUnavailable
from harness.config import ConfigError, ExecutionConfig, load_execution
from harness.data import (
    PROJECT_ROOT,
    BarPanel,
    DataError,
    DataLoader,
    PanelError,
    PaperLogError,
)
from harness.engine import Engine, EquityPoint, StrategyError
from harness.inbox import Inbox
from harness.journal import Journal, JournalError, JournalRecord
from harness.keys import KeysError
from harness.killswitch import KillSwitch, kill
from harness.models import MARKET_TZ, FrozenModel, PortfolioSnapshot, Signal
from harness.orders import (
    TERMINAL_PHASES,
    OrderManager,
    OrderManagerError,
    SessionIncomplete,
    StagedOrderChanged,
    WorkingOrder,
    flatten_number,
)
from harness.paper_report import PAPER_TRADE_COLUMNS, paper_report, unpriced_fills
from harness.positions import Ledger, LedgerError, PnL, ReconcileReport, reconcile_venue
from harness.profiles import (
    KILL_FILE,
    ActiveProfile,
    FrozenProfile,
    ProfileError,
    ProfileTampered,
    _write_atomic,
    activate_profile,
    active_profile,
    build_profile_strategy,
    load_profile,
    load_risk_state,
    paper_access_for,
    paper_access_log_problems,
    promote_run,
    save_risk_state,
)
from harness.ratelimit import RateLimiter
from harness.report import EQUITY_COLUMNS, ReportError, _json_safe, _write_csv
from harness.risk import RiskManager, RiskStateError
from harness.run import USER_ERRORS, _git_commit
from harness.view import PRICE_FIELDS, MarketView
from strategies.base import Strategy

LOCK_FILE = "runner.lock"
ALERTS_FILE = "alerts.jsonl"
JOURNAL_FILE = "journal.jsonl"
LEDGER_FILE = "ledger.json"
EQUITY_FILE = "equity.csv"
TRADES_FILE = "trades.csv"
SUMMARY_FILE = "summary.json"
SNAPSHOT_FILE = "config_snapshot.yaml"
PENDING_FILE = "pending_approvals.json"
ORDERS_FILE = "orders.json"
LAST_RUN_FILE = "last_run.json"
INBOX_DIR = "inbox"
LOG_DIR = "logs"
# How far back the venue's calendar is read to find the session before today. Longer
# than any market closure on record.
CALENDAR_LOOKBACK_DAYS = 14
_SOURCE = "auto:runner"

Outcome = Literal[
    "traded", "no_session", "already_done", "skipped_stale_data", "killed", "failed"
]  # fmt: skip


class PaperError(RuntimeError):
    """The runner was not started: another one is running, or the profile is not the
    active one. Nothing was read from the venue and nothing was sent."""


class DayReport(FrozenModel):
    """What one run of the runner did."""

    profile_id: str
    session: dt.date  # the session traded (today in New York)
    decided_on: dt.date | None  # the close the decision used; None if it was never reached
    outcome: Outcome
    resumed: bool = False  # the decision was made by an earlier run of the same day
    orders: tuple[WorkingOrder, ...] = ()  # the orders decided on `decided_on`
    reconcile_before: ReconcileReport | None = None
    reconcile_after: ReconcileReport | None = None
    pnl: PnL | None = None
    alerts: tuple[Alert, ...] = ()


class LastRun(FrozenModel):
    """`last_run.json`: how the runner's latest run ended, for `harness.service`."""

    finished_at: AwareDatetime
    report: DayReport


def write_orders(folder: Path, orders: Sequence[WorkingOrder]) -> None:
    """`orders.json`: every order the order manager knows, as of now. Only the process
    that holds the order manager can write it; readers get the last snapshot."""
    _write_atomic(folder / ORDERS_FILE, _json([o.model_dump(mode="json") for o in orders]))


class _End(Exception):
    """Ends the day early with this outcome. Whatever needed saying was alerted."""

    def __init__(self, outcome: Outcome) -> None:
        super().__init__(outcome)
        self.outcome: Outcome = outcome


class _Collector:
    """Passes alerts on to the real sinks and keeps them for the day report."""

    def __init__(self, sinks: Alerter, kept: list[Alert]) -> None:
        self._sinks = sinks
        self._kept = kept

    def send(self, alert: Alert) -> None:
        self._kept.append(alert)
        send_alert(self._sinks, alert)


class _ProfileStrategy(Strategy):
    """The profile's strategy, with anything it raises turned into a `StrategyError`.

    The runner treats the two differently: a strategy that raises costs the day its
    orders and nothing else, while an error from the harness's own code during the
    decision means the runner cannot vouch for itself, and kills. The engine still runs
    these calls inside its guard."""

    def __init__(self, inner: Strategy) -> None:
        super().__init__()
        self._inner = inner

    def on_start(self, symbols: tuple[str, ...]) -> None:
        try:
            self._inner.on_start(symbols)
        except Exception as exc:
            raise StrategyError(f"{type(self._inner).__name__}.on_start raised {exc!r}") from exc

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        try:
            return self._inner.on_bar(view, portfolio)
        except Exception as exc:
            raise StrategyError(f"{type(self._inner).__name__}.on_bar raised {exc!r}") from exc


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _alpaca_venue(cfg: ExecutionConfig, kill_switch: KillSwitch) -> Venue:
    return AlpacaBroker(cfg, kill_switch, RateLimiter(cfg.max_requests_per_minute))


@contextlib.contextmanager
def _runner_lock(path: Path) -> Iterator[None]:
    """One runner per account. Held for the whole run; the OS drops it if the process
    dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise PaperError(f"another runner holds {path}") from exc
        yield
    finally:
        os.close(fd)


def run_trading_day(
    profile_id: str,
    *,
    venue: Venue | None = None,
    loader: DataLoader | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
    execution: ExecutionConfig | None = None,
    alerts: Alerter | None = None,
    git_commit: Callable[[], str | None] = lambda: _git_commit(PROJECT_ROOT),
) -> DayReport:
    """Run one trading day for a frozen profile: decide on the last completed session,
    trade the next open, reconcile, exit. Safe to run twice: the second run resumes or
    reports `already_done`. Started by the OS scheduler, never by the backend.

    `execution` (default: `config/execution.yaml`) only says where profiles and state
    live. Everything about trading comes from the profile's own frozen settings.
    Raises PaperError if another runner is running or the profile is not the active one.
    """
    execution = execution if execution is not None else load_execution()
    with _runner_lock(execution.state_dir / LOCK_FILE):
        day = _Day(profile_id, execution, venue, loader, sleep, now, alerts, git_commit)
        return day.run()


class _Day:
    """One run of the runner. The methods are the steps, in the order `_day` calls them."""

    def __init__(
        self,
        profile_id: str,
        execution: ExecutionConfig,
        venue: Venue | None,
        loader: DataLoader | None,
        sleep: Callable[[float], None],
        now: Callable[[], dt.datetime],
        alerts: Alerter | None,
        git_commit: Callable[[], str | None],
    ) -> None:
        self.profile_id = profile_id
        self.execution = execution  # where profiles and state live
        self.cfg = execution  # replaced by the profile's own frozen settings once loaded
        self.state_dir = execution.state_dir
        self.folder = self.state_dir / profile_id
        self.venue = venue
        self.loader = loader
        self.sleep = sleep
        self.now = now
        self.git_commit = git_commit
        self.kill_switch = KillSwitch(self.state_dir / KILL_FILE, now)
        self.sent: list[Alert] = []
        sinks = alerts or Alerters(LogAlerter(), JsonlAlerter(self.state_dir / ALERTS_FILE))
        self.alerts = _Collector(sinks, self.sent)
        self.session = now().astimezone(MARKET_TZ).date()  # replaced by the venue's date
        self.decided_on: dt.date | None = None
        self.resumed = False
        self.strategy_name = ""  # as a backtest's summary names it; set when it is built
        # Set once the venue has confirmed this is the account the profile was
        # activated on. Until then a kill sets the latch and cancels nothing: the keys
        # may be for another account.
        self.account_verified = False
        self.journal: Journal | None = None
        self.om: OrderManager | None = None
        self.reconcile_before: ReconcileReport | None = None
        self.reconcile_after: ReconcileReport | None = None
        self.pnl: PnL | None = None

    # ------------------------------------------------------------------ the frame

    def run(self) -> DayReport:
        try:
            outcome = self._guarded()
        finally:
            if self.journal is not None:
                self.journal.close()
        orders: tuple[WorkingOrder, ...] = ()
        if self.om is not None and self.decided_on is not None:
            orders = tuple(o for o in self.om.orders() if o.order.created_on == self.decided_on)
        report = DayReport(
            profile_id=self.profile_id,
            session=self.session,
            decided_on=self.decided_on,
            outcome=outcome,
            resumed=self.resumed,
            orders=orders,
            reconcile_before=self.reconcile_before,
            reconcile_after=self.reconcile_after,
            pnl=self.pnl,
            alerts=tuple(self.sent),
        )
        self._write_last_run(report)
        return report

    def _write_last_run(self, report: DayReport) -> None:
        """How the run ended and the orders as they are now, for readers
        (`harness.service`). Written however the day ended; not needed to trade, so a
        failure here is a warning."""
        try:
            if self.om is not None:
                write_orders(self.folder, self.om.orders())
            finished = self.now().astimezone(dt.UTC)
            last = LastRun(finished_at=finished, report=report)
            _write_atomic(self.folder / LAST_RUN_FILE, last.model_dump_json(indent=2).encode())
        except Exception as exc:
            self._alert("warn", "state", f"{LAST_RUN_FILE} or {ORDERS_FILE} was not written: {exc}")

    def _guarded(self) -> Outcome:
        try:
            return self._day()
        except _End as end:
            return end.outcome
        except PaperError:
            raise
        except KillSwitchEngaged as exc:
            self._alert(
                "critical",
                "kill_switch",
                f"the kill switch is engaged, so the day stopped ({exc}). After it is "
                "reset, start the runner again.",
            )
            return "killed"
        except SessionIncomplete:
            return "killed"  # the order manager engaged the switch and said why
        except Exception as exc:
            # Not one of the failures this module knows how to stop for. Whatever state
            # the day is in, nothing more may be sent until someone has looked.
            self._engage(f"the runner failed unexpectedly: {exc!r}")
            raise

    def _day(self) -> Outcome:
        if self.kill_switch.engaged():
            state = self.kill_switch.state()
            self._alert(
                "warn",
                "kill_switch",
                f"the kill switch is engaged ({state.source or 'unknown'}: {state.reason}); "
                "the day was not run. Reset it after a clean reconciliation, then start "
                "the runner again.",
            )
            return "killed"
        profile, active = self._load_profile()
        self.cfg = cfg = profile.execution
        if self.venue is None:
            try:
                self.venue = _alpaca_venue(cfg, self.kill_switch)
            except (KeysError, VenueDataError) as exc:
                self._fail("venue_setup", f"the paper venue cannot be used: {exc}")
        venue = self.venue

        venue_now, today, t = self._calendar(profile, active, venue)
        if t is None:
            return "no_session"
        self.kill_switch.check()

        try:
            self.journal = journal = Journal(self.folder / JOURNAL_FILE, self.now)
            records = list(journal.read())
        except JournalError as exc:
            self._fail("state", f"the profile's journal cannot be used: {exc}")
        done = sorted(_date(r, "session") for r in records if r.kind == "day_complete")
        if today in done:
            return "already_done"
        self.resumed = any(r.kind == "decision" and _date(r, "decided_on") == t for r in records)
        if done and done[-1] < t:
            self._alert_missed(venue, done[-1], today)
        if not self.resumed and venue_now >= dt.datetime.combine(
            today, cfg.session_deadline, MARKET_TZ
        ):
            # Deciding now would stage orders only to cancel them all at once. A day that
            # was decided earlier is different: it is carried on and closed below.
            self._fail(
                "late_start",
                f"the runner started at {venue_now.astimezone(MARKET_TZ):%H:%M} New York, "
                f"after the session deadline ({cfg.session_deadline:%H:%M}). Nothing was "
                f"decided or sent for {today}; positions are as they were.",
            )

        ledger, risk, om = self._open_state(profile, active, venue, journal)
        self._settle_earlier_orders(om, t)
        om.ingest_inbox(dt.datetime.combine(today, cfg.approval_deadline, MARKET_TZ))
        om.poll_updates()  # nothing found so far belongs to the engine built below
        self.kill_switch.check()

        commit = self.git_commit()
        journal.append(
            "day_start",
            {
                "session": today.isoformat(),
                "decided_on": t.isoformat(),
                "resumed": self.resumed,
                "harness_git_commit": commit,
            },
        )
        self._write_snapshot(profile, commit)
        if commit != profile.harness_git_commit:
            self._alert(
                "warn",
                "harness_changed",
                f"the harness is at {commit}; profile {profile.profile_id} was frozen at "
                f"{profile.harness_git_commit}",
            )
        self.reconcile_before = self._reconcile(
            profile, active, venue, journal, ledger, om, "before"
        )
        self._check_access_log(profile)
        self.kill_switch.check()

        engine, closes = self._decide(profile, journal, ledger, risk, om, records, today, t)
        self._write_pending(om, today)
        self.kill_switch.check()
        engine.on_session_open(today, {})  # the open workflow; blocks until all are final
        self.kill_switch.check()

        self.reconcile_after = self._reconcile(profile, active, venue, journal, ledger, om, "after")
        self._finish(profile, journal, ledger, risk, om, closes, today, t)
        return "traded"

    # ------------------------------------------------------------------ steps

    def _load_profile(self) -> tuple[FrozenProfile, ActiveProfile]:
        execution = self.execution
        try:
            profile = load_profile(self.profile_id, profiles_dir=execution.profiles_dir)
        except ProfileTampered as exc:
            # The latch only: no account has been verified yet, so nothing is canceled.
            self._kill(f"profile {self.profile_id} does not verify: {exc}")
        except ProfileError as exc:
            raise PaperError(str(exc)) from exc
        try:
            active = active_profile(self.state_dir)
        except ProfileError as exc:
            raise PaperError(str(exc)) from exc
        if active is None or active.profile_id != self.profile_id:
            current = active.profile_id if active is not None else "none"
            raise PaperError(
                f"profile {self.profile_id} is not the active profile (active: {current}); "
                "activate it first"
            )
        frozen = profile.execution
        if (frozen.state_dir, frozen.profiles_dir) != (execution.state_dir, execution.profiles_dir):
            raise PaperError(
                f"profile {self.profile_id} was frozen with state_dir {frozen.state_dir} and "
                f"profiles_dir {frozen.profiles_dir}, which are not the ones in use now; "
                "promote a new profile"
            )
        return profile, active

    def _calendar(
        self, profile: FrozenProfile, active: ActiveProfile, venue: Venue
    ) -> tuple[dt.datetime, dt.date, dt.date | None]:
        """The venue's time, today's session and the one before it (None if the market
        is closed today), after the account and the clock have been checked."""
        cfg = profile.execution
        account = self._ask("verify the account", venue.verify_account, "account_setup")
        if account.account_id != active.account_id:
            self._fail(
                "wrong_account",
                f"the keys are for account {account.account_id}, but profile "
                f"{profile.profile_id} was activated on {active.account_id}; nothing was sent",
            )
        self.account_verified = True
        venue_now = self._ask("read the venue's clock", venue.now)
        skew = (venue_now - self.now()).total_seconds()
        if abs(skew) > cfg.max_clock_skew_seconds:
            self._kill(
                f"clock skew: the venue's clock is {skew:+.0f} seconds from the local clock "
                f"(limit {cfg.max_clock_skew_seconds:.0f})"
            )
        today = venue_now.astimezone(MARKET_TZ).date()
        self.session = today
        start = today - dt.timedelta(days=CALENDAR_LOOKBACK_DAYS)
        dates = [
            s.date for s in self._ask("read the calendar", lambda: venue.sessions(start, today))
        ]
        if today not in dates:
            self._alert("info", "no_session", f"{today} is not a trading session; nothing to do")
            return venue_now, today, None
        earlier = [d for d in dates if d < today]
        if not earlier:
            self._fail("calendar", f"the venue lists no session in the {CALENDAR_LOOKBACK_DAYS} "
                       f"days before {today}")  # fmt: skip
        self.decided_on = earlier[-1]
        return venue_now, today, earlier[-1]

    def _alert_missed(self, venue: Venue, last_done: dt.date, today: dt.date) -> None:
        """A notice only: a calendar that cannot be read here does not cost the day."""
        day = dt.timedelta(days=1)
        try:
            missed = venue.sessions(last_done + day, today - day)
        except (VenueUnavailable, VenueDataError) as exc:
            self._alert(
                "warn",
                "missed_sessions",
                f"could not check for sessions missed since {last_done}: {exc}",
            )
            return
        if missed:
            self._alert(
                "warn",
                "missed_sessions",
                f"the runner did not complete {len(missed)} session(s) since {last_done}: "
                + ", ".join(s.date.isoformat() for s in missed)
                + ". Carrying on from the latest close.",
            )

    def _open_state(
        self, profile: FrozenProfile, active: ActiveProfile, venue: Venue, journal: Journal
    ) -> tuple[Ledger, RiskManager, OrderManager]:
        recipe, cfg = profile.recipe, profile.execution
        assert recipe.risk is not None  # FrozenProfile
        try:
            ledger = Ledger.open(journal, recipe.backtest.starting_cash)
            state = load_risk_state(self.state_dir, self.profile_id)
            risk = RiskManager(recipe.risk, cfg.limits, cfg.entry_limit_pct, state)
            self.om = om = OrderManager(
                venue,
                ledger,
                journal,
                self.kill_switch,
                self.alerts,
                cfg,
                self.profile_id,
                risk,
                recipe.construction.max_positions,
                active.approval,
                Inbox(self.folder / INBOX_DIR),
                sleep=self.sleep,
                now=self.now,
            )
            om.recover()  # first: resolves what may be at the venue before anything else
        except (JournalError, LedgerError, ProfileError, OrderManagerError) as exc:
            self._fail("state", f"the profile's saved state cannot be used: {exc}")
        return ledger, risk, om

    def _settle_earlier_orders(self, om: OrderManager, t: dt.date) -> None:
        """The order manager works any open order decided before the session; it does
        not know the calendar. Only orders decided on `t` may be worked today, so what
        an unfinished earlier day left behind is closed here, before anything else.

        So is a sell of the operator's flatten that an interrupted flatten left open,
        whatever its date (a flatten dates its sells `t` too). Selling everything was
        asked of that command, not of this run: the runner never sends it."""
        earlier = [
            o
            for o in om.orders()
            if o.phase not in TERMINAL_PHASES
            and (o.order.created_on != t or flatten_number(o.order) is not None)
        ]
        for order in earlier:
            if order.order.created_on > t or order.phase not in ("staged", "idle"):
                if flatten_number(order.order) is not None:
                    self._kill(
                        f"flatten sell {order.order_id}, left by an interrupted flatten, is "
                        f"{order.phase}: it may still be at the venue. Run flatten again, or "
                        "engage the kill switch to cancel it"
                    )
                self._kill(
                    f"order {order.order_id}, decided on {order.order.created_on}, is "
                    f"{order.phase}; only orders decided on {t} can be open today"
                )
        for order in earlier:
            unsold = order.quantity - order.filled_quantity
            if flatten_number(order.order) is not None:
                om.close(order.order_id, "canceled", "flatten")
                self._alert(
                    "critical",
                    "flatten_unfinished",
                    f"flatten sell {order.order_id} was left unfinished by an interrupted "
                    f"flatten: {unsold} of {order.quantity} shares unsold. It is closed and "
                    "was not sent; the position remains. To sell it, run flatten again.",
                )
                continue
            om.close(order.order_id, "canceled", "session_deadline")
            if order.order.action == "exit":
                self._alert(
                    "critical",
                    "exit_unfilled",
                    f"exit {order.order_id}, decided on {order.order.created_on}, was never "
                    f"completed: {unsold} of {order.quantity} shares unsold. The position "
                    "remains; today's decision starts from it.",
                )
            else:
                self._alert(
                    "warn",
                    "entry_dropped",
                    f"entry {order.order_id}, decided on {order.order.created_on}, was never "
                    "completed and is closed",
                )

    def _reconcile(
        self,
        profile: FrozenProfile,
        active: ActiveProfile,
        venue: Venue,
        journal: Journal,
        ledger: Ledger,
        om: OrderManager,
        stage: Literal["before", "after"],
    ) -> ReconcileReport:
        expected = [
            o.venue_orders[-1].client_order_id
            for o in om.orders()
            if o.phase in ("working", "submitting", "unknown") and o.venue_orders
        ]
        report = self._ask(
            "reconcile",
            lambda: reconcile_venue(
                ledger.account(),
                expected,
                venue,
                active.cash_offset,
                profile.execution.cash_tolerance,
            ),
        )
        journal.append("reconcile", {"stage": stage, "report": report.model_dump(mode="json")})
        if report.severity == "stop":
            if stage == "before":
                self.reconcile_before = report
            else:
                self.reconcile_after = report
            self._kill(f"reconciliation {stage} the open: " + "; ".join(report.reasons))
        if report.severity == "warn":
            self._alert("warn", "reconcile", f"{stage} the open: " + "; ".join(report.reasons))
        return report

    def _check_access_log(self, profile: FrozenProfile) -> None:
        log = profile.recipe.data.paper_access_log
        try:
            problems = paper_access_log_problems(log, profile.execution.profiles_dir)
        except (OSError, ValueError) as exc:  # csv.Error and UnicodeDecodeError are ValueErrors
            self._fail("paper_access_log", f"the paper access log {log} cannot be read: {exc}")
        for problem in problems:
            self._alert(
                "critical",
                "paper_access_log",
                f"holdout data was read without a real profile: {problem}",
            )

    def _decide(
        self,
        profile: FrozenProfile,
        journal: Journal,
        ledger: Ledger,
        risk: RiskManager,
        om: OrderManager,
        records: Sequence[JournalRecord],
        today: dt.date,
        t: dt.date,
    ) -> tuple[Engine, dict[str, float]]:
        """The engine the open runs on, and the closes of `t` positions are marked at.
        Makes the decision for `t` unless the journal already holds it."""
        recipe = profile.recipe
        universe = recipe.universe.symbols
        try:
            inner = build_profile_strategy(profile, profiles_dir=profile.execution.profiles_dir)
        except ProfileTampered as exc:
            self._kill(f"profile {self.profile_id} does not verify: {exc}")
        except ProfileError as exc:
            self._fail("strategy", f"the profile's strategy cannot be used: {exc}")
        strategy = _ProfileStrategy(inner)
        # The frozen copy is imported from the profile's folder, so its module name says
        # nothing: without a `name` of its own it is called what the recipe calls it.
        self.strategy_name = type(inner).name or recipe.strategy.strategy

        def engine_for(capacity: int) -> Engine:
            try:
                return Engine(
                    strategy, om, universe, recipe.construction, recipe.costs, capacity, risk
                )
            except StrategyError as exc:
                self._fail(
                    "decision_failed",
                    f"the strategy failed: {exc}. No orders today; positions are untouched.",
                )

        if self.resumed:
            # Decided by an earlier run of today. The strategy is not asked again: the
            # orders and the closes are in the journal, the risk state on disk.
            if risk.state.marked_on != t:
                self._fail(
                    "state",
                    f"the journal holds the decision for {t}, but the saved risk state was "
                    f"last marked on {risk.state.marked_on}",
                )
            engine = engine_for(1)
            engine.orders.update(
                {o.order_id: o.order for o in om.orders() if o.order.created_on == t}
            )
            return engine, _journaled_closes(records)

        panel = self._load_bars(profile, t)
        closes = _last_closes(panel, universe)
        held = ledger.account().positions
        strays = sorted(s for s in held if s not in universe or s not in closes)
        if strays:
            self._fail(
                "unknown_position",
                f"the ledger holds {', '.join(strays)}, not in the profile's universe or "
                "without any bar; the engine cannot value it. No orders today.",
            )
        try:
            om.set_closes(closes)  # before the engine submits: a missing close fails it
        except OrderManagerError as exc:
            self._fail("decision_failed", f"the closes of {t} cannot be used: {exc}")

        engine = engine_for(panel.n_dates)
        cols = [panel.column(s) for s in universe]
        fields = {f: panel.field(f)[:, cols] for f in PRICE_FIELDS}
        for i, session in enumerate(panel.dates[:-1]):
            engine.warm_up(session, {f: fields[f][i] for f in PRICE_FIELDS})
        # What an earlier attempt at this decision staged. An operator's flatten order
        # carries the same date and was never the strategy's to decide again.
        staged_before = {
            o.order_id
            for o in om.orders()
            if o.order.created_on == t and flatten_number(o.order) is None
        }
        try:
            engine.on_session_close(t, {f: fields[f][-1] for f in PRICE_FIELDS})
        except RiskStateError as exc:
            self._fail(
                "account_changed",
                f"account changed since the first attempt, reconcile first ({exc})",
            )
        except StagedOrderChanged as exc:
            self._fail(
                "decision_changed",
                f"an earlier attempt at the decision for {t} staged a different order "
                f"({exc}): the bars or the account changed in between. Nothing is sent today.",
            )
        except StrategyError as exc:
            # The strategy's own failure. Anything else raised in there is the harness's
            # (construction, risk, the order manager) and is not caught: it kills.
            self._fail(
                "decision_failed",
                f"the strategy failed on {t}: {exc}. No orders today; positions are untouched.",
            )
        ids: list[JsonValue] = list(engine.orders)
        dropped = sorted(staged_before - set(engine.orders))
        if dropped:
            self._fail(
                "decision_changed",
                f"an earlier attempt at the decision for {t} staged {', '.join(dropped)}, "
                "which this attempt did not decide: the bars or the account changed in "
                "between. Nothing is sent today.",
            )
        fill_opens = self._fill_opens(profile, journal, t)
        benchmark = recipe.backtest.benchmark
        benchmark_close = panel.field("close")[-1, panel.column(benchmark)]
        # The mark first, then the record that says the decision is made: a rerun that
        # finds no record marks again, and the same equity changes nothing.
        save_risk_state(risk.state, self.state_dir, self.profile_id)
        journal.append(
            "decision",
            {
                "session": today.isoformat(),
                "decided_on": t.isoformat(),
                "orders": ids,
                "equity": engine.equity[-1].model_dump(mode="json"),
                "events": [e.model_dump(mode="json") for e in engine.events],
                "data_fingerprint": panel.fingerprint(),
                "fill_opens": fill_opens,
                "benchmark_close": None if np.isnan(benchmark_close) else float(benchmark_close),
            },
        )
        return engine, closes

    def _fill_opens(
        self, profile: FrozenProfile, journal: Journal, t: dt.date
    ) -> dict[str, JsonValue]:
        """The official open of every fill through `t` that has none on record yet, as
        the `decision` record carries it: `{session: {symbol: open}}`. Unadjusted, so it
        is asked of the data source and not read from the bars just loaded (those are
        adjusted, and a dividend or split that goes ex today has already moved them).

        Only the report needs these. If they cannot be had, that is a warning, the
        fills stay unmeasured, and the next decision asks again."""
        recipe = profile.recipe
        loader = self.loader or DataLoader(recipe.dates, recipe.data, now=self.now)
        try:
            wanted = unpriced_fills(list(journal.read()), t)
            opens = loader.raw_opens(wanted, paper=paper_access_for(profile))
        except (ReportError, DataError, PaperLogError) as exc:
            self._alert(
                "warn",
                "report",
                f"the fills through {t} could not be compared with their official opens "
                f"today: {exc}",
            )
            return {}
        found: dict[str, dict[str, JsonValue]] = {}
        for (session, symbol), value in opens.items():
            found.setdefault(session.isoformat(), {})[symbol] = value
        return dict(found)

    def _load_bars(self, profile: FrozenProfile, t: dt.date) -> BarPanel:
        recipe = profile.recipe
        benchmark = recipe.backtest.benchmark
        loader = self.loader or DataLoader(recipe.dates, recipe.data, now=self.now)
        symbols = list(dict.fromkeys([*recipe.universe.symbols, benchmark]))
        try:
            panel = loader.load(symbols, paper=paper_access_for(profile))
        except PaperLogError as exc:
            self._fail("paper_access_log", f"no bars and no orders today: {exc}")
        except (DataError, PanelError) as exc:
            self._fail("data", f"bars could not be loaded, so no orders today: {exc}")
        seen = np.flatnonzero(~np.isnan(panel.field("close")[:, panel.column(benchmark)]))
        latest = panel.dates[int(seen[-1])] if seen.size else None
        if latest != t or panel.dates[-1] != t:
            self._alert(
                "warn",
                "stale_data",
                f"the latest {benchmark} bar is {latest} and the latest bar of any symbol is "
                f"{panel.dates[-1]}; the decision needs {t}. No decision on stale data: "
                "nothing is sent. Run again once the bar is there.",
            )
            raise _End("skipped_stale_data")
        return panel

    def _write_pending(self, om: OrderManager, today: dt.date) -> None:
        """The entries waiting for approval and every order as it stands, where an
        operator's tools can read them: only this process holds the order manager."""
        pending = [r.model_dump(mode="json") for r in om.pending_approvals(today)]
        _write_atomic(self.folder / PENDING_FILE, _json(pending))
        write_orders(self.folder, om.orders())

    def _write_snapshot(self, profile: FrozenProfile, commit: str | None) -> None:
        """Every value the day runs with, the whole of the execution settings included."""
        snapshot = {
            "run": {
                "profile_id": profile.profile_id,
                "content_hash": profile.content_hash,
                "strategy": profile.recipe.strategy.strategy,
                "strategy_source_sha256": profile.strategy_source_sha256,
                "harness_git_commit": commit,
                "profile_harness_git_commit": profile.harness_git_commit,
                "python": sys.version.split()[0],
            },
            "recipe": profile.recipe.model_dump(mode="json"),
            "execution": profile.execution.model_dump(mode="json"),
        }
        _write_atomic(
            self.folder / SNAPSHOT_FILE, yaml.safe_dump(snapshot, sort_keys=False).encode()
        )

    def _finish(
        self,
        profile: FrozenProfile,
        journal: Journal,
        ledger: Ledger,
        risk: RiskManager,
        om: OrderManager,
        closes: Mapping[str, float],
        today: dt.date,
        t: dt.date,
    ) -> None:
        """Derived files first, then the record that says the day is done: a crash in
        between repeats this step, which only rewrites the same files."""
        ledger.write_snapshot(self.folder / LEDGER_FILE)
        records = list(journal.read())
        points: dict[dt.date, EquityPoint] = {}
        for record in records:
            if record.kind == "decision":
                point = EquityPoint.model_validate(record.payload["equity"])
                points[point.date] = point
        _write_csv_atomic(
            self.folder / EQUITY_FILE,
            EQUITY_COLUMNS,
            [points[d].model_dump() for d in sorted(points)],
        )
        self._write_report(profile, records, closes, t)
        self._write_pending(om, today)
        try:
            self.pnl = ledger.pnl(closes, t, prior_equity=risk.state.prior_equity)
        except LedgerError as exc:
            self._alert("warn", "pnl", f"the day's P&L could not be computed: {exc}")
        journal.append(
            "day_complete",
            {
                "session": today.isoformat(),
                "decided_on": t.isoformat(),
                "orders": {o.order_id: o.phase for o in om.orders() if o.order.created_on == t},
            },
        )

    def _write_report(
        self,
        profile: FrozenProfile,
        records: Sequence[JournalRecord],
        closes: Mapping[str, float],
        t: dt.date,
    ) -> None:
        """`trades.csv` and `summary.json` for sessions through `t`. Derived, and not
        needed to trade: a journal they cannot be built from is a warning, and the
        files of the day before stay as they are."""
        try:
            trades, summary = paper_report(
                records,
                profile.recipe,
                profile_id=self.profile_id,
                strategy=self.strategy_name,
                through=t,
                marks=closes,
            )
        except ReportError as exc:
            self._alert(
                "warn",
                "report",
                f"{TRADES_FILE} and {SUMMARY_FILE} were not updated for {t}: {exc}",
            )
            return
        _write_csv_atomic(self.folder / TRADES_FILE, PAPER_TRADE_COLUMNS, trades)
        _write_atomic(
            self.folder / SUMMARY_FILE,
            (json.dumps(_json_safe(summary), indent=2) + "\n").encode(),
        )

    # ------------------------------------------------------------------ helpers

    def _ask[T](self, what: str, call: Callable[[], T], data_kind: str = "venue_data") -> T:
        """A read from the venue, tried a few times. A venue that does not answer, or
        answers with something that cannot be trusted (alerted as `data_kind`), fails
        the day: nothing has been decided on what it would have said."""
        tries = self.cfg.max_submit_attempts
        problem = ""
        for tried in range(tries):
            if tried:
                self.sleep(self.cfg.poll_seconds)
            try:
                return call()
            except VenueUnavailable as exc:
                problem = str(exc)
            except VenueDataError as exc:
                self._fail(data_kind, f"could not {what}: {exc}")
        self._fail(
            "venue_unavailable",
            f"could not {what}: the venue did not answer in {tries} tries ({problem}). "
            "Run again once it does.",
        )

    def _alert(self, severity: Severity, kind: str, message: str) -> None:
        try:
            alert = Alert(at=self.now(), severity=severity, kind=kind, message=message)
        except Exception:
            alert = Alert(at=_utc_now(), severity=severity, kind=kind, message=message)
        self.alerts.send(alert)

    def _fail(self, kind: str, message: str) -> NoReturn:
        self._alert("critical", kind, message)
        raise _End("failed")

    def _engage(self, reason: str) -> None:
        kill(
            self.kill_switch,
            reason,
            _SOURCE,
            venue=self.venue if self.account_verified else None,
            alerts=self.alerts,
            wait_seconds=self.cfg.poll_seconds,
            sleep=self.sleep,
            now=self.now,
        )

    def _kill(self, reason: str) -> NoReturn:
        self._engage(reason)
        raise _End("killed")


def _date(record: JournalRecord, key: str) -> dt.date:
    return dt.date.fromisoformat(str(record.payload[key]))


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _last_closes(panel: BarPanel, universe: Sequence[str]) -> dict[str, float]:
    """The most recent close of each symbol that has one: what the engine marks at."""
    close = panel.field("close")
    closes: dict[str, float] = {}
    for symbol in universe:
        column = close[:, panel.column(symbol)]
        seen = np.flatnonzero(~np.isnan(column))
        if seen.size:
            closes[symbol] = float(column[seen[-1]])
    return closes


def _journaled_closes(records: Sequence[JournalRecord]) -> dict[str, float]:
    """The closes the order manager journaled last (those of the decision)."""
    for record in reversed(records):
        if record.kind == "closes":
            closes = record.payload["closes"]
            if isinstance(closes, dict):
                return {s: float(c) for s, c in closes.items() if isinstance(c, int | float)}
    return {}


def _write_csv_atomic(
    path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        _write_csv(Path(tmp), columns, rows)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- CLI

_OK: frozenset[Outcome] = frozenset({"traded", "no_session", "already_done"})


def _print_report(report: DayReport) -> None:
    print(f"profile {report.profile_id}  session {report.session}  outcome {report.outcome}")
    if report.decided_on is not None:
        resumed = "  (resumed)" if report.resumed else ""
        print(f"decided on the close of {report.decided_on}{resumed}")
    for order in report.orders:
        leg = order.order.legs[0]
        print(
            f"  {order.order_id}  {leg.side} {order.order.symbol}  "
            f"{order.filled_quantity}/{order.quantity}  {order.phase}  {order.reason}".rstrip()
        )
    for stage, reconcile in (
        ("before", report.reconcile_before),
        ("after", report.reconcile_after),
    ):
        if reconcile is not None:
            print(
                f"reconcile {stage}: {reconcile.severity}  {'; '.join(reconcile.reasons)}".rstrip()
            )
    if report.pnl is not None:
        print(f"equity {report.pnl.equity:.2f}  cash {report.pnl.cash:.2f}")
    for alert in report.alerts:
        print(f"[{alert.severity}] {alert.kind}: {alert.message}")


def _log_to_file(state_dir: Path) -> None:
    """The human-readable log: `<state_dir>/logs/<New York date>.log`, and stderr."""
    folder = state_dir / LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    day = _utc_now().astimezone(MARKET_TZ).date().isoformat()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(folder / f"{day}.log"), logging.StreamHandler()],
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness.paper", description="Paper trading: the daily runner."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run today's trading day")
    run.add_argument("--profile", default=None, help="profile id (default: the active profile)")
    promote = commands.add_parser("promote", help="freeze a backtest run into a profile")
    promote.add_argument("run_dir", type=Path, help="the in-sample run folder in backtests/")
    promote.add_argument("name", help="profile name")
    promote.add_argument(
        "--plumbing", action="store_true", help="no holdout run needed (example strategy only)"
    )
    activate = commands.add_parser("activate", help="bind the paper account to a profile")
    activate.add_argument("profile_id")
    activate.add_argument("--approval", choices=("off", "entries"), required=True)
    activate.add_argument("--operator", required=True, help="who is activating it")
    activate.add_argument(
        "--replace", action="store_true", help="replace another profile that is active"
    )
    args = parser.parse_args(argv)
    try:
        execution = load_execution()
        if args.command == "promote":
            profile = promote_run(
                args.run_dir, args.name, plumbing=args.plumbing, execution=execution
            )
            print(f"profile {profile.profile_id} written to {execution.profiles_dir}")
            return 0
        if args.command == "activate":
            kill_switch = KillSwitch(execution.state_dir / KILL_FILE)
            # Under the runner's lock: the active profile cannot change under a runner
            # that is working the open.
            with _runner_lock(execution.state_dir / LOCK_FILE):
                active = activate_profile(
                    args.profile_id,
                    args.approval,
                    operator=args.operator,
                    venue=_alpaca_venue(execution, kill_switch),
                    replace=args.replace,
                    execution=execution,
                )
            print(
                f"profile {active.profile_id} is active on account {active.account_id} "
                f"(approval {active.approval}, cash offset {active.cash_offset:.2f})"
            )
            return 0
        profile_id = args.profile
        if profile_id is None:
            active = active_profile(execution.state_dir)
            if active is None:
                raise PaperError("no profile is active; activate one first")
            profile_id = active.profile_id
        _log_to_file(execution.state_dir)
        report = run_trading_day(profile_id, execution=execution)
    except (PaperError, ProfileError, ConfigError, KeysError, VenueDataError, VenueUnavailable,
            *USER_ERRORS) as exc:  # fmt: skip
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_report(report)
    return 0 if report.outcome in _OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
