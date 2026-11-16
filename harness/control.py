"""Operator commands for paper trading: the functions that act.

The read-only views are in `harness.service`. These are kept apart from them on
purpose: a UI backend gives an operator both, and gives the chat agent only the views.
That includes the kill switch. An agent that can stop trading can be talked into
stopping it at a bad moment.

Every command takes the operator's name and leaves a record that carries it.

How a command reaches the books (the journal has one writer at a time):

- `engage_kill_switch` writes the latch and cancels at the venue itself. It takes no
  lock and waits for nothing, so it works while a runner is hung.
- `approve_orders` and `reject_orders` only drop a command file in the inbox. The
  runner reads it, checks it against its own state, and journals what it did.
- Everything else (`reset_kill_switch`, `reset_risk_halt`, `flatten`,
  `accept_broker_state`, `record_adjustment`, `activate_profile`) takes the runner's
  lock for as long as it runs and is then the journal's one writer. While a runner is
  running these refuse, and say so: none of them belongs in the middle of a trading
  day. To stop a running day, engage the kill switch; the runner cancels and exits.

A command either finishes or raises `ControlError` with what was and was not done.

`flatten` is the one thing that can send an order while the kill switch is engaged, and
it can only sell what the ledger holds (see `OrderManager.flatten`). It blocks while
the sells are worked, the way the runner blocks during the open: started before the
open it waits for the opening auction. It engages the kill switch first (if it is not
engaged already): an operator who flattens wants out, and without the latch the runner
would buy again at its next run. `reset_kill_switch` is the deliberate way back in.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from harness import profiles, service
from harness.alerts import Alert, Alerter, Alerters, JsonlAlerter, LogAlerter, Severity, send_alert
from harness.broker.venue import KillSwitchEngaged, Venue, VenueDataError, VenueUnavailable
from harness.config import MARKET_OPEN, ConfigError, ExecutionConfig, load_execution
from harness.inbox import ApproveOrders, Inbox, RejectOrders
from harness.journal import Journal, JournalError
from harness.keys import KeysError
from harness.killswitch import KillState, KillSwitch, KillSwitchError, kill
from harness.models import MARKET_TZ, FrozenModel
from harness.orders import OrderManager, OrderManagerError, SessionIncomplete, WorkingOrder
from harness.paper import (
    ALERTS_FILE,
    CALENDAR_LOOKBACK_DAYS,
    INBOX_DIR,
    JOURNAL_FILE,
    LEDGER_FILE,
    LOCK_FILE,
    PaperError,
    _alpaca_venue,
    _runner_lock,
    write_orders,
)
from harness.positions import (
    Ledger,
    LedgerAdjustment,
    LedgerError,
    ReconcileReport,
    reconcile_venue,
)
from harness.profiles import KILL_FILE, ActiveProfile, FrozenProfile, ProfileError
from harness.risk import RiskManager, RiskState, RiskStateError

# What `confirm` must be to reset a kill switch that was engaged with no profile active.
NO_PROFILE = "none"
_CENT = 0.005


class ControlError(RuntimeError):
    """A command was refused, or stopped part way. The message says which, and what to
    do next."""


class KillReset(FrozenModel):
    cleared: KillState  # the latch that was removed: who set it, when and why
    reconcile: ReconcileReport | None  # the check the reset passed; None with no profile
    message: str  # what to do next


class FlattenReport(FrozenModel):
    session: dt.date
    orders: tuple[WorkingOrder, ...]  # the sells, one per position held when it started
    remaining: dict[str, int]  # what the ledger still holds; empty when flat
    # Positions the ledger and the venue disagreed about before the sells, and what was
    # sold of each. Not empty: settle with `accept_broker_state`.
    differences: tuple[str, ...]
    reconcile: ReconcileReport  # the ledger against the venue afterwards


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# --------------------------------------------------------------------------- kill switch


def engage_kill_switch(
    reason: str,
    operator: str,
    *,
    venue: Venue | None = None,
    execution: ExecutionConfig | None = None,
    alerts: Alerter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
) -> KillState:
    """Set the latch, then cancel every open order at the venue and confirm none
    remain. Never sells. Never raises for anything that happens after the arguments are
    checked: read `cancel_confirmed` on the result, and the alert.

    The latch is written first, before the venue is asked anything. Orders are canceled
    only if the keys are for the account the active profile was activated on; otherwise
    the latch is set and the alert says nothing was canceled.
    """
    _named(operator)
    if not reason.strip():
        raise ControlError("say why the kill switch is being engaged")
    execution = execution if execution is not None else load_execution()
    state_dir = execution.state_dir
    switch = KillSwitch(state_dir / KILL_FILE, now)
    sinks = alerts or _sinks(state_dir)
    source = f"operator:{operator}"
    try:
        switch.engage(reason, source)
    except Exception:
        pass  # `kill` below engages again and reports what went wrong
    target: Venue | None = None
    try:
        active = profiles.active_profile(state_dir)
        if active is not None:
            try:
                cfg = _load(active.profile_id, execution).execution
            except ProfileError:
                cfg = execution  # a profile that does not verify must not stop a cancel
            candidate = venue if venue is not None else _alpaca_venue(cfg, switch)
            account = candidate.verify_account()
            if account.account_id != active.account_id:
                raise ControlError(
                    f"the keys are for account {account.account_id}, not {active.account_id}"
                )
            target = candidate
    except Exception as exc:
        _alert(sinks, now, "critical", "kill_switch", f"the venue cannot be used to cancel: {exc}")
    return kill(
        switch,
        reason,
        source,
        venue=target,
        alerts=sinks,
        wait_seconds=execution.poll_seconds,
        sleep=sleep,
        now=now,
    )


def reset_kill_switch(
    operator: str,
    confirm: str,
    *,
    venue: Venue | None = None,
    execution: ExecutionConfig | None = None,
    alerts: Alerter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
) -> KillReset:
    """Remove the latch, after a clean reconciliation. `confirm` is the active profile's
    id, typed by the operator (`NO_PROFILE` if none is active).

    Refused while a runner is running, while the venue lists any open order, and while
    the ledger and the venue differ by more than reconciliation accepts
    (`accept_broker_state` records such a difference first). Orders the journal left
    unresolved are looked up at the venue before the comparison, so a fill that
    happened while trading was stopped is on the books. The journal records who reset
    the switch and the latch that was cleared. Afterwards, start the runner again.
    """
    _named(operator)
    execution = execution if execution is not None else load_execution()
    state_dir = execution.state_dir
    sinks = alerts or _sinks(state_dir)
    with _exclusive(execution), _refusals():
        switch = KillSwitch(state_dir / KILL_FILE, now)
        if not switch.engaged():
            raise ControlError("the kill switch is not engaged")
        active = profiles.active_profile(state_dir)
        if active is None:
            cleared = switch.reset(operator, confirm, NO_PROFILE)
            message = f"kill switch reset by {operator}; no profile is active"
            _alert(sinks, now, "warn", "kill_switch", message)
            return KillReset(cleared=cleared, reconcile=None, message=message)
        if confirm != active.profile_id:
            raise ControlError("reset refused: the confirmation does not match the profile id")
        with _book(execution, active, now) as book:
            checked = _verified(venue, book, switch)
            om = _order_manager(book, checked, switch, sinks, sleep, now)
            report = _reconcile(book, checked, "reset_kill_switch")
            if report.severity == "stop":
                raise ControlError(
                    "reset refused, the kill switch is still engaged. Reconciliation is "
                    "not clean: " + "; ".join(report.reasons) + ". Cancel open orders by "
                    "engaging the kill switch again; record a difference in cash or "
                    "positions with accept_broker_state."
                )
            cleared = switch.state()
            book.journal.append(
                "control",
                {
                    "command": "reset_kill_switch",
                    "operator": operator,
                    "cleared": cleared.model_dump(mode="json"),
                },
            )
            switch.reset(operator, confirm, active.profile_id)
            _write_views(book, om)
        message = (
            f"kill switch reset by {operator} (it was set by {cleared.source or 'unknown'}: "
            f"{cleared.reason}). Start the runner again."
        )
        _alert(sinks, now, "warn", "kill_switch", message)
        return KillReset(cleared=cleared, reconcile=report, message=message)


# --------------------------------------------------------------------------- approvals


def approve_orders(
    approvals: Mapping[str, str],
    operator: str,
    *,
    execution: ExecutionConfig | None = None,
    now: Callable[[], dt.datetime] = _utc_now,
) -> Path:
    """Approve entries: order id -> the `content_hash` shown by
    `service.list_pending_approvals`. Returns the command file left for the runner.

    Refused here if an id is not waiting for approval or its hash is not the one last
    shown. That check is a courtesy, for an answer now. The runner makes the real one
    against its own state and the deadline, and alerts if it refuses.
    """
    _named(operator)
    execution = execution if execution is not None else load_execution()
    active = _active(execution)
    waiting = {r.order_id: r for r in service.list_pending_approvals(execution=execution, now=now)}
    for order_id, content_hash in approvals.items():
        request = waiting.get(order_id)
        if request is None:
            raise ControlError(f"{order_id} is not waiting for approval; nothing was approved")
        if request.content_hash != content_hash:
            raise ControlError(
                f"{order_id} is not the order that was shown (its terms changed); look at "
                "the pending approvals again. Nothing was approved"
            )
    try:
        command = ApproveOrders(approvals=dict(approvals), operator=operator, at=now())
    except ValueError as exc:
        raise ControlError(f"not a usable approval: {exc}") from exc
    return Inbox(execution.state_dir / active.profile_id / INBOX_DIR).put(command)


def reject_orders(
    order_ids: Sequence[str],
    operator: str,
    reason: str,
    *,
    execution: ExecutionConfig | None = None,
    now: Callable[[], dt.datetime] = _utc_now,
) -> Path:
    """Reject entries. A rejection counts until the entry is sent, approved or not.
    Returns the command file left for the runner."""
    _named(operator)
    execution = execution if execution is not None else load_execution()
    active = _active(execution)
    try:
        command = RejectOrders(
            order_ids=tuple(order_ids), operator=operator, reason=reason, at=now()
        )
    except ValueError as exc:
        raise ControlError(f"not a usable rejection: {exc}") from exc
    return Inbox(execution.state_dir / active.profile_id / INBOX_DIR).put(command)


# --------------------------------------------------------------------------- risk


def reset_risk_halt(
    operator: str,
    reason: str,
    *,
    execution: ExecutionConfig | None = None,
    now: Callable[[], dt.datetime] = _utc_now,
) -> RiskState:
    """Lift the drawdown halt, so entries are allowed again from the next decision.

    The peak is rebased to the last marked equity (otherwise the next close would halt
    again at once), which erases the old peak from the saved state. So the journal
    records it, with who reset the halt and why.
    """
    _named(operator)
    if not reason.strip():
        raise ControlError("say why the halt is being reset")
    execution = execution if execution is not None else load_execution()
    with _exclusive(execution), _refusals():
        active = _active(execution)
        with _book(execution, active, now) as book:
            state = profiles.load_risk_state(execution.state_dir, active.profile_id)
            manager = _risk_manager(book, state)
            manager.reset_halt()
            book.journal.append(
                "control",
                {
                    "command": "reset_risk_halt",
                    "operator": operator,
                    "reason": reason,
                    "halt_reason": state.halt_reason,
                    "old_peak": state.peak_equity,
                    "new_peak": manager.state.peak_equity,
                },
            )
            profiles.save_risk_state(manager.state, execution.state_dir, active.profile_id)
            return manager.state


# --------------------------------------------------------------------------- flatten


def flatten(
    operator: str,
    confirm: str,
    *,
    venue: Venue | None = None,
    execution: ExecutionConfig | None = None,
    alerts: Alerter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
) -> FlattenReport:
    """Sell every position of the active profile. `confirm` is its id, typed by the
    operator.

    The kill switch is engaged first, as soon as the confirmation matches (source
    `operator:<name>`, reason "flatten"; a latch that is already set stays as it is).
    It stays engaged afterwards, also when the flatten is then refused or fails:
    trading starts again only through `reset_kill_switch`.

    Open orders are canceled next. Each position is then sold: into the opening
    auction if this starts before `opg_cutoff`, with a market order otherwise. It keeps
    trying until every sell is final, or until the session deadline (started after the
    open: for as long as the open workflow gets, and never past the close). Sending
    nothing, it is refused on a day without a session or after the close.

    Where the venue holds fewer shares of a position than the ledger, only the venue's
    are sold; `differences` names each such case, and `accept_broker_state` settles
    the ledger afterwards. A kill engaged from elsewhere while this runs stops it
    (ControlError). The sells count against `max_orders_per_session`: on a session that
    has used the cap up, the flatten stops with positions still held.

    Returns the sells and what is still held. A position that could not be sold is a
    critical alert and is in `remaining`; calling this again tries again.
    """
    _named(operator)
    execution = execution if execution is not None else load_execution()
    state_dir = execution.state_dir
    sinks = alerts or _sinks(state_dir)
    with _exclusive(execution), _refusals():
        active = _active(execution)
        if confirm != active.profile_id:
            raise ControlError("flatten refused: the confirmation does not match the profile id")
        switch = KillSwitch(state_dir / KILL_FILE, now)
        # First, before the venue is asked anything: whatever happens next, nothing
        # buys behind the flatten. A latch that cannot be written refuses the flatten.
        switch.engage("flatten", f"operator:{operator}")
        with _book(execution, active, now) as book:
            checked = _verified(venue, book, switch)
            cfg = book.profile.execution
            venue_now = checked.now()
            today = venue_now.astimezone(MARKET_TZ).date()
            start = today - dt.timedelta(days=CALENDAR_LOOKBACK_DAYS)
            sessions = checked.sessions(start, today)
            current = next((s for s in sessions if s.date == today), None)
            earlier = [s.date for s in sessions if s.date < today]
            if current is None or venue_now >= current.close or not earlier:
                raise ControlError(
                    f"flatten refused, nothing was sent: the market is closed "
                    f"({venue_now.astimezone(MARKET_TZ):%Y-%m-%d %H:%M} New York). Run it "
                    "on a session day, before the close. The kill switch is engaged."
                )
            deadline = dt.datetime.combine(today, cfg.session_deadline, MARKET_TZ)
            window = deadline - dt.datetime.combine(today, MARKET_OPEN, MARKET_TZ)
            until = min(current.close, max(deadline, venue_now + window))
            om = _order_manager(book, checked, switch, sinks, sleep, now)
            book.journal.append(
                "control",
                {
                    "command": "flatten",
                    "operator": operator,
                    "session": today.isoformat(),
                    "until": until.isoformat(),
                },
            )
            try:
                result = om.flatten(today, earlier[-1], until)
            finally:
                _write_views(book, om)
            ids = set(result.order_ids)
            report = _reconcile(book, checked, "flatten")
            remaining = {s: p.quantity for s, p in book.ledger.account().positions.items()}
            severity: Severity = "critical" if remaining or report.severity == "stop" else "warn"
            held = ", ".join(f"{s} {q}" for s, q in sorted(remaining.items())) or "nothing"
            _alert(
                sinks,
                now,
                severity,
                "flatten",
                f"flatten by {operator}: {len(ids)} sell order(s); still held: {held}; "
                f"reconciliation {report.severity}"
                + (": " + "; ".join(report.reasons) if report.reasons else "")
                + ". The kill switch is engaged; reset it to trade again.",
            )
            return FlattenReport(
                session=today,
                orders=tuple(o for o in om.orders() if o.order_id in ids),
                remaining=remaining,
                differences=result.differences,
                reconcile=report,
            )


# --------------------------------------------------------------------------- the books


def accept_broker_state(
    operator: str,
    note: str,
    *,
    venue: Venue | None = None,
    execution: ExecutionConfig | None = None,
    alerts: Alerter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
) -> ReconcileReport:
    """Make the ledger agree with the venue, on the operator's word: what clears a
    reconciliation stop. Returns the reconciliation afterwards.

    Each position difference becomes one `manual` adjustment (shares added come in at
    the venue's average entry price; shares removed take their cost with them), and the
    cash difference becomes one more. Orders the journal left unresolved are looked up
    first, so a real fill is booked as a fill and not as an adjustment.

    Refused, changing nothing, while the venue lists an open order (cancel it first),
    for a short position, and for a position in a symbol outside the profile's
    universe, which the runner could not value (sell it at the broker, then accept).
    A known event (a dividend, a split) is better recorded under its own name with
    `record_adjustment`.
    """
    _named(operator)
    if not note.strip():
        raise ControlError("say why the broker's state is being accepted")
    execution = execution if execution is not None else load_execution()
    state_dir = execution.state_dir
    sinks = alerts or _sinks(state_dir)
    with _exclusive(execution), _refusals():
        active = _active(execution)
        switch = KillSwitch(state_dir / KILL_FILE, now)
        with _book(execution, active, now) as book:
            checked = _verified(venue, book, switch)
            om = _order_manager(book, checked, switch, sinks, sleep, now)
            entry_price = {p.symbol: p.avg_entry_price for p in checked.positions()}
            before = _reconcile(book, checked, "accept_broker_state")
            if before.unexpected_orders:
                raise ControlError(
                    "refused, nothing was changed: the venue lists open orders ("
                    + ", ".join(before.unexpected_orders)
                    + "). Cancel them first (engaging the kill switch does)."
                )
            if not before.position_diffs and abs(before.cash_diff) < _CENT:
                if before.severity == "stop":  # venue data that cannot be trusted
                    raise ControlError("refused: " + "; ".join(before.reasons))
                return before
            universe = set(book.profile.recipe.universe.symbols)
            for diff in before.position_diffs:
                if diff.venue_quantity < 0:
                    raise ControlError(
                        f"refused, nothing was changed: the venue is short {diff.symbol}; "
                        "close the short at the broker first"
                    )
                if diff.venue_quantity > 0 and diff.symbol not in universe:
                    raise ControlError(
                        f"refused, nothing was changed: {diff.symbol} is not in the "
                        "profile's universe; sell it at the broker first"
                    )
                if diff.venue_quantity > diff.ledger_quantity and diff.symbol not in entry_price:
                    raise ControlError(f"refused: the venue gave no entry price for {diff.symbol}")
            today = checked.now().astimezone(MARKET_TZ).date()
            book.journal.append(
                "control",
                {
                    "command": "accept_broker_state",
                    "operator": operator,
                    "note": note,
                    "report": before.model_dump(mode="json"),
                },
            )
            for diff in before.position_diffs:
                delta = diff.venue_quantity - diff.ledger_quantity
                book.ledger.apply_adjustment(
                    LedgerAdjustment(
                        adjustment_id=_accept_id(book, today, diff.symbol),
                        as_of=today,
                        kind="manual",
                        symbol=diff.symbol,
                        quantity_delta=delta,
                        cost_basis_delta=delta * entry_price[diff.symbol] if delta > 0 else 0.0,
                        operator=operator,
                        note=note,
                    )
                )
            if abs(before.cash_diff) >= _CENT:
                book.ledger.apply_adjustment(
                    LedgerAdjustment(
                        adjustment_id=_accept_id(book, today, "cash"),
                        as_of=today,
                        kind="manual",
                        cash_delta=before.cash_diff,
                        operator=operator,
                        note=note,
                    )
                )
            after = _reconcile(book, checked, "accept_broker_state")
            _write_views(book, om)
            _alert(
                sinks,
                now,
                "warn",
                "broker_state_accepted",
                f"{operator} accepted the broker's state ({note}): "
                + "; ".join(before.reasons)
                + f". Reconciliation is now {after.severity}.",
            )
            if after.severity == "stop":
                raise ControlError(
                    "the adjustments were recorded, but the ledger and the venue still "
                    "differ: " + "; ".join(after.reasons)
                )
            return after


def record_adjustment(
    adjustment: LedgerAdjustment,
    *,
    execution: ExecutionConfig | None = None,
    now: Callable[[], dt.datetime] = _utc_now,
) -> bool:
    """Record a correction that did not come from a fill: a dividend, a split, a fee.

    `adjustment_id` names the event (for example `2024-01-04-AAPL-dividend`), and the
    ledger applies each id once. Entering the same adjustment again returns False and
    changes nothing; the same id with different content is refused.
    """
    execution = execution if execution is not None else load_execution()
    with _exclusive(execution), _refusals():
        active = _active(execution)
        with _book(execution, active, now) as book:
            applied = book.ledger.apply_adjustment(adjustment)
            book.ledger.write_snapshot(book.folder / LEDGER_FILE)
            return applied


# --------------------------------------------------------------------------- profiles


def promote_run(
    run_dir: Path,
    name: str,
    *,
    plumbing: bool = False,
    execution: ExecutionConfig | None = None,
) -> FrozenProfile:
    """Freeze a validated backtest run into a profile (`profiles.promote_run`, with
    every refusal it makes)."""
    execution = execution if execution is not None else load_execution()
    with _refusals():
        return profiles.promote_run(run_dir, name, plumbing=plumbing, execution=execution)


def activate_profile(
    profile_id: str,
    approval: Literal["off", "entries"],
    operator: str,
    *,
    replace: bool = False,
    venue: Venue | None = None,
    execution: ExecutionConfig | None = None,
    now: Callable[[], dt.datetime] = _utc_now,
) -> ActiveProfile:
    """Bind the paper account to a profile (`profiles.activate_profile`), under the
    runner's lock: the active profile cannot change under a running day."""
    _named(operator)
    execution = execution if execution is not None else load_execution()
    with _exclusive(execution), _refusals():
        if venue is None:
            venue = _alpaca_venue(execution, KillSwitch(execution.state_dir / KILL_FILE, now))
        return profiles.activate_profile(
            profile_id,
            approval,
            operator=operator,
            venue=venue,
            replace=replace,
            execution=execution,
            now=now,
        )


# --------------------------------------------------------------------------- helpers


@dataclass
class _Book:
    """The active profile's journal and ledger, held by this process as their writer."""

    execution: ExecutionConfig
    active: ActiveProfile
    profile: FrozenProfile
    folder: Path
    journal: Journal
    ledger: Ledger


def _named(operator: str) -> None:
    if not operator.strip():
        raise ControlError("an operator command needs the operator's name")


def _sinks(state_dir: Path) -> Alerter:
    return Alerters(LogAlerter(), JsonlAlerter(state_dir / ALERTS_FILE))


def _alert(
    sinks: Alerter, now: Callable[[], dt.datetime], severity: Severity, kind: str, message: str
) -> None:
    try:
        alert = Alert(at=now(), severity=severity, kind=kind, message=message)
    except Exception:
        alert = Alert(at=_utc_now(), severity=severity, kind=kind, message=message)
    send_alert(sinks, alert)


@contextlib.contextmanager
def _exclusive(execution: ExecutionConfig) -> Iterator[None]:
    """The runner's lock, for the whole command. Refuses if a runner (or another
    command) holds it."""
    lock = _runner_lock(execution.state_dir / LOCK_FILE)
    try:
        lock.__enter__()
    except PaperError as exc:
        raise ControlError(
            "refused, nothing was changed: the runner (or another command) is running. "
            "Wait for it to finish; to stop a trading day, engage the kill switch."
        ) from exc
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


@contextlib.contextmanager
def _refusals() -> Iterator[None]:
    """Turn the errors the layers below raise into a ControlError that reads as an
    answer to the operator."""
    try:
        yield
    except ControlError:
        raise
    except SessionIncomplete as exc:
        raise ControlError(
            f"stopped part way: {exc}. The kill switch is engaged; see the alerts."
        ) from exc
    except VenueUnavailable as exc:
        raise ControlError(f"the venue did not answer ({exc}); try again") from exc
    except KillSwitchEngaged as exc:
        raise ControlError(f"refused: {exc}") from exc
    except (
        VenueDataError,
        KeysError,
        ConfigError,
        ProfileError,
        JournalError,
        LedgerError,
        OrderManagerError,
        RiskStateError,
        KillSwitchError,
    ) as exc:
        raise ControlError(str(exc)) from exc


def _active(execution: ExecutionConfig) -> ActiveProfile:
    try:
        active = profiles.active_profile(execution.state_dir)
    except ProfileError as exc:
        raise ControlError(str(exc)) from exc
    if active is None:
        raise ControlError("no profile is active")
    return active


def _load(profile_id: str, execution: ExecutionConfig) -> FrozenProfile:
    return profiles.load_profile(profile_id, profiles_dir=execution.profiles_dir)


@contextlib.contextmanager
def _book(
    execution: ExecutionConfig, active: ActiveProfile, now: Callable[[], dt.datetime]
) -> Iterator[_Book]:
    profile = _load(active.profile_id, execution)
    frozen = profile.execution
    if (frozen.state_dir, frozen.profiles_dir) != (execution.state_dir, execution.profiles_dir):
        raise ControlError(
            f"profile {profile.profile_id} was frozen with other state and profile folders "
            "than the ones in use now"
        )
    folder = execution.state_dir / active.profile_id
    journal = Journal(folder / JOURNAL_FILE, now)
    try:
        ledger = Ledger.open(journal, profile.recipe.backtest.starting_cash)
        yield _Book(execution, active, profile, folder, journal, ledger)
    finally:
        journal.close()


def _verified(venue: Venue | None, book: _Book, switch: KillSwitch) -> Venue:
    """The venue, once it has confirmed the keys are for the activated account."""
    if venue is None:
        venue = _alpaca_venue(book.profile.execution, switch)
    account = venue.verify_account()
    if account.account_id != book.active.account_id:
        raise ControlError(
            f"refused, nothing was sent: the keys are for account {account.account_id}, but "
            f"profile {book.active.profile_id} was activated on {book.active.account_id}"
        )
    return venue


def _risk_manager(book: _Book, state: RiskState) -> RiskManager:
    recipe, cfg = book.profile.recipe, book.profile.execution
    assert recipe.risk is not None  # FrozenProfile
    return RiskManager(recipe.risk, cfg.limits, cfg.entry_limit_pct, state)


def _order_manager(
    book: _Book,
    venue: Venue,
    switch: KillSwitch,
    sinks: Alerter,
    sleep: Callable[[float], None],
    now: Callable[[], dt.datetime],
) -> OrderManager:
    """The profile's order manager, recovered: every attempt the journal is unsure of
    has been looked up at the venue, and its fills are in the ledger."""
    state = profiles.load_risk_state(book.execution.state_dir, book.active.profile_id)
    om = OrderManager(
        venue,
        book.ledger,
        book.journal,
        switch,
        sinks,
        book.profile.execution,
        book.active.profile_id,
        _risk_manager(book, state),
        book.profile.recipe.construction.max_positions,
        book.active.approval,
        Inbox(book.folder / INBOX_DIR),
        sleep=sleep,
        now=now,
    )
    om.recover()
    return om


def _reconcile(book: _Book, venue: Venue, stage: str) -> ReconcileReport:
    """The ledger against the venue, expecting no open order at all, on the record."""
    report = reconcile_venue(
        book.ledger.account(),
        [],
        venue,
        book.active.cash_offset,
        book.profile.execution.cash_tolerance,
    )
    book.journal.append("reconcile", {"stage": stage, "report": report.model_dump(mode="json")})
    return report


def _write_views(book: _Book, om: OrderManager) -> None:
    """Refresh what `harness.service` reads, after a command changed it."""
    book.ledger.write_snapshot(book.folder / LEDGER_FILE)
    write_orders(book.folder, om.orders())


def _accept_id(book: _Book, day: dt.date, what: str) -> str:
    """Names one accepted difference. The journal position makes it unique: the same
    difference accepted on another occasion is another event."""
    return f"{day.isoformat()}-{what}-accepted-{book.journal.last_seq + 1}"
