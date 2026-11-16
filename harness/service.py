"""Read-only views of paper trading, for a UI backend and for the chat agent.

Every function here reads files the runner (or an operator command) wrote. None of them
calls the broker, takes a lock, writes a file, or changes anything, so they are safe to
call at any time, from any process, as often as wanted, and safe to hand to an agent
that must not be able to act. The functions that act are in `harness.control`, a
separate module on purpose: this one does not import it.

What each view is built from, and so how fresh it is:

- `get_status`: `active.json`, the kill latch, the saved risk state, the journal and
  `last_run.json`. Current as of the call.
- `get_positions`: the journal's fills and adjustments (replayed, not the `ledger.json`
  snapshot), so it is current as of the last fill the runner recorded. Marks are the
  closes the last decision used.
- `list_orders`, `list_pending_approvals`: snapshots the runner writes after the
  decision and when its run ends (only the runner holds the order manager). During the
  open they show the orders as staged, not their progress; fills show up in
  `get_positions` at once.
- `list_alerts`: `alerts.jsonl`.

A view never raises because a state file is missing or damaged: `get_status` lists what
it could not read in `problems`, and the others return what there is.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, TypeAdapter, ValidationError

from harness import profiles
from harness.alerts import Alert, read_alerts
from harness.config import ExecutionConfig, load_execution
from harness.journal import JournalError, JournalRecord, read_journal
from harness.killswitch import KillState, KillSwitch
from harness.models import FrozenModel
from harness.orders import ApprovalRequest, WorkingOrder
from harness.paper import (
    ALERTS_FILE,
    JOURNAL_FILE,
    LAST_RUN_FILE,
    ORDERS_FILE,
    PENDING_FILE,
    LastRun,
)
from harness.positions import Ledger, LedgerError, PnL, ReconcileReport
from harness.profiles import KILL_FILE, ActiveProfile, FrozenProfile, ProfileError
from harness.risk import RiskState

# Weakest first (1.5 of the slice 2 design). "none": nothing is stopping entries.
StopLevel = Literal["none", "entries_blocked", "entries_halted", "killed"]
# How far the runner got with the latest session it started.
DayStage = Literal["none", "started", "decided", "complete"]

_ORDERS: TypeAdapter[list[WorkingOrder]] = TypeAdapter(list[WorkingOrder])
_PENDING: TypeAdapter[list[ApprovalRequest]] = TypeAdapter(list[ApprovalRequest])


class TradingStatus(FrozenModel):
    """Where paper trading stands. Every part says when it is from."""

    mode: Literal["paper"]
    as_of: AwareDatetime  # when this was read
    active: ActiveProfile | None  # the profile the account is bound to, and since when
    kill: KillState
    stop_level: StopLevel
    stop_reason: str
    risk: RiskState | None  # as saved after the last decision (`marked_on` dates it)
    session: dt.date | None  # the latest session the runner started
    day_stage: DayStage
    last_reconcile: ReconcileReport | None  # its `at` dates it
    last_reconcile_stage: str  # "before" or "after" the open, or the command that ran it
    last_run: LastRun | None  # how the runner's latest run ended, and when
    problems: tuple[str, ...]  # state that could not be read; what is shown may be partial


class PositionView(FrozenModel):
    symbol: str
    quantity: int
    avg_price: float
    entry_date: dt.date
    entry_order_id: str
    cost_basis: float  # total cost of the shares held
    mark: float | None  # the close it is valued at; None if the last decision had none
    market_value: float | None
    unrealized: float | None


class PositionsView(FrozenModel):
    """The ledger's cash and positions, valued at the closes the last decision used."""

    profile_id: str | None
    as_of: dt.datetime | None  # the journal record this is current to
    journal_seq: int
    cash: float
    realized: float
    positions: tuple[PositionView, ...]
    marks_as_of: dt.date | None  # the session whose closes the marks are
    pnl: PnL | None  # None unless every position has a mark


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def trading_mode() -> Literal["paper"]:
    """Paper, always, in this slice. The mode is a property of the keys and the active
    profile, never something a caller passes in."""
    return "paper"


def get_status(
    *, execution: ExecutionConfig | None = None, now: Callable[[], dt.datetime] = _utc_now
) -> TradingStatus:
    """The active profile, the kill switch, what is stopping entries, how far today's
    run got, the last reconciliation and how the last run ended."""
    execution = execution if execution is not None else load_execution()
    state_dir = execution.state_dir
    problems: list[str] = []
    kill = KillSwitch(state_dir / KILL_FILE).state()
    active = _active(state_dir, problems)

    risk: RiskState | None = None
    records: list[JournalRecord] = []
    last_run: LastRun | None = None
    if active is not None:
        folder = state_dir / active.profile_id
        try:
            risk = profiles.load_risk_state(state_dir, active.profile_id)
        except ProfileError as exc:
            problems.append(str(exc))
        records = _records(folder, problems)
        try:
            last_run = LastRun.model_validate_json((folder / LAST_RUN_FILE).read_bytes())
        except FileNotFoundError:
            pass
        except (OSError, ValueError, ValidationError) as exc:
            problems.append(f"{folder / LAST_RUN_FILE} cannot be read: {type(exc).__name__}")

    session: dt.date | None = None
    stage: DayStage = "none"
    reconcile: ReconcileReport | None = None
    reconcile_stage = ""
    for record in records:
        try:
            if record.kind == "day_start":
                session, stage = _date(record, "session"), "started"
            elif record.kind == "decision" and _date(record, "session") == session:
                stage = "decided"
            elif record.kind == "day_complete" and _date(record, "session") == session:
                stage = "complete"
            elif record.kind == "reconcile":
                reconcile = ReconcileReport.model_validate(record.payload["report"])
                reconcile_stage = str(record.payload.get("stage", ""))
        except (KeyError, ValueError, ValidationError) as exc:
            problems.append(f"journal record {record.seq} ({record.kind}) does not read: {exc!r}")

    level: StopLevel = "none"
    reason = ""
    if kill.engaged:
        level, reason = "killed", f"{kill.source or 'unknown'}: {kill.reason}"
    elif risk is not None and risk.halted:
        level, reason = "entries_halted", risk.halt_reason
    elif (
        risk is not None
        and risk.entries_blocked_for is not None
        and risk.entries_blocked_for == risk.marked_on
    ):
        level = "entries_blocked"
        reason = f"daily loss cap on {risk.entries_blocked_for}; cleared at the next close"
    return TradingStatus(
        mode=trading_mode(),
        as_of=now().astimezone(dt.UTC),
        active=active,
        kill=kill,
        stop_level=level,
        stop_reason=reason,
        risk=risk,
        session=session,
        day_stage=stage,
        last_reconcile=reconcile,
        last_reconcile_stage=reconcile_stage,
        last_run=last_run,
        problems=tuple(problems),
    )


def get_positions(*, execution: ExecutionConfig | None = None) -> PositionsView:
    """Cash and positions from the journal, valued at the last decision's closes.
    Raises LedgerError or JournalError if the journal does not replay: a book that
    cannot be rebuilt is not shown as an empty one."""
    execution = execution if execution is not None else load_execution()
    problems: list[str] = []
    active = _active(execution.state_dir, problems)
    if problems:
        raise LedgerError(problems[0])
    empty = PositionsView(
        profile_id=active.profile_id if active else None,
        as_of=None,
        journal_seq=0,
        cash=0.0,
        realized=0.0,
        positions=(),
        marks_as_of=None,
        pnl=None,
    )
    if active is None:
        return empty
    records = read_journal(execution.state_dir / active.profile_id / JOURNAL_FILE)
    if not records:
        return empty  # activated, never run
    ledger = Ledger.from_records(records)

    marks: dict[str, float] = {}
    marks_as_of: dt.date | None = None
    decided_on: dt.date | None = None
    for record in records:
        if record.kind == "day_start":
            decided_on = _date(record, "decided_on")
        elif record.kind == "closes":
            closes = record.payload.get("closes")
            if isinstance(closes, dict):
                marks = {s: float(c) for s, c in closes.items() if isinstance(c, int | float)}
                marks_as_of = decided_on

    snapshot = ledger.snapshot()
    views = []
    for symbol, held in sorted(snapshot.positions.items()):
        mark = marks.get(symbol)
        cost = snapshot.cost_basis.get(symbol, 0.0)
        value = held.quantity * mark if mark is not None else None
        views.append(
            PositionView(
                symbol=symbol,
                quantity=held.quantity,
                avg_price=held.avg_price,
                entry_date=held.entry_date,
                entry_order_id=held.entry_order_id,
                cost_basis=cost,
                mark=mark,
                market_value=value,
                unrealized=value - cost if value is not None else None,
            )
        )
    pnl: PnL | None = None
    if marks_as_of is not None and all(v.mark is not None for v in views):
        pnl = ledger.pnl(marks, marks_as_of)
    return PositionsView(
        profile_id=active.profile_id,
        as_of=records[-1].at,
        journal_seq=records[-1].seq,
        cash=snapshot.cash,
        realized=snapshot.realized,
        positions=tuple(views),
        marks_as_of=marks_as_of,
        pnl=pnl,
    )


def list_orders(
    decided_on: dt.date | None = None, *, execution: ExecutionConfig | None = None
) -> list[WorkingOrder]:
    """Every order of the active profile as of the runner's last snapshot, oldest
    first; with `decided_on`, only the orders decided on that session's close (they
    trade at the next open)."""
    execution = execution if execution is not None else load_execution()
    orders = _snapshot(execution.state_dir, ORDERS_FILE, _ORDERS)
    if decided_on is not None:
        orders = [o for o in orders if o.order.created_on == decided_on]
    return orders


def list_pending_approvals(
    *, execution: ExecutionConfig | None = None, now: Callable[[], dt.datetime] = _utc_now
) -> list[ApprovalRequest]:
    """The entries waiting for the operator, as of the runner's last snapshot, without
    those whose deadline has passed. An approval must quote each one's `content_hash`."""
    execution = execution if execution is not None else load_execution()
    pending = _snapshot(execution.state_dir, PENDING_FILE, _PENDING)
    moment = now()
    return [request for request in pending if request.expires_at > moment]


def list_alerts(
    since: dt.datetime | None = None, *, execution: ExecutionConfig | None = None
) -> list[Alert]:
    """Alerts, oldest first, optionally only those at or after `since`."""
    execution = execution if execution is not None else load_execution()
    return read_alerts(execution.state_dir / ALERTS_FILE, since)


def list_profiles(*, execution: ExecutionConfig | None = None) -> list[FrozenProfile]:
    """Every frozen profile that loads and verifies, by id."""
    execution = execution if execution is not None else load_execution()
    return profiles.list_profiles(execution.profiles_dir)


# --------------------------------------------------------------------------- helpers


def _active(state_dir: Path, problems: list[str]) -> ActiveProfile | None:
    try:
        return profiles.active_profile(state_dir)
    except ProfileError as exc:
        problems.append(str(exc))
        return None


def _records(folder: Path, problems: list[str]) -> list[JournalRecord]:
    try:
        return read_journal(folder / JOURNAL_FILE)
    except (JournalError, OSError) as exc:
        problems.append(f"the journal cannot be read: {exc}")
        return []


def _date(record: JournalRecord, key: str) -> dt.date:
    return dt.date.fromisoformat(str(record.payload[key]))


def _snapshot[T](state_dir: Path, name: str, adapter: TypeAdapter[list[T]]) -> list[T]:
    """A list the runner wrote into the active profile's state folder, or nothing."""
    try:
        active = profiles.active_profile(state_dir)
        if active is None:
            return []
        return adapter.validate_python(
            json.loads((state_dir / active.profile_id / name).read_bytes())
        )
    except (ProfileError, OSError, ValueError, ValidationError):
        return []


__all__: Sequence[str] = (
    "PositionView",
    "PositionsView",
    "TradingStatus",
    "get_positions",
    "get_status",
    "list_alerts",
    "list_orders",
    "list_pending_approvals",
    "list_profiles",
    "trading_mode",
)
