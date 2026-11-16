"""The operator's command line for paper trading: `python -m harness.cli <command>`.

A thin layer over `harness.service` (the views) and `harness.control` (the commands).
It parses arguments, calls one function, and prints what came back. Every check, lock
and journal record is in the function it calls; nothing here decides anything.

Views (read files only, safe at any time, also while the runner is running):

    status | positions | orders | pending | alerts | profiles

Commands (each needs `--operator`, and leaves a record that carries the name):

    kill        engage the kill switch and cancel open orders; never waits for the runner
    reset-kill  remove the latch after a clean reconciliation (`--confirm <profile id>`)
    approve     approve entries: `ORDER_ID=HASH`, both as `pending` shows them
    reject      reject entries
    reset-halt  lift the drawdown halt
    flatten     sell every position (`--confirm <profile id>`); engages the kill switch
    accept      make the ledger agree with the broker
    adjust      record a dividend, a split, a fee

`promote`, `activate` and `run` are in `python -m harness.paper`.

Exit status: 0 the command did what it says; 1 it ran but something is left for the
operator (open orders not confirmed canceled, positions still held); 2 it was refused
or failed, and the message says what was and was not done (a usage error is also 2);
3 it crashed: the traceback is on stderr, and nothing is known about how far it got.

An approval quotes the hash on purpose. `pending` prints each entry with its terms and
its `ORDER_ID=HASH`; approving by id alone would approve whatever the order is by the
time the command runs, not what the operator looked at.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from harness import control, profiles, service
from harness.alerts import Alert, Alerter
from harness.broker.venue import Venue
from harness.config import ConfigError, ExecutionConfig, load_execution
from harness.control import ControlError
from harness.journal import JournalError
from harness.models import MARKET_TZ
from harness.orders import WorkingOrder
from harness.positions import LedgerAdjustment, LedgerError, ReconcileReport
from harness.profiles import ProfileError

_ADJUSTMENT_KINDS = ("split", "dividend", "fee", "delisting", "bust", "manual")


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _local(moment: dt.datetime) -> str:
    return f"{moment.astimezone(MARKET_TZ):%Y-%m-%d %H:%M:%S} New York"


# --------------------------------------------------------------------------- printing


def _print_order(order: WorkingOrder) -> None:
    leg = order.order.legs[0]
    print(
        f"  {order.order_id}  {leg.side} {order.order.symbol}  "
        f"{order.filled_quantity}/{order.quantity}  {order.phase}  {order.reason}".rstrip()
    )


def _print_reconcile(label: str, report: ReconcileReport) -> None:
    print(f"{label}: {report.severity}  {'; '.join(report.reasons)}".rstrip())


def _print_alert(alert: Alert) -> None:
    print(f"{_local(alert.at)}  [{alert.severity}] {alert.kind}: {alert.message}")


# --------------------------------------------------------------------------- views


def _status(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    status = service.get_status(**files)
    print(f"mode {status.mode}  as of {_local(status.as_of)}")
    if status.active is None:
        print("profile: none is active")
    else:
        a = status.active
        print(
            f"profile {a.profile_id} on account {a.account_id} (approval {a.approval}, "
            f"activated {_local(a.activated_at)} by {a.operator})"
        )
    kill = status.kill
    if kill.engaged:
        since = _local(kill.at) if kill.at is not None else "an unknown time"
        canceled = "confirmed" if kill.cancel_confirmed else "NOT confirmed"
        print(
            f"kill switch: ENGAGED since {since} by {kill.source or 'unknown'}: {kill.reason} "
            f"(open orders canceled: {canceled})"
        )
    else:
        print("kill switch: not engaged")
    print(f"entries: {status.stop_level}  {status.stop_reason}".rstrip())
    if status.risk is not None:
        r = status.risk
        print(
            f"risk: peak equity {r.peak_equity:.2f}, marked on {r.marked_on}"
            + (f", HALTED: {r.halt_reason}" if r.halted else "")
        )
    if status.session is not None:
        print(f"session {status.session}: {status.day_stage}")
    if status.last_reconcile is not None:
        _print_reconcile(
            f"last reconcile ({status.last_reconcile_stage}, {_local(status.last_reconcile.at)})",
            status.last_reconcile,
        )
    if status.last_run is not None:
        report = status.last_run.report
        print(
            f"last run: session {report.session}, outcome {report.outcome}, "
            f"finished {_local(status.last_run.finished_at)}"
        )
    for problem in status.problems:
        print(f"problem: {problem}")
    return 0


def _positions(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    view = service.get_positions(execution=files["execution"])
    if view.profile_id is None:
        print("no profile is active")
        return 0
    current = _local(view.as_of) if view.as_of is not None else "an empty journal"
    print(f"profile {view.profile_id}  current to {current}")
    print(f"cash {view.cash:.2f}  realized {view.realized:.2f}")
    for p in view.positions:
        marked = (
            f"mark {p.mark:.2f}  value {p.market_value:.2f}  unrealized {p.unrealized:.2f}"
            if p.mark is not None and p.market_value is not None and p.unrealized is not None
            else "no mark"
        )
        print(
            f"  {p.symbol}  {p.quantity} @ {p.avg_price:.2f}  since {p.entry_date}  "
            f"cost {p.cost_basis:.2f}  {marked}"
        )
    if not view.positions:
        print("  no positions")
    if view.pnl is not None:
        print(f"equity {view.pnl.equity:.2f} at the close of {view.pnl.as_of}")
    return 0


def _orders(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    orders = service.list_orders(args.decided_on, execution=files["execution"])
    for order in orders:
        _print_order(order)
    if not orders:
        print("no orders")
    return 0


def _pending(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    pending = service.list_pending_approvals(**files)
    for request in pending:
        print(
            f"{request.side} {request.symbol}  at most {request.quantity} shares at "
            f"{request.limit_price:.2f} or less (at most {request.est_notional:.2f})  "
            f"expires {_local(request.expires_at)}"
        )
        print(f"  {request.order_id}={request.content_hash}")
    if not pending:
        print("nothing is waiting for approval")
    return 0


def _alerts(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    since: dt.datetime | None = args.since
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=MARKET_TZ)
    alerts = service.list_alerts(since, execution=files["execution"])
    if args.last:
        alerts = alerts[-args.last :]
    for alert in alerts:
        _print_alert(alert)
    if not alerts:
        print("no alerts")
    return 0


def _profiles(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    execution: ExecutionConfig = files["execution"]
    found = service.list_profiles(execution=execution)
    for profile in found:
        kind = "plumbing" if profile.plumbing else "validated"
        print(
            f"{profile.profile_id}  {profile.recipe.strategy.strategy}  {kind}  "
            f"created {profile.created_at:%Y-%m-%d}"
        )
    if not found:
        print(f"no profiles in {execution.profiles_dir}")
    return 0


# --------------------------------------------------------------------------- commands


def _kill(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    state = control.engage_kill_switch(args.reason, args.operator, **live)
    print(f"kill switch engaged by {state.source or 'unknown'}: {state.reason}")
    if not state.latch_written:
        canceled = "were canceled" if state.cancel_confirmed else "are NOT confirmed canceled"
        print(
            "THE LATCH FILE WAS NOT WRITTEN, so nothing stops the next run. Open orders "
            f"{canceled}. Do not start the runner; see the alerts, make the latch "
            "writable, and run kill again."
        )
        return 1
    if state.cancel_confirmed:
        print("every open order at the venue is canceled. Positions were not touched.")
        return 0
    execution: ExecutionConfig = files["execution"]
    try:
        no_profile = profiles.active_profile(execution.state_dir) is None
    except Exception:
        # The kill is done; this read only picks the wording. Unreadable: the alert
        # says what happened.
        no_profile = False
    if no_profile:
        print(
            "open orders are NOT confirmed canceled: no profile is active, so no venue was "
            "asked to cancel."
        )
    else:
        print(
            "open orders are NOT confirmed canceled: see the alerts, check the broker, and "
            "run kill again."
        )
    return 1


def _reset_kill(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    reset = control.reset_kill_switch(args.operator, args.confirm, **live)
    if reset.reconcile is not None:
        _print_reconcile("reconcile", reset.reconcile)
    print(reset.message)
    return 0


def _approve(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    approvals: dict[str, str] = {}
    for item in args.approvals:
        order_id, sep, content_hash = item.partition("=")
        if not sep or not order_id or not content_hash:
            raise ControlError(
                f"{item!r} is not ORDER_ID=HASH (copy it from `pending`); nothing was approved"
            )
        if approvals.setdefault(order_id, content_hash) != content_hash:
            raise ControlError(f"{order_id} is given twice with two hashes; nothing was approved")
    path = control.approve_orders(approvals, args.operator, **files)
    print(f"approval of {', '.join(approvals)} left for the runner ({path.name})")
    print("The runner checks it against its own state; a refusal is an alert.")
    return 0


def _reject(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    path = control.reject_orders(args.order_ids, args.operator, args.reason, **files)
    print(f"rejection of {', '.join(args.order_ids)} left for the runner ({path.name})")
    print("The runner checks it against its own state; a refusal is an alert.")
    return 0


def _reset_halt(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    state = control.reset_risk_halt(args.operator, args.reason, **files)
    print(
        f"drawdown halt reset by {args.operator}; the peak is now {state.peak_equity:.2f}. "
        "Entries are allowed again from the next decision."
    )
    return 0


def _flatten(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    report = control.flatten(args.operator, args.confirm, **live)
    print(f"flatten on session {report.session}: {len(report.orders)} sell order(s)")
    for order in report.orders:
        _print_order(order)
    for difference in report.differences:
        print(f"difference: {difference}")
    _print_reconcile("reconcile", report.reconcile)
    print("The kill switch is engaged; reset-kill is the way back to trading.")
    if report.remaining:
        held = ", ".join(f"{s} {q}" for s, q in sorted(report.remaining.items()))
        print(f"STILL HELD: {held}. Run flatten again, or sell at the broker and accept.")
        return 1
    print("The ledger holds nothing.")
    return 1 if report.reconcile.severity == "stop" else 0


def _accept(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    report = control.accept_broker_state(args.operator, args.note, **live)
    _print_reconcile("reconcile after", report)
    return 0


def _adjust(args: argparse.Namespace, files: dict[str, Any], live: dict[str, Any]) -> int:
    try:
        adjustment = LedgerAdjustment(
            adjustment_id=args.id,
            as_of=args.as_of,
            kind=args.kind,
            symbol=args.symbol,
            cash_delta=args.cash,
            quantity_delta=args.quantity,
            cost_basis_delta=args.cost_basis,
            operator=args.operator,
            note=args.note,
        )
    except ValidationError as exc:
        problems = "; ".join(str(e["msg"]) for e in exc.errors())
        raise ControlError(f"not a usable adjustment, nothing was recorded: {problems}") from exc
    if control.record_adjustment(adjustment, **files):
        print(f"adjustment {adjustment.adjustment_id} recorded")
    else:
        print(f"adjustment {adjustment.adjustment_id} was already recorded; nothing changed")
    return 0


_Handler = Callable[[argparse.Namespace, dict[str, Any], dict[str, Any]], int]

_HANDLERS: dict[str, _Handler] = {
    "status": _status,
    "positions": _positions,
    "orders": _orders,
    "pending": _pending,
    "alerts": _alerts,
    "profiles": _profiles,
    "kill": _kill,
    "reset-kill": _reset_kill,
    "approve": _approve,
    "reject": _reject,
    "reset-halt": _reset_halt,
    "flatten": _flatten,
    "accept": _accept,
    "adjust": _adjust,
}


# --------------------------------------------------------------------------- parser


def _count(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli",
        description="Paper trading: the operator's views and commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def acting(name: str, summary: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=summary)
        sub.add_argument("--operator", required=True, help="who is doing this")
        return sub

    commands.add_parser("status", help="the profile, the kill switch, how far today got")
    commands.add_parser("positions", help="what the ledger holds")
    orders = commands.add_parser("orders", help="orders as of the runner's last snapshot")
    orders.add_argument(
        "--decided-on", type=dt.date.fromisoformat, default=None, metavar="YYYY-MM-DD",
        help="only the orders decided on this session's close",
    )  # fmt: skip
    commands.add_parser("pending", help="entries waiting for approval, with their hashes")
    alerts = commands.add_parser("alerts", help="alerts, oldest first")
    alerts.add_argument(
        "--since", type=dt.datetime.fromisoformat, default=None, metavar="ISO",
        help="only alerts at or after this time (New York time if no offset is given)",
    )  # fmt: skip
    alerts.add_argument("--last", type=_count, default=20, help="how many (0: all; default 20)")
    commands.add_parser("profiles", help="every frozen profile that verifies")

    kill = acting("kill", "engage the kill switch and cancel every open order")
    kill.add_argument("--reason", required=True)
    reset_kill = acting("reset-kill", "remove the kill switch after a clean reconciliation")
    reset_kill.add_argument(
        "--confirm", required=True, metavar="PROFILE_ID",
        help=f"the active profile's id, typed out ({control.NO_PROFILE!r} if none is active)",
    )  # fmt: skip
    approve = acting("approve", "approve entries that are waiting")
    approve.add_argument(
        "approvals", nargs="+", metavar="ORDER_ID=HASH", help="as `pending` prints them"
    )
    reject = acting("reject", "reject entries")
    reject.add_argument("order_ids", nargs="+", metavar="ORDER_ID")
    reject.add_argument("--reason", required=True)
    reset_halt = acting("reset-halt", "lift the drawdown halt")
    reset_halt.add_argument("--reason", required=True)
    flatten = acting("flatten", "sell every position; engages the kill switch first")
    flatten.add_argument(
        "--confirm", required=True, metavar="PROFILE_ID",
        help="the active profile's id, typed out",
    )  # fmt: skip
    accept = acting("accept", "make the ledger agree with the broker's cash and positions")
    accept.add_argument("--note", required=True, help="why the difference exists")
    adjust = acting("adjust", "record a dividend, a split, a fee (once per --id)")
    adjust.add_argument(
        "--id", required=True, help="names the event, for example 2024-01-04-AAPL-dividend"
    )
    adjust.add_argument("--as-of", required=True, type=dt.date.fromisoformat, metavar="YYYY-MM-DD")
    adjust.add_argument("--kind", required=True, choices=_ADJUSTMENT_KINDS)
    adjust.add_argument("--symbol", default=None)
    adjust.add_argument("--cash", type=float, default=0.0, help="cash received (+) or paid (-)")
    adjust.add_argument("--quantity", type=int, default=0, help="shares added (+) or removed (-)")
    adjust.add_argument(
        "--cost-basis", type=float, default=0.0, help="cost the added shares bring with them"
    )
    adjust.add_argument("--note", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    venue: Venue | None = None,
    execution: ExecutionConfig | None = None,
    alerts: Alerter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
) -> int:
    """Run one command. The keyword arguments are the same seams the functions in
    `harness.control` have, for tests; the command line never sets them."""
    args = _parser().parse_args(argv)
    try:
        execution = execution if execution is not None else load_execution()
        files: dict[str, Any] = {"execution": execution, "now": now}
        live: dict[str, Any] = {**files, "venue": venue, "alerts": alerts, "sleep": sleep}
        return _HANDLERS[args.command](args, files, live)
    except (ControlError, ConfigError, ProfileError, LedgerError, JournalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:
        # Not 1: Python's own exit status for an uncaught exception is 1, which here
        # means "it ran, and something is left for the operator".
        traceback.print_exc()
        print("error: the command crashed; how far it got is not known", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
