"""The order manager: every decision about how an approved order reaches the venue.

This module is the core: staging, order ids, the submission protocol, turning venue
states into ledger fills, cancel with confirmation, and recovery after a restart. The
timed open workflow (when exits and entries are sent, deadlines) is built on these
operations: `on_session_open`, at the bottom of the class.

How a duplicate order is made impossible:

- **Ids come from the decision, not from time or a counter.** An order's id is
  `<profile tag>-<signal date>-<SYMBOL>-<e|x>`; a venue attempt appends `-a1`, `-a2`.
  The same decision always yields the same id. (An operator's flatten order ends in
  `f1`, `f2`, ... instead: the number is part of the staged order, so the id is still a
  function of what the journal holds.)
- **Write-ahead journal.** The intent to send an attempt is journaled and fsynced before
  the network call, so a restart knows what may be at the venue.
- **Lookup before resend.** After a timeout the attempt is looked up by its client id.
  Found: adopted. Not found: placed again with the same client id, which the venue
  treats as the same order.
- **"Not found" right after a timeout proves nothing.** The venue may have accepted the
  order and not show it yet. So once a send has timed out, a rejection of the resend
  (a venue may reject a repeated client id) is looked up before it is believed, and an
  attempt that still cannot be found stays `unknown`. It is written off only by a
  lookup after a restart. One still not found when it is canceled, after
  `cancel_confirm_seconds` of looking, kills: the session does not complete, and the
  lookup in `recover()` after `reset-kill` books a late fill as a fill.
- **One attempt at a time.** A new attempt is created only after the previous one is
  final at the venue, for the quantity still unfilled.

Fills reach the ledger (and its journal) before anything else records them, and the
ledger applies each fill once, so a crash at any point replays to the same book.

Venue errors never raise out of this module: they become order updates and alerts. Two
things do raise. `KillSwitchEngaged` when the latch is set (nothing is sent), and
`SessionIncomplete` when the manager can no longer vouch for its own state. Before
raising `SessionIncomplete` it engages the kill switch itself.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn

from pydantic import JsonValue, ValidationError

from harness.alerts import Alert, Alerter, Severity, send_alert
from harness.broker.base import Broker, LookAheadError
from harness.broker.venue import (
    KillSwitchEngaged,
    TimeInForce,
    Venue,
    VenueDataError,
    VenueOrder,
    VenueOrderState,
    VenueOrderType,
    VenueRejected,
    VenueStatus,
)
from harness.config import MARKET_OPEN, ExecutionConfig
from harness.inbox import ApproveOrders, Inbox, RejectOrders
from harness.journal import Journal
from harness.killswitch import KillSwitch, kill
from harness.models import (
    MARKET_TZ,
    AccountState,
    Fill,
    ForcedReason,
    FrozenModel,
    Leg,
    Order,
    OrderUpdate,
    OrderUpdateReason,
    Side,
    Stock,
)
from harness.positions import Ledger, LedgerError
from harness.risk import RiskDecision, RiskManager

# "staged": journaled, nothing sent yet. "idle": an attempt ended with shares still
# unfilled and nothing is at the venue; the caller sends again or closes the order.
# "submitting" and "unknown": an attempt may or may not exist at the venue.
OrderPhase = Literal[
    "staged", "submitting", "working", "unknown", "idle",
    "filled", "canceled", "rejected", "expired",
]  # fmt: skip
ClosedStatus = Literal["canceled", "rejected", "expired"]
TERMINAL_PHASES: frozenset[OrderPhase] = frozenset({"filled", "canceled", "rejected", "expired"})

_PROFILE_TAG = re.compile(r"^[A-Za-z0-9]{8}$")
# The `reason` of an order staged by the operator's flatten command: the n-th such sell
# of one symbol for one session. No strategy order has it.
_FLATTEN_REASON = re.compile(r"^operator:flatten:([1-9][0-9]*)$")
_SOURCE = "auto:orders"


class OrderManagerError(RuntimeError):
    """The order manager was used incorrectly. Nothing was sent."""


class StagedOrderChanged(OrderManagerError):
    """An order was submitted again with different content than the one already staged
    under its id: what the decision was made from changed between two attempts."""


class SessionIncomplete(RuntimeError):
    """The manager cannot vouch for its orders (a cancel that will not confirm, venue
    data that contradicts the journal). The kill switch is already engaged."""


ApprovalMode = Literal["off", "entries"]
# "none": no approval needed (an exit, or approval is off). "pending": waiting for the
# operator. "approved" and "denied" are the operator's answers.
ApprovalState = Literal["none", "pending", "approved", "denied"]


class ApprovalRefused(Exception):
    """An approval or rejection does not fit the order it names. Nothing changed."""


class ApprovalRequest(FrozenModel):
    """An entry waiting for the operator, as it should be shown to them."""

    order_id: str
    content_hash: str  # of what is shown here; an approval must quote it
    symbol: str
    side: Side
    quantity: int  # at most this many shares
    limit_price: float  # at most this price per share
    est_notional: float  # quantity * limit_price: the most the order can cost
    expires_at: dt.datetime  # unapproved by then, the entry is canceled


def approval_hash(order_id: str, symbol: str, side: Side, quantity: int, limit: float) -> str:
    """Hash of the terms an operator approves. An approval quotes it, so it cannot be
    applied to an order whose terms are different from the ones that were shown."""
    terms = [order_id, symbol, side, quantity, f"{limit:.2f}"]
    return hashlib.sha256(json.dumps(terms).encode()).hexdigest()


class FlattenResult(FrozenModel):
    """What `OrderManager.flatten` did."""

    order_ids: tuple[str, ...]  # the sells, one per position sold
    # Where the ledger and the venue disagreed about a position, and what was sold.
    differences: tuple[str, ...]


class WorkingOrder(FrozenModel):
    """Order-manager state for one order. Rebuilt from the journal after a restart."""

    order_id: str  # deterministic; what the engine sees
    order: Order  # as submitted
    quantity: int  # what is being worked: less than the order's if it was cut at release
    reference_price: float  # the close the order was decided on
    phase: OrderPhase
    venue_orders: tuple[VenueOrder, ...]  # attempts, oldest first; at most one not final
    filled_quantity: int
    reason: OrderUpdateReason = ""
    approval: ApprovalState = "none"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def profile_tag(profile_id: str) -> str:
    """The last 8 characters of a profile id (the start of its content hash)."""
    tag = profile_id[-8:]
    if not _PROFILE_TAG.match(tag):
        raise OrderManagerError(
            f"profile id {profile_id!r} must end in 8 letters or digits (its content hash)"
        )
    return tag


def id_symbol(symbol: str) -> str:
    """`symbol` as it appears in an order id: letters and digits only, so the id does
    not depend on which characters a venue allows in a client order id.

    A plain ticker is itself. Anything else (`BRK.B`) drops the other characters and
    adds `x` and four hex digits of the symbol's hash, so `BRK.B` and `BRK-B` stay
    different. Tickers are upper case, so the `x` cannot collide with a real one.
    """
    if symbol.isalnum():
        return symbol
    letters = "".join(c for c in symbol if c.isalnum())
    return f"{letters}x{hashlib.sha256(symbol.encode()).hexdigest()[:4]}"


def flatten_number(order: Order) -> int | None:
    """n if `order` is the n-th sell the operator's flatten command staged for its
    symbol and session; None for an order that came from a decision."""
    if order.action != "exit":
        return None
    match = _FLATTEN_REASON.match(order.reason)
    return int(match.group(1)) if match else None


def order_id_for(profile_id: str, order: Order) -> str:
    """The id of `order` under `profile_id`: a function of the decision only."""
    tag = profile_tag(profile_id)
    number = flatten_number(order)
    if number is not None:
        kind = f"f{number}"  # never the id of the strategy's own exit of that session
    else:
        kind = "e" if order.action == "enter" else "x"
    return f"{tag}-{order.created_on:%Y%m%d}-{id_symbol(order.symbol)}-{kind}"


@dataclass
class _Attempt:
    number: int
    venue_order: VenueOrder
    filled_before: int  # the order's filled total when this attempt was created
    # "sending": journaled, may or may not be at the venue. "open": seen at the venue,
    # not final. "final": the venue will not change it again. "void": never existed.
    state: Literal["sending", "open", "final", "void"] = "sending"
    venue_status: VenueStatus | None = None
    filled: int = 0
    value: float = 0.0  # dollars of the shares filled by this attempt
    # A send of this attempt ended without an answer in this process, so the venue may
    # hold it even while a lookup says it does not.
    maybe_sent: bool = False


@dataclass
class _Tracked:
    order_id: str
    order: Order
    reference_price: float
    phase: OrderPhase = "staged"
    reason: OrderUpdateReason = ""
    attempts: list[_Attempt] = field(default_factory=list)
    quantity: int = 0  # shares being worked; the order's own quantity unless cut at release
    fill_reason: OrderUpdateReason = ""  # reported with the final fill of a cut order
    approval: ApprovalState = "none"
    approved_hash: str = ""

    def __post_init__(self) -> None:
        self.quantity = self.quantity or self.order.quantity

    @property
    def side(self) -> Side:
        return self.order.legs[0].side

    @property
    def filled(self) -> int:
        return sum(a.filled for a in self.attempts)

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled

    @property
    def last(self) -> _Attempt | None:
        return self.attempts[-1] if self.attempts else None


class OrderManager(Broker):
    """The engine-facing Broker for paper trading. Stages orders, sends them without
    duplicates, and keeps the ledger in step with the venue. Call `recover()` once after
    construction, before anything else."""

    def __init__(
        self,
        venue: Venue,
        ledger: Ledger,
        journal: Journal,
        kill_switch: KillSwitch,
        alerts: Alerter | None,
        cfg: ExecutionConfig,
        profile_id: str,
        risk: RiskManager,
        max_positions: int,
        approval: ApprovalMode,
        inbox: Inbox | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], dt.datetime] = _utc_now,
    ) -> None:
        profile_tag(profile_id)  # validates the profile id
        if approval == "entries" and inbox is None:
            raise OrderManagerError("approval of entries needs an inbox to receive it")
        self._approval = approval
        self._inbox = inbox
        self._venue = venue
        self._ledger = ledger
        self._journal = journal
        self._kill = kill_switch
        self._alerts = alerts
        self._cfg = cfg
        self._profile_id = profile_id
        self._risk = risk
        self._max_positions = max_positions
        self._clock_offset = dt.timedelta(0)  # venue clock minus local clock
        self._sleep = sleep
        self._now = now
        self._orders: dict[str, _Tracked] = {}  # in staging order
        self._closes: dict[str, float] = {}
        self._updates: list[OrderUpdate] = []
        self._sent: dict[dt.date, int] = {}  # session -> venue attempts journaled
        self._blocked: set[dt.date] = set()  # signal dates whose entries are blocked
        self._recovered = False

    # ------------------------------------------------------------------ staging

    def set_closes(self, closes: Mapping[str, float]) -> None:
        """The closes of the signal session, by symbol. Each order staged afterwards
        records its symbol's close as the reference price of its fills, and held
        positions are valued at these closes when an entry is released.

        They must cover every symbol the ledger holds, or no entry could be judged at
        release. Journaled, so a restarted manager has them without being told again.
        """
        self._require_recovered()
        for symbol, close in closes.items():
            if not (math.isfinite(close) and close > 0):
                raise OrderManagerError(f"close for {symbol} must be a positive number")
        missing = sorted(s for s in self._ledger.account().positions if s not in closes)
        if missing:
            raise OrderManagerError(f"no close for held symbols: {', '.join(missing)}")
        if dict(closes) == self._closes:
            return  # the same closes again (a rerun after a restart): nothing new to record
        self._journal.append("closes", {"closes": dict(closes)})
        self._closes = dict(closes)

    def submit(self, orders: Sequence[Order]) -> list[str]:
        """Journal each order as staged and return its id. Sends nothing. Submitting the
        same order again (same signal date, symbol and action) returns the existing id."""
        self._require_recovered()
        staged: list[tuple[str, Order, float | None]] = []
        for order in orders:
            _check_supported(order)
            order_id = order_id_for(self._profile_id, order)
            existing = self._orders.get(order_id)
            if existing is not None:
                if existing.order != order:
                    raise StagedOrderChanged(
                        f"order {order_id} was already staged with different content"
                    )
                staged.append((order_id, order, None))
                continue
            if any(order_id == other for other, _, _ in staged):
                raise OrderManagerError(f"order {order_id} appears twice in one submit")
            close = self._closes.get(order.symbol)
            if close is None:
                raise OrderManagerError(f"no close set for {order.symbol}; call set_closes first")
            staged.append((order_id, order, close))
        for order_id, order, close in staged:  # nothing is journaled unless all are valid
            if close is None:
                continue
            self._journal.append(
                "order_staged",
                {
                    "order_id": order_id,
                    "order": order.model_dump(mode="json"),
                    "reference_price": close,
                    "approval": self._needs_approval(order),
                },
            )
            self._orders[order_id] = _Tracked(
                order_id, order, close, approval=self._needs_approval(order)
            )
        return [order_id for order_id, _, _ in staged]

    # ------------------------------------------------------------------ sending

    def send(
        self,
        order_id: str,
        session: dt.date,
        *,
        type: VenueOrderType,
        time_in_force: TimeInForce,
        limit_price: float | None = None,
    ) -> OrderPhase:
        """Send the unfilled part of `order_id` to the venue as one attempt, in
        `session`, and return the order's phase afterwards.

        Allowed when nothing of the order is at the venue (`staged`, `idle`) or when its
        last attempt's outcome is `unknown`, in which case the attempt is looked up
        first and adopted if it exists. Raises KillSwitchEngaged, sending nothing, if
        the latch is set (a sell inside `KillSwitch.flattening()` is the one exception).
        """
        self._require_recovered()
        tracked = self._get(order_id)
        if tracked.phase not in ("staged", "idle", "unknown"):
            raise OrderManagerError(f"order {order_id} is {tracked.phase}; it cannot be sent")
        if session <= tracked.order.created_on:
            raise OrderManagerError(
                f"look-ahead: order {order_id} decided on {tracked.order.created_on} "
                f"cannot be sent in session {session}"
            )
        is_entry = tracked.order.action == "enter"
        if is_entry and tracked.order.created_on in self._blocked:
            raise OrderManagerError(f"entries are blocked; order {order_id} cannot be sent")
        self._kill.check(selling=tracked.side == "sell")

        if tracked.side == "sell":
            held = self._ledger.account().positions.get(tracked.order.symbol)
            if tracked.remaining > (held.quantity if held else 0):
                self._alert(
                    "critical",
                    "order_rejected",
                    f"exit {order_id} for {tracked.remaining} shares exceeds the "
                    f"{held.quantity if held else 0} the ledger holds; not sent",
                )
                self._close(tracked, "rejected", "not_held")
                return tracked.phase

        last = tracked.last
        reuse = last is not None and last.state in ("sending", "void")
        number = last.number if last is not None and reuse else len(tracked.attempts) + 1
        try:
            venue_order = VenueOrder(
                client_order_id=f"{order_id}-a{number}",
                symbol=tracked.order.symbol,
                side=tracked.side,
                quantity=tracked.remaining,
                type=type,
                limit_price=limit_price,
                time_in_force=time_in_force,
            )
        except ValidationError as exc:
            raise OrderManagerError(f"order {order_id} cannot be sent: {exc}") from exc

        lookup_first = tracked.phase == "unknown"
        if lookup_first and last is not None and self._adopt_if_found(tracked, last):
            return tracked.phase

        if self._sent.get(session, 0) >= self._cfg.limits.max_orders_per_session:
            self._violation(
                f"order cap reached: session {session} already has {self._sent[session]} "
                f"venue orders (max_orders_per_session); refusing to send {order_id}"
            )
        intent: dict[str, JsonValue] = {
            "order_id": order_id,
            "attempt": number,
            "session": session.isoformat(),
            "venue_order": venue_order.model_dump(mode="json"),
            "filled_before": tracked.filled,
        }
        maybe_sent = last is not None and reuse and last.maybe_sent
        if maybe_sent and last is not None and last.venue_order != venue_order:
            # The earlier send under this client id may have reached the venue. If it
            # did, the venue holds that order, not the one being sent now.
            intent["may_be_at_venue_as"] = last.venue_order.model_dump(mode="json")
        self._journal.append("order_intent", intent)
        self._sent[session] = self._sent.get(session, 0) + 1
        attempt = _Attempt(number, venue_order, tracked.filled, maybe_sent=maybe_sent)
        if reuse:
            tracked.attempts[-1] = attempt
        else:
            tracked.attempts.append(attempt)
        tracked.phase = "submitting"
        self._place(tracked, attempt)
        return tracked.phase

    def _place(self, tracked: _Tracked, attempt: _Attempt) -> None:
        """Steps 2 and 3 of the protocol: place, and on an unknown outcome look the
        attempt up before placing it again under the same client id."""
        client_id = attempt.venue_order.client_order_id
        lookup_first = False
        for tried in range(self._cfg.max_submit_attempts):
            if tried:
                self._sleep(self._cfg.poll_seconds * tried)
            self._kill.check(selling=tracked.side == "sell")
            if lookup_first:
                try:
                    found = self._venue.get_order(client_id)
                except VenueDataError as exc:
                    self._violation(f"the venue's data cannot be trusted: {exc}")
                except Exception:
                    continue  # cannot look: do not resend blind
                if found is not None:
                    self._observe(tracked, attempt, found)
                    return
            try:
                state = self._venue.place(attempt.venue_order)
            except KillSwitchEngaged:
                raise
            except VenueRejected as exc:
                if attempt.maybe_sent:
                    # An earlier send of this client id got no answer, so this may be
                    # the venue refusing a repeated id for an order it already holds.
                    self._adopt_if_found(tracked, attempt)
                    if attempt.state == "sending":  # not found, or the lookup failed
                        self._mark_unknown(tracked, attempt, f"the resend was rejected ({exc})")
                    return
                self._void(tracked, attempt, f"rejected by the venue: {exc}")
                self._alert(
                    "critical" if tracked.side == "sell" else "warn",
                    "order_rejected",
                    f"the venue rejected {client_id}: {exc}",
                )
                self._close(tracked, "rejected", "broker_rejected")
                return
            except VenueDataError as exc:
                self._violation(f"the venue's data cannot be trusted: {exc}")
            except Exception:
                tracked.phase = "unknown"
                attempt.maybe_sent = True
                lookup_first = True
                continue
            self._observe(tracked, attempt, state)
            return
        self._mark_unknown(tracked, attempt, f"{self._cfg.max_submit_attempts} tries got no answer")

    def _mark_unknown(self, tracked: _Tracked, attempt: _Attempt, why: str) -> None:
        client_id = attempt.venue_order.client_order_id
        tracked.phase = "unknown"
        self._journal.append(
            "order_unknown",
            {"order_id": tracked.order_id, "client_order_id": client_id, "why": why},
        )
        message = f"outcome of {client_id} is unknown: {why}. It may or may not be at the venue."
        if tracked.order.action == "enter":
            self._block_entries(tracked.order.created_on)
            message += " Entries are blocked for the session."
        self._alert("critical", "order_unknown", message)

    def _adopt_if_found(self, tracked: _Tracked, attempt: _Attempt) -> bool:
        """Look up an attempt whose outcome is unknown. True if the venue has it (it is
        adopted) or could not be asked (the order stays unknown)."""
        try:
            found = self._venue.get_order(attempt.venue_order.client_order_id)
        except VenueDataError as exc:
            self._violation(f"the venue's data cannot be trusted: {exc}")
        except Exception:
            return True
        if found is None:
            return False
        self._observe(tracked, attempt, found)
        return True

    # ------------------------------------------------------------------ venue states

    def refresh(self) -> None:
        """Read the venue's state of every order that is, or may be, at the venue, and
        record new fills. A venue that does not answer changes nothing.

        One request lists every order still open at the venue. Only an order that is
        not in that list (it just ended, or it is not visible yet) is asked about by
        itself, so a quiet poll costs one request however many orders are working.
        """
        self._require_recovered()
        watched = [
            (t, t.last)
            for t in self._orders.values()
            if t.last is not None and t.phase in ("working", "unknown", "submitting")
        ]
        if not watched:
            return
        try:
            listed = {s.client_order_id: s for s in self._venue.open_orders()}
        except VenueDataError as exc:
            self._violation(f"the venue's data cannot be trusted: {exc}")
        except Exception:
            listed = {}  # fall back to asking about each order
        for tracked, attempt in watched:
            state = listed.get(attempt.venue_order.client_order_id)
            if state is None:
                try:
                    state = self._venue.get_order(attempt.venue_order.client_order_id)
                except VenueDataError as exc:
                    self._violation(f"the venue's data cannot be trusted: {exc}")
                except Exception:
                    continue
            if state is not None:
                self._observe(tracked, attempt, state)
            elif attempt.state == "open":
                self._violation(
                    f"the venue no longer knows {attempt.venue_order.client_order_id}, "
                    f"which it reported earlier"
                )

    def _observe(self, tracked: _Tracked, attempt: _Attempt, state: VenueOrderState) -> None:
        """Apply one venue state: new fills to the ledger first, then the journal, then
        the order's phase."""
        sent = attempt.venue_order
        if state.client_order_id != sent.client_order_id or state.quantity != sent.quantity:
            self._violation(
                f"the venue reports {state.client_order_id} for {state.quantity} shares, "
                f"but {sent.client_order_id} was sent for {sent.quantity}"
            )
        if state.filled_quantity < attempt.filled:
            self._violation(
                f"{sent.client_order_id}: filled quantity went from {attempt.filled} "
                f"down to {state.filled_quantity}"
            )
        fills: tuple[Fill, ...] = ()
        delta = state.filled_quantity - attempt.filled
        if delta:
            assert state.filled_avg_price is not None  # VenueOrderState
            value = state.filled_quantity * state.filled_avg_price
            price = (value - attempt.value) / delta
            if not (math.isfinite(price) and price > 0):
                self._alert(
                    "warn",
                    "fill_price",
                    f"{sent.client_order_id}: cannot derive the price of the last {delta} "
                    f"shares from the venue's average; using the average",
                )
                price = state.filled_avg_price
            fill_date = state.updated_at.astimezone(MARKET_TZ).date()
            try:
                fill = Fill(
                    order_id=tracked.order_id,
                    kind="venue",
                    signal_date=tracked.order.created_on,
                    fill_date=fill_date,
                    price_date=fill_date,
                    instrument=tracked.order.legs[0].instrument,
                    side=tracked.side,
                    quantity=delta,
                    reference_price=tracked.reference_price,
                    price=price,
                    slippage_cost=0.0,
                    commission=0.0,
                    filled_at=state.updated_at,
                )
                applied = self._ledger.apply_fill(
                    fill, attempt.filled_before + state.filled_quantity
                )
            except (ValidationError, LedgerError) as exc:
                self._violation(f"a fill of {sent.client_order_id} does not fit the book: {exc}")
            attempt.filled = state.filled_quantity
            attempt.value += price * delta
            fills = (fill,) if applied else ()

        if state.status != attempt.venue_status or delta:
            self._journal.append(
                "venue_state",
                {
                    "order_id": tracked.order_id,
                    "attempt": attempt.number,
                    "state": state.model_dump(mode="json"),
                },
            )
        attempt.venue_status = state.status
        attempt.state = "final" if state.is_terminal else "open"
        self._settle(tracked, attempt, fills)

    def _settle(self, tracked: _Tracked, attempt: _Attempt, fills: tuple[Fill, ...]) -> None:
        """Set the order's phase from its last attempt, and report."""
        if tracked.remaining == 0:
            attempt.state = "final"
            self._close(tracked, "filled", tracked.fill_reason, fills)
            return
        if fills:
            self._updates.append(
                OrderUpdate(order_id=tracked.order_id, status="partially_filled", fills=fills)
            )
        if attempt.state != "final":
            tracked.phase = "working"
        elif attempt.venue_status == "rejected":
            self._alert(
                "critical" if tracked.side == "sell" else "warn",
                "order_rejected",
                f"the venue rejected {attempt.venue_order.client_order_id} after accepting it",
            )
            self._close(tracked, "rejected", "broker_rejected")
        else:
            # Canceled or expired with shares unfilled: which of the two the venue used
            # does not matter. Nothing is at the venue; the caller decides what is next.
            tracked.phase = "idle"

    # ------------------------------------------------------------------ closing and canceling

    def close(self, order_id: str, status: ClosedStatus, reason: OrderUpdateReason) -> None:
        """End an order that has nothing at the venue (`staged` or `idle`) without
        sending it again."""
        self._require_recovered()
        tracked = self._get(order_id)
        if tracked.phase not in ("staged", "idle"):
            raise OrderManagerError(f"order {order_id} is {tracked.phase}; it cannot be closed")
        self._close(tracked, status, reason)

    def cancel_all(self, reason: OrderUpdateReason) -> list[str]:
        """Cancel everything not final and confirm with the venue before returning.
        Returns the ids of the orders that ended canceled.

        A cancel is a request: shares can still fill before it takes effect, and those
        fills are recorded. Works while the kill switch is engaged. Raises
        SessionIncomplete if the venue does not confirm within `cancel_confirm_seconds`.
        """
        self._require_recovered()
        return self._pull(list(self._orders.values()), lambda tracked: reason)

    def _pull(
        self,
        targets: Sequence[_Tracked],
        reason_for: Callable[[_Tracked], OrderUpdateReason] | None,
    ) -> list[str]:
        """Take `targets` off the venue and confirm it. With `reason_for`, an order left
        unfilled is closed as canceled with that reason, and the ids of those orders
        are returned. Without it, the order is left `idle` (or `staged`) to be sent
        again."""
        canceled: list[str] = []
        pending: list[_Tracked] = []

        def finish(tracked: _Tracked) -> None:
            if reason_for is not None and tracked.phase in ("staged", "idle"):
                self._close(tracked, "canceled", reason_for(tracked))
                canceled.append(tracked.order_id)

        for tracked in targets:
            if tracked.phase in ("staged", "idle"):
                finish(tracked)
            elif tracked.phase not in TERMINAL_PHASES:
                pending.append(tracked)

        waited = 0.0
        while pending:
            still: list[_Tracked] = []
            for tracked in pending:
                attempt = tracked.last
                assert attempt is not None  # working, unknown and submitting have one
                if self._cancel_one(tracked, attempt):
                    finish(tracked)  # staged here means it never reached the venue
                else:
                    still.append(tracked)
            pending = still
            if not pending:
                break
            if waited >= self._cfg.cancel_confirm_seconds:
                unseen = [t.order_id for t in pending if _unseen(t)]
                if unseen:
                    # Left unknown, not voided: the lookup in recover() after a reset
                    # books a late fill as a fill.
                    self._violation(
                        f"a send got no answer and the venue still does not show it after "
                        f"{waited:.0f} seconds: {', '.join(unseen)}. It may be at the venue"
                    )
                ids = ", ".join(t.order_id for t in pending)
                self._violation(
                    f"cancel not confirmed by the venue after {waited:.0f} seconds for: {ids}"
                )
            self._sleep(self._cfg.poll_seconds)
            waited += self._cfg.poll_seconds
        return canceled

    def _cancel_one(self, tracked: _Tracked, attempt: _Attempt) -> bool:
        """Request a cancel and read the order back. True once nothing of the order is
        left at the venue."""
        client_id = attempt.venue_order.client_order_id
        try:
            state = self._venue.get_order(client_id)
        except VenueDataError as exc:
            self._violation(f"the venue's data cannot be trusted: {exc}")
        except Exception:
            return False
        if state is None:
            if attempt.state == "open":
                self._violation(f"the venue no longer knows {client_id}, which it reported earlier")
            if attempt.maybe_sent:
                # A send got no answer, and "not found" may only mean "not visible yet".
                # Keep looking; at the end of the window _pull kills instead of writing
                # it off.
                return False
            # Journaled but never seen: the venue has no such order.
            self._void(tracked, attempt, "not at the venue when canceled")
            tracked.phase = "idle" if tracked.filled else "staged"
            return True
        if not state.is_terminal:
            try:
                self._venue.cancel(client_id)
                state = self._venue.get_order(client_id) or state
            except VenueDataError as exc:
                self._violation(f"the venue's data cannot be trusted: {exc}")
            except Exception:
                pass  # asked again on the next round
        self._observe(tracked, attempt, state)
        return tracked.phase != "working"

    # ------------------------------------------------------------------ recovery

    def recover(self) -> None:
        """Rebuild from the journal, then resolve every attempt whose outcome is not
        known by looking it up at the venue, before anything new is sent.

        Updates for what recovery finds are not queued: whoever asked for those orders
        is gone, and the ledger already holds every fill. Raises SessionIncomplete if
        the venue cannot be asked.
        """
        if self._recovered:
            raise OrderManagerError("recover() was already called")
        self._replay()
        self._recovered = True
        for tracked in self._orders.values():
            if tracked.phase in TERMINAL_PHASES:
                continue
            attempt = tracked.last
            if attempt is None or attempt.state == "void":
                tracked.phase = "idle" if tracked.filled else "staged"
                if tracked.approval == "denied":  # died between the denial and the close
                    self._close(tracked, "canceled", "approval_denied")
            elif attempt.state == "final":
                self._settle(tracked, attempt, ())
            else:
                self._resolve(tracked, attempt)
        self._updates.clear()

    def _replay(self) -> None:
        for record in self._journal.read():
            payload = record.payload
            try:
                if record.kind == "order_staged":
                    order = Order.model_validate(payload["order"])
                    order_id = str(payload["order_id"])
                    if order_id != order_id_for(self._profile_id, order):
                        raise OrderManagerError(f"{order_id} is not this profile's id for it")
                    reference = payload["reference_price"]
                    assert isinstance(reference, int | float)
                    wait = payload.get("approval", "none")
                    assert wait in ("none", "pending")
                    self._orders[order_id] = _Tracked(
                        order_id,
                        order,
                        float(reference),
                        approval=wait,  # type: ignore[arg-type]
                    )
                elif record.kind == "approval":
                    tracked = self._orders[str(payload["order_id"])]
                    decision = payload["decision"]
                    assert decision in ("approved", "denied")
                    tracked.approval = decision  # type: ignore[assignment]
                    tracked.approved_hash = str(payload.get("content_hash", ""))
                elif record.kind == "closes":
                    closes = payload["closes"]
                    assert isinstance(closes, dict)
                    self._closes = {
                        symbol: float(close)
                        for symbol, close in closes.items()
                        if isinstance(close, int | float)
                    }
                    assert len(self._closes) == len(closes)
                elif record.kind == "order_reduced":
                    tracked = self._orders[str(payload["order_id"])]
                    quantity = payload["quantity"]
                    assert isinstance(quantity, int) and 0 < quantity < tracked.quantity
                    tracked.quantity = quantity
                    tracked.fill_reason = _reduced_reason(str(payload["rule"]))
                elif record.kind == "order_intent":
                    tracked = self._orders[str(payload["order_id"])]
                    number, before = payload["attempt"], payload["filled_before"]
                    assert isinstance(number, int) and isinstance(before, int)
                    attempt = _Attempt(
                        number, VenueOrder.model_validate(payload["venue_order"]), before
                    )
                    if tracked.attempts and tracked.attempts[-1].number == number:
                        tracked.attempts[-1] = attempt
                    else:
                        tracked.attempts.append(attempt)
                    session = dt.date.fromisoformat(str(payload["session"]))
                    self._sent[session] = self._sent.get(session, 0) + 1
                elif record.kind == "venue_state":
                    attempt = self._attempt(payload)
                    state = VenueOrderState.model_validate(payload["state"])
                    attempt.venue_status = state.status
                    attempt.state = "final" if state.is_terminal else "open"
                elif record.kind == "attempt_void":
                    self._attempt(payload).state = "void"
                elif record.kind == "entries_blocked":
                    self._blocked.add(dt.date.fromisoformat(str(payload["signal_date"])))
                elif record.kind == "order_closed":
                    tracked = self._orders[str(payload["order_id"])]
                    tracked.phase = payload["status"]  # type: ignore[assignment]
                    tracked.reason = payload["reason"]  # type: ignore[assignment]
                elif record.kind == "fill":
                    fill = Fill.model_validate(payload["fill"])
                    owner = self._orders.get(fill.order_id)
                    if owner is not None and owner.last is not None:
                        owner.last.filled += fill.quantity
                        owner.last.value += fill.price * fill.quantity
            except (KeyError, ValueError, AssertionError, OrderManagerError) as exc:
                raise OrderManagerError(
                    f"journal record {record.seq} ({record.kind}) does not replay: {exc!r}"
                ) from exc
        for tracked in self._orders.values():
            if tracked.filled != self._ledger.filled_quantity(tracked.order_id):
                raise OrderManagerError(
                    f"journal does not replay: order {tracked.order_id} has "
                    f"{tracked.filled} shares in its attempts but "
                    f"{self._ledger.filled_quantity(tracked.order_id)} in the ledger"
                )

    def _attempt(self, payload: Mapping[str, object]) -> _Attempt:
        tracked = self._orders[str(payload["order_id"])]
        for attempt in reversed(tracked.attempts):
            if attempt.number == payload["attempt"]:
                return attempt
        raise KeyError(f"attempt {payload['attempt']} of {tracked.order_id}")

    def _resolve(self, tracked: _Tracked, attempt: _Attempt) -> None:
        """Find out what became of an attempt the journal says may be at the venue."""
        client_id = attempt.venue_order.client_order_id
        for tried in range(self._cfg.max_submit_attempts):
            if tried:
                self._sleep(self._cfg.poll_seconds * tried)
            try:
                state = self._venue.get_order(client_id)
            except VenueDataError as exc:
                self._violation(f"the venue's data cannot be trusted: {exc}")
            except Exception:
                continue
            if state is not None:
                self._observe(tracked, attempt, state)
            elif attempt.state == "open":
                self._violation(f"the venue no longer knows {client_id}, which it reported earlier")
            else:
                # Journaled, then the process died before the venue saw it.
                self._void(tracked, attempt, "not at the venue after a restart")
                tracked.phase = "idle" if tracked.filled else "staged"
            return
        self._violation(
            f"cannot recover: the venue did not answer about {client_id}, which may be open"
        )

    # ------------------------------------------------------------------ Broker-facing

    def poll_updates(self) -> list[OrderUpdate]:
        """Return and clear order updates since the last call, in execution order."""
        updates, self._updates = self._updates, []
        return updates

    def account(self) -> AccountState:
        """The ledger's cash and positions, not a raw venue read."""
        return self._ledger.account()

    def force_close(
        self,
        symbol: str,
        price: float,
        price_date: dt.date,
        session: dt.date,
        *,
        reason: ForcedReason,
    ) -> str | None:
        """Never fabricates a fill. Alerts; reconciliation picks up what the venue did."""
        why = "left the universe's membership and " if reason == "index_removal" else ""
        self._alert(
            "warn",
            "position_stale_no_data",
            f"{symbol} {why}has had no bar since {price_date}; the position is left alone",
        )
        return None

    # ------------------------------------------------------------------ the open workflow

    def on_session_open(self, session: dt.date, opens: Mapping[str, float]) -> None:
        """Work every open order to a final state, in `session`. Blocks until all are
        final or `session_deadline` passes. Ignores `opens`: real fills come from the
        venue.

        0. With approval of entries on, commands are read from the inbox on every pass.
           An entry still unapproved at `approval_deadline` is canceled
           (`approval_timeout`); a rejected one is canceled at once
           (`approval_denied`). Exits never wait for approval.
        1. Exits go into the opening auction (market, `opg`) until `opg_cutoff`; later
           ones go as market `day` orders. Whatever an auction order leaves unfilled is
           sent again as a market `day` order, and an auction order still working at
           `exit_deadline` is canceled and sent that way too.
        2. Entries wait for the open, and for every exit to be final or `exit_deadline`.
           Each is then checked against the actual account (`RiskManager.review_release`),
           cut or refused if it no longer fits, and sent as a limit `day` order capped
           at its close times `1 + entry_limit_pct`. Entries are never sent twice.
        3. At `session_deadline` whatever is still working is canceled and confirmed.

        Safe to call again after a restart (and `recover()`): each step looks at what
        the order's phase says is left to do. Raises KillSwitchEngaged, after canceling
        what is open, if the latch is set; SessionIncomplete as elsewhere; and
        LookAheadError, doing nothing, for an order decided in `session` or later.
        """
        self._require_recovered()
        live = [t for t in self._orders.values() if t.phase not in TERMINAL_PHASES]
        if not live:
            return
        for tracked in live:
            if tracked.order.created_on >= session:
                raise LookAheadError(
                    f"order {tracked.order_id} was decided on {tracked.order.created_on}; "
                    f"it cannot be worked in session {session}"
                )
        self._sync_clock()
        today = self._time().astimezone(MARKET_TZ).date()
        if today != session:
            raise OrderManagerError(f"session {session} is not today ({today} in New York)")
        try:
            self._work_session(session, live)
        except KillSwitchEngaged:
            self._pull(live, lambda tracked: "kill_switch")
            raise

    def _work_session(self, session: dt.date, live: Sequence[_Tracked]) -> None:
        def at(time_of_day: dt.time) -> dt.datetime:
            return dt.datetime.combine(session, time_of_day, MARKET_TZ)

        cfg = self._cfg
        open_at, deadline = at(MARKET_OPEN), at(cfg.session_deadline)
        opg_cutoff, exit_deadline = at(cfg.opg_cutoff), at(cfg.exit_deadline)
        approval_deadline = at(cfg.approval_deadline)
        exits = [t for t in live if t.side == "sell"]
        entries = [t for t in live if t.side == "buy"]  # staging order is rank order
        sends: dict[str, int] = {}  # exit order id -> sends this session

        while True:
            self._kill.check()
            now = self._time()
            if now >= deadline:
                break
            if now >= open_at:
                self.refresh()
            self._work_approvals(entries, approval_deadline, now)
            for tracked in exits:
                self._work_exit(tracked, session, sends, now < opg_cutoff, now >= exit_deadline)
            exits_final = all(t.phase in TERMINAL_PHASES for t in exits)
            if now >= open_at and (exits_final or now >= exit_deadline):
                for tracked in entries:
                    self._work_entry(tracked, session)
            if all(t.phase in TERMINAL_PHASES for t in live):
                break
            upcoming = min(
                b for b in (approval_deadline, open_at, exit_deadline, deadline) if b > now
            )
            self._sleep(max(min(cfg.poll_seconds, (upcoming - now).total_seconds()), 0.001))

        def reason(tracked: _Tracked) -> OrderUpdateReason:
            if tracked.side == "sell":
                return "session_deadline"
            sent = any(a.state != "void" for a in tracked.attempts)
            return "limit_not_reached" if sent else "session_deadline"

        self._pull(live, reason)
        for tracked in exits:
            if tracked.phase in ("canceled", "expired"):
                self._alert(
                    "critical",
                    "exit_unfilled",
                    f"exit {tracked.order_id} ended {tracked.phase} with {tracked.remaining} "
                    f"of {tracked.quantity} shares unsold; the position remains",
                )

    def _work_exit(
        self,
        tracked: _Tracked,
        session: dt.date,
        sends: dict[str, int],
        auction_open: bool,
        past_exit_deadline: bool,
    ) -> None:
        attempt = tracked.last
        if (
            tracked.phase == "working"
            and past_exit_deadline
            and attempt is not None
            and attempt.venue_order.time_in_force == "opg"
        ):
            self._pull([tracked], None)  # the auction order is taken back; the rest goes below
        if tracked.phase not in ("staged", "idle", "unknown"):
            return
        # The first send, plus a bounded number of further ones: an exit keeps trying,
        # but a venue that ends every attempt must not turn into a loop.
        if sends.get(tracked.order_id, 0) > self._cfg.max_submit_attempts:
            return
        sends[tracked.order_id] = sends.get(tracked.order_id, 0) + 1
        self.send(
            tracked.order_id,
            session,
            type="market",
            time_in_force="opg" if auction_open else "day",
        )

    def _work_entry(self, tracked: _Tracked, session: dt.date) -> None:
        if tracked.phase == "idle":  # the venue ended it early; entries are not sent twice
            self._close(tracked, "canceled", "limit_not_reached")
            return
        if tracked.phase != "staged" or tracked.approval == "pending":
            return
        if tracked.order.created_on in self._blocked:
            self._close(tracked, "canceled", "unknown_outcome")
            return
        limit = entry_limit_price(tracked.reference_price, self._cfg.entry_limit_pct)
        if limit is None:
            self._refuse(tracked, f"no limit price from a close of {tracked.reference_price}")
            return
        working = [
            t
            for t in self._orders.values()
            if t.side == "buy" and t.phase in ("working", "unknown", "submitting") and t.last
        ]
        reserved = sum(
            t.remaining * (t.last.venue_order.limit_price or 0.0) for t in working if t.last
        )
        order = tracked.order
        if tracked.quantity != order.quantity:
            order = order.model_copy(update={"quantity": tracked.quantity})
        decision = self._risk.review_release(
            order,
            self._ledger.account(),
            self._closes,
            price=limit,
            reserved_cash=reserved,
            pending_entries=sum(1 for t in working if t.filled == 0),
            max_positions=self._max_positions,
        )
        if decision.approved_quantity == 0:
            if decision.rule == "no_leverage":
                self._close(tracked, "canceled", "insufficient_cash")
            else:
                self._refuse(tracked, f"{decision.rule}: {decision.detail}")
            return
        if decision.approved_quantity < tracked.quantity:
            self._reduce(tracked, decision)
        self.send(tracked.order_id, session, type="limit", time_in_force="day", limit_price=limit)

    # ------------------------------------------------------------------ flatten

    def flatten(self, session: dt.date, decided_on: dt.date, until: dt.datetime) -> FlattenResult:
        """The operator's flatten: sell every share the ledger holds, in `session`.
        Runs with the kill switch engaged (the caller engages it first), and sends
        sells only.

        `decided_on` is the session before `session` (what the orders are dated, so a
        fill is always in a later session than its order). `until` is when to stop
        trying.

        1. Everything still open is canceled and confirmed first (`flatten`), so no
           entry can fill behind the sells and no other sell holds the shares.
        2. The ledger is compared with the venue's positions. Where the venue holds
           fewer shares than the ledger, only the venue's are sold (the venue would
           reject a sell for more, and then none of the position would be sold). Every
           difference is named in the result and in a critical alert; the ledger is
           not corrected here (`accept_broker_state` does that).
        3. One sell is staged per position and worked as the open workflow works an
           exit: into the opening auction before `opg_cutoff`, as a market `day` order
           after it, sent again if an attempt ends with shares left.
        4. At `until` whatever is still working is canceled and confirmed, and each
           sell that did not fill in full is a critical alert.

        The sells go through `send` like any exit: journaled before the venue is
        called, never for more than the ledger holds. Only the latch check differs, and
        only for sells (`KillSwitch.flattening`).

        A kill from anywhere else stops it, whether or not that kill confirms its
        cancel: every kill changes the latch file before it cancels. The file is read
        on every pass and again before every send (`KillSwitch.flattening`), and once
        it is not the latch this flatten started under, nothing more is sent, the sells
        are canceled and SessionIncomplete is raised. What remains is a request already
        on its way to the venue when the kill began: it can still arrive, and this
        flatten cancels it when it stops.

        The sells count against `max_orders_per_session` like any other order. If the
        session's orders have used the cap up, the next send stops everything
        (SessionIncomplete, as elsewhere) with positions still held: sell those at the
        broker and record it with `accept_broker_state`.
        """
        self._require_recovered()
        if decided_on >= session:
            raise LookAheadError(f"a flatten in session {session} cannot be dated {decided_on}")
        if not self._kill.engaged():
            raise OrderManagerError("a flatten runs with the kill switch engaged; engage it first")
        self._sync_clock()
        today = self._time().astimezone(MARKET_TZ).date()
        if today != session:
            raise OrderManagerError(f"session {session} is not today ({today} in New York)")
        live = [t for t in self._orders.values() if t.phase not in TERMINAL_PHASES]
        self._pull(live, lambda tracked: "flatten")

        ledger = self._ledger.account().positions
        at_venue = self._venue_quantities()
        differences: list[str] = []
        if at_venue is None:
            differences.append(
                "the venue's positions could not be read; selling what the ledger holds"
            )
        else:
            for symbol in sorted(set(ledger) | set(at_venue)):
                ours = ledger[symbol].quantity if symbol in ledger else 0
                theirs = at_venue.get(symbol, 0)
                if ours != theirs:
                    selling = max(min(ours, theirs), 0)
                    differences.append(
                        f"{symbol}: ledger holds {ours}, venue holds {theirs}; selling {selling}"
                    )
        if differences:
            self._alert(
                "critical",
                "flatten_mismatch",
                "flatten: the ledger and the venue differ ("
                + "; ".join(differences)
                + "). Only shares both hold are sold. Settle the rest at the broker and "
                "record it with accept_broker_state.",
            )

        # From here on orders may be open at the venue again. A kill run by anyone else
        # changes the latch back, and the loop below stops when it sees that.
        latch = self._kill.orders_sent()
        if not latch.engaged or latch.at is None or latch.cancel_confirmed:
            # Without the note a kill from elsewhere would leave the latch file as it
            # is, and this flatten would send after that kill reported its cancel.
            raise OrderManagerError(
                "flatten refused, nothing was sent: the kill latch could not be marked "
                f"'orders may be open' (it reads: {latch.reason or 'no reason'}). Check "
                "that the latch file can be written, or reset the kill switch, and try again"
            )

        sells: list[_Tracked] = []
        for symbol, held in sorted(ledger.items()):
            quantity = held.quantity
            if at_venue is not None:
                quantity = max(min(quantity, at_venue.get(symbol, 0)), 0)
            if quantity == 0:
                continue
            earlier = sum(
                1
                for t in self._orders.values()
                if t.order.created_on == decided_on
                and t.order.symbol == symbol
                and flatten_number(t.order) is not None
            )
            order = Order(
                created_on=decided_on,
                action="exit",
                legs=(Leg(instrument=held.instrument, side="sell"),),
                quantity=quantity,
                reason=f"operator:flatten:{earlier + 1}",
            )
            _check_supported(order)
            order_id = order_id_for(self._profile_id, order)
            # The close the position was last valued at; one adopted by an adjustment
            # since then has none, and its cost stands in.
            reference = self._closes.get(symbol, held.avg_price)
            self._journal.append(
                "order_staged",
                {
                    "order_id": order_id,
                    "order": order.model_dump(mode="json"),
                    "reference_price": reference,
                    "approval": "none",
                },
            )
            tracked = self._orders[order_id] = _Tracked(order_id, order, reference)
            sells.append(tracked)

        def at(time_of_day: dt.time) -> dt.datetime:
            return dt.datetime.combine(session, time_of_day, MARKET_TZ)

        open_at, opg_cutoff = at(MARKET_OPEN), at(self._cfg.opg_cutoff)
        exit_deadline = at(self._cfg.exit_deadline)
        sends: dict[str, int] = {}
        stopped = False
        with self._kill.flattening(latch):
            while True:
                if self._kill.on_disk() != latch:
                    stopped = True
                    break
                now = self._time()
                if now >= until:
                    break
                if now >= open_at:
                    self.refresh()
                try:
                    for tracked in sells:
                        self._work_exit(
                            tracked, session, sends, now < opg_cutoff, now >= exit_deadline
                        )
                except KillSwitchEngaged:
                    # The latch changed between the look above and a send: the pass for
                    # sells ended with it, and the sell was not sent.
                    stopped = True
                    break
                if all(t.phase in TERMINAL_PHASES for t in sells):
                    break
                self._sleep(max(min(self._cfg.poll_seconds, (until - now).total_seconds()), 0.001))
        self._pull(sells, lambda tracked: "kill_switch" if stopped else "flatten")
        if stopped:
            message = (
                "the kill switch was engaged again or reset while a flatten was running; "
                "the flatten stopped and its unfilled sells are canceled"
            )
            self._alert("critical", "flatten_stopped", message + ". Positions may remain.")
            raise SessionIncomplete(message)
        self._kill.confirm_cancel()  # nothing of the flatten is left at the venue
        for tracked in sells:
            if tracked.phase != "filled":
                self._alert(
                    "critical",
                    "flatten_unfilled",
                    f"flatten: {tracked.order_id} ended {tracked.phase} with {tracked.remaining} "
                    f"of {tracked.quantity} {tracked.order.symbol} unsold; the position remains",
                )
        return FlattenResult(
            order_ids=tuple(t.order_id for t in sells), differences=tuple(differences)
        )

    def _venue_quantities(self) -> dict[str, int] | None:
        """Shares per symbol at the venue, or None if it cannot be asked."""
        for tried in range(self._cfg.max_submit_attempts):
            if tried:
                self._sleep(self._cfg.poll_seconds)
            try:
                positions = self._venue.positions()
            except (VenueDataError, ValidationError) as exc:
                self._violation(f"the venue's data cannot be trusted: {exc}")
            except Exception:
                continue
            quantities: dict[str, int] = {}
            for position in positions:
                quantities[position.symbol] = quantities.get(position.symbol, 0) + position.quantity
            return quantities
        return None

    # ------------------------------------------------------------------ approval

    def pending_approvals(self, session: dt.date) -> list[ApprovalRequest]:
        """The entries waiting for the operator, for the open of `session`."""
        expires = dt.datetime.combine(session, self._cfg.approval_deadline, MARKET_TZ)
        requests = []
        for tracked in self._orders.values():
            terms = self._approval_terms(tracked)
            if tracked.approval != "pending" or tracked.phase != "staged" or terms is None:
                continue
            content_hash, limit = terms
            requests.append(
                ApprovalRequest(
                    order_id=tracked.order_id,
                    content_hash=content_hash,
                    symbol=tracked.order.symbol,
                    side=tracked.side,
                    quantity=tracked.quantity,
                    limit_price=limit,
                    est_notional=tracked.quantity * limit,
                    expires_at=expires,
                )
            )
        return requests

    def ingest_inbox(self, approval_deadline: dt.datetime | None = None) -> None:
        """Read and act on every command waiting in the inbox. A command that does not
        fit is refused with an alert and a journal record; it never raises.

        Safe to repeat: a command read twice (the process died before removing its
        file) changes nothing the second time.
        """
        self._require_recovered()
        if self._inbox is None:
            return
        for item in self._inbox.read():
            command = item.command
            if command is None:
                self._refuse_command(item.path.name, f"cannot be read: {item.error}")
                self._inbox.quarantine(item.path)
                continue
            if isinstance(command, ApproveOrders):
                for order_id, content_hash in command.approvals.items():
                    try:
                        self._approve(
                            order_id, content_hash, command, item.written_at, approval_deadline
                        )
                    except ApprovalRefused as exc:
                        self._refuse_command(item.path.name, f"approve {order_id}: {exc}")
            else:
                for order_id in command.order_ids:
                    try:
                        self._deny(order_id, command)
                    except ApprovalRefused as exc:
                        self._refuse_command(item.path.name, f"reject {order_id}: {exc}")
            self._inbox.remove(item.path)

    def _approve(
        self,
        order_id: str,
        content_hash: str,
        command: ApproveOrders,
        written_at: dt.datetime | None,
        approval_deadline: dt.datetime | None,
    ) -> None:
        tracked = self._orders.get(order_id)
        if tracked is None:
            raise ApprovalRefused("unknown order")
        if tracked.side != "buy":
            raise ApprovalRefused("exits are never gated; there is nothing to approve")
        if tracked.approval == "approved" and tracked.approved_hash == content_hash:
            return  # the same approval again
        if tracked.approval != "pending" or tracked.phase != "staged":
            raise ApprovalRefused(
                f"the order is not waiting for approval ({tracked.phase}, {tracked.approval})"
            )
        # When was this decided? The command says (`at`), but a writer can say anything.
        # The file's own time is the later bound: nothing was decided after it was
        # written, and nothing written after the deadline counts, whatever it claims.
        if written_at is None:
            raise ApprovalRefused("the command file's time cannot be read")
        skew = dt.timedelta(seconds=self._cfg.max_clock_skew_seconds)
        if command.at > self._time() + skew:
            raise ApprovalRefused(f"decided at {command.at.isoformat()}, which is in the future")
        decided = max(command.at, written_at)
        if approval_deadline is not None and decided > approval_deadline:
            said = command.at.astimezone(MARKET_TZ).isoformat()
            written = written_at.astimezone(MARKET_TZ).isoformat()
            raise ApprovalRefused(
                f"decided after the deadline (it says {said}; its file was written {written})"
            )
        terms = self._approval_terms(tracked)
        if terms is None or terms[0] != content_hash:
            raise ApprovalRefused("the order is not the one that was shown (hash mismatch)")
        self._journal.append(
            "approval",
            {
                "order_id": order_id,
                "decision": "approved",
                "operator": command.operator,
                "content_hash": content_hash,
                "at": command.at.isoformat(),
                "written_at": written_at.isoformat(),
            },
        )
        tracked.approval = "approved"
        tracked.approved_hash = content_hash

    def _deny(self, order_id: str, command: RejectOrders) -> None:
        tracked = self._orders.get(order_id)
        if tracked is None:
            raise ApprovalRefused("unknown order")
        if tracked.side != "buy":
            raise ApprovalRefused("exits are never gated; an exit cannot be rejected")
        if tracked.approval == "denied":
            return  # the same rejection again
        if tracked.phase != "staged":
            raise ApprovalRefused(f"the order is already {tracked.phase}")
        self._journal.append(
            "approval",
            {
                "order_id": order_id,
                "decision": "denied",
                "operator": command.operator,
                "reason": command.reason,
                "at": command.at.isoformat(),
            },
        )
        tracked.approval = "denied"
        self._close(tracked, "canceled", "approval_denied")

    def _work_approvals(
        self, entries: Sequence[_Tracked], approval_deadline: dt.datetime, now: dt.datetime
    ) -> None:
        if self._inbox is None:
            return
        # Every pass, whatever is waiting: a rejection counts until the entry is sent,
        # and a command that fits nothing is refused now, not left for another day.
        self.ingest_inbox(approval_deadline)
        if now < approval_deadline:
            return
        late = [t for t in entries if t.phase == "staged" and t.approval == "pending"]
        for tracked in late:
            self._close(tracked, "canceled", "approval_timeout")
        if late:
            self._alert(
                "warn",
                "approval_timeout",
                f"{len(late)} entries were not approved by the deadline and were canceled: "
                + ", ".join(t.order_id for t in late),
            )

    def _needs_approval(self, order: Order) -> Literal["none", "pending"]:
        return "pending" if self._approval == "entries" and order.action == "enter" else "none"

    def _approval_terms(self, tracked: _Tracked) -> tuple[str, float] | None:
        """The hash an approval of this entry must quote, and its limit price."""
        limit = entry_limit_price(tracked.reference_price, self._cfg.entry_limit_pct)
        if limit is None:
            return None
        return (
            approval_hash(
                tracked.order_id, tracked.order.symbol, tracked.side, tracked.quantity, limit
            ),
            limit,
        )

    def _refuse_command(self, name: str, why: str) -> None:
        self._journal.append("command_refused", {"file": name, "why": why})
        self._alert("warn", "command_refused", f"inbox command {name} refused: {why}")

    def _refuse(self, tracked: _Tracked, why: str) -> None:
        self._alert("warn", "entry_refused", f"entry {tracked.order_id} not sent: {why}")
        self._close(tracked, "rejected", "risk_rejected")

    def _reduce(self, tracked: _Tracked, decision: RiskDecision) -> None:
        """Cut the quantity being worked, on the record, before the cut order is sent."""
        self._journal.append(
            "order_reduced",
            {
                "order_id": tracked.order_id,
                "quantity": decision.approved_quantity,
                "rule": decision.rule,
                "detail": decision.detail,
            },
        )
        tracked.quantity = decision.approved_quantity
        tracked.fill_reason = _reduced_reason(str(decision.rule))

    def _sync_clock(self) -> None:
        """Measure the local clock against the venue's. All deadlines use the venue's
        time; a local clock too far from it stops trading."""
        for tried in range(self._cfg.max_submit_attempts):
            if tried:
                self._sleep(self._cfg.poll_seconds)
            try:
                offset = self._venue.now() - self._now()
            except VenueDataError as exc:
                self._violation(f"the venue's data cannot be trusted: {exc}")
            except Exception:
                continue
            if abs(offset.total_seconds()) > self._cfg.max_clock_skew_seconds:
                self._violation(
                    f"clock skew: the venue's clock is {offset.total_seconds():+.0f} seconds "
                    f"from the local clock (limit {self._cfg.max_clock_skew_seconds:.0f})"
                )
            self._clock_offset = offset
            return
        self._alert("warn", "clock", "could not read the venue's clock; using the local clock")

    def _time(self) -> dt.datetime:
        return self._now() + self._clock_offset

    def orders(self) -> list[WorkingOrder]:
        """Every order the manager knows, in staging order."""
        return [
            WorkingOrder(
                order_id=t.order_id,
                order=t.order,
                quantity=t.quantity,
                reference_price=t.reference_price,
                phase=t.phase,
                venue_orders=tuple(a.venue_order for a in t.attempts),
                filled_quantity=t.filled,
                reason=t.reason,
                approval=t.approval,
            )
            for t in self._orders.values()
        ]

    def entries_blocked(self, signal_date: dt.date) -> bool:
        """True if entries decided on `signal_date` may no longer be sent."""
        return signal_date in self._blocked

    # ------------------------------------------------------------------ internals

    def _require_recovered(self) -> None:
        if not self._recovered:
            raise OrderManagerError("call recover() before using the order manager")

    def _get(self, order_id: str) -> _Tracked:
        tracked = self._orders.get(order_id)
        if tracked is None:
            raise OrderManagerError(f"unknown order {order_id}")
        return tracked

    def _void(self, tracked: _Tracked, attempt: _Attempt, why: str) -> None:
        self._journal.append(
            "attempt_void",
            {"order_id": tracked.order_id, "attempt": attempt.number, "why": why},
        )
        attempt.state = "void"

    def _block_entries(self, signal_date: dt.date) -> None:
        if signal_date not in self._blocked:
            self._journal.append("entries_blocked", {"signal_date": signal_date.isoformat()})
            self._blocked.add(signal_date)

    def _close(
        self,
        tracked: _Tracked,
        status: Literal["filled", "canceled", "rejected", "expired"],
        reason: OrderUpdateReason,
        fills: tuple[Fill, ...] = (),
    ) -> None:
        self._journal.append(
            "order_closed", {"order_id": tracked.order_id, "status": status, "reason": reason}
        )
        tracked.phase = status
        tracked.reason = reason
        if status != "filled" or fills:  # a fill found during recovery has no update
            self._updates.append(
                OrderUpdate(order_id=tracked.order_id, status=status, fills=fills, reason=reason)
            )

    def _alert(self, severity: Severity, kind: str, message: str) -> None:
        try:
            alert = Alert(at=self._now(), severity=severity, kind=kind, message=message)
        except Exception:
            alert = Alert(at=_utc_now(), severity=severity, kind=kind, message=message)
        send_alert(self._alerts, alert)

    def _violation(self, message: str) -> NoReturn:
        """The manager's picture of its orders is no longer trustworthy: stop everything."""
        kill(
            self._kill,
            message,
            _SOURCE,
            venue=self._venue,
            alerts=self._alerts,
            wait_seconds=self._cfg.poll_seconds,
            sleep=self._sleep,
            now=self._now,
        )
        raise SessionIncomplete(message)


def _unseen(tracked: _Tracked) -> bool:
    """A send of the last attempt got no answer and the venue has never shown it."""
    last = tracked.last
    return last is not None and last.maybe_sent and last.state == "sending"


def _check_supported(order: Order) -> None:
    if len(order.legs) != 1:
        raise OrderManagerError("the order manager sends one-leg orders only")
    leg = order.legs[0]
    if not isinstance(leg.instrument, Stock) or leg.ratio != 1:
        raise OrderManagerError("the order manager sends stock orders only")
    expected: Side = "buy" if order.action == "enter" else "sell"
    if leg.side != expected:
        raise OrderManagerError(f"long-only: an {order.action} order must {expected}")


def _reduced_reason(rule: str) -> OrderUpdateReason:
    """What the final fill of a cut entry reports: the sim's reason when cash was the
    limit, nothing otherwise (the journal's `order_reduced` record names the rule)."""
    return "reduced_for_cash" if rule == "no_leverage" else ""


def entry_limit_price(close: float, entry_limit_pct: float) -> float | None:
    """The entry cap: `close * (1 + entry_limit_pct)`, rounded down to a whole cent.
    None if that is less than one cent."""
    cents = math.floor(close * (1 + entry_limit_pct) * 100 + 1e-9)
    return cents / 100 if cents >= 1 else None
