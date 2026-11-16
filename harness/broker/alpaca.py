"""`AlpacaBroker`: the `Venue` for the Alpaca PAPER account. The only trading network code.

It makes no decisions. It translates the harness's venue calls into Alpaca requests and
Alpaca's answers into the venue models, and it sorts every failure into one of three
kinds the order manager knows how to handle:

- `VenueRejected`: Alpaca refused the request (a 4xx answer). Definitive.
- `VenueUnavailable`: no answer, or an answer that says "try later" (a timeout, a
  connection error, 429, 5xx). The outcome is unknown.
- `VenueDataError`: an answer that cannot be trusted (an order status this module does
  not know, a fractional quantity, a list that may be cut short, an order under one of
  our ids with terms we did not send). Stops trading.
- `VenueSetupError`, a kind of `VenueDataError`: the keys are refused (401, or 403 on
  anything but an order) or a request was about to go somewhere other than the paper
  host. Stops trading too: neither is a reason to reject one order and carry on.

**Live trading is impossible here, in layers.** There is no mode argument and no URL
argument. The client is built with `paper=True` as a literal. Only the paper key names
are read (`harness/keys.py`). The key id must start with `PK`. After the client is
built, its base URL is checked to be exactly the paper host. A test fails if the live
host or a live key name appears anywhere in `harness/`.

**What this module takes over from the SDK** (alpaca-py 0.45):

- Retries. The SDK retries 429 and 504 by itself, three times with a fixed wait. A
  silent retry of an order would be invisible to the journal and the order cap, so it
  is switched off and the order manager's own counts are the only ones.
- Timeouts. The SDK sends requests with none. Every request here has one.

Both rely on private attributes of the SDK's client, which are checked when the client is
built: if a new SDK version moves them, construction fails instead of trading without
them.

**The kill latch blocks `place`, and only `place`.** Cancels and reads must work while
the latch is set, or the kill path could never cancel anything.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, Protocol

import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.trading.requests import (
    GetCalendarRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)
from pydantic import ValidationError

from harness.broker.venue import (
    KillSwitchEngaged,
    MarketSession,
    VenueAccount,
    VenueDataError,
    VenueOrder,
    VenueOrderState,
    VenuePosition,
    VenueRejected,
    VenueStatus,
    VenueUnavailable,
)
from harness.config import ExecutionConfig
from harness.data import PROJECT_ROOT
from harness.keys import read_alpaca_paper_keys
from harness.killswitch import KillSwitch
from harness.models import MARKET_TZ
from harness.ratelimit import RateLimiter

PAPER_URL: Final = "https://paper-api.alpaca.markets"
MAX_OPEN_ORDERS: Final = 500  # the most Alpaca returns in one list of orders

# Every Alpaca order status, and what it means here. "open" is anything Alpaca may still
# change: it becomes `pending` or `partially_filled` depending on what has filled, so the
# deadline cancel still runs and confirms. Only the four final statuses end an order.
# `replaced` is not in the table: the harness never replaces an order, so seeing it is an
# error, as is any status Alpaca adds later.
STATUS_MAP: Final[dict[str, VenueStatus | None]] = {
    "new": None,
    "accepted": None,
    "pending_new": None,
    "accepted_for_bidding": None,
    "held": None,
    "pending_review": None,  # no longer sent by Alpaca; still in the SDK
    "partially_filled": None,
    "pending_cancel": None,  # a cancel was asked for, not done
    "pending_replace": None,
    "done_for_day": None,  # may trade again the next day unless canceled
    "stopped": None,  # a fill is promised, not booked
    "suspended": None,
    "calculated": None,  # done for the day, settlement not computed yet
    "filled": "filled",
    "canceled": "canceled",
    "expired": "expired",
    "rejected": "rejected",
}


class VenueSetupError(VenueDataError):
    """The adapter or the account is not set up to trade safely: keys that are refused,
    an account that can short or borrow, a request that is not to the paper host.

    A kind of `VenueDataError`, so wherever it surfaces (even in the middle of sending
    an order) it stops trading instead of being read as "no answer". The message never
    contains key material."""


class _Refused(VenueRejected):
    """A rejection that remembers Alpaca's HTTP status, for the callers in this module
    that treat "not found" differently from other refusals."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class Transport(Protocol):
    """The one method of `requests.Session` the SDK uses."""

    def request(self, *args: Any, **kwargs: Any) -> requests.Response: ...


class _TimedSession:
    """Adds the timeout the SDK leaves out, and refuses any URL but the paper host."""

    def __init__(self, inner: Transport, timeout: float) -> None:
        self._inner = inner
        self._timeout = timeout

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        if not url.startswith(PAPER_URL + "/"):
            raise VenueSetupError("refusing a request that is not to the Alpaca paper host")
        kwargs["timeout"] = self._timeout
        return self._inner.request(method, url, **kwargs)


class AlpacaBroker:
    """`Venue` for the Alpaca PAPER account. There is no mode parameter and no live URL."""

    def __init__(
        self,
        cfg: ExecutionConfig,
        kill: KillSwitch,
        limiter: RateLimiter,
        *,
        dotenv_path: Path = PROJECT_ROOT / ".env",
        session: Transport | None = None,
    ) -> None:
        """`session` is the HTTP transport, replaced by a fake in tests. It cannot change
        where requests go: the URL comes from the client, which is always the paper one."""
        self._kill = kill
        self._limiter = limiter
        self._venue_ids: dict[str, str] = {}  # client order id -> Alpaca's order id
        keys = read_alpaca_paper_keys(dotenv_path)
        if not keys.key_id.startswith("PK"):
            raise VenueSetupError(
                "the key id in .env does not look like an Alpaca paper key (paper key ids "
                "start with PK); refusing to build a trading client"
            )
        client = TradingClient(keys.key_id, keys.secret_key, paper=True)
        for attribute in ("_base_url", "_retry", "_session"):
            if not hasattr(client, attribute):
                raise VenueSetupError(
                    f"this alpaca-py version has no TradingClient.{attribute}; the adapter "
                    "cannot check the endpoint, switch off retries, or set a timeout"
                )
        if str(getattr(client._base_url, "value", client._base_url)) != PAPER_URL:
            raise VenueSetupError("the trading client is not pointed at the Alpaca paper host")
        client._retry = 0
        client._session = _TimedSession(  # pyright: ignore[reportAttributeAccessIssue]
            session or requests.Session(), cfg.request_timeout_seconds
        )
        self._client = client

    def __repr__(self) -> str:
        return "AlpacaBroker(paper)"

    # ------------------------------------------------------------------ setup

    def verify_account(self) -> VenueAccount:
        """Startup gate: the account can trade, cannot short, and cannot borrow.

        The last two are Alpaca account settings (`no_shorting`, a margin multiplier of
        1), set once by the operator. With them, an oversized sell cannot open a short
        and a buy cannot borrow even if the harness is wrong. Raises VenueSetupError if
        any is missing; the runner calls this before anything else.
        """
        account = self.account()
        settings = self._call(self._client.get_account_configurations)
        problems = []
        try:
            multiplier = float(settings.max_margin_multiplier)
            no_shorting = bool(settings.no_shorting)
            suspended = bool(settings.suspend_trade)
        except (AttributeError, TypeError, ValueError) as exc:
            raise VenueDataError(f"account settings cannot be read: {exc}") from exc
        if account.trading_blocked:
            problems.append("the account is blocked from trading")
        if suspended:
            problems.append("trading is suspended in the account settings (suspend_trade)")
        if not no_shorting:
            problems.append("shorting is allowed; set no_shorting in the account settings")
        if multiplier != 1:
            problems.append(
                f"the margin multiplier is {multiplier:g}; set max_margin_multiplier to 1"
            )
        elif account.margin_multiplier != 1:
            # The setting says 1 but the account itself still reports more. Believe the
            # account: it is what orders are checked against.
            problems.append(
                f"the account reports a margin multiplier of {account.margin_multiplier:g} "
                "although max_margin_multiplier is 1"
            )
        if problems:
            raise VenueSetupError("the paper account is not set up safely: " + "; ".join(problems))
        return account  # as Alpaca reported it; nothing is overwritten

    # ------------------------------------------------------------------ Venue: reads

    def now(self) -> dt.datetime:
        clock = self._call(self._client.get_clock)
        try:
            stamp = clock.timestamp
            if stamp.utcoffset() is None:
                raise ValueError("the clock has no time zone")
        except (AttributeError, ValueError) as exc:
            raise VenueDataError(f"the venue clock cannot be read: {exc}") from exc
        return stamp

    def sessions(self, start: dt.date, end: dt.date) -> list[MarketSession]:
        days = self._call(
            lambda: self._client.get_calendar(GetCalendarRequest(start=start, end=end))
        )
        try:
            return [
                MarketSession(date=d.date, open=_new_york(d.open), close=_new_york(d.close))
                for d in days
            ]
        except (AttributeError, TypeError, ValueError) as exc:
            raise VenueDataError(f"the market calendar cannot be read: {exc}") from exc

    def account(self) -> VenueAccount:
        raw = self._call(self._client.get_account)
        try:
            status = str(getattr(raw.status, "value", raw.status))
            return VenueAccount(
                account_id=str(raw.account_number),
                cash=_number(raw.cash),
                buying_power=_number(raw.buying_power),
                equity=_number(raw.equity),
                # Anything but a plainly active, unblocked account counts as blocked.
                trading_blocked=bool(
                    raw.trading_blocked or raw.account_blocked or status != "ACTIVE"
                ),
                shorting_enabled=bool(raw.shorting_enabled),
                margin_multiplier=_number(raw.multiplier),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise VenueDataError(f"the account cannot be read: {exc}") from exc

    def positions(self) -> list[VenuePosition]:
        raw = self._call(self._client.get_all_positions)
        try:
            positions = []
            for p in raw:
                quantity = _whole(p.qty, f"position in {p.symbol}")
                if _value(p.side) == "short":  # reported as negative; make sure of it
                    quantity = -abs(quantity)
                positions.append(
                    VenuePosition(
                        symbol=p.symbol,
                        quantity=quantity,
                        avg_entry_price=_number(p.avg_entry_price),
                    )
                )
            return positions
        except (AttributeError, TypeError, ValueError) as exc:
            raise VenueDataError(f"positions cannot be read: {exc}") from exc

    def open_orders(self) -> list[VenueOrderState]:
        """Every order that is not final. Alpaca returns at most 500 in one list; a full
        list may be cut short, so it is an error, not an answer."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=MAX_OPEN_ORDERS)
        raw = self._call(lambda: self._client.get_orders(request))
        if len(raw) >= MAX_OPEN_ORDERS:
            raise VenueDataError(
                f"{len(raw)} open orders, the most one list can hold: the list may be cut short"
            )
        return [self._state(order) for order in raw]

    def get_order(self, client_order_id: str) -> VenueOrderState | None:
        raw = self._lookup(client_order_id)
        return self._state(raw) if raw is not None else None

    # ------------------------------------------------------------------ Venue: writes

    def place(self, order: VenueOrder) -> VenueOrderState:
        """Send one order. The latch is read last, immediately before the request."""
        self._kill.check(selling=order.side == "sell")
        self._limiter.acquire()  # may sleep, so the latch is read again after it
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        tif = AlpacaTimeInForce.OPG if order.time_in_force == "opg" else AlpacaTimeInForce.DAY
        if order.type == "limit":
            request: MarketOrderRequest | LimitOrderRequest = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                client_order_id=order.client_order_id,
            )
        else:
            request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.quantity,
                side=side,
                time_in_force=tif,
                client_order_id=order.client_order_id,
            )
        self._kill.check(selling=order.side == "sell")
        try:
            raw = self._call(
                lambda: self._client.submit_order(request), throttle=False, placing=True
            )
        except VenueRejected as rejection:
            # Alpaca rejects a repeated client order id. If that is what this is, the
            # order already exists, and the protocol asks for it to be reported as the
            # existing order. So look before believing the rejection.
            existing = self._lookup(order.client_order_id)
            if existing is None:
                raise
            if not _same_terms(existing, order):
                # Not a rejection: saying so would close our order while a different one
                # is live at the venue under its id. Nothing here can be trusted.
                raise VenueDataError(
                    f"an order with client id {order.client_order_id} already exists at the "
                    f"venue with different terms from the one being sent ({rejection})"
                ) from rejection
            return self._state(existing)
        return self._state(raw)

    def cancel(self, client_order_id: str) -> None:
        """Request cancellation. Not throttled: a cancel must not wait behind reads."""
        venue_id = self._venue_ids.get(client_order_id)
        if venue_id is None:
            found = self._lookup(client_order_id, throttle=False)
            if found is None:
                raise VenueRejected(f"no order {client_order_id}")
            venue_id = self._venue_ids[client_order_id] = str(found.id)
        try:
            self._call(lambda: self._client.cancel_order_by_id(venue_id), throttle=False)
        except _Refused as exc:
            if exc.status == 404:
                raise
            # 422: the order is already final. Canceling a final order does nothing.

    def cancel_all(self) -> None:
        """Request cancellation of every open order. Not throttled."""
        self._call(self._client.cancel_orders, throttle=False)

    # ------------------------------------------------------------------ internals

    def _lookup(self, client_order_id: str, throttle: bool = True) -> AlpacaOrder | None:
        """Alpaca's order with this client id, or None if it has never seen it."""
        try:
            raw = self._call(
                lambda: self._client.get_order_by_client_id(client_order_id), throttle=throttle
            )
        except _Refused as exc:
            if exc.status == 404:
                return None
            raise VenueUnavailable(f"lookup of {client_order_id} failed: {exc}") from exc
        if not isinstance(raw, AlpacaOrder):
            raise VenueDataError(f"lookup of {client_order_id} did not return an order")
        if raw.client_order_id != client_order_id:
            raise VenueDataError(
                f"asked for {client_order_id}, the venue answered with {raw.client_order_id}"
            )
        return raw

    def _call(
        self, request: Callable[[], Any], throttle: bool = True, placing: bool = False
    ) -> Any:
        """Run one SDK call and sort its failure. Never lets an SDK or HTTP exception
        out, so no caller can mistake one for something else.

        401 means the keys are refused: a setup failure, which stops trading. 403 means
        the same on any request but an order. On an order (`placing`) Alpaca also uses
        403 to refuse the order itself (not enough buying power, for one), so there it
        is a rejection, and the lookup that follows every rejection tells the two apart:
        if the keys are the problem, the lookup is refused too, and that is a stop.

        The exception raised here is built from the status and type of the original
        only, and raised outside the handler, so the original (whose text can carry
        request details) is not attached to it in any way.
        """
        if throttle:
            self._limiter.acquire()
        failure: Exception
        try:
            return request()
        except (KillSwitchEngaged, VenueDataError):  # VenueSetupError is one of these
            raise
        except APIError as exc:
            status = exc.status_code
            detail = _api_message(exc)
            if status == 401 or (status == 403 and not placing):
                failure = VenueSetupError(
                    f"Alpaca refused the keys or the account ({status}): {detail}"
                )
            elif status is not None and 400 <= status < 500 and status != 429:
                failure = _Refused(f"Alpaca refused the request ({status}): {detail}", status)
            else:
                failure = VenueUnavailable(
                    f"Alpaca did not complete the request ({status}): {detail}"
                )
        except requests.RequestException as exc:
            if isinstance(exc, requests.JSONDecodeError):  # an answer came, and it is not JSON
                failure = VenueDataError("Alpaca's answer is not JSON")
            else:  # a timeout or a connection error: no answer
                failure = VenueUnavailable(f"no answer from Alpaca ({type(exc).__name__})")
        except (ValidationError, TypeError, ValueError, KeyError, AttributeError) as exc:
            # The request went out and something came back that the SDK cannot parse.
            failure = VenueDataError(f"Alpaca's answer cannot be parsed ({type(exc).__name__})")
        except Exception as exc:
            failure = VenueUnavailable(f"no answer from Alpaca ({type(exc).__name__})")
        raise failure

    def _state(self, raw: AlpacaOrder) -> VenueOrderState:
        try:
            status = _value(raw.status)
            if status not in STATUS_MAP:
                raise ValueError(f"order status {status!r} is not one the harness handles")
            quantity = _whole(raw.qty, "order quantity")
            filled = _whole(raw.filled_qty or 0, "filled quantity")
            mapped = STATUS_MAP[status]
            if mapped is None:  # still open at the venue
                mapped = "partially_filled" if 0 < filled < quantity else "pending"
            price = _number(raw.filled_avg_price) if filled else None
            when = (raw.filled_at if mapped == "filled" else None) or raw.updated_at
            state = VenueOrderState(
                client_order_id=raw.client_order_id,
                venue_order_id=str(raw.id),
                status=mapped,
                quantity=quantity,
                filled_quantity=filled,
                filled_avg_price=price,
                updated_at=when,
            )
        except (AttributeError, TypeError, ValueError) as exc:  # ValidationError is a ValueError
            raise VenueDataError(
                f"order {getattr(raw, 'client_order_id', '?')} cannot be read: {exc}"
            ) from exc
        self._venue_ids[state.client_order_id] = state.venue_order_id
        return state


def _same_terms(existing: AlpacaOrder, order: VenueOrder) -> bool:
    """True if the order Alpaca holds under this client id is for the same symbol, side
    and quantity as the one being sent."""
    try:
        return (
            existing.symbol == order.symbol
            and _value(existing.side) == order.side
            and _whole(existing.qty, "quantity") == order.quantity
        )
    except (TypeError, ValueError):
        return False


def _api_message(exc: APIError) -> str:
    """Alpaca's own description of an error: its message if the body is the usual JSON,
    otherwise nothing. The raw body is not passed on."""
    try:
        return str(exc.message)[:200]
    except Exception:
        return "no details"


def _value(item: object) -> str:
    return str(getattr(item, "value", item))


def _number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{value!r} is not a finite number")
    return number


def _whole(value: Any, what: str) -> int:
    number = _number(value)
    if number != int(number):
        raise ValueError(f"{what} {value!r} is not a whole number of shares")
    return int(number)


def _new_york(when: dt.datetime) -> dt.datetime:
    """Alpaca's calendar gives open and close as New York wall-clock times with no zone."""
    return when.replace(tzinfo=MARKET_TZ) if when.tzinfo is None else when
