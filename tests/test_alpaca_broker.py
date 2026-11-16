"""AlpacaBroker: paper only, no keys in sight, and every failure sorted correctly.

No network: the SDK's real client runs against a fake HTTP session that answers with
canned responses, so Alpaca's JSON goes through the SDK's own parsing.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest
import requests
from alpaca.trading.enums import OrderStatus

from harness.broker.alpaca import (
    MAX_OPEN_ORDERS,
    PAPER_URL,
    STATUS_MAP,
    AlpacaBroker,
    VenueSetupError,
    _TimedSession,
)
from harness.broker.venue import (
    KillSwitchEngaged,
    Venue,
    VenueDataError,
    VenueOrder,
    VenueRejected,
    VenueUnavailable,
)
from harness.keys import KeysError
from harness.killswitch import KillSwitch, kill
from harness.models import MARKET_TZ
from harness.ratelimit import RateLimiter
from tests.test_orders import CFG

KEY_ID = "PKSENTINELKEYID0001"
SECRET = "sentinel-secret-value-do-not-print"
HARNESS = Path(__file__).resolve().parent.parent / "harness"
T0 = "2024-01-03T14:30:05Z"


class FakeSession:
    """Stands in for `requests.Session`. Answers are queued per (method, path); a path
    with nothing queued answers 404, as Alpaca does for an order it has never seen."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._answers: dict[tuple[str, str], list[Any]] = {}

    def on(self, method: str, path: str, status: int = 200, body: Any = None) -> None:
        self._answers.setdefault((method, path), []).append((status, body))

    def fail(self, method: str, path: str, error: Exception) -> None:
        self._answers.setdefault((method, path), []).append(error)

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        path = url.removeprefix(PAPER_URL)
        queued = self._answers.get((method, path))
        answer = queued.pop(0) if queued else (404, {"code": 40410000, "message": "not found"})
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        response = requests.Response()
        response.status_code = status
        response.url = url
        response._content = b"" if body is None else json.dumps(body).encode()
        return response

    def seen(self) -> list[tuple[str, str]]:
        return [(c["method"], c["url"].removeprefix(PAPER_URL)) for c in self.calls]


def env_file(tmp_path: Path, key_id: str = KEY_ID) -> Path:
    path = tmp_path / ".env"
    path.write_text(f"ALPACA_PAPER_API_KEY={key_id}\nALPACA_PAPER_SECRET_KEY={SECRET}\n")
    return path


class Rig:
    def __init__(self, tmp_path: Path, max_per_minute: int = 1_000) -> None:
        self.session = FakeSession()
        self.kill = KillSwitch(tmp_path / "KILL")
        self.slept: list[float] = []
        self.clock = 0.0

        def sleep(seconds: float) -> None:
            self.slept.append(seconds)
            self.clock += seconds

        self.limiter = RateLimiter(max_per_minute, clock=lambda: self.clock, sleep=sleep)
        self.broker = AlpacaBroker(
            CFG, self.kill, self.limiter, dotenv_path=env_file(tmp_path), session=self.session
        )


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def order_json(
    client_order_id: str = "0a1b2c3d-20240102-AAPL-x-a1",
    status: str = "new",
    qty: str = "10",
    filled: str = "0",
    price: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, client_order_id)),
        "client_order_id": client_order_id,
        "created_at": "2024-01-03T13:00:00Z",
        "updated_at": T0,
        "submitted_at": "2024-01-03T13:00:00Z",
        "filled_at": None,
        "order_class": "simple",
        "time_in_force": "opg",
        "status": status,
        "extended_hours": False,
        "symbol": "AAPL",
        "qty": qty,
        "filled_qty": filled,
        "filled_avg_price": price,
        "side": "sell",
        "type": "market",
        "order_type": "market",
        **extra,
    }


def exit_order(quantity: int = 10) -> VenueOrder:
    return VenueOrder(
        client_order_id="0a1b2c3d-20240102-AAPL-x-a1", symbol="AAPL", side="sell",
        quantity=quantity, type="market", time_in_force="opg",
    )  # fmt: skip


def entry_order() -> VenueOrder:
    return VenueOrder(
        client_order_id="0a1b2c3d-20240102-MSFT-e-a1", symbol="MSFT", side="buy",
        quantity=3, type="limit", limit_price=306.0, time_in_force="day",
    )  # fmt: skip


BY_CLIENT_ID = "/v2/orders:by_client_order_id"
ERROR = {"code": 40310000, "message": "insufficient buying power"}


# --------------------------------------------------------------------------- paper only


def test_client_is_the_paper_client_with_retries_off_and_a_timeout(rig: Rig) -> None:
    venue: Venue = rig.broker  # satisfies the protocol (checked by pyright)
    client = rig.broker._client
    assert str(getattr(client._base_url, "value", client._base_url)) == PAPER_URL
    assert client._retry == 0  # the SDK's own retries are off
    assert isinstance(client._session, _TimedSession)
    rig.session.on("GET", "/v2/clock", body={
        "timestamp": "2024-01-03T09:00:00-05:00", "is_open": False,
        "next_open": "2024-01-03T09:30:00-05:00", "next_close": "2024-01-03T16:00:00-05:00",
    })  # fmt: skip
    venue.now()
    (call,) = rig.session.calls
    assert call["url"] == PAPER_URL + "/v2/clock"
    assert call["timeout"] == CFG.request_timeout_seconds
    assert call["allow_redirects"] is False
    assert repr(rig.broker) == "AlpacaBroker(paper)"


def test_there_is_no_way_to_ask_for_anything_but_paper() -> None:
    import inspect

    parameters = set(inspect.signature(AlpacaBroker.__init__).parameters)
    assert parameters == {"self", "cfg", "kill", "limiter", "dotenv_path", "session"}


def test_a_request_to_any_other_host_is_refused() -> None:
    inner = FakeSession()
    timed = _TimedSession(inner, 10.0)
    for url in (
        "https://api.alpaca.markets/v2/orders",  # the live host
        "https://paper-api.alpaca.markets.evil.example/v2/orders",
        "http://paper-api.alpaca.markets/v2/orders",  # not https
        "https://paper-api.alpaca.markets",  # no path
    ):
        with pytest.raises(VenueSetupError, match="not to the Alpaca paper host"):
            timed.request("POST", url)
    assert inner.calls == []


def test_key_that_is_not_a_paper_key_is_refused(tmp_path: Path) -> None:
    kill_switch = KillSwitch(tmp_path / "KILL")
    for key_id in ("AKLIVEKEYID00000001", "pksentinel", "XPK123"):
        with pytest.raises(VenueSetupError, match="does not look like an Alpaca paper key") as info:
            AlpacaBroker(
                CFG, kill_switch, RateLimiter(100), dotenv_path=env_file(tmp_path, key_id),
                session=FakeSession(),
            )  # fmt: skip
        assert key_id not in str(info.value) and SECRET not in str(info.value)
    with pytest.raises(KeysError):
        AlpacaBroker(CFG, kill_switch, RateLimiter(100), dotenv_path=tmp_path / "missing.env")


def test_sdk_that_moved_its_internals_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alpaca.trading.client import TradingClient

    real_init = TradingClient.__init__

    def init_without_retry(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        del self._retry

    monkeypatch.setattr(TradingClient, "__init__", init_without_retry)
    with pytest.raises(VenueSetupError, match="no TradingClient._retry"):
        Rig(tmp_path)


def test_sdk_client_that_is_not_pointed_at_paper_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alpaca.trading.client import TradingClient

    real_init = TradingClient.__init__

    def init_elsewhere(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self._base_url = "https://example.invalid"  # as if the SDK's default ever changed

    monkeypatch.setattr(TradingClient, "__init__", init_elsewhere)
    with pytest.raises(VenueSetupError, match="not pointed at the Alpaca paper host"):
        Rig(tmp_path)


def test_harness_source_has_no_way_to_reach_live() -> None:
    live_host = re.compile(r"(?<!paper-)api\.alpaca\.markets")
    forbidden = ("ALPACA_LIVE", "paper=False", "url_override", "TRADING_LIVE", "BaseURL")
    checked = 0
    for path in HARNESS.rglob("*.py"):
        text = path.read_text()
        checked += 1
        assert not live_host.search(text), f"{path.name} names the live host"
        for word in forbidden:
            assert word not in text, f"{path.name} contains {word}"
    assert checked > 15
    assert "paper=True" in (HARNESS / "broker" / "alpaca.py").read_text()


# --------------------------------------------------------------------------- keys stay hidden


def test_keys_never_appear_in_errors_or_reprs(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    session = rig.session
    leaky = requests.ConnectionError(f"failed with headers APCA-API-SECRET-KEY={SECRET} {KEY_ID}")
    session.fail("POST", "/v2/orders", leaky)
    session.on("GET", "/v2/account", 500, {"code": 50010000, "message": f"oops {SECRET}"[:8]})
    session.on("GET", "/v2/positions", 200, [{"symbol": "AAPL"}])  # unparsable
    session.fail("GET", "/v2/clock", requests.Timeout(f"timed out {SECRET}"))
    texts = [repr(rig.broker), str(rig.broker), repr(rig.broker.__dict__.keys())]
    for call in (
        lambda: rig.broker.place(exit_order()),
        rig.broker.account,
        rig.broker.positions,
        rig.broker.now,
    ):
        with pytest.raises((VenueUnavailable, VenueDataError)) as info:
            call()
        error = info.value
        texts += [str(error), repr(error), repr(error.__cause__), repr(error.__context__)]
    for text in texts:
        assert SECRET not in text and KEY_ID not in text
    # The keys did go where they belong: the request headers, and nowhere else.
    headers = session.calls[0]["headers"]
    assert headers["APCA-API-KEY-ID"] == KEY_ID and headers["APCA-API-SECRET-KEY"] == SECRET
    assert SECRET not in json.dumps(
        [{k: v for k, v in c.items() if k != "headers"} for c in session.calls], default=str
    )


# --------------------------------------------------------------------------- place


def test_place_sends_the_order_alpaca_expects(rig: Rig) -> None:
    rig.session.on("POST", "/v2/orders", body=order_json())
    state = rig.broker.place(exit_order())
    assert rig.session.calls[0]["json"] == {
        "symbol": "AAPL", "qty": 10.0, "side": "sell", "type": "market",
        "time_in_force": "opg", "client_order_id": "0a1b2c3d-20240102-AAPL-x-a1",
    }  # fmt: skip
    assert (state.status, state.quantity, state.filled_quantity) == ("pending", 10, 0)
    assert state.client_order_id == "0a1b2c3d-20240102-AAPL-x-a1"

    entry = order_json("0a1b2c3d-20240102-MSFT-e-a1", symbol="MSFT", side="buy", qty="3")
    rig.session.on("POST", "/v2/orders", body=entry)
    rig.broker.place(entry_order())
    assert rig.session.calls[1]["json"] == {
        "symbol": "MSFT", "qty": 3.0, "side": "buy", "type": "limit", "time_in_force": "day",
        "limit_price": 306.0, "client_order_id": "0a1b2c3d-20240102-MSFT-e-a1",
    }  # fmt: skip


def test_latch_engaged_makes_place_raise_without_any_request(rig: Rig) -> None:
    rig.kill.engage("test", "operator:cli")
    with pytest.raises(KillSwitchEngaged):
        rig.broker.place(exit_order())
    assert rig.session.calls == []


def test_the_flatten_pass_lets_a_sell_through_the_latch_and_never_a_buy(rig: Rig) -> None:
    rig.kill.engage("test", "operator:cli")
    rig.session.on("POST", "/v2/orders", body=order_json())
    with rig.kill.flattening():
        assert rig.broker.place(exit_order()).client_order_id == "0a1b2c3d-20240102-AAPL-x-a1"
        with pytest.raises(KillSwitchEngaged):
            rig.broker.place(entry_order())
    assert len(rig.session.calls) == 1  # the sell; the buy never left
    with pytest.raises(KillSwitchEngaged):
        rig.broker.place(exit_order())


def test_latched_place_does_not_even_wait_for_the_rate_limiter(tmp_path: Path) -> None:
    rig = Rig(tmp_path, max_per_minute=1)
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    rig.broker.account()  # uses the only token
    rig.kill.engage("stop", "operator:cli")
    with pytest.raises(KillSwitchEngaged):
        rig.broker.place(exit_order())
    assert rig.slept == []  # refused at once, without a minute's wait for a token


def test_latch_set_while_place_waits_for_the_rate_limiter_stops_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path, max_per_minute=1)
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    rig.broker.account()  # uses the only token
    real_sleep = rig.limiter._sleep

    def sleep_then_kill(seconds: float) -> None:
        real_sleep(seconds)
        rig.kill.engage("operator stop", "operator:cli")

    rig.limiter._sleep = sleep_then_kill
    with pytest.raises(KillSwitchEngaged):
        rig.broker.place(exit_order())
    assert rig.session.seen() == [("GET", "/v2/account")]  # the order never went out


def test_refused_keys_stop_trading_instead_of_rejecting_one_order(rig: Rig) -> None:
    denied = {"code": 40110000, "message": "request is not authorized"}
    rig.session.on("POST", "/v2/orders", 401, denied)
    with pytest.raises(VenueSetupError, match="refused the keys or the account"):
        rig.broker.place(exit_order())
    assert rig.session.seen() == [("POST", "/v2/orders")]
    # 403 on a read or a cancel is the keys too.
    for method, path, call in (
        ("GET", "/v2/account", rig.broker.account),
        ("GET", "/v2/orders", rig.broker.open_orders),
        ("DELETE", "/v2/orders", rig.broker.cancel_all),
        ("GET", BY_CLIENT_ID, lambda: rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")),
    ):
        rig.session.on(method, path, 403, {"code": 40310000, "message": "forbidden"})
        with pytest.raises(VenueSetupError):
            call()


def test_403_on_an_order_is_told_apart_by_the_lookup_that_follows(rig: Rig) -> None:
    # Alpaca uses 403 to refuse an order (not enough buying power) and to refuse keys.
    # If the lookup after it is refused as well, it was the keys: stop.
    rig.session.on("POST", "/v2/orders", 403, {"code": 40310000, "message": "forbidden"})
    rig.session.on("GET", BY_CLIENT_ID, 403, {"code": 40310000, "message": "forbidden"})
    with pytest.raises(VenueSetupError):
        rig.broker.place(entry_order())


def test_request_that_is_not_to_the_paper_host_stops_trading(rig: Rig) -> None:
    rig.broker._client._base_url = "https://example.invalid"  # as if something moved it
    for call in (lambda: rig.broker.place(exit_order()), rig.broker.account, rig.broker.cancel_all):
        with pytest.raises(VenueSetupError, match="not to the Alpaca paper host"):
            call()
    assert rig.session.calls == []
    assert issubclass(VenueSetupError, VenueDataError)  # so the order manager stops on it


@pytest.mark.parametrize("status", [400, 403, 422])
def test_refused_order_that_does_not_exist_is_a_rejection(rig: Rig, status: int) -> None:
    rig.session.on("POST", "/v2/orders", status, ERROR)
    with pytest.raises(VenueRejected, match="insufficient buying power"):
        rig.broker.place(entry_order())
    # The rejection was not believed until a lookup said the order is not there.
    assert rig.session.seen() == [("POST", "/v2/orders"), ("GET", BY_CLIENT_ID)]
    assert rig.session.calls[1]["params"] == {"client_order_id": "0a1b2c3d-20240102-MSFT-e-a1"}


def test_refusal_of_a_repeated_client_id_returns_the_existing_order(rig: Rig) -> None:
    duplicate = {"code": 42210000, "message": "client_order_id must be unique"}
    rig.session.on("POST", "/v2/orders", 422, duplicate)
    rig.session.on(
        "GET", BY_CLIENT_ID, body=order_json(status="partially_filled", filled="4", price="101")
    )
    state = rig.broker.place(exit_order())
    assert (state.status, state.filled_quantity, state.filled_avg_price) == (
        "partially_filled",
        4,
        101.0,
    )


@pytest.mark.parametrize(
    "different", [{"qty": "9"}, {"side": "buy"}, {"symbol": "MSFT"}, {"qty": "10.5"}]
)
def test_existing_order_with_other_terms_is_not_adopted(
    rig: Rig, different: dict[str, str]
) -> None:
    rig.session.on("POST", "/v2/orders", 422, {"code": 42210000, "message": "must be unique"})
    rig.session.on("GET", BY_CLIENT_ID, body=order_json(**different))
    # Not a rejection: that would close our order while a different one is live under
    # its id. It is data that cannot be trusted, which stops trading.
    with pytest.raises(VenueDataError, match="already exists at the venue with different terms"):
        rig.broker.place(exit_order())


def test_refused_order_whose_lookup_fails_is_unknown_not_rejected(rig: Rig) -> None:
    rig.session.on("POST", "/v2/orders", 422, ERROR)
    rig.session.on("GET", BY_CLIENT_ID, 500, {"code": 50010000, "message": "internal"})
    with pytest.raises(VenueUnavailable):
        rig.broker.place(exit_order())


@pytest.mark.parametrize(
    "failure",
    [429, 500, 502, 503, 504, requests.Timeout("slow"), requests.ConnectionError("reset")],
)
def test_no_answer_is_unknown_and_is_never_retried_here(rig: Rig, failure: Any) -> None:
    if isinstance(failure, Exception):
        rig.session.fail("POST", "/v2/orders", failure)
    else:
        rig.session.on("POST", "/v2/orders", failure, {"code": failure * 100000, "message": "x"})
    rig.session.on("POST", "/v2/orders", body=order_json())  # a retry would get this
    with pytest.raises(VenueUnavailable):
        rig.broker.place(exit_order())
    assert rig.session.seen() == [("POST", "/v2/orders")]  # one attempt, no silent retry
    assert rig.slept == []


@pytest.mark.parametrize("body", [{"id": "not-a-uuid"}, {}, [], "ok", order_json(status="frozen")])
def test_answer_that_cannot_be_parsed_stops_trading(rig: Rig, body: Any) -> None:
    rig.session.on("POST", "/v2/orders", body=body)
    with pytest.raises(VenueDataError):
        rig.broker.place(exit_order())


def test_error_body_that_is_not_json_is_still_sorted_by_its_status(rig: Rig) -> None:
    rig.session.on("POST", "/v2/orders", 503, "<html>bad gateway</html>")
    with pytest.raises(VenueUnavailable, match="no details"):
        rig.broker.place(exit_order())


# --------------------------------------------------------------------------- order states


OPEN_STATUSES = [
    "new", "accepted", "pending_new", "accepted_for_bidding", "held", "pending_review",
    "pending_cancel", "pending_replace", "done_for_day", "stopped", "suspended", "calculated",
]  # fmt: skip


def test_every_alpaca_status_is_handled_explicitly() -> None:
    sdk = {status.value for status in OrderStatus}
    assert len(sdk) == 18  # a status the SDK adds must be added to the table by hand
    assert set(STATUS_MAP) | {"replaced"} == sdk
    final = {status for status, mapped in STATUS_MAP.items() if mapped is not None}
    assert final == {"filled", "canceled", "expired", "rejected"}
    assert all(STATUS_MAP[status] == status for status in final)
    open_statuses = {status for status, mapped in STATUS_MAP.items() if mapped is None}
    assert open_statuses == {*OPEN_STATUSES, "partially_filled"}


# The SDK warns when Alpaca sends `pending_review`, which it no longer should.
@pytest.mark.filterwarnings("ignore:OrderStatus.PENDING_REVIEW:DeprecationWarning")
@pytest.mark.parametrize("status", OPEN_STATUSES)
def test_status_that_is_not_final_is_never_reported_as_final(rig: Rig, status: str) -> None:
    rig.session.on("GET", BY_CLIENT_ID, body=order_json(status=status))
    rig.session.on("GET", BY_CLIENT_ID, body=order_json(status=status, filled="4", price="101.5"))
    untouched = rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")
    partly = rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")
    assert untouched is not None and partly is not None
    assert (untouched.status, untouched.is_terminal) == ("pending", False)
    assert (partly.status, partly.filled_quantity, partly.is_terminal) == (
        "partially_filled",
        4,
        False,
    )


@pytest.mark.parametrize(
    ("status", "filled", "price", "expected"),
    [
        ("partially_filled", "4", "101.5", "partially_filled"),
        ("filled", "10", "101.5", "filled"),
        ("canceled", "0", None, "canceled"),
        ("canceled", "4", "101.5", "canceled"),
        ("expired", "0", None, "expired"),
        ("rejected", "0", None, "rejected"),
    ],
)
def test_final_and_partial_statuses(
    rig: Rig, status: str, filled: str, price: str | None, expected: str
) -> None:
    rig.session.on("GET", BY_CLIENT_ID, body=order_json(status=status, filled=filled, price=price))
    state = rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")
    assert state is not None
    assert (state.status, state.filled_quantity) == (expected, int(filled))
    assert state.filled_avg_price == (float(price) if price else None)
    assert state.is_terminal == (expected != "partially_filled")


@pytest.mark.parametrize(
    "bad",
    [
        {"status": "replaced"},  # the harness never replaces an order
        {"status": "frozen"},  # a status Alpaca might add
        {"qty": "10.5"},
        {"qty": None},
        {"filled": "0.5"},
        {"filled": "11"},
        {"status": "filled", "filled": "4", "price": "100"},  # filled, but not in full
        {"status": "rejected", "filled": "4", "price": "100"},
        {"filled": "4", "price": None},  # fills with no price
        {"filled": "4", "price": "0"},
        {"filled": "4", "price": "NaN"},
        {"client_order_id": "someone-elses-order"},
    ],
)
def test_order_that_cannot_be_trusted_is_an_error_not_a_guess(
    rig: Rig, bad: dict[str, Any]
) -> None:
    rig.session.on("GET", BY_CLIENT_ID, body=order_json(**bad))
    with pytest.raises(VenueDataError):
        rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")


def test_fill_time_is_the_execution_time(rig: Rig) -> None:
    filled = order_json(status="filled", filled="10", price="104", filled_at="2024-01-03T14:30:01Z")
    rig.session.on("GET", BY_CLIENT_ID, body=filled)
    state = rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")
    assert state is not None
    assert state.updated_at == dt.datetime(2024, 1, 3, 14, 30, 1, tzinfo=dt.UTC)
    assert state.updated_at.astimezone(MARKET_TZ).date() == dt.date(2024, 1, 3)


def test_order_alpaca_has_never_seen_is_none(rig: Rig) -> None:
    assert rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1") is None
    rig.session.on("GET", BY_CLIENT_ID, 422, {"code": 42210000, "message": "bad request"})
    with pytest.raises(VenueUnavailable):  # any other refusal of a lookup is not "never seen"
        rig.broker.get_order("0a1b2c3d-20240102-AAPL-x-a1")


def test_open_orders_asks_for_every_open_order(rig: Rig) -> None:
    rig.session.on("GET", "/v2/orders", body=[
        order_json("0a1b2c3d-20240102-AAPL-x-a1"),
        order_json("0a1b2c3d-20240102-TSLA-x-a1", status="partially_filled", filled="2", price="9"),
    ])  # fmt: skip
    states = rig.broker.open_orders()
    assert [(s.client_order_id, s.status) for s in states] == [
        ("0a1b2c3d-20240102-AAPL-x-a1", "pending"),
        ("0a1b2c3d-20240102-TSLA-x-a1", "partially_filled"),
    ]
    assert rig.session.calls[0]["params"] == {"status": "open", "limit": MAX_OPEN_ORDERS}
    rig.session.on("GET", "/v2/orders", body=[])
    assert rig.broker.open_orders() == []


def test_list_that_may_be_cut_short_is_an_error(rig: Rig) -> None:
    full = [order_json(f"0a1b2c3d-20240102-S{n}-x-a1") for n in range(MAX_OPEN_ORDERS)]
    rig.session.on("GET", "/v2/orders", body=full)
    with pytest.raises(VenueDataError, match="may be cut short"):
        rig.broker.open_orders()
    rig.session.on("GET", "/v2/orders", body=full[:-1])
    assert len(rig.broker.open_orders()) == MAX_OPEN_ORDERS - 1


# --------------------------------------------------------------------------- cancel


def test_cancel_uses_alpacas_order_id_and_works_while_latched(rig: Rig) -> None:
    placed = order_json()
    rig.session.on("POST", "/v2/orders", body=placed)
    rig.broker.place(exit_order())
    rig.kill.engage("stop", "operator:cli")
    rig.session.on("DELETE", f"/v2/orders/{placed['id']}", 204)
    rig.broker.cancel("0a1b2c3d-20240102-AAPL-x-a1")
    assert rig.session.seen()[-1] == ("DELETE", f"/v2/orders/{placed['id']}")
    assert len(rig.session.calls) == 2  # the id was remembered: no lookup needed


def test_cancel_looks_the_order_up_when_it_has_not_seen_it(rig: Rig) -> None:
    known = order_json()
    rig.session.on("GET", BY_CLIENT_ID, body=known)
    rig.session.on("DELETE", f"/v2/orders/{known['id']}", 204)
    rig.broker.cancel("0a1b2c3d-20240102-AAPL-x-a1")
    assert rig.session.seen() == [("GET", BY_CLIENT_ID), ("DELETE", f"/v2/orders/{known['id']}")]
    with pytest.raises(VenueRejected, match="no order"):
        rig.broker.cancel("0a1b2c3d-20240102-ZZZ-x-a1")  # never seen


def test_cancel_of_a_final_order_does_nothing_and_of_a_missing_one_raises(rig: Rig) -> None:
    known = order_json()
    rig.session.on("GET", BY_CLIENT_ID, body=known)
    rig.session.on(
        "DELETE", f"/v2/orders/{known['id']}", 422, {"code": 42210000, "message": "final"}
    )
    rig.broker.cancel("0a1b2c3d-20240102-AAPL-x-a1")  # already final: nothing to do
    rig.session.on(
        "DELETE", f"/v2/orders/{known['id']}", 404, {"code": 40410000, "message": "gone"}
    )
    with pytest.raises(VenueRejected):
        rig.broker.cancel("0a1b2c3d-20240102-AAPL-x-a1")
    rig.session.on("DELETE", f"/v2/orders/{known['id']}", 500, {"code": 50010000, "message": "x"})
    with pytest.raises(VenueUnavailable):
        rig.broker.cancel("0a1b2c3d-20240102-AAPL-x-a1")


def test_cancels_are_not_held_back_by_the_rate_limiter(tmp_path: Path) -> None:
    rig = Rig(tmp_path, max_per_minute=1)
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    rig.broker.account()  # the only token is gone
    known = order_json()
    rig.session.on("GET", BY_CLIENT_ID, body=known)
    rig.session.on("DELETE", f"/v2/orders/{known['id']}", 204)
    rig.session.on("DELETE", "/v2/orders", 207, [])
    rig.broker.cancel("0a1b2c3d-20240102-AAPL-x-a1")
    rig.broker.cancel_all()
    assert rig.slept == []
    # A read does wait: the limiter slows a call, it never drops one.
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    rig.broker.account()
    assert rig.slept == [60.0] and rig.session.seen()[-1] == ("GET", "/v2/account")


def test_kill_run_while_already_latched_cancels_through_the_adapter(rig: Rig) -> None:
    rig.kill.engage("reconciliation stop", "auto:reconcile")
    rig.session.on("DELETE", "/v2/orders", 207, [{"id": str(uuid.uuid4()), "status": 200}])
    rig.session.on("GET", "/v2/orders", body=[])
    state = kill(rig.kill, "operator stop", "operator:cli", venue=rig.broker, sleep=lambda s: None)
    assert state.cancel_confirmed
    assert rig.session.seen() == [("DELETE", "/v2/orders"), ("GET", "/v2/orders")]


# --------------------------------------------------------------------------- account


ACCOUNT = {
    "id": str(uuid.uuid4()), "account_number": "PA3TEST00001", "status": "ACTIVE",
    "cash": "9100.50", "buying_power": "9100.50", "equity": "10100.50",
    "trading_blocked": False, "account_blocked": False, "shorting_enabled": True,
    "multiplier": "1",
}  # fmt: skip
SETTINGS = {
    "fractional_trading": True, "max_margin_multiplier": "1", "no_shorting": True,
    "suspend_trade": False, "trade_confirm_email": "all", "ptp_no_exception_entry": False,
}  # fmt: skip


def test_account_is_read_from_alpaca(rig: Rig) -> None:
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    account = rig.broker.account()
    assert (account.account_id, account.cash, account.equity) == ("PA3TEST00001", 9100.5, 10100.5)
    assert (account.trading_blocked, account.shorting_enabled, account.margin_multiplier) == (
        False,
        True,
        1.0,
    )


@pytest.mark.parametrize(
    "change",
    [{"trading_blocked": True}, {"account_blocked": True}, {"status": "ACCOUNT_UPDATED"}],
)
def test_anything_but_a_plainly_active_account_counts_as_blocked(
    rig: Rig, change: dict[str, Any]
) -> None:
    rig.session.on("GET", "/v2/account", body={**ACCOUNT, **change})
    assert rig.broker.account().trading_blocked


@pytest.mark.parametrize("missing", ["cash", "buying_power", "equity", "multiplier"])
def test_account_with_a_missing_number_is_an_error(rig: Rig, missing: str) -> None:
    rig.session.on("GET", "/v2/account", body={k: v for k, v in ACCOUNT.items() if k != missing})
    with pytest.raises(VenueDataError):
        rig.broker.account()


def test_verify_account_passes_a_safely_set_up_account(rig: Rig) -> None:
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    rig.session.on("GET", "/v2/account/configurations", body=SETTINGS)
    account = rig.broker.verify_account()
    # Returned as Alpaca reported it: nothing is overwritten with what the settings say.
    assert (account.shorting_enabled, account.margin_multiplier, account.cash) == (
        True,
        1.0,
        9100.5,
    )
    assert {m for m, _ in rig.session.seen()} == {"GET"}  # it checks; it changes nothing


@pytest.mark.parametrize(
    ("account", "settings", "problem"),
    [
        ({}, {"no_shorting": False}, "shorting is allowed"),
        ({}, {"max_margin_multiplier": "2"}, "margin multiplier is 2"),
        ({}, {"max_margin_multiplier": "4"}, "margin multiplier is 4"),
        ({}, {"suspend_trade": True}, "suspended"),
        ({"trading_blocked": True}, {}, "blocked from trading"),
        ({"status": "INACTIVE"}, {}, "blocked from trading"),
        # The setting says 1, but the account itself still reports 2.
        ({"multiplier": "2"}, {}, "reports a margin multiplier of 2 although"),
    ],
)
def test_verify_account_refuses_an_account_that_is_not_set_up_safely(
    rig: Rig, account: dict[str, Any], settings: dict[str, Any], problem: str
) -> None:
    rig.session.on("GET", "/v2/account", body={**ACCOUNT, **account})
    rig.session.on("GET", "/v2/account/configurations", body={**SETTINGS, **settings})
    with pytest.raises(VenueSetupError, match=problem):
        rig.broker.verify_account()


def test_verify_account_lists_every_problem_at_once(rig: Rig) -> None:
    rig.session.on("GET", "/v2/account", body=ACCOUNT)
    unsafe = {**SETTINGS, "no_shorting": False, "max_margin_multiplier": "2"}
    rig.session.on("GET", "/v2/account/configurations", body=unsafe)
    with pytest.raises(VenueSetupError, match="shorting is allowed.*margin multiplier is 2"):
        rig.broker.verify_account()


def position_json(symbol: str, qty: str, side: str = "long") -> dict[str, Any]:
    return {
        "asset_id": str(uuid.uuid4()), "symbol": symbol, "exchange": "NASDAQ",
        "asset_class": "us_equity", "avg_entry_price": "100.25", "qty": qty, "side": side,
        "cost_basis": "1002.50",
    }  # fmt: skip


def test_positions_keep_their_sign(rig: Rig) -> None:
    rig.session.on("GET", "/v2/positions", body=[
        position_json("AAPL", "10"),
        position_json("TSLA", "-5", "short"),
        position_json("NVDA", "5", "short"),  # reported unsigned: still a short
    ])  # fmt: skip
    assert [(p.symbol, p.quantity, p.avg_entry_price) for p in rig.broker.positions()] == [
        ("AAPL", 10, 100.25),
        ("TSLA", -5, 100.25),
        ("NVDA", -5, 100.25),
    ]
    rig.session.on("GET", "/v2/positions", body=[])
    assert rig.broker.positions() == []


@pytest.mark.parametrize("qty", ["0.5", "10.25", "0", "abc"])
def test_position_that_is_not_whole_shares_is_an_error(rig: Rig, qty: str) -> None:
    rig.session.on("GET", "/v2/positions", body=[position_json("AAPL", qty)])
    with pytest.raises(VenueDataError):
        rig.broker.positions()


def test_clock_and_calendar(rig: Rig) -> None:
    rig.session.on("GET", "/v2/clock", body={
        "timestamp": "2024-01-03T09:25:00.123-05:00", "is_open": False,
        "next_open": "2024-01-03T09:30:00-05:00", "next_close": "2024-01-03T16:00:00-05:00",
    })  # fmt: skip
    now = rig.broker.now()
    assert now.utcoffset() is not None
    assert now == dt.datetime(2024, 1, 3, 14, 25, 0, 123000, tzinfo=dt.UTC)

    rig.session.on("GET", "/v2/calendar", body=[
        {"date": "2024-01-03", "open": "09:30", "close": "16:00"},
        {"date": "2024-07-03", "open": "09:30", "close": "13:00"},  # a half day
    ])  # fmt: skip
    first, half = rig.broker.sessions(dt.date(2024, 1, 3), dt.date(2024, 7, 3))
    assert rig.session.calls[1]["params"] == {"start": "2024-01-03", "end": "2024-07-03"}
    assert first.open == dt.datetime(2024, 1, 3, 9, 30, tzinfo=MARKET_TZ)
    assert first.close == dt.datetime(2024, 1, 3, 16, 0, tzinfo=MARKET_TZ)
    assert half.close == dt.datetime(2024, 7, 3, 13, 0, tzinfo=MARKET_TZ)
    assert half.open.utcoffset() == dt.timedelta(hours=-4)  # summer time, from the zone


def test_every_read_goes_through_the_rate_limiter(tmp_path: Path) -> None:
    rig = Rig(tmp_path, max_per_minute=2)
    for _ in range(3):
        rig.session.on("GET", "/v2/account", body=ACCOUNT)
        rig.broker.account()
    assert rig.slept == [60.0]  # the third call waited; none was dropped
    assert len(rig.session.calls) == 3
