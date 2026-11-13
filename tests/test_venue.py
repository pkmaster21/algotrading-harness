"""Venue models: the shapes the harness refuses to send to, or accept from, a venue."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from harness.broker.venue import MarketSession, VenueOrder, VenueOrderState, VenuePosition
from harness.models import MARKET_TZ

NOW = dt.datetime(2024, 1, 3, 9, 31, tzinfo=MARKET_TZ)


def order(**changes: Any) -> VenueOrder:
    base: dict[str, Any] = {
        "client_order_id": "plumbing-20240102-AAPL-e-a1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "type": "limit",
        "limit_price": 102.0,
        "time_in_force": "day",
    }
    return VenueOrder(**{**base, **changes})


def state(**changes: Any) -> VenueOrderState:
    base: dict[str, Any] = {
        "client_order_id": "c1",
        "venue_order_id": "v1",
        "status": "pending",
        "quantity": 10,
        "filled_quantity": 0,
        "filled_avg_price": None,
        "updated_at": NOW,
    }
    return VenueOrderState(**{**base, **changes})


# --------------------------------------------------------------------------- VenueOrder


def test_the_three_order_shapes_the_harness_sends() -> None:
    order()  # entry: limit buy, day
    order(side="sell", type="market", limit_price=None, time_in_force="opg")  # exit in the auction
    order(side="sell", type="market", limit_price=None)  # exit after the open


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"type": "market", "limit_price": None}, "a buy must be a limit order"),
        ({"time_in_force": "opg"}, "opg is for market sells only"),
        ({"side": "sell", "time_in_force": "opg"}, "opg is for market sells only"),  # limit sell
        ({"limit_price": None}, "needs a positive limit_price"),
        ({"limit_price": 0.0}, "needs a positive limit_price"),
        ({"limit_price": -1.0}, "needs a positive limit_price"),
        ({"limit_price": 102.005}, "whole cents"),
        ({"side": "sell", "type": "market"}, "a market order has no limit_price"),
        ({"quantity": 0}, "quantity"),
        ({"quantity": 1.5}, "quantity"),
        ({"symbol": "aapl"}, "symbol"),
        ({"client_order_id": ""}, "client_order_id"),
        ({"client_order_id": "x" * 49}, "client_order_id"),
        ({"type": "stop"}, "type"),
        ({"time_in_force": "gtc"}, "time_in_force"),
    ],
)
def test_order_shapes_the_harness_never_sends_are_refused(
    changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        order(**changes)


def test_limit_price_in_cents_survives_float_representation() -> None:
    for price in (0.01, 19.99, 102.03, 4999.99, 100.0 * 1.02):
        order(limit_price=round(price, 2))


# --------------------------------------------------------------------------- VenueOrderState


def test_valid_states() -> None:
    assert not state().is_terminal
    assert not state(status="partially_filled", filled_quantity=4, filled_avg_price=10).is_terminal
    assert state(status="filled", filled_quantity=10, filled_avg_price=10.0).is_terminal
    assert state(status="canceled").is_terminal
    assert state(status="canceled", filled_quantity=4, filled_avg_price=10.0).is_terminal
    assert state(status="expired", filled_quantity=4, filled_avg_price=10.0).is_terminal
    assert state(status="rejected", reject_reason="insufficient buying power").is_terminal
    # A cancel in flight after a partial fill is still pending, with fills.
    state(status="pending", filled_quantity=4, filled_avg_price=10.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"filled_quantity": 11, "filled_avg_price": 10.0}, "exceeds quantity"),
        ({"status": "filled", "filled_quantity": 9, "filled_avg_price": 10.0}, "filled in full"),
        ({"status": "filled"}, "filled in full"),
        ({"status": "partially_filled"}, "some, not all"),
        (
            {"status": "partially_filled", "filled_quantity": 10, "filled_avg_price": 10.0},
            "some, not all",
        ),
        ({"status": "rejected", "filled_quantity": 1, "filled_avg_price": 10.0}, "rejected"),
        ({"filled_quantity": 4}, "positive filled_avg_price"),
        ({"filled_quantity": 4, "filled_avg_price": 0.0}, "positive filled_avg_price"),
        ({"filled_avg_price": 10.0}, "without fills"),
        ({"updated_at": NOW.replace(tzinfo=None)}, "timezone-aware"),
        ({"status": "replaced"}, "status"),
        ({"quantity": 0}, "quantity"),
    ],
)
def test_impossible_states_are_refused(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        state(**changes)


# --------------------------------------------------------------------------- others


def test_position_and_session_rules() -> None:
    VenuePosition(symbol="AAPL", quantity=-5, avg_entry_price=10.0)  # a short is representable
    with pytest.raises(ValueError, match="nonzero"):
        VenuePosition(symbol="AAPL", quantity=0, avg_entry_price=10.0)

    day = dt.date(2024, 1, 3)
    opens = dt.datetime.combine(day, dt.time(9, 30), MARKET_TZ)
    closes = dt.datetime.combine(day, dt.time(16, 0), MARKET_TZ)
    MarketSession(date=day, open=opens, close=closes)
    MarketSession(date=day, open=opens.astimezone(dt.UTC), close=closes.astimezone(dt.UTC))
    with pytest.raises(ValueError, match="before it closes"):
        MarketSession(date=day, open=closes, close=opens)
    with pytest.raises(ValueError, match="New York date"):
        MarketSession(date=day + dt.timedelta(days=1), open=opens, close=closes)
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketSession(date=day, open=opens.replace(tzinfo=None), close=closes)
