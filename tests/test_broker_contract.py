"""Runs the shared Broker contract against every Broker implementation.

`SimBroker`, and the order manager over the fake venue. Both must pass the same tests.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import pytest

from harness.broker.sim import SimBroker
from harness.journal import Journal
from harness.killswitch import KillSwitch
from harness.orders import OrderManager
from harness.positions import Ledger, reconcile_venue
from harness.risk import RiskManager, RiskState
from tests.broker_contract import BrokerContract, BrokerRig, RigFactory
from tests.fake_venue import FakeVenue, FillStep, OrderScript, at
from tests.synthetic import COSTS
from tests.test_orders import CFG, LOOSE, PROFILE

SYMBOLS = ("AAPL", "MSFT", "NVDA")


class SimRig(BrokerRig):
    """The simulated market is just the opening prices handed to the broker."""

    def __init__(self, starting_cash: float) -> None:
        self.broker = SimBroker(starting_cash, COSTS)

    def open_session(self, session: dt.date, opens: Mapping[str, float]) -> None:
        self.broker.on_session_open(session, opens)


class TestSimBroker(BrokerContract):
    @pytest.fixture
    def make_rig(self) -> RigFactory:
        return SimRig


class OrderManagerRig(BrokerRig):
    """The order manager over the fake venue. The market is the venue's prices: a symbol
    with no price does not trade. Every symbol closed at $100 the session before."""

    def __init__(self, starting_cash: float, directory: Path) -> None:
        self.venue = FakeVenue(clock=at(dt.date(2024, 1, 2), 16, 30), cash=starting_cash)
        journal = Journal(directory / "journal.jsonl")
        self.ledger = Ledger.open(journal, starting_cash)
        risk = RiskManager(LOOSE, CFG.limits, CFG.entry_limit_pct, RiskState.initial(starting_cash))
        self.broker = OrderManager(
            self.venue, self.ledger, journal, KillSwitch(directory / "KILL"), None, CFG, PROFILE,
            risk, 10, sleep=self.venue.advance, now=lambda: self.venue.clock,
        )  # fmt: skip
        self.broker.recover()
        self.broker.set_closes(dict.fromkeys(SYMBOLS, 100.0))

    def open_session(self, session: dt.date, opens: Mapping[str, float]) -> None:
        self.venue.set_now(max(self.venue.clock, at(session, 9, 0)))
        for symbol in SYMBOLS:
            self.venue.set_price(symbol, opens.get(symbol))
        self.broker.on_session_open(session, opens)
        # What the ledger says must be what the venue holds, after every session.
        report = reconcile_venue(self.ledger.account(), [], self.venue, 0.0, 0.01)
        assert report.severity == "ok", report.reasons

    def split_fill(self, symbol: str, first: int, *, rest_fills: bool) -> bool:
        steps = [FillStep(0, first)] + ([FillStep(30, None)] if rest_fills else [])
        self.venue.script(symbol, OrderScript(fills=tuple(steps)))
        return True


class TestOrderManager(BrokerContract):
    @pytest.fixture
    def make_rig(self, tmp_path: Path) -> RigFactory:
        made = 0

        def make(starting_cash: float) -> BrokerRig:
            nonlocal made
            made += 1
            directory = tmp_path / f"rig{made}"
            directory.mkdir()
            return OrderManagerRig(starting_cash, directory)

        return make
