"""Runs the shared Broker contract against every Broker implementation.

`SimBroker` today. The order manager over a fake venue joins in slice 2, and must pass
the same tests.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import pytest

from harness.broker.sim import SimBroker
from tests.broker_contract import BrokerContract, BrokerRig, RigFactory
from tests.synthetic import COSTS


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
