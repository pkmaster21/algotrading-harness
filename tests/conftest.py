"""Suite-wide fixtures.

Every test runs with outbound network access blocked, so no test can reach Alpaca (or
anything else) by mistake, whatever keys are on the machine. The block covers this
process only: a subprocess started by a test is not covered.
"""

from __future__ import annotations

import pytest

from tests import network_block


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    network_block.install(monkeypatch)
