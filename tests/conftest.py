"""Suite-wide fixtures.

Every test runs with outbound network access blocked, so no test can reach Alpaca (or
anything else) by mistake, whatever keys are on the machine. The block covers this
process only: a subprocess started by a test is not covered.
"""

from __future__ import annotations

import pytest

# The Broker contract tests live in a helper module (not test_*.py), so pytest must be told
# to give its asserts the same detailed failure messages as a test module.
pytest.register_assert_rewrite("tests.broker_contract")

from tests import network_block  # noqa: E402


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    network_block.install(monkeypatch)
