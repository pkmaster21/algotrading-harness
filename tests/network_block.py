"""Blocks outbound network access for the test process (installed by conftest.py)."""

from __future__ import annotations

import socket
import sys
from typing import Any, NoReturn

import pytest


class NetworkBlockedError(RuntimeError):
    """A test tried to use the network. Use a fake instead."""


def _blocked(event: str, *args: Any) -> NoReturn:
    # Raise the audit event first, as the real call would, so `harness.guard` still sees
    # (and rejects) network use by strategy code with its own error.
    sys.audit(event, *args)
    raise NetworkBlockedError(f"network access is blocked in tests ({event})")


def install(monkeypatch: pytest.MonkeyPatch) -> None:
    def connect(self: socket.socket, address: Any) -> NoReturn:
        _blocked("socket.connect", self, address)

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> NoReturn:
        _blocked("socket.getaddrinfo", host, port)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
