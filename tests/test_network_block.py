"""The test suite cannot reach the network (see conftest.py)."""

from __future__ import annotations

import socket
import urllib.request

import pytest

from tests.network_block import NetworkBlockedError


def test_socket_connect_is_blocked() -> None:
    with socket.socket() as sock:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("127.0.0.1", 9))
        with pytest.raises(NetworkBlockedError):
            sock.connect_ex(("127.0.0.1", 9))


def test_name_lookup_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("paper-api.alpaca.markets", 443)
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("paper-api.alpaca.markets", 443), timeout=1)


def test_http_request_is_blocked() -> None:
    # urllib wraps socket errors in URLError only for OSError; ours passes through.
    with pytest.raises(NetworkBlockedError):
        urllib.request.urlopen("https://paper-api.alpaca.markets/v2/account", timeout=1)
