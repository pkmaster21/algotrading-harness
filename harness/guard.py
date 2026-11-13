"""Runtime guard: no file, network, or process access while strategy code runs.

The engine wraps every call into a strategy (`on_start`, `on_bar`) in `strategy_code()`.
While it is active, a Python audit hook rejects:

- opening a file, except read-only opens inside the Python installation (the interpreter,
  the standard library, installed packages, the time zone database). Imports and
  libraries need those; nothing in them is run data or a secret.
- any network activity (sockets, urllib, http.client).
- starting a process (subprocess, `os.system`, exec, spawn, fork), since a child process
  could do either of the above.

This is the second layer behind the static check in `harness.run`, which only sees a
strategy module's source. It covers every strategy the engine runs, including test
strategies that skip the static check.

Known limits (this is a guard against accidents, not a sandbox):

- Libraries that do their own I/O in native code never raise the `open` audit event.
  pyarrow is one: `pd.read_parquet` and `DataFrame.to_parquet` are NOT caught here. The
  static check rejects those names in strategy source instead.
- The hook is per thread: work a strategy hands to another thread is not covered.
- `StrategyIOError` is a `PermissionError`, because libraries probe optional files and
  recover from `OSError` (the import system writing bytecode, for example). A strategy
  that catches `OSError` therefore hides the attempt, but the access is still refused.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import threading
import zoneinfo
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class StrategyIOError(PermissionError):
    """Strategy code tried to open a file, use the network, or start a process."""


_NETWORK_PREFIXES = ("socket.", "urllib.", "http.client.", "ftplib.", "smtplib.")
# A child process could do any of the I/O refused here, so starting one is refused too.
_PROCESS_EVENTS = frozenset(
    {"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
     "os.forkpty", "os.startfile"}
)  # fmt: skip
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_WRITE_MODE_CHARS = frozenset("wax+")

_state = threading.local()
_install_lock = threading.Lock()
_installed = False
_readable_roots: tuple[str, ...] = ()


def _python_install_roots() -> tuple[str, ...]:
    candidates = [
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        *sysconfig.get_paths().values(),
        *zoneinfo.TZPATH,
    ]
    roots = {os.path.realpath(c) for c in candidates if c}
    return tuple(sorted(r.rstrip(os.sep) + os.sep for r in roots))


def _is_install_read(path: Any, mode: Any, flags: Any) -> bool:
    """True for a read-only open of a file inside the Python installation."""
    if isinstance(path, int):
        return True  # an already-open descriptor; whatever opened it was checked
    if isinstance(mode, str) and _WRITE_MODE_CHARS & set(mode):
        return False
    if isinstance(flags, int) and flags & _WRITE_FLAGS:
        return False
    try:
        resolved = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        return False
    return resolved.startswith(_readable_roots)


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not getattr(_state, "depth", 0):
        return
    if event == "open":
        path, mode, flags = (*args, None, None, None)[:3]
        if not _is_install_read(path, mode, flags):
            raise StrategyIOError(
                f"strategy code tried to open {path!r}: strategies may not read or write "
                "files (use the MarketView and PortfolioSnapshot they are given)"
            )
    elif event.startswith(_NETWORK_PREFIXES):
        raise StrategyIOError(f"strategy code tried to use the network ({event})")
    elif event in _PROCESS_EVENTS:
        raise StrategyIOError(f"strategy code tried to start a process ({event})")


def _install() -> None:
    global _installed, _readable_roots
    with _install_lock:
        if not _installed:
            _readable_roots = _python_install_roots()
            sys.addaudithook(_audit)  # audit hooks cannot be removed; this one idles when off
            _installed = True


@contextmanager
def strategy_code() -> Iterator[None]:
    """Reject file opens, network use, and process starts on this thread until the block
    exits."""
    _install()
    _state.depth = getattr(_state, "depth", 0) + 1
    try:
        yield
    finally:
        _state.depth -= 1
