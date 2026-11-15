"""Alerts: things an operator should know about, and where they are sent.

An alert is a notice, not a control: nothing in the trading path depends on one being
delivered. So a sink must never raise. `send_alert` enforces that for any sink, and the
sinks here also handle their own failures.

Sinks in this slice: the log and `alerts.jsonl`. Neither reaches a phone. A push channel
is a gate before any unattended run (see `docs/slice2-design.md`, D8).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Literal, Protocol

from pydantic import AwareDatetime, Field, ValidationError

from harness.models import FrozenModel

Severity = Literal["info", "warn", "critical"]

_LEVELS: dict[Severity, int] = {
    "info": logging.INFO,
    "warn": logging.WARNING,
    "critical": logging.CRITICAL,
}
_log = logging.getLogger(__name__)


class Alert(FrozenModel):
    at: AwareDatetime
    severity: Severity
    kind: str = Field(min_length=1)  # short and stable, for filtering: "kill_switch", ...
    message: str = Field(min_length=1)


class Alerter(Protocol):
    def send(self, alert: Alert) -> None:
        """Must not raise: a failing alert sink never blocks trading logic or the kill path."""
        ...


def send_alert(alerter: Alerter | None, alert: Alert) -> bool:
    """Send `alert`, containing any failure of the sink. Returns False if the sink raised.

    Callers on a path that must finish (the kill path, the runner's shutdown) use this
    instead of `alerter.send`, so a sink that breaks its promise still cannot stop them.
    """
    if alerter is None:
        return True
    try:
        alerter.send(alert)
    except Exception:
        try:
            _log.exception("alert sink failed; the alert was: %s", alert)
        except Exception:
            pass
        return False
    return True


class LogAlerter:
    """Writes each alert to the Python log at a level matching its severity."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("harness.alert")

    def send(self, alert: Alert) -> None:
        try:
            self._logger.log(_LEVELS[alert.severity], "[%s] %s", alert.kind, alert.message)
        except Exception:
            pass


class JsonlAlerter:
    """Appends each alert as one JSON line to a file such as `state/paper/alerts.jsonl`.

    Several processes may append to the same file (the runner, and a kill issued from
    another terminal). Each alert is one `write` to a file opened for append, which keeps
    lines whole.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, alert: Alert) -> None:
        try:
            line = json.dumps(alert.model_dump(mode="json"), sort_keys=True) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            try:
                _log.exception("could not write the alert to %s: %s", self.path, alert)
            except Exception:
                pass


class Alerters:
    """Sends each alert to every sink. One sink failing does not stop the others."""

    def __init__(self, *sinks: Alerter) -> None:
        self._sinks = sinks

    def send(self, alert: Alert) -> None:
        for sink in self._sinks:
            send_alert(sink, alert)


def read_alerts(path: Path, since: dt.datetime | None = None) -> list[Alert]:
    """Alerts in `path`, oldest first, optionally only those at or after `since`.

    The file is a notice board, not a source of truth, so a line that does not parse
    (a write cut short by a crash) is skipped instead of failing the read.
    """
    try:
        lines = path.read_bytes().splitlines()
    except FileNotFoundError:
        return []
    alerts: list[Alert] = []
    for line in lines:
        try:
            alert = Alert.model_validate(json.loads(line))
        except (ValueError, ValidationError):
            continue
        if since is None or alert.at >= since:
            alerts.append(alert)
    return alerts
