"""Alerts: sinks never raise, and one failing sink does not silence the others."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path

import pytest

from harness.alerts import (
    Alert,
    Alerter,
    Alerters,
    JsonlAlerter,
    LogAlerter,
    read_alerts,
    send_alert,
)

T0 = dt.datetime(2024, 1, 3, 14, 0, tzinfo=dt.UTC)


def alert(n: int = 0, severity: str = "warn", kind: str = "test") -> Alert:
    return Alert(
        at=T0 + dt.timedelta(minutes=n),
        severity=severity,  # type: ignore[arg-type]
        kind=kind,
        message=f"message {n}",
    )


class Recorder:
    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.alerts.append(alert)


class Broken:
    def send(self, alert: Alert) -> None:
        raise RuntimeError("webhook down")


def test_alert_needs_an_aware_time_a_kind_and_a_message() -> None:
    for bad in (
        {"at": dt.datetime(2024, 1, 3, 14, 0)},  # naive
        {"kind": ""},
        {"message": ""},
        {"severity": "fatal"},
    ):
        with pytest.raises(ValueError):
            Alert(**{"at": T0, "severity": "info", "kind": "k", "message": "m", **bad})  # type: ignore[arg-type]


def test_jsonl_alerter_appends_one_line_per_alert(tmp_path: Path) -> None:
    path = tmp_path / "state" / "alerts.jsonl"  # the directory is created
    sink: Alerter = JsonlAlerter(path)  # satisfies the protocol (checked by pyright)
    sink.send(alert(0, "info"))
    JsonlAlerter(path).send(alert(1, "critical", "kill_switch"))  # another writer, same file
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1]) == {
        "at": "2024-01-03T14:01:00Z",
        "kind": "kill_switch",
        "message": "message 1",
        "severity": "critical",
    }
    assert read_alerts(path) == [alert(0, "info"), alert(1, "critical", "kill_switch")]


def test_read_alerts_filters_by_time_and_skips_damaged_lines(tmp_path: Path) -> None:
    path = tmp_path / "alerts.jsonl"
    assert read_alerts(path) == []  # no file yet
    sink = JsonlAlerter(path)
    for n in range(3):
        sink.send(alert(n))
    with path.open("ab") as fh:
        fh.write(b'garbage\n{"at":"2024-01-03T14:09:00Z","severity":"warn"}\n{"at":"2024')
    sink.send(alert(5))  # lands after the torn line, on the same line: lost, not fatal
    sink.send(alert(6))
    assert [a.message for a in read_alerts(path)] == [f"message {n}" for n in (0, 1, 2, 6)]
    since = T0 + dt.timedelta(minutes=1)
    assert [a.message for a in read_alerts(path, since)] == ["message 1", "message 2", "message 6"]


def test_jsonl_alerter_that_cannot_write_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    JsonlAlerter(blocker / "alerts.jsonl").send(alert())  # the parent is a file

    def disk_error(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", disk_error)
    with caplog.at_level(logging.ERROR):
        JsonlAlerter(tmp_path / "alerts.jsonl").send(alert(3))
    assert "could not write the alert" in caplog.text and "message 3" in caplog.text


def test_log_alerter_uses_the_severity_as_the_level(caplog: pytest.LogCaptureFixture) -> None:
    sink = LogAlerter()
    with caplog.at_level(logging.INFO):
        for severity in ("info", "warn", "critical"):
            sink.send(alert(0, severity, "reconcile"))
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.WARNING, logging.CRITICAL]
    assert caplog.records[0].getMessage() == "[reconcile] message 0"


def test_log_alerter_with_a_broken_logger_does_not_raise() -> None:
    class BadLogger(logging.Logger):
        def log(self, *args: object, **kw: object) -> None:  # type: ignore[override]
            raise RuntimeError("handler exploded")

    LogAlerter(BadLogger("bad")).send(alert())


def test_one_failing_sink_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    first, last = Recorder(), Recorder()
    with caplog.at_level(logging.ERROR):
        Alerters(first, Broken(), last).send(alert(2))
    assert first.alerts == last.alerts == [alert(2)]
    assert "alert sink failed" in caplog.text and "message 2" in caplog.text


def test_send_alert_contains_a_sink_that_raises() -> None:
    recorder = Recorder()
    assert send_alert(recorder, alert()) is True
    assert send_alert(Broken(), alert()) is False
    assert send_alert(None, alert()) is True  # no sink configured
    assert recorder.alerts == [alert()]
