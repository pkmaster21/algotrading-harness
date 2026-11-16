"""Inbox: commands arrive whole, in order, and a bad file never breaks the reader."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from harness.inbox import ApproveOrders, Inbox, RejectOrders

T0 = dt.datetime(2024, 1, 3, 14, 0, tzinfo=dt.UTC)


def approve(n: int = 1) -> ApproveOrders:
    return ApproveOrders(approvals={f"order-{n}": f"hash-{n}"}, operator="david", at=T0)


def test_commands_come_back_in_the_order_they_were_written(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "state" / "inbox")  # the directory is created on the first put
    assert inbox.read() == []
    reject = RejectOrders(order_ids=("order-9",), operator="david", reason="too big", at=T0)
    paths = [inbox.put(approve(1)), inbox.put(reject), inbox.put(approve(3))]
    assert len(set(paths)) == 3
    items = Inbox(inbox.directory).read()  # another process reads them
    assert [i.path for i in items] == paths
    assert [i.command for i in items] == [approve(1), reject, approve(3)]
    assert all(i.error == "" for i in items)
    # Each item carries the time its file was written, taken from the file system.
    now = dt.datetime.now(dt.UTC)
    assert all(i.written_at and abs(now - i.written_at) < dt.timedelta(minutes=1) for i in items)
    stamp = dt.datetime(2024, 1, 3, 14, 10, tzinfo=dt.UTC).timestamp()
    os.utime(paths[0], (stamp, stamp))
    assert inbox.read()[0].written_at == dt.datetime(2024, 1, 3, 14, 10, tzinfo=dt.UTC)
    assert sorted(os.listdir(inbox.directory)) == sorted(p.name for p in paths)  # no temp files


def test_removed_command_is_not_read_again(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path)
    first, second = inbox.put(approve(1)), inbox.put(approve(2))
    inbox.remove(first)
    inbox.remove(first)  # removing twice is harmless
    assert [i.path for i in inbox.read()] == [second]


GOOD = {"kind": "approve", "approvals": {"a": "h"}, "operator": "d", "at": "2024-01-03T14:00:00Z"}
REJECT = {"kind": "reject", "order_ids": ["a"], "operator": "d", "reason": "r", "at": GOOD["at"]}


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json",
        b'{"kind": "approve"',  # cut short
        json.dumps({"kind": "launch", "operator": "david"}).encode(),  # not a command
        json.dumps({**GOOD, "approvals": {}}).encode(),
        json.dumps({**GOOD, "operator": ""}).encode(),
        json.dumps({**GOOD, "at": "2024-01-03T14:00:00"}).encode(),  # no time zone
        json.dumps({**REJECT, "reason": ""}).encode(),
        json.dumps({**REJECT, "extra": 1}).encode(),
    ],
)
def test_bad_file_is_reported_not_raised_and_can_be_moved_aside(
    tmp_path: Path, content: bytes
) -> None:
    inbox = Inbox(tmp_path)
    good = inbox.put(approve(1))
    bad = tmp_path / "00000000-bad.json"  # sorts first
    bad.write_bytes(content)
    first, second = inbox.read()
    assert (first.path, first.command) == (bad, None) and first.error
    assert (second.path, second.command) == (good, approve(1))
    inbox.quarantine(bad)
    assert [i.path for i in inbox.read()] == [good]
    assert (tmp_path / "00000000-bad.bad").read_bytes() == content  # kept for inspection


def test_the_shapes_used_for_bad_files_are_otherwise_valid(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path)
    for n, shape in enumerate((GOOD, REJECT)):
        (tmp_path / f"{n}.json").write_text(json.dumps(shape))
    assert [i.error for i in inbox.read()] == ["", ""]


def test_files_that_are_not_commands_are_ignored(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path)
    (tmp_path / ".incoming-abc.tmp").write_text("half written")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "old.bad").write_text("x")
    assert inbox.read() == []


def test_failed_put_leaves_nothing_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inbox = Inbox(tmp_path)

    def disk_full(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", disk_full)
    with pytest.raises(OSError, match="disk full"):
        inbox.put(approve(1))
    assert os.listdir(tmp_path) == [] and inbox.read() == []
