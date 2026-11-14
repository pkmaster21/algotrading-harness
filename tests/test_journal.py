"""Journal: durable appends, crash artifacts ignored, corruption refused."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harness.journal import Journal, JournalError, read_journal

T0 = dt.datetime(2024, 1, 3, 13, 0, tzinfo=dt.UTC)


def journal(path: Path) -> Journal:
    ticks = iter(T0 + dt.timedelta(seconds=i) for i in range(1_000))
    return Journal(path, now=lambda: next(ticks))


def test_append_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "state" / "journal.jsonl"  # the directory is created
    j = journal(path)
    assert j.last_seq == 0 and list(j.read()) == []
    assert j.append("ledger_open", {"starting_cash": 10_000.0}) == 1
    assert j.append("fill", {"nested": {"a": [1, 2.5, None, True, "x"]}}) == 2
    first, second = j.read()
    assert (first.seq, first.kind, first.payload) == (1, "ledger_open", {"starting_cash": 10_000.0})
    assert (second.seq, second.at) == (2, T0 + dt.timedelta(seconds=1))
    assert second.payload == {"nested": {"a": [1, 2.5, None, True, "x"]}}
    assert j.last_seq == 2
    # One compact JSON object per line, keys sorted: the same records give the same bytes.
    lines = path.read_text().splitlines()
    assert lines[0] == (
        '{"at":"2024-01-03T13:00:00Z","kind":"ledger_open",'
        '"payload":{"starting_cash":10000.0},"seq":1}'
    )
    assert read_journal(path) == [first, second]


def test_reopening_continues_the_sequence(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal(path).append("fill", {"n": 1})
    reopened = journal(path)
    assert reopened.last_seq == 1
    assert reopened.append("fill", {"n": 2}) == 2
    assert [r.payload["n"] for r in read_journal(path)] == [1, 2]


def test_times_are_stored_in_utc(tmp_path: Path) -> None:
    new_york = dt.timezone(dt.timedelta(hours=-5))
    j = Journal(tmp_path / "j.jsonl", now=lambda: dt.datetime(2024, 1, 3, 8, 0, tzinfo=new_york))
    j.append("fill", {})
    (record,) = j.read()
    assert record.at == T0 and record.at.utcoffset() == dt.timedelta(0)
    naive = Journal(tmp_path / "naive.jsonl", now=lambda: dt.datetime(2024, 1, 3, 8, 0))
    with pytest.raises(JournalError, match="timezone-aware"):
        naive.append("fill", {})


def test_every_append_is_fsynced_before_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    j = journal(tmp_path / "j.jsonl")
    assert len(synced) == 1  # the directory entry of the new file
    j.append("fill", {})
    assert len(synced) == 2
    j.append("fill", {})
    assert len(synced) == 3


# --------------------------------------------------------------------------- crash artifacts


@pytest.mark.parametrize(
    "torn",
    [
        b'{"at":"2024-01-03T13:00:05Z","kind":"fi',  # cut mid-record
        b'{"at":"2024-01-03T13:00:05Z","kind":"fill","payload":{},"seq":3}',  # no newline yet
        b"\x00\x00\x00",
    ],
)
def test_unfinished_last_line_is_ignored_by_readers_and_removed_by_the_writer(
    tmp_path: Path, torn: bytes
) -> None:
    path = tmp_path / "j.jsonl"
    j = journal(path)
    j.append("fill", {"n": 1})
    j.append("fill", {"n": 2})
    intact = path.read_bytes()
    j.close()
    path.write_bytes(intact + torn)

    assert [r.seq for r in read_journal(path)] == [1, 2]
    assert path.read_bytes() == intact + torn  # reading never modifies the file

    reopened = journal(path)
    assert path.read_bytes() == intact
    assert reopened.append("fill", {"n": 3}) == 3
    assert [r.payload["n"] for r in read_journal(path)] == [1, 2, 3]


def test_file_that_is_only_a_torn_line_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    path.write_bytes(b'{"at":')
    assert read_journal(path) == []
    assert journal(path).append("fill", {}) == 1
    assert read_journal(tmp_path / "missing.jsonl") == []


# --------------------------------------------------------------------------- corruption


def _three(path: Path) -> list[bytes]:
    j = journal(path)
    for n in range(3):
        j.append("fill", {"n": n})
    return path.read_bytes().splitlines(keepends=True)


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        (lambda lines: [lines[0], b"garbage\n", lines[2]], "line 2 is not a valid record"),
        (lambda lines: [lines[0], lines[2]], "line 2 has seq 3"),  # a record is missing
        (lambda lines: [lines[1], lines[2]], "line 1 has seq 2"),
        (lambda lines: [lines[0], lines[0], lines[2]], "line 2 has seq 1"),  # duplicated
        (lambda lines: [lines[0], b"\n", lines[1], lines[2]], "line 2 is not a valid record"),
        (
            lambda lines: [lines[0].replace(b'"fill"', b'"mystery"'), lines[1], lines[2]],
            "line 1 is not a valid record",
        ),
        (lambda lines: [lines[0], lines[1], b"garbage\n"], "line 3 is not a valid record"),
    ],
)
def test_corruption_is_an_error_never_skipped(tmp_path: Path, damage, match: str) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "j.jsonl"
    path.write_bytes(b"".join(damage(_three(path))))
    with pytest.raises(JournalError, match=match):
        read_journal(path)
    with pytest.raises(JournalError, match=match):
        Journal(path)  # the writer refuses to append to a corrupt journal


# --------------------------------------------------------------------------- bad writes


@pytest.mark.parametrize("payload", [{"x": float("nan")}, {"x": float("inf")}, {"x": dt.date.min}])
def test_payload_that_is_not_json_is_refused_and_nothing_is_written(
    tmp_path: Path, payload: dict
) -> None:  # type: ignore[type-arg]
    path = tmp_path / "j.jsonl"
    j = journal(path)
    j.append("fill", {"ok": 1})
    before = path.read_bytes()
    with pytest.raises(JournalError, match="cannot journal"):
        j.append("fill", payload)
    assert path.read_bytes() == before and j.last_seq == 1
    assert j.append("fill", {"ok": 2}) == 2  # a refused record does not break the journal


def test_failed_write_stops_the_writer_until_it_is_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "j.jsonl"
    j = journal(path)
    j.append("fill", {"n": 1})

    def disk_error(fd: int) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", disk_error)
        with pytest.raises(JournalError, match="write failed"):
            j.append("fill", {"n": 2})
    assert j.last_seq == 1
    with pytest.raises(JournalError, match="reopen"):
        j.append("fill", {"n": 3})  # even though the disk works again
    j.close()
    # Reopening works. The record whose write failed may or may not have survived; here
    # the bytes reached the file, so it is there, exactly once.
    reopened = journal(path)
    assert [r.payload["n"] for r in reopened.read()] == [1, 2]
    assert reopened.append("fill", {"n": 3}) == 3


def test_second_writer_is_refused_until_the_first_closes(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    first = journal(path)
    first.append("fill", {"n": 1})
    with pytest.raises(JournalError, match="another writer has this journal open"):
        journal(path)
    assert read_journal(path)[0].payload == {"n": 1}  # readers do not need the lock
    assert first.append("fill", {"n": 2}) == 2  # the refused writer did not disturb the first
    first.close()
    first.close()  # closing twice is harmless
    with pytest.raises(JournalError, match="closed"):
        first.append("fill", {"n": 3})
    with journal(path) as second:
        assert second.append("fill", {"n": 3}) == 3
    assert journal(path).last_seq == 3  # leaving the `with` block released the lock


def test_second_process_cannot_write_the_journal(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    held = journal(path)
    held.append("fill", {"n": 1})
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from harness.journal import Journal, JournalError\n"
        "try:\n"
        "    Journal(Path(sys.argv[1]))\n"
        "except JournalError as exc:\n"
        "    print(exc)\n"
        "    sys.exit(3)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 3 and "another writer" in result.stdout
    held.close()
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0


def test_journal_that_fails_to_open_does_not_keep_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    path.write_bytes(b"garbage\n")
    with pytest.raises(JournalError, match="not a valid record"):
        Journal(path)
    path.write_bytes(b"")
    assert journal(path).append("fill", {}) == 1


def test_records_are_plain_json(tmp_path: Path) -> None:
    path = tmp_path / "j.jsonl"
    journal(path).append("adjustment", {"note": "dividend é"})
    (line,) = path.read_text().splitlines()
    assert json.loads(line)["payload"] == {"note": "dividend é"}
