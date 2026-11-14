"""The journal: an append-only log that is the source of truth for paper trading state.

Everything else on disk (the ledger snapshot, reports) is derived from it and can be
rebuilt. One process writes it; any process may read it. The writer holds an exclusive
lock on the file for as long as it is open, so a second writer fails at once instead of
interleaving lines with the first.

Format: one JSON object per line, `{"at": ..., "kind": ..., "payload": {...}, "seq": n}`,
with `seq` counting from 1 with no gaps.

Durability rules:

- `append` returns only after the record is written and fsynced. A caller that got a
  sequence number back can rely on the record surviving a crash.
- A record counts only once its line ends with a newline. If the process dies mid-write,
  the unfinished last line is ignored by readers and removed by the next writer. Its
  caller never got a sequence number, so nothing depended on it.
- Anything else wrong (a bad line in the middle, a gap in `seq`) is corruption, not a
  crash artifact. It raises `JournalError`. Nothing is skipped or repaired silently.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, ValidationError

from harness.models import FrozenModel

# Later slices of the paper path add their own kinds (order intents, venue states, ...).
JournalKind = Literal["ledger_open", "fill", "adjustment"]


class JournalError(RuntimeError):
    """The journal is corrupt or was used incorrectly."""


class JournalRecord(FrozenModel):
    seq: int = Field(ge=1)
    at: dt.datetime  # UTC
    kind: JournalKind
    payload: dict[str, JsonValue]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _scan(path: Path) -> tuple[list[JournalRecord], int]:
    """All complete records, and the byte length of the complete part of the file."""
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    complete = data.rfind(b"\n") + 1  # bytes after the last newline are a torn write
    records: list[JournalRecord] = []
    for number, line in enumerate(data[:complete].split(b"\n")[:-1], start=1):
        try:
            record = JournalRecord.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:
            raise JournalError(f"{path}: line {number} is not a valid record: {exc}") from exc
        if record.seq != number:
            raise JournalError(f"{path}: line {number} has seq {record.seq}")
        records.append(record)
    return records, complete


def read_journal(path: Path) -> list[JournalRecord]:
    """Every complete record in `path`, oldest first. Safe to call from any process:
    it never modifies the file, and an unfinished last line is ignored."""
    return _scan(path)[0]


class Journal:
    """The single writer of one journal file.

    Opening takes an exclusive lock on the file (`flock`), held until `close`. A second
    `Journal` on the same file, in this process or another, raises `JournalError`.
    Readers (`read_journal`) do not take the lock.
    """

    def __init__(self, path: Path, now: Callable[[], dt.datetime] = _utc_now) -> None:
        self.path = path
        self._now = now
        self._fd: int | None = None
        self._broken = False
        path.parent.mkdir(parents=True, exist_ok=True)
        created = not path.exists()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise JournalError(f"{path}: another writer has this journal open") from exc
            records, complete = _scan(path)
            if os.fstat(fd).st_size != complete:
                os.ftruncate(fd, complete)  # drop the torn last line before appending after it
                os.fsync(fd)
            if created:
                self._sync_directory()
        except BaseException:
            os.close(fd)  # also releases the lock
            raise
        self._fd = fd
        self._seq = len(records)

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Release the file and its lock. Safe to call more than once."""
        fd, self._fd = getattr(self, "_fd", None), None
        if fd is not None:
            os.close(fd)

    @property
    def last_seq(self) -> int:
        """Sequence number of the last record written (0 if the journal is empty)."""
        return self._seq

    def append(self, kind: JournalKind, payload: Mapping[str, JsonValue]) -> int:
        """Write one record and fsync it. Returns its sequence number."""
        if self._fd is None:
            raise JournalError(f"{self.path}: the journal is closed")
        if self._broken:
            raise JournalError(
                f"{self.path}: an earlier write failed; close and reopen the journal"
            )
        at = self._now()
        if at.tzinfo is None:
            raise JournalError("journal clock must be timezone-aware")
        try:
            record = JournalRecord(
                seq=self._seq + 1, at=at.astimezone(dt.UTC), kind=kind, payload=dict(payload)
            )
            line = json.dumps(
                record.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (ValueError, ValidationError) as exc:
            raise JournalError(f"cannot journal a {kind} record: {exc}") from exc
        data = line.encode() + b"\n"
        try:
            while data:
                data = data[os.write(self._fd, data) :]
            os.fsync(self._fd)
        except OSError as exc:
            # The file may now end in a partial line. Appending after it would corrupt the
            # journal, so this writer stops; reopening repairs the tail.
            self._broken = True
            raise JournalError(f"{self.path}: write failed: {exc}") from exc
        self._seq = record.seq
        return record.seq

    def read(self) -> Iterator[JournalRecord]:
        return iter(read_journal(self.path))

    def _sync_directory(self) -> None:
        """Make the new file's directory entry durable too."""
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
