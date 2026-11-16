"""The inbox: how another process hands a control command to the runner.

The journal has one writer, the runner. An operator's command (approve these entries,
reject those) comes from a different process, so it is dropped here as a file and the
runner reads it, acts on it and records it in the journal.

- One command per file, written whole: a temp file renamed into place, so a reader never
  sees half a command.
- File names sort in the order the commands were written.
- A file is removed only after its command has been handled. If the runner dies in
  between, the command is read again, so handling a command must be safe to repeat.
- A file that does not parse is moved aside (`.bad`), not deleted and not retried.

Nothing in the inbox is trusted. A command is only a request: the order manager checks
it against its own state before acting. That includes the time: a command says when it
was decided (`at`), but a writer can say anything, so each item also carries the time
its file was written, which the reader takes from the file system.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from harness.models import FrozenModel


class ApproveOrders(FrozenModel):
    """Approve entries. Each approval names the order and the hash of what was shown to
    the operator, so it cannot apply to an order that has changed since."""

    kind: Literal["approve"] = "approve"
    approvals: dict[str, str] = Field(min_length=1)  # order id -> content hash
    operator: str = Field(min_length=1)
    at: AwareDatetime  # when the operator decided


class RejectOrders(FrozenModel):
    kind: Literal["reject"] = "reject"
    order_ids: tuple[str, ...] = Field(min_length=1)
    operator: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    at: AwareDatetime


Command = Annotated[ApproveOrders | RejectOrders, Field(discriminator="kind")]
_COMMAND: TypeAdapter[Command] = TypeAdapter(Command)


class InboxItem(FrozenModel):
    """One file in the inbox: its command, or why it could not be read."""

    path: Path
    command: Command | None = None
    error: str = ""
    written_at: AwareDatetime | None = None  # the file's modification time


class Inbox:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def put(self, command: Command) -> Path:
        """Write one command. Safe to call from any process."""
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S%f")
        path = self.directory / f"{stamp}-{uuid.uuid4().hex[:12]}.json"
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".incoming-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(command.model_dump_json() + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

    def read(self) -> list[InboxItem]:
        """Every command file waiting, oldest first. Never raises for a bad file: it
        comes back as an item with `error` set."""
        try:
            paths = sorted(p for p in self.directory.iterdir() if p.suffix == ".json")
        except FileNotFoundError:
            return []
        items = []
        for path in paths:
            written_at = None
            try:
                # A rename keeps the modification time, so this is when `put` wrote it.
                written_at = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
                command = _COMMAND.validate_python(json.loads(path.read_bytes()))
            except FileNotFoundError:
                continue  # removed by someone else between the listing and the read
            except (OSError, ValueError, ValidationError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                items.append(InboxItem(path=path, error=error, written_at=written_at))
            else:
                items.append(InboxItem(path=path, command=command, written_at=written_at))
        return items

    def remove(self, path: Path) -> None:
        """Forget a command that has been handled."""
        path.unlink(missing_ok=True)

    def quarantine(self, path: Path) -> None:
        """Move a file that cannot be read out of the way, keeping it for inspection."""
        try:
            os.replace(path, path.with_suffix(".bad"))
        except OSError:
            pass
