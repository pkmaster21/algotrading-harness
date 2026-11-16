"""The kill switch: a latch file that stops every order, plus `kill`, which sets it and
cancels what is open at the venue.

The latch is a file so that any process can trip it, including while the runner is hung,
and so that it survives a crash or reboot. Everything about it fails closed:

- The latch counts as engaged if the file exists in any form (empty, garbled, a
  directory) or if its presence cannot be determined.
- `engage` that cannot write the file still latches this process in memory, then raises.
- Only an operator `reset` clears it. Nothing in the harness resets it automatically.

The kill switch never sells. It cancels orders and stops; flattening is a separate,
confirmed operator command (D6). That command is the one thing allowed past a set latch:
inside `KillSwitch.flattening()` a sell passes `check`, and nothing else does. The pass
lives in memory on the one `KillSwitch` object the command built, so no other process
and no other object has it. It holds only while the latch file is the one the command
started under, and every `kill` changes the file (`note_kill`) before it cancels
anything: a kill from anywhere ends the pass, whether or not its cancel is confirmed.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import AwareDatetime, Field, ValidationError

from harness.alerts import Alert, Alerter, send_alert
from harness.broker.venue import KillSwitchEngaged, Venue
from harness.models import FrozenModel


class KillSwitchError(RuntimeError):
    """The latch could not be written, or a reset was refused."""


class KillState(FrozenModel):
    engaged: bool
    at: AwareDatetime | None = None
    reason: str = ""
    source: str = ""  # "operator:cli", "auto:reconcile", ...
    cancel_confirmed: bool = False
    # How many times `kill` has run against this latch. It exists so that the file
    # changes on every kill, which is how a running flatten learns of one.
    kills: int = Field(default=0, ge=0)
    # False only on what `kill` returns when no latch is on disk afterwards: this
    # process is stopped, but another process, or the next run, is not. Not part of the
    # latch file.
    latch_written: bool = Field(default=True, exclude=True)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class KillSwitch:
    """A latch file. Engaged means: no order may be sent, by anything, until an operator
    resets it."""

    def __init__(self, path: Path, now: Callable[[], dt.datetime] = _utc_now) -> None:
        self.path = path
        self._now = now
        self._latched: KillState | None = None  # set when this process engaged the switch
        # Set only inside `flattening()`: the latch file the sell pass is valid under.
        self._sells_pass: KillState | None = None

    def engaged(self) -> bool:
        """True if the latch exists or cannot be read. Fails closed."""
        if self._latched is not None:
            return True
        try:
            os.lstat(self.path)
        except FileNotFoundError:
            return False
        except OSError:
            return True
        return True

    def check(self, *, selling: bool = False) -> None:
        """Raise KillSwitchEngaged if the latch is set. Call before anything that sends.

        `selling` says the order about to be sent is a sell. It makes no difference
        except inside `flattening()`, where a sell passes a set latch for as long as
        the latch file is still the one the pass was given under."""
        if selling and self._sells_pass is not None and self.on_disk() == self._sells_pass:
            return
        if self.engaged() or self._sells_pass is not None:
            raise KillSwitchEngaged(f"kill latch {self.path} is set")

    @contextlib.contextmanager
    def flattening(self, under: KillState) -> Iterator[None]:
        """Let sells through this object's `check` while the block runs and the latch
        file still reads `under` (what `orders_sent` returned). For the operator's
        `flatten` command only: the order manager still refuses a sell larger than the
        position, so what passes can only reduce risk. Buys are refused as ever.

        The file is read again on every `check`. Once it is anything but `under` (a
        kill ran, the switch was reset, the file cannot be read), sells are refused
        like everything else, for the rest of the block."""
        self._sells_pass = under
        try:
            yield
        finally:
            self._sells_pass = None

    def state(self) -> KillState:
        """Who engaged the latch, when and why. Always agrees with `engaged`: a latch
        that exists but cannot be read is reported as engaged, with a reason saying so."""
        problem = "latch file cannot be read"
        try:
            state = KillState.model_validate(json.loads(self.path.read_bytes()))
            if state.engaged:
                return state
            problem += " (it says not engaged)"
        except FileNotFoundError:
            if not self.engaged():
                return KillState(engaged=False)
        except (OSError, ValueError, ValidationError) as exc:
            problem += f" ({type(exc).__name__})"
        return self._latched or KillState(engaged=True, reason=problem, source="unknown")

    def engage(self, reason: str, source: str) -> KillState:
        """Write the latch atomically. Idempotent. Does no network I/O.

        If the latch is already set, the first reason stays and its state is returned.
        If the file cannot be written, this process is latched in memory anyway and
        KillSwitchError is raised: other processes may not see the latch.
        """
        try:
            state = KillState(engaged=True, at=self._now(), reason=reason, source=source)
        except Exception:  # a broken clock must not stop the latch
            state = KillState(engaged=True, at=_utc_now(), reason=reason, source=source)
        self._latched = self._latched or state
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(state.model_dump_json() + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                try:
                    os.link(tmp, self.path)  # fails if the latch exists: the first one wins
                except FileExistsError:
                    return self.state()
            finally:
                Path(tmp).unlink(missing_ok=True)
            self._sync_directory()
        except OSError as exc:
            self._engage_bare()
            raise KillSwitchError(f"could not write the kill latch {self.path}: {exc}") from exc
        return state

    def confirm_cancel(self) -> KillState:
        """Record in the latch that no open orders remain at the venue.

        If an operator resets the switch between the read and the write here, the write
        sets the latch again. That is the safe direction, and the window is milliseconds.
        """
        return self._note_cancel(True)

    def orders_sent(self) -> KillState:
        """Record in the latch that orders may be open at the venue again: the
        operator's flatten is about to send sells past it. A kill run after this
        cancels them and confirms, which sets the note back. That change, and the one
        every kill makes before it cancels (`note_kill`), is how a running flatten
        learns it has been stopped (`on_disk`).

        Returns what the latch file holds afterwards, from the one read made before the
        write. The caller must look at it: if the note could not be written, or the
        file is not a readable latch, the result does not say "engaged, cancel not
        confirmed", and a later kill would then leave the file as it is."""
        return self._note_cancel(False, self.on_disk())

    def on_disk(self) -> KillState:
        """What the latch file says now, whatever this object remembers: a process that
        engaged the switch itself would otherwise never see another process reset it."""
        return KillSwitch(self.path, self._now).state()

    def note_kill(self) -> bool:
        """Count one more kill in the latch file. `kill` calls this right after it
        engages and before it asks the venue anything, so the file changes on every
        kill, confirmed or not, and a flatten running in another process stops sending
        (its pass ends at the next `check`, its loop at the next pass).

        False only if a readable latch could not be rewritten: a flatten running under
        it has not been told. A latch that is not readable needs no note, because no
        flatten can be running under one (it refuses to start, and stops if the file
        turns unreadable). As in `confirm_cancel`, a reset that lands between the read
        and the write is undone by the write, which is the safe direction.
        """
        state = self.on_disk()
        if not state.engaged or state.at is None:
            return True
        return self._rewrite(state.model_copy(update={"kills": state.kills + 1}))

    def _note_cancel(self, value: bool, state: KillState | None = None) -> KillState:
        state = state if state is not None else self.state()
        if not state.engaged or state.at is None:  # not engaged, or an unreadable latch
            return state
        confirmed = state.model_copy(update={"cancel_confirmed": value})
        if self._latched is not None:
            self._latched = self._latched.model_copy(update={"cancel_confirmed": value})
        if not self._rewrite(confirmed):
            return state  # the latch itself is still set; only the note is missing
        return confirmed

    def _rewrite(self, state: KillState) -> bool:
        """Replace the latch file with `state`, atomically. False if it could not be
        written; the file is then as it was."""
        try:
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    fh.write(state.model_dump_json() + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            finally:
                Path(tmp).unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def reset(self, operator: str, confirm: str, profile_id: str) -> KillState:
        """Remove the latch. `confirm` must equal `profile_id`, the active profile, typed
        by the operator. Returns the state that was cleared, for the audit trail.

        This only removes the latch. Checking that reconciliation is clean first is the
        caller's job.

        A process that engaged the switch itself also holds the latch in memory, and a
        reset from another process does not reach it: that process keeps refusing until
        it is restarted. After a reset, restart the runner.
        """
        if not operator.strip():
            raise KillSwitchError("a reset needs an operator name")
        if not profile_id or confirm != profile_id:
            raise KillSwitchError("reset refused: the confirmation does not match the profile id")
        if not self.engaged():
            raise KillSwitchError("the kill switch is not engaged")
        cleared = self.state()
        try:
            self.path.unlink(missing_ok=True)
            self._sync_directory()
        except OSError as exc:
            raise KillSwitchError(f"could not remove the kill latch {self.path}: {exc}") from exc
        self._latched = None
        return cleared

    def _engage_bare(self) -> None:
        """Last resort when the full latch cannot be written: any file at the path
        counts as engaged, so try to leave an empty one."""
        try:
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o644))
        except OSError:
            pass

    def _sync_directory(self) -> None:
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def kill(
    switch: KillSwitch,
    reason: str,
    source: str,
    *,
    venue: Venue | None = None,
    alerts: Alerter | None = None,
    attempts: int = 5,
    wait_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], dt.datetime] = _utc_now,
) -> KillState:
    """Engage the latch first, then cancel every open order at the venue and confirm
    none remain. Safe to call from any process, including while the runner is hung.

    Never raises: every failure ends in a critical alert and a returned state whose
    `cancel_confirmed` is False. If the cancel cannot be confirmed, the latch stays set
    and the alert says so. Positions are never touched.

    Between the latch and the cancel it counts itself in the latch file (`note_kill`).
    That is what stops a flatten running in another process, also when the cancel is
    not confirmed afterwards.

    If no latch is on disk afterwards (it could not be written), the returned state
    has `latch_written` False and the alert says so: this process is stopped, but
    nothing stops the next one. A caller that reports to an operator must pass that on.
    """
    problems: list[str] = []
    try:
        state = switch.engage(reason, source)
    except Exception as exc:
        problems.append(f"{exc} (this process is stopped; other processes may not be)")
        state = switch.state()
        if not state.engaged:  # cannot happen after engage, but never report "not engaged"
            state = KillState(engaged=True, at=now(), reason=reason, source=source)
    try:
        noted = switch.note_kill()  # before the venue is asked anything
    except Exception:
        noted = False
    if noted:
        fresh = switch.state()
        state = fresh if fresh.engaged and fresh.at is not None else state
    else:
        problems.append(
            "this kill could not be noted in the latch file, so a flatten running in "
            "another terminal may NOT have stopped; end that process"
        )

    confirmed = False
    if venue is None:
        problems.append("no venue was given, so open orders were NOT canceled")
    else:
        last = "the venue still lists open orders"
        for attempt in range(max(attempts, 1)):
            try:
                venue.cancel_all()
                remaining = venue.open_orders()
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if not remaining:
                    confirmed = True
                    break
                last = f"{len(remaining)} order(s) still open at the venue"
            if attempt + 1 < max(attempts, 1):
                try:
                    sleep(wait_seconds)
                except Exception:
                    break
        if confirmed:
            try:
                state = switch.confirm_cancel()
            except Exception:
                pass
            if not state.cancel_confirmed:
                state = state.model_copy(update={"cancel_confirmed": True})
        else:
            problems.append(f"open orders are NOT confirmed canceled: {last}")

    # Looked at with a fresh object: what any other process, and the next run, will see.
    if not KillSwitch(switch.path).engaged():
        state = state.model_copy(update={"latch_written": False})
        problems.append(
            "THE LATCH IS NOT ON DISK, so the next run will NOT be refused. Do not start "
            "the runner; make the latch writable and engage the kill switch again"
        )

    message = f"kill switch engaged by {source}: {reason}."
    if state.reason != reason or state.source != source:
        message += f" It was already engaged by {state.source or 'unknown'}: {state.reason}."
    message += " No open orders remain at the venue." if confirmed else ""
    message += "".join(f" PROBLEM: {p}." for p in problems)
    message += " Positions were not touched."
    try:
        alert = Alert(at=now(), severity="critical", kind="kill_switch", message=message)
    except Exception:
        alert = Alert(at=_utc_now(), severity="critical", kind="kill_switch", message=message)
    send_alert(alerts, alert)
    return state
