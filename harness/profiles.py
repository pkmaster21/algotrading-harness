"""Frozen profiles: what paper trading runs, and how a backtest run becomes one.

A profile is a validated run frozen into values: the fully resolved recipe (risk preset
included), the execution settings, and the hash of the strategy source. Editing a preset
or a strategy file later never changes what is trading; changing anything means a new
profile with a new id. The profile id is `<name>-<first 8 of content_hash>`, and the
content hash covers every field, so a profile whose file was edited no longer matches
its own id and is refused. A field added to the models after profiles were frozen is left
out of the hash while it holds its default (`HASH_OMITTED_AT`), so a profile frozen before
it existed keeps loading.

Promotion refuses a run that is not ready: the strategy file changed since the run, the
tree was dirty, no risk preset, no holdout run for the same recipe (a plumbing profile
of the tracked example strategy is the one exception, decision D4), or a strategy whose
signals depend on state that a daily process would lose (the restart-equivalence check).

What trades is the frozen copy of the strategy (`profiles/<id>/strategy.py`), imported
from there by `build_profile_strategy`, never the module in `strategies/`. So the
strategy file can keep changing while a profile trades, and "tampered" means the profile
folder changed: the profile file or its copy of the strategy.

Activation binds the paper account to one profile: an account the venue has verified,
flat, no open orders, cash at least the profile's starting cash. Every activation is
appended to `activations.jsonl`; replacing another active profile has to be asked for.
The difference between the account's cash and the
starting cash is recorded once as `cash_offset`; the risk state is seeded from the
account's equity net of that offset and saved, so the next day starts from it.

This module is also the only place a `PaperAccess` is built (`paper_access_for`), from a
profile that has been loaded and verified. Nothing else should build one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import yaml
from pydantic import AwareDatetime, Field, ValidationError, model_validator

from harness.broker.sim import SimBroker
from harness.broker.venue import Venue, VenueDataError
from harness.config import ExecutionConfig, Recipe, load_execution
from harness.data import PROFILE_ID_PATTERN, PROJECT_ROOT, BarPanel, DataLoader, PaperAccess
from harness.data import read_holdout_log as _read_holdout_log
from harness.data import read_paper_access_log as _read_paper_access_log
from harness.engine import Engine
from harness.killswitch import KillSwitch
from harness.models import FrozenModel, Order
from harness.risk import RiskState
from harness.run import (
    StrategyLoadError,
    _git_commit,
    build_risk,
    build_strategy,
    load_strategy_file,
    strategy_source_hash,
)
from harness.view import PRICE_FIELDS
from strategies.base import Strategy

PLUMBING_STRATEGY = "example_buy_and_hold"  # the only strategy a plumbing profile may run
PROFILE_FILE = "profile.json"
STRATEGY_FILE = "strategy.py"
ACTIVE_FILE = "active.json"
ACTIVATIONS_FILE = "activations.jsonl"
RISK_STATE_FILE = "risk_state.json"
KILL_FILE = "KILL"
# Sessions the restart-equivalence check rebuilds the engine for (the last ones of the
# in-sample data). Rebuilding every session would replay the history once per session.
RESTART_CHECK_SESSIONS = 60

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*(-[a-z0-9_]+)*$")
_PROFILE_ID_RE = re.compile(PROFILE_ID_PATTERN)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HASHED_EXCLUDED = {"profile_id", "content_hash"}

# Fields added to what a profile holds after profiles had been frozen, by dotted path in
# the profile's JSON, each with the value (as JSON) at which it is left out of the content
# hash. A profile frozen before the field existed has no such key, so hashing it with the
# key would break every one of them. tests/profile_fields.json pins every field path
# and these values: a new field fails that test until it is listed here, and so does a
# changed default, which would otherwise change what an old profile loads with no
# mismatch to show for it.
HASH_OMITTED_AT: dict[str, Any] = {
    "recipe.universe.membership": None,
}
# From #57 (d0b77be) until this list existed, the hash included `membership` as null.
# Profiles frozen then verify against the hash with that key kept.
_KEPT_BEFORE_OMITTING = frozenset({"recipe.universe.membership"})


class ProfileError(RuntimeError):
    """A promotion or activation was refused, or a profile cannot be used."""


class ProfileTampered(ProfileError):
    """A profile's file or its strategy's source no longer matches what was frozen."""


class RestartEquivalenceError(ProfileError):
    """The strategy's signals depend on state a daily process would lose."""


# --------------------------------------------------------------------------- profile


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class FrozenProfile(FrozenModel):
    """A run frozen into values. Validation recomputes the content hash, so an instance
    that exists is one whose fields match its id."""

    profile_id: str = Field(pattern=PROFILE_ID_PATTERN)
    name: str = Field(pattern=_NAME_RE.pattern, max_length=40)
    mode: Literal["paper"]
    created_at: AwareDatetime
    recipe: Recipe  # fully resolved values, including the risk preset
    execution: ExecutionConfig
    strategy_source_sha256: str = Field(pattern=_SHA256_RE.pattern)
    harness_git_commit: str | None  # None if git was unavailable at promotion
    source_run_id: str | None  # the in-sample run it was promoted from
    holdout_run_id: str | None  # None only for a plumbing profile
    plumbing: bool  # True only for the tracked example strategy
    content_hash: str = Field(pattern=_SHA256_RE.pattern)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.recipe.risk is None or self.recipe.limits is None:
            raise ValueError("a profile needs a risk preset")
        if self.recipe.limits != self.execution.limits:
            raise ValueError("the recipe's order limits differ from the execution settings")
        if self.plumbing and self.recipe.strategy.strategy != PLUMBING_STRATEGY:
            raise ValueError(f"only {PLUMBING_STRATEGY} may be a plumbing profile")
        if not self.plumbing and self.holdout_run_id is None:
            raise ValueError("a profile that is not plumbing needs a holdout run")
        if self.content_hash not in self.accepted_hashes():
            raise ValueError("content hash does not match the profile's fields")
        if self.profile_id != profile_id_for(self.name, self.content_hash):
            raise ValueError("profile id does not match the name and content hash")
        return self

    def hashed_content(self) -> str:
        """sha256 of the canonical JSON of every field but the id and the hash itself,
        less the `HASH_OMITTED_AT` fields that hold their omitted value."""
        return self._hash(HASH_OMITTED_AT.keys())

    def accepted_hashes(self) -> frozenset[str]:
        """The content hashes this profile verifies against: the current one, and the
        one profiles frozen between #57 and the omitted-field list were given."""
        later = HASH_OMITTED_AT.keys() - _KEPT_BEFORE_OMITTING
        return frozenset({self.hashed_content(), self._hash(later)})

    def _hash(self, omit: Iterable[str]) -> str:
        payload = self.model_dump(mode="json", exclude=_HASHED_EXCLUDED)
        for path in omit:
            _drop_at(payload, path, HASH_OMITTED_AT[path])
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def verify(self) -> None:
        """Raise ProfileTampered unless the fields still match the id and hash."""
        if self.content_hash not in self.accepted_hashes() or self.profile_id != profile_id_for(
            self.name, self.content_hash
        ):
            raise ProfileTampered(f"profile {self.profile_id} does not match its content")


def _drop_at(node: Any, path: str, omitted: Any) -> None:
    """Remove the key at dotted `path` from nested dicts where its value equals `omitted`.
    A part ending in `[]` names a list, and the rest of the path applies to each item."""
    part, _, rest = path.partition(".")
    name, many = (part[:-2], True) if part.endswith("[]") else (part, False)
    if not isinstance(node, dict) or name not in node:
        return
    if not rest:
        if not many and node[name] == omitted:
            del node[name]
        return
    children = node[name] if many else [node[name]]
    if isinstance(children, list):
        for child in children:
            _drop_at(child, rest, omitted)


def profile_id_for(name: str, content_hash: str) -> str:
    return f"{name}-{content_hash[:8]}"


def paper_access_for(profile: FrozenProfile) -> PaperAccess:
    """The loader's proof that `profile` is being paper traded. The one place that
    builds a `PaperAccess`; a test fails if another harness module does."""
    profile.verify()
    return PaperAccess(profile_id=profile.profile_id)


# --------------------------------------------------------------------------- promotion


def _read_snapshot(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config_snapshot.yaml"
    if not path.is_file():
        raise ProfileError(f"{run_dir} is not a run folder (no config_snapshot.yaml)")
    try:
        loaded = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"cannot read {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ProfileError(f"{path} must contain a mapping")
    return loaded


def _run_recipe(snapshot: dict[str, Any], run_dir: Path) -> Recipe:
    try:
        return Recipe.model_validate(snapshot["recipe"])
    except (KeyError, ValidationError) as exc:
        raise ProfileError(f"{run_dir}: the saved recipe does not load: {exc}") from exc


def _run_field(snapshot: dict[str, Any], key: str) -> Any:
    run = snapshot.get("run")
    return run.get(key) if isinstance(run, dict) else None


def _logged_holdout_runs(log_path: Path) -> set[str]:
    try:
        return {row.get("run_id") or "" for row in _read_holdout_log(log_path)}
    except (OSError, ValueError) as exc:  # csv.Error and UnicodeDecodeError are ValueErrors
        raise ProfileError(f"cannot read the holdout log {log_path}: {exc}") from exc


def _find_holdout_run(
    runs_dir: Path, recipe: Recipe, strategy_hash: str, exclude: str, logged: set[str]
) -> str | None:
    """The latest holdout run of exactly this recipe and strategy source, if any. The
    run must be in the holdout log, which the loader writes before it reads holdout
    data: a folder alone can be made by hand."""
    found: list[str] = []
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name != exclude):
        if folder.name not in logged:
            continue
        try:
            snapshot = _read_snapshot(folder)
        except ProfileError:
            continue
        if snapshot.get("holdout") is not True:
            continue
        if _run_field(snapshot, "strategy_source_sha256") != strategy_hash:
            continue
        try:
            other = Recipe.model_validate(snapshot.get("recipe"))
        except ValidationError:
            continue
        if other == recipe:
            found.append(folder.name)
    return found[-1] if found else None


def promote_run(
    run_dir: Path,
    name: str,
    *,
    plumbing: bool = False,
    execution: ExecutionConfig | None = None,
    loader: DataLoader | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    git_commit: Callable[[], str | None] = lambda: _git_commit(PROJECT_ROOT),
) -> FrozenProfile:
    """Freeze a validated in-sample run. Refuses unless: the run's strategy hash matches
    the file on disk; the run was made from a clean git tree; the run used a risk preset
    whose order limits are the current execution settings; a holdout run exists for the
    same recipe and is in the holdout log (or `plumbing` and the strategy is the tracked
    example); and the
    restart-equivalence check passes. Writes profile.json and a read-only copy of the
    strategy source under `execution.profiles_dir`.
    """
    if not _NAME_RE.match(name) or len(name) > 40:
        raise ProfileError(
            f"invalid profile name {name!r}: lowercase letters, digits, '_' and single '-'"
        )
    execution = execution if execution is not None else load_execution()
    run_dir = Path(run_dir)
    snapshot = _read_snapshot(run_dir)
    recipe = _run_recipe(snapshot, run_dir)
    if recipe.universe.membership is not None:
        # The restart check below and the paper runner build the engine without the
        # member mask or the bad-bars windows, so they would trade every ticker that was
        # ever a member. Paper trading a point-in-time universe has its own design first.
        raise ProfileError(
            f"universe {recipe.universe.name!r} is point-in-time; it cannot be promoted "
            "until paper trading supports membership (docs/pit-universe-design.md, "
            "Paper trading)"
        )
    if snapshot.get("holdout") is not False:
        raise ProfileError(
            "promote the in-sample run, not the holdout run; the holdout run of the same "
            "recipe is found and recorded automatically"
        )
    if recipe.risk is None or recipe.limits is None:
        raise ProfileError("the run used no risk preset; rerun it with --risk <preset>")
    if recipe.limits != execution.limits:
        raise ProfileError("the order limits in execution.yaml changed since the run; rerun it")
    commit = _run_field(snapshot, "git_commit")
    if not isinstance(commit, str) or not commit or commit.endswith("-dirty"):
        raise ProfileError("the run was not made from a clean git tree; commit and rerun")
    if plumbing and recipe.strategy.strategy != PLUMBING_STRATEGY:
        raise ProfileError(f"only {PLUMBING_STRATEGY} may be promoted as plumbing")
    strategy_hash = strategy_source_hash(build_strategy(recipe))  # the file on disk now
    if _run_field(snapshot, "strategy_source_sha256") != strategy_hash:
        raise ProfileError(
            f"strategies/{recipe.strategy.strategy}.py changed since the run; rerun it"
        )
    logged = _logged_holdout_runs(recipe.data.holdout_log)
    holdout_run = _find_holdout_run(run_dir.parent, recipe, strategy_hash, run_dir.name, logged)
    if holdout_run is None and not plumbing:
        raise ProfileError(
            "no holdout run for this recipe and strategy source (a run folder that is not "
            "in the holdout log does not count); run it with --holdout and the same "
            "presets first"
        )
    check_restart_equivalence(recipe, loader)

    fields: dict[str, Any] = {
        "name": name,
        "mode": "paper",
        "created_at": now().astimezone(dt.UTC),
        "recipe": recipe,
        "execution": execution,
        "strategy_source_sha256": strategy_hash,
        "harness_git_commit": git_commit(),
        "source_run_id": run_dir.name,
        "holdout_run_id": holdout_run,
        "plumbing": plumbing,
    }
    unsigned = FrozenProfile.model_construct(**fields)
    content_hash = unsigned.hashed_content()
    profile = FrozenProfile(
        **fields, profile_id=profile_id_for(name, content_hash), content_hash=content_hash
    )
    _write_profile(profile, execution.profiles_dir, _strategy_file(recipe))
    return profile


def _strategy_file(recipe: Recipe) -> Path:
    return PROJECT_ROOT / "strategies" / f"{recipe.strategy.strategy}.py"


def _write_profile(profile: FrozenProfile, profiles_dir: Path, strategy_file: Path) -> None:
    folder = profiles_dir / profile.profile_id
    try:
        folder.mkdir(parents=True)
    except FileExistsError:
        raise ProfileError(f"profile {profile.profile_id} already exists") from None
    read_only = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        source = strategy_file.read_bytes()
        if hashlib.sha256(source).hexdigest() != profile.strategy_source_sha256:
            raise ProfileError(f"{strategy_file} changed while the profile was being written")
        _write_atomic(folder / STRATEGY_FILE, source, mode=read_only)
        data = profile.model_dump_json(indent=2).encode()
        _write_atomic(folder / PROFILE_FILE, data, mode=read_only)
    except BaseException:
        # A half-written folder would block every retry ("already exists") and never load.
        shutil.rmtree(folder, ignore_errors=True)
        raise


def _write_atomic(path: Path, data: bytes, *, mode: int | None = None) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- loading


def load_profile(profile_id: str, *, profiles_dir: Path) -> FrozenProfile:
    """Load and verify a profile: the content hash and id, and the frozen copy of the
    strategy (what the runner imports, through `build_profile_strategy`). Any mismatch
    raises ProfileTampered; the runner turns that into a kill. The module in
    `strategies/` is not looked at: it may have changed since, and it is not what runs."""
    if not _PROFILE_ID_RE.match(profile_id):
        raise ProfileError(f"{profile_id!r} is not a profile id")
    folder = profiles_dir / profile_id
    path = folder / PROFILE_FILE
    if not path.is_file():
        raise ProfileError(f"no profile {profile_id} in {profiles_dir}")
    try:
        profile = FrozenProfile.model_validate_json(path.read_bytes())
    except (OSError, ValueError, ValidationError) as exc:
        raise ProfileTampered(f"{path} does not verify: {exc}") from exc
    if profile.profile_id != profile_id:
        raise ProfileTampered(f"{path} holds profile {profile.profile_id}, not {profile_id}")
    _verified_strategy_copy(profile, profiles_dir)
    return profile


def _verified_strategy_copy(profile: FrozenProfile, profiles_dir: Path) -> Path:
    file = profiles_dir / profile.profile_id / STRATEGY_FILE
    try:
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProfileTampered(f"cannot read the strategy source {file}: {exc}") from exc
    if digest != profile.strategy_source_sha256:
        raise ProfileTampered(
            f"{file} does not match the source frozen in profile {profile.profile_id}"
        )
    return file


def build_profile_strategy(profile: FrozenProfile, *, profiles_dir: Path) -> Strategy:
    """The strategy a profile trades, built from the frozen copy of its source with the
    frozen params. The copy is hashed again here and goes through the same import
    checks as any strategy. `strategies/<name>.py` is never imported."""
    profile.verify()
    file = _verified_strategy_copy(profile, profiles_dir)
    # One module per profile: the id ends in the content hash, which covers the source
    # hash, so the name is unique to these bytes. Its last part is the strategy's own
    # module name, which is what reports call the strategy.
    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", profile.profile_id)
    module_name = f"frozen_profile_strategies.{safe_id}.{profile.recipe.strategy.strategy}"
    try:
        cls = load_strategy_file(file, module_name)
        return cls.from_params(profile.recipe.strategy.params)
    except (StrategyLoadError, ValidationError) as exc:
        raise ProfileError(
            f"profile {profile.profile_id}: its strategy does not load: {exc}"
        ) from exc


def list_profiles(profiles_dir: Path) -> list[FrozenProfile]:
    """Every profile in `profiles_dir` that loads and verifies, by id."""
    if not profiles_dir.is_dir():
        return []
    profiles = []
    for folder in sorted(p for p in profiles_dir.iterdir() if p.is_dir()):
        try:
            profiles.append(load_profile(folder.name, profiles_dir=profiles_dir))
        except ProfileError:
            continue
    return profiles


def paper_access_log_problems(log_path: Path, profiles_dir: Path) -> list[str]:
    """Lines of the paper access log whose profile does not exist or does not verify.
    An empty list is the expected state: a line there without a real profile is the
    record that something read holdout data without one."""
    problems: list[str] = []
    verdicts: dict[str, str | None] = {}
    for row in _read_paper_access_log(log_path):
        profile_id = row.get("profile_id") or ""
        if profile_id not in verdicts:
            try:
                load_profile(profile_id, profiles_dir=profiles_dir)
            except ProfileError as exc:
                verdicts[profile_id] = str(exc)
            else:
                verdicts[profile_id] = None
        if verdicts[profile_id] is not None:
            problems.append(
                f"{row.get('day', '?')}: {profile_id!r} read data through "
                f"{row.get('data_end', '?')}: {verdicts[profile_id]}"
            )
    return problems


# --------------------------------------------------------------------------- equivalence


def check_restart_equivalence(
    recipe: Recipe,
    loader: DataLoader | None = None,
    *,
    build: Callable[[], Strategy] | None = None,
    sessions: int = RESTART_CHECK_SESSIONS,
) -> None:
    """Run the in-sample backtest twice: once continuously, once rebuilding the strategy
    and the engine for each of the last `sessions` sessions, the way the daily paper
    runner does (warm up on every bar before the session, then one close). Raises
    RestartEquivalenceError if the orders or the equity differ: the strategy keeps state
    across `on_bar` calls that a daily process would lose."""
    if sessions < 1:
        raise ValueError("sessions must be at least 1")
    build = build or (lambda: build_strategy(recipe))
    loader = loader or DataLoader(recipe.dates, recipe.data)
    symbols = list(dict.fromkeys([*recipe.universe.symbols, recipe.backtest.benchmark]))
    panel = loader.load(symbols)  # in-sample only: no holdout, no paper access
    continuous = _run(recipe, panel, build, rebuild_from=panel.n_dates)
    rebuilt = _run(recipe, panel, build, rebuild_from=max(0, panel.n_dates - sessions))
    for session in panel.dates:
        a = [_order_key(o) for o in continuous.orders if o.created_on == session]
        b = [_order_key(o) for o in rebuilt.orders if o.created_on == session]
        if a != b:
            raise RestartEquivalenceError(
                f"{recipe.strategy.strategy}: orders on {session} differ when the strategy "
                f"is rebuilt each session (continuous {a}, rebuilt {b}); the strategy "
                "keeps state across on_bar calls instead of computing from view.history"
            )
    if continuous.equity != rebuilt.equity:
        raise RestartEquivalenceError(
            f"{recipe.strategy.strategy}: the equity curve differs when the engine is "
            "rebuilt each session"
        )


class _RunTrace(FrozenModel):
    orders: tuple[Order, ...]
    equity: tuple[float, ...]


def _order_key(order: Order) -> tuple[str, str, int]:
    return (order.symbol, order.legs[0].side, order.quantity)


def _run(
    recipe: Recipe, panel: BarPanel, build: Callable[[], Strategy], *, rebuild_from: int
) -> _RunTrace:
    universe = recipe.universe.symbols
    cols = [panel.column(s) for s in universe]
    fields = {f: panel.field(f)[:, cols] for f in PRICE_FIELDS}
    rows = [{f: fields[f][i] for f in PRICE_FIELDS} for i in range(panel.n_dates)]
    broker = SimBroker(recipe.backtest.starting_cash, recipe.costs)
    risk = build_risk(recipe)  # one risk state throughout, as the saved state would be

    def fresh() -> Engine:
        return Engine(
            build(), broker, universe, recipe.construction, recipe.costs, panel.n_dates, risk
        )

    engine = fresh()
    orders: list[Order] = []
    equity: list[float] = []
    for i, session in enumerate(panel.dates):
        if i >= rebuild_from:
            previous = engine
            engine = fresh()
            for j in range(i):
                engine.warm_up(panel.dates[j], rows[j])
            # What the journal would tell a restarted process about yesterday's orders.
            engine.orders.update(previous.orders)
        row = rows[i]
        opens = {
            s: float(row["open"][j]) for j, s in enumerate(universe) if not np.isnan(row["open"][j])
        }
        before = set(engine.orders)
        engine.on_session_open(session, opens)
        engine.on_session_close(session, row)
        orders.extend(o for k, o in engine.orders.items() if k not in before)
        equity.append(engine.equity[-1].equity)
    return _RunTrace(orders=tuple(orders), equity=tuple(equity))


# --------------------------------------------------------------------------- activation


class ActiveProfile(FrozenModel):
    """What `active.json` records: the one profile the paper account is bound to."""

    profile_id: str = Field(pattern=PROFILE_ID_PATTERN)
    account_id: str = Field(min_length=1)
    cash_offset: float = Field(ge=0)  # account cash above the profile's starting cash
    approval: Literal["off", "entries"]
    activated_at: AwareDatetime
    operator: str = Field(min_length=1)


class Activation(FrozenModel):
    """One line of `activations.jsonl`: what was activated, and what it replaced."""

    active: ActiveProfile
    replaced: str | None  # the profile id that was active before, if it was another one


def activate_profile(
    profile_id: str,
    approval: Literal["off", "entries"],
    *,
    operator: str,
    venue: Venue,
    replace: bool = False,
    execution: ExecutionConfig | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> ActiveProfile:
    """Bind the paper account to this profile. Requires a verified profile, the kill
    switch clear, and an account that passes the venue's own setup gate
    (`verify_account`: it can trade, cannot short, cannot borrow), is flat with no open
    orders, and holds at least the profile's starting cash. Records the account id and
    the cash offset in active.json and seeds the profile's risk state from the account's
    equity net of the offset. A profile is activated once: one that has a journal
    already is refused (promote a new one). If another profile is active, activation is
    refused unless `replace` is set. Every activation is appended to activations.jsonl
    with the profile it replaced."""
    execution = execution if execution is not None else load_execution()
    if not operator.strip():
        raise ProfileError("activation needs an operator name")
    profile = load_profile(profile_id, profiles_dir=execution.profiles_dir)
    state_dir = execution.state_dir
    if KillSwitch(state_dir / KILL_FILE).engaged():
        raise ProfileError("the kill switch is engaged; reset it before activating a profile")
    if (state_dir / profile_id / "journal.jsonl").exists():
        raise ProfileError(
            f"profile {profile_id} has traded before (its journal exists); promote a new one"
        )
    previous = active_profile(state_dir)
    replaced = previous.profile_id if previous and previous.profile_id != profile_id else None
    if replaced is not None and not replace:
        raise ProfileError(
            f"profile {replaced} is active; pass replace=True to activate {profile_id} instead"
        )
    # The venue's gate, not the account's flags: whether shorting is really off is an
    # account setting, and the `shorting_enabled` field may not follow it. The runner
    # uses the same gate every morning, so the two cannot disagree.
    try:
        account = venue.verify_account()
    except VenueDataError as exc:
        raise ProfileError(f"the account did not pass the venue's setup check: {exc}") from exc
    if positions := venue.positions():
        held = ", ".join(f"{p.symbol} {p.quantity}" for p in positions)
        raise ProfileError(f"the account is not flat: it holds {held}")
    if open_orders := venue.open_orders():
        ids = ", ".join(o.client_order_id for o in open_orders)
        raise ProfileError(f"the account has open orders: {ids}")
    starting_cash = profile.recipe.backtest.starting_cash
    if not (math.isfinite(account.cash) and math.isfinite(account.equity)):
        raise ProfileError("the account's cash or equity is not a number")
    if abs(account.equity - account.cash) > execution.cash_tolerance:
        raise ProfileError(
            f"the account is not flat: equity {account.equity:.2f} differs from cash "
            f"{account.cash:.2f}"
        )
    if account.cash < starting_cash:
        raise ProfileError(
            f"the account holds {account.cash:.2f}, less than the profile's starting cash "
            f"{starting_cash:.2f}"
        )
    cash_offset = account.cash - starting_cash
    active = ActiveProfile(
        profile_id=profile_id,
        account_id=account.account_id,
        cash_offset=cash_offset,
        approval=approval,
        activated_at=now().astimezone(dt.UTC),
        operator=operator,
    )
    # The initial peak is what the ledger will say the account is worth, not the recipe's
    # number: the account's own equity, net of the cash that is not the profile's.
    save_risk_state(RiskState.initial(account.equity - cash_offset), state_dir, profile_id)
    # The record first, then the switch: a crash between the two leaves a line for an
    # activation that did not take effect, never a swap with no record.
    _append_line(
        state_dir / ACTIVATIONS_FILE,
        Activation(active=active, replaced=replaced).model_dump_json(),
    )
    _write_atomic(state_dir / ACTIVE_FILE, active.model_dump_json(indent=2).encode())
    return active


def _append_line(path: Path, line: str) -> None:
    with path.open("a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_activations(state_dir: Path) -> list[Activation]:
    """Every activation recorded for this account, oldest first."""
    path = state_dir / ACTIVATIONS_FILE
    if not path.is_file():
        return []
    try:
        return [Activation.model_validate_json(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError, ValidationError) as exc:
        raise ProfileError(f"{path} cannot be read: {exc}") from exc


def active_profile(state_dir: Path) -> ActiveProfile | None:
    """The profile the account is bound to, or None if none has been activated."""
    path = state_dir / ACTIVE_FILE
    if not path.is_file():
        return None
    try:
        return ActiveProfile.model_validate_json(path.read_bytes())
    except (OSError, ValueError, ValidationError) as exc:
        raise ProfileError(f"{path} cannot be read: {exc}") from exc


def save_risk_state(state: RiskState, state_dir: Path, profile_id: str) -> None:
    """Write the risk state the next trading day starts from (atomic replace)."""
    if not _PROFILE_ID_RE.match(profile_id):
        raise ProfileError(f"{profile_id!r} is not a profile id")
    folder = state_dir / profile_id
    folder.mkdir(parents=True, exist_ok=True)
    _write_atomic(folder / RISK_STATE_FILE, state.model_dump_json(indent=2).encode())


def load_risk_state(state_dir: Path, profile_id: str) -> RiskState:
    """The saved risk state. A missing or unreadable file is an error, never a fresh
    state: that would erase a drawdown halt."""
    if not _PROFILE_ID_RE.match(profile_id):
        raise ProfileError(f"{profile_id!r} is not a profile id")
    path = state_dir / profile_id / RISK_STATE_FILE
    try:
        return RiskState.model_validate_json(path.read_bytes())
    except (OSError, ValueError, ValidationError) as exc:
        raise ProfileError(f"no usable risk state for {profile_id} at {path}: {exc}") from exc
