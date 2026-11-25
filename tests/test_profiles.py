"""Frozen profiles: promotion refusals, tamper detection, the restart-equivalence check,
the one place a PaperAccess is built, the paper access log check, and activation."""

from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import stat
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from harness.broker.venue import VenueOrder
from harness.config import ExecutionConfig, load_execution, load_recipe
from harness.data import DataLoader, PaperAccess, read_paper_access_log
from harness.killswitch import KillSwitch
from harness.profiles import (
    Activation,
    ActiveProfile,
    FrozenProfile,
    ProfileError,
    ProfileTampered,
    RestartEquivalenceError,
    activate_profile,
    active_profile,
    build_profile_strategy,
    check_restart_equivalence,
    list_profiles,
    load_profile,
    load_risk_state,
    paper_access_for,
    paper_access_log_problems,
    promote_run,
    read_activations,
    save_risk_state,
)
from harness.risk import RiskState
from harness.run import StrategyLoadError, load_strategy_file, run_backtest
from tests.fake_alpaca import NOW, FakeFetcher, recipe, tmp_config
from tests.fake_venue import FakeVenue
from tests.fixture_strategies import CallCounter, Churn

COMMIT = "0123456789abcdef0123456789abcdef01234567"
T0 = dt.datetime(2026, 7, 1, 16, 0, tzinfo=dt.UTC)


# --------------------------------------------------------------------------- helpers


class Rig:
    """A config copy with a fake data source, a backtest run of the example strategy with
    the balanced risk preset, and (unless told otherwise) a holdout run of the same."""

    def __init__(
        self, tmp_path: Path, *, holdout: bool = True, spinoffs: Sequence[str] = ()
    ) -> None:
        self.config = tmp_config(tmp_path, spinoffs=spinoffs)
        self.fetcher = FakeFetcher()
        self.execution: ExecutionConfig = load_execution(self.config)
        self.recipe = load_recipe("example_buy_and_hold", "test-3", self.config, risk="balanced")
        self.run_dir = self.run(holdout=False)
        self.holdout_dir = self.run(holdout=True) if holdout else None
        self.now = NOW

    def loader(self) -> DataLoader:
        r = recipe(self.config)
        return DataLoader(r.dates, r.data, fetcher_factory=lambda: self.fetcher, now=lambda: NOW)

    def run(self, *, holdout: bool, commit: str | None = COMMIT) -> Path:
        summary = run_backtest(
            self.recipe,
            holdout=holdout,
            loader=self.loader(),
            now=lambda: NOW,
            git_commit=lambda: commit,
        )
        return Path(summary["run_dir"])

    def promote(self, name: str = "plumbing", **kw) -> FrozenProfile:  # type: ignore[no-untyped-def]
        kw.setdefault("execution", self.execution)
        kw.setdefault("loader", self.loader())
        kw.setdefault("now", lambda: self.now)
        kw.setdefault("git_commit", lambda: COMMIT)
        return promote_run(self.run_dir, name, **kw)

    def load(self, profile_id: str) -> FrozenProfile:
        return load_profile(profile_id, profiles_dir=self.execution.profiles_dir)

    def edit_snapshot(self, mutate) -> None:  # type: ignore[no-untyped-def]
        path = self.run_dir / "config_snapshot.yaml"
        snapshot = yaml.safe_load(path.read_text())
        mutate(snapshot)
        path.write_text(yaml.safe_dump(snapshot, sort_keys=False))

    def venue(self, cash: float = 12_000.0) -> FakeVenue:
        return FakeVenue(clock=T0, cash=cash)

    def activate(self, profile_id: str, venue: FakeVenue, **kw) -> ActiveProfile:  # type: ignore[no-untyped-def]
        kw.setdefault("operator", "david")
        kw.setdefault("approval", "entries")
        return activate_profile(
            profile_id, venue=venue, execution=self.execution, now=lambda: T0, **kw
        )


def _writable(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


# --------------------------------------------------------------------------- promotion


def test_promotion_freezes_the_run_into_a_verified_profile(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("plumbing")
    assert profile.profile_id == f"plumbing-{profile.content_hash[:8]}"
    assert PaperAccess(profile_id=profile.profile_id)  # the id shape the loader accepts
    assert profile.mode == "paper"
    assert profile.recipe == rig.recipe and profile.recipe.risk is not None
    assert profile.execution == rig.execution
    assert profile.source_run_id == rig.run_dir.name
    assert profile.holdout_run_id == rig.holdout_dir.name  # type: ignore[union-attr]
    assert profile.harness_git_commit == COMMIT
    assert profile.plumbing is False
    assert profile.created_at == NOW
    # The copy on disk reloads to the same profile, and both files are read-only.
    folder = rig.execution.profiles_dir / profile.profile_id
    assert rig.load(profile.profile_id) == profile
    assert (folder / "strategy.py").read_bytes() == (
        Path(__file__).resolve().parent.parent / "strategies" / "example_buy_and_hold.py"
    ).read_bytes()
    for name in ("profile.json", "strategy.py"):
        assert not os.access(folder / name, os.W_OK)
    assert list_profiles(rig.execution.profiles_dir) == [profile]


def test_a_plumbing_profile_needs_no_holdout_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdout=False)
    with pytest.raises(ProfileError, match="no holdout run"):
        rig.promote("strat")
    assert list_profiles(rig.execution.profiles_dir) == []
    profile = rig.promote("plumbing", plumbing=True)
    assert profile.plumbing is True and profile.holdout_run_id is None


def test_the_holdout_run_must_be_of_the_same_recipe_and_source(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert rig.holdout_dir is not None
    snapshot = rig.holdout_dir / "config_snapshot.yaml"
    text = snapshot.read_text()
    snapshot.write_text(text.replace("max_position_pct: 0.25", "max_position_pct: 0.3"))
    with pytest.raises(ProfileError, match="no holdout run"):
        rig.promote("strat")
    hashed = yaml.safe_load(text)["run"]["strategy_source_sha256"]
    snapshot.write_text(text.replace(hashed, "0" * 64))  # the same recipe on other code
    with pytest.raises(ProfileError, match="no holdout run"):
        rig.promote("strat")
    snapshot.write_text(text.replace(COMMIT, "f" * 40))  # another commit is still the same run
    rig.promote("strat")


def test_the_latest_matching_holdout_run_is_recorded(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    later = rig.run(holdout=True)
    assert rig.promote("strat").holdout_run_id == later.name


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s.__setitem__("holdout", True), "promote the in-sample run"),
        (lambda s: s["recipe"].update({"risk": None, "limits": None}), "no risk preset"),
        (lambda s: s["recipe"]["limits"].__setitem__("max_order_notional", 1.0), "limits"),
        (lambda s: s["run"].__setitem__("git_commit", COMMIT + "-dirty"), "clean git tree"),
        (lambda s: s["run"].__setitem__("git_commit", None), "clean git tree"),
        (lambda s: s["run"].__setitem__("strategy_source_sha256", "0" * 64), "changed since"),
        (lambda s: s["run"].pop("strategy_source_sha256"), "changed since"),
        (lambda s: s.pop("recipe"), "does not load"),
    ],
)
def test_promotion_refuses_a_run_that_is_not_ready(tmp_path: Path, mutate, match: str) -> None:  # type: ignore[no-untyped-def]
    rig = Rig(tmp_path)
    rig.edit_snapshot(mutate)
    with pytest.raises(ProfileError, match=match):
        rig.promote("strat")
    assert not rig.execution.profiles_dir.exists()


def test_promotion_refuses_when_the_execution_limits_changed(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    path = rig.config / "execution.yaml"
    path.write_text(
        path.read_text().replace("max_order_notional: 5000.0", "max_order_notional: 4000.0")
    )
    with pytest.raises(ProfileError, match="limits in execution.yaml changed"):
        rig.promote("strat", execution=load_execution(rig.config))


def test_only_the_example_strategy_can_be_plumbing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.edit_snapshot(lambda s: s["recipe"]["strategy"].__setitem__("strategy", "momentum"))
    with pytest.raises(ProfileError, match="only example_buy_and_hold may be promoted"):
        rig.promote("strat", plumbing=True)


def test_promotion_refuses_a_folder_that_is_not_a_run(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(ProfileError, match="not a run folder"):
        promote_run(tmp_path, "strat", execution=rig.execution, loader=rig.loader())


@pytest.mark.parametrize("name", ["", "Strat", "my strat", "-x", "x-", "a--b", "a.b", "x" * 41])
def test_promotion_refuses_a_bad_name(tmp_path: Path, name: str) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(ProfileError, match="invalid profile name"):
        rig.promote(name)


def test_promotion_refuses_a_strategy_that_fails_the_restart_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import harness.profiles as module

    rig = Rig(tmp_path)

    def failing(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RestartEquivalenceError("keeps state")

    monkeypatch.setattr(module, "check_restart_equivalence", failing)
    with pytest.raises(RestartEquivalenceError):
        rig.promote("strat")
    assert not rig.execution.profiles_dir.exists()


def test_the_same_content_is_the_same_profile_and_is_not_written_twice(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    first = rig.promote("strat")
    with pytest.raises(ProfileError, match="already exists"):
        rig.promote("strat")
    rig.now = NOW + dt.timedelta(minutes=1)
    second = rig.promote("strat")  # a later promotion is a different profile
    assert second.profile_id != first.profile_id
    assert [p.profile_id for p in list_profiles(rig.execution.profiles_dir)] == sorted(
        [first.profile_id, second.profile_id]
    )


def test_a_failed_promotion_leaves_no_folder_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import harness.profiles as module

    rig = Rig(tmp_path)
    real = module._write_atomic

    def failing(path: Path, data: bytes, **kw) -> None:  # type: ignore[no-untyped-def]
        if path.name == "profile.json":
            raise OSError("disk full")
        real(path, data, **kw)

    monkeypatch.setattr(module, "_write_atomic", failing)
    with pytest.raises(OSError, match="disk full"):
        rig.promote("strat")
    assert list(rig.execution.profiles_dir.iterdir()) == []
    monkeypatch.setattr(module, "_write_atomic", real)
    profile = rig.promote("strat")  # the retry is not blocked by a leftover folder
    assert rig.load(profile.profile_id) == profile


def test_a_holdout_run_folder_that_is_not_in_the_holdout_log_does_not_count(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    log = rig.recipe.data.holdout_log
    lines = log.read_text().splitlines(keepends=True)
    assert len(lines) == 2 and rig.holdout_dir is not None and rig.holdout_dir.name in lines[1]
    log.write_text(lines[0])  # the folder is still there; the record is not
    with pytest.raises(ProfileError, match="not in the holdout log"):
        rig.promote("strat")
    log.write_bytes(b"run_id\n\xff\xfe\x00\n")
    with pytest.raises(ProfileError, match="cannot read the holdout log"):
        rig.promote("strat")
    log.write_text("".join(lines))
    assert rig.promote("strat").holdout_run_id == rig.holdout_dir.name


# --------------------------------------------------------------------------- tampering


def test_editing_the_profile_file_is_detected(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    path = rig.execution.profiles_dir / profile.profile_id / "profile.json"
    original = path.read_text()
    edits = {
        "max_position_pct": original.replace(
            '"max_position_pct": 0.25', '"max_position_pct": 0.26'
        ),
        "plumbing": original.replace('"plumbing": false', '"plumbing": true'),
        "hash": original.replace(profile.content_hash, "0" * 64),
        "id": original.replace(
            f'"profile_id": "{profile.profile_id}"', '"profile_id": "strat-00000000"'
        ),
        "truncated": original[:-40],
    }
    for label, edited in edits.items():
        assert edited != original, label
        _writable(path)
        path.write_text(edited)
        with pytest.raises(ProfileTampered):
            rig.load(profile.profile_id)
        assert paper_access_log_problems(
            _log_with(rig, profile.profile_id), rig.execution.profiles_dir
        )
    _writable(path)
    path.write_text(original)
    assert rig.load(profile.profile_id) == profile


def test_editing_the_frozen_strategy_copy_is_detected(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    path = rig.execution.profiles_dir / profile.profile_id / "strategy.py"
    _writable(path)
    path.write_text(path.read_text() + "\n")  # one byte
    with pytest.raises(ProfileTampered, match="strategy.py does not match"):
        rig.load(profile.profile_id)


def test_a_changed_strategy_module_does_not_stop_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Research goes on while a profile trades: `strategies/<name>.py` may change. The
    # profile runs its own frozen copy, so it still loads, and what is built is the copy.
    import harness.profiles as module

    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    changed = tmp_path / "example_buy_and_hold.py"
    changed.write_text("# edited after promotion\n")
    monkeypatch.setattr(module, "_strategy_file", lambda recipe: changed)
    assert rig.load(profile.profile_id) == profile


def test_the_profile_trades_its_frozen_copy_of_the_strategy(tmp_path: Path) -> None:
    from strategies.example_buy_and_hold import ExampleBuyAndHold

    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    profiles_dir = rig.execution.profiles_dir
    copy = profiles_dir / profile.profile_id / "strategy.py"
    strategy = build_profile_strategy(profile, profiles_dir=profiles_dir)
    assert Path(inspect.getsourcefile(type(strategy)) or "") == copy
    assert type(strategy) is not ExampleBuyAndHold  # not the module in strategies/
    assert type(strategy).__name__ == "ExampleBuyAndHold"
    assert strategy.strategy_name() == "example_buy_and_hold"  # what reports call it
    assert strategy.params.model_dump() == profile.recipe.strategy.params
    # Built twice, it is the same class: the module is imported once per profile.
    assert type(build_profile_strategy(profile, profiles_dir=profiles_dir)) is type(strategy)

    _writable(copy)
    copy.write_text(copy.read_text() + "\n")
    with pytest.raises(ProfileTampered, match="strategy.py does not match"):
        build_profile_strategy(profile, profiles_dir=profiles_dir)
    with pytest.raises(ProfileTampered):
        build_profile_strategy(profile.model_copy(update={"name": "x"}), profiles_dir=profiles_dir)


def test_a_strategy_file_goes_through_the_same_import_checks(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("import os\nfrom strategies.base import Strategy\n")
    with pytest.raises(StrategyLoadError, match="import os is not allowed"):
        load_strategy_file(bad, "frozen_test.bad")
    with pytest.raises(StrategyLoadError, match="cannot read"):
        load_strategy_file(tmp_path / "missing.py", "frozen_test.missing")
    empty = tmp_path / "empty.py"
    empty.write_text("X = 1\n")
    with pytest.raises(StrategyLoadError, match="exactly one Strategy subclass"):
        load_strategy_file(empty, "frozen_test.empty")
    broken = tmp_path / "broken.py"
    broken.write_text("raise RuntimeError('at import')\n")
    with pytest.raises(RuntimeError, match="at import"):
        load_strategy_file(broken, "frozen_test.broken")
    import sys

    assert "frozen_test.bad" not in sys.modules and "frozen_test.broken" not in sys.modules
    del sys.modules["frozen_test.empty"]


def test_a_renamed_profile_folder_is_detected(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    folder = rig.execution.profiles_dir / profile.profile_id
    folder.rename(folder.with_name("other-" + profile.content_hash[:8]))
    with pytest.raises(ProfileTampered, match="holds profile"):
        rig.load("other-" + profile.content_hash[:8])


def test_unknown_and_malformed_profile_ids_are_refused_before_any_file_is_read(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(ProfileError, match="no profile"):
        rig.load("strat-00000000")
    for bad in ("strat", "../strat-00000000", ""):
        with pytest.raises(ProfileError, match="not a profile id"):
            rig.load(bad)


def test_a_profile_model_cannot_be_built_with_a_wrong_hash_or_id(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    fields = profile.model_dump(mode="json")
    with pytest.raises(ValueError, match="content hash"):
        FrozenProfile.model_validate({**fields, "created_at": NOW + dt.timedelta(seconds=1)})
    with pytest.raises(ValueError, match="profile id"):
        FrozenProfile.model_validate({**fields, "profile_id": "strat-00000000"})
    with pytest.raises(ValueError, match="plumbing"):
        FrozenProfile.model_validate({**fields, "holdout_run_id": None, "content_hash": "0" * 64})


# --------------------------------------------------------------------------- paper access


def _log_with(rig: Rig, *profile_ids: str) -> Path:
    log = recipe(rig.config).data.paper_access_log
    loader = rig.loader()
    for profile_id in profile_ids:
        loader.load(["SPY"], paper=PaperAccess(profile_id=profile_id))
    return log


def test_paper_access_comes_only_from_a_verified_profile(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    access = paper_access_for(profile)
    assert access == PaperAccess(profile_id=profile.profile_id)
    forged = profile.model_copy(update={"name": "other"})  # bypasses validation
    with pytest.raises(ProfileTampered):
        paper_access_for(forged)


def test_the_paper_access_log_is_checked_against_the_profiles(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    log = _log_with(rig, profile.profile_id)
    assert paper_access_log_problems(log, rig.execution.profiles_dir) == []
    _log_with(rig, "ghost-12345678")
    assert len(read_paper_access_log(log)) == 2
    (problem,) = paper_access_log_problems(log, rig.execution.profiles_dir)
    assert problem.startswith("2026-07-01: 'ghost-12345678' read data through 2026-06-30")
    assert "no profile" in problem
    assert paper_access_log_problems(tmp_path / "missing.csv", rig.execution.profiles_dir) == []


# --------------------------------------------------------------------------- restart check


def test_restart_check_passes_for_strategies_that_compute_from_the_view(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    check_restart_equivalence(rig.recipe, rig.loader())  # the example strategy
    check_restart_equivalence(recipe(rig.config), rig.loader(), build=Churn)
    with pytest.raises(ValueError):
        check_restart_equivalence(rig.recipe, rig.loader(), sessions=0)


def test_restart_check_survives_an_order_canceled_for_a_rebuilt_engine(tmp_path: Path) -> None:
    # A symbol that stops trading inside the rebuilt window: its exit is canceled at the
    # next open with no fill. A fresh engine never submitted that order, so it must be
    # told about it, or the cancel would be reported for an unknown order.
    config = tmp_config(tmp_path)
    fetcher = FakeFetcher()
    fetcher.delisted["MSFT"] = dt.date(2026, 3, 10)
    r = recipe(config)
    loader = DataLoader(r.dates, r.data, fetcher_factory=lambda: fetcher, now=lambda: NOW)
    check_restart_equivalence(r, loader, build=Churn)


def test_restart_check_fails_for_a_strategy_that_keeps_state(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    with pytest.raises(RestartEquivalenceError, match="orders on .* differ"):
        check_restart_equivalence(recipe(rig.config), rig.loader(), build=CallCounter)


def test_restart_check_reads_in_sample_data_only(tmp_path: Path) -> None:
    rig = Rig(tmp_path, holdout=False)
    rig.fetcher.requests.clear()
    check_restart_equivalence(rig.recipe, rig.loader())
    assert all(end < dt.date(2026, 4, 1) for _, _, end in rig.fetcher.requests)
    assert read_paper_access_log(recipe(rig.config).data.paper_access_log) == []
    assert not recipe(rig.config).data.holdout_log.exists()


# --------------------------------------------------------------------------- activation


def test_activation_binds_the_account_and_seeds_the_risk_state(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    state_dir = rig.execution.state_dir
    assert active_profile(state_dir) is None
    active = rig.activate(profile.profile_id, rig.venue(cash=12_000.0))
    assert active == ActiveProfile(
        profile_id=profile.profile_id,
        account_id="FAKE-PAPER",
        cash_offset=2_000.0,
        approval="entries",
        activated_at=T0,
        operator="david",
    )
    assert active_profile(state_dir) == active
    # Risk starts from what the ledger will say the account is worth: 12,000 less the
    # 2,000 that is not the profile's, not from the recipe's number by name.
    assert load_risk_state(state_dir, profile.profile_id) == RiskState.initial(10_000.0)
    assert json.loads((state_dir / "active.json").read_text())["profile_id"] == profile.profile_id


def test_activation_with_exactly_the_starting_cash_has_no_offset(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    active = rig.activate(profile.profile_id, rig.venue(cash=10_000.0), approval="off")
    assert active.cash_offset == 0.0 and active.approval == "off"


def test_risk_is_seeded_from_the_accounts_equity_not_the_recipe(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")

    class Accrued(FakeVenue):
        """Flat within the cash tolerance: a few cents of interest not yet in cash."""

        def account(self):  # type: ignore[no-untyped-def]
            return super().account().model_copy(update={"equity": self.cash + 0.5})

    rig.activate(profile.profile_id, Accrued(clock=T0, cash=12_000.0))
    state = load_risk_state(rig.execution.state_dir, profile.profile_id)
    assert state == RiskState.initial(10_000.5)


def test_activation_refuses_an_account_that_is_not_flat_or_not_funded(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")

    held = rig.venue()
    held.hold("AAPL", 3, 100.0)
    with pytest.raises(ProfileError, match="not flat: it holds AAPL 3"):
        rig.activate(profile.profile_id, held)

    working = rig.venue()
    working.set_price("SPY", 100.0)
    working.place(
        VenueOrder(
            client_order_id="x-1",
            symbol="SPY",
            side="buy",
            quantity=1,
            type="limit",
            limit_price=90.0,  # below the price: stays open
            time_in_force="day",
        )  # fmt: skip
    )
    with pytest.raises(ProfileError, match="open orders: x-1"):
        rig.activate(profile.profile_id, working)

    with pytest.raises(ProfileError, match="less than the profile's starting cash"):
        rig.activate(profile.profile_id, rig.venue(cash=9_999.0))

    class Unbalanced(FakeVenue):
        def account(self):  # type: ignore[no-untyped-def]
            return super().account().model_copy(update={"equity": self.cash + 5.0})

    with pytest.raises(ProfileError, match="equity .* differs from cash"):
        rig.activate(profile.profile_id, Unbalanced(clock=T0, cash=12_000.0))
    assert active_profile(rig.execution.state_dir) is None
    assert not (rig.execution.state_dir / profile.profile_id).exists()


def test_activation_uses_the_venues_own_setup_gate(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    for field, value, match in [
        ("trading_blocked", True, "blocked"),
        ("no_shorting", False, "shorting is allowed"),
        ("margin_multiplier", 2.0, "margin multiplier is 2"),
    ]:
        venue = rig.venue()
        setattr(venue, field, value)
        with pytest.raises(ProfileError, match=f"did not pass the venue's setup check.*{match}"):
            rig.activate(profile.profile_id, venue)
        assert ("verify_account", "") in venue.calls
    assert active_profile(rig.execution.state_dir) is None

    # The account field alone is not the gate: a margin account can report shorting
    # enabled with the no_shorting setting on, and the venue's check is what counts.
    reports_shorting = rig.venue()
    reports_shorting.shorting_enabled = True
    rig.activate(profile.profile_id, reports_shorting)


def test_another_active_profile_is_replaced_only_on_request_and_on_the_record(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path)
    first = rig.promote("strat")
    rig.now = NOW + dt.timedelta(minutes=1)
    second = rig.promote("strat")
    state_dir = rig.execution.state_dir
    assert read_activations(state_dir) == []
    a = rig.activate(first.profile_id, rig.venue())

    venue = rig.venue()
    with pytest.raises(ProfileError, match=f"profile {first.profile_id} is active"):
        rig.activate(second.profile_id, venue)
    assert venue.calls == []
    assert active_profile(state_dir) == a
    assert not (state_dir / second.profile_id).exists()

    b = rig.activate(second.profile_id, rig.venue(), replace=True, operator="opus")
    assert active_profile(state_dir) == b
    assert read_activations(state_dir) == [
        Activation(active=a, replaced=None),
        Activation(active=b, replaced=first.profile_id),
    ]
    # The same profile again (to change the approval mode) replaces nothing.
    c = rig.activate(second.profile_id, rig.venue(), approval="off")
    assert read_activations(state_dir)[-1] == Activation(active=c, replaced=None)


def test_activation_refuses_while_killed_or_after_the_profile_has_traded(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    state_dir = rig.execution.state_dir
    switch = KillSwitch(state_dir / "KILL")
    switch.engage("test", "operator:test")
    with pytest.raises(ProfileError, match="kill switch is engaged"):
        rig.activate(profile.profile_id, rig.venue())
    switch.reset("david", profile.profile_id, profile.profile_id)

    (state_dir / profile.profile_id).mkdir(parents=True)
    (state_dir / profile.profile_id / "journal.jsonl").write_text("")
    with pytest.raises(ProfileError, match="traded before"):
        rig.activate(profile.profile_id, rig.venue())
    assert active_profile(state_dir) is None


def test_activation_needs_a_verified_profile_and_an_operator(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    profile = rig.promote("strat")
    venue = rig.venue()
    with pytest.raises(ProfileError, match="no profile"):
        rig.activate("ghost-12345678", venue)
    with pytest.raises(ProfileError, match="operator"):
        rig.activate(profile.profile_id, venue, operator="  ")
    assert venue.calls == []  # refused before the venue was asked anything


def test_risk_state_round_trips_and_a_missing_one_is_an_error(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    halted = RiskState(peak_equity=10_000.0, prior_equity=8_000.0, marked_on=dt.date(2026, 7, 1),
                       halted=True, halt_reason="drawdown")  # fmt: skip
    save_risk_state(halted, state_dir, "strat-0a1b2c3d")
    assert load_risk_state(state_dir, "strat-0a1b2c3d") == halted
    with pytest.raises(ProfileError, match="no usable risk state"):
        load_risk_state(state_dir, "other-0a1b2c3d")
    (state_dir / "strat-0a1b2c3d" / "risk_state.json").write_text("{")
    with pytest.raises(ProfileError, match="no usable risk state"):
        load_risk_state(state_dir, "strat-0a1b2c3d")
    with pytest.raises(ProfileError, match="not a profile id"):
        save_risk_state(halted, state_dir, "../x")


def test_a_broken_active_file_is_an_error_not_none(tmp_path: Path) -> None:
    (tmp_path / "active.json").write_text("{}")
    with pytest.raises(ProfileError, match="cannot be read"):
        active_profile(tmp_path)
