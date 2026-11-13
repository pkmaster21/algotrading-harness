import datetime as dt
import shutil
from pathlib import Path

import pytest

from harness.config import (
    ConfigError,
    ExecutionConfig,
    OrderLimits,
    Recipe,
    RiskConfig,
    load_execution,
    load_recipe,
)

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"

# The repo has no strategy presets until the first strategy lands, so tests add one.
STRATEGY_PRESET = "strategy: buy_and_hold\nparams: {}\n"


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, dest)
    (dest / "strategies").mkdir(exist_ok=True)
    (dest / "strategies" / "test-preset.yaml").write_text(STRATEGY_PRESET)
    return dest


def load(config_dir: Path, strategy: str = "test-preset", universe: str = "plumbing-10"):
    return load_recipe(strategy, universe, config_dir)


def test_repo_config_loads_with_agreed_values(config_copy: Path) -> None:
    recipe = load(config_copy)
    assert recipe.universe.name == "plumbing-10"
    assert recipe.universe.symbols == (
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
    )  # fmt: skip
    assert recipe.dates.data_start == dt.date(2015, 1, 1)
    assert recipe.dates.in_sample_end == dt.date(2026, 3, 31)
    assert recipe.dates.holdout_start == dt.date(2026, 4, 1)
    assert recipe.costs.slippage_bps == 5.0
    assert recipe.costs.commission_per_order == 0.0
    assert recipe.construction.max_positions == 5
    assert recipe.construction.max_missing_sessions == 5
    assert recipe.backtest.starting_cash == 10_000.0
    assert recipe.data.feed == "sip"


def test_strategy_preset_resolves_name_module_and_params(config_copy: Path) -> None:
    (config_copy / "strategies" / "rsi-tight.yaml").write_text(
        "strategy: rsi_pullback\nparams:\n  period: 14\n  entry_below: 30\n"
    )
    recipe = load(config_copy, strategy="rsi-tight")
    assert recipe.strategy.name == "rsi-tight"
    assert recipe.strategy.strategy == "rsi_pullback"
    assert recipe.strategy.params == {"period": 14, "entry_below": 30}


def test_recipe_round_trips_through_json(config_copy: Path) -> None:
    # The run saves the resolved recipe; it must reload to an identical recipe.
    recipe = load(config_copy)
    assert Recipe.model_validate_json(recipe.model_dump_json()) == recipe


def test_recipe_does_not_change_when_preset_is_edited_later(config_copy: Path) -> None:
    recipe = load(config_copy)
    saved = recipe.model_dump_json()
    (config_copy / "strategies" / "test-preset.yaml").write_text(
        "strategy: buy_and_hold\nparams: {extra: 1}\n"
    )
    assert recipe.model_dump_json() == saved
    assert load(config_copy).strategy.params == {"extra": 1}


@pytest.mark.parametrize("name", ["../costs", "Upper", ".hidden", "a/b", "", "a b"])
def test_unsafe_preset_names_rejected(config_copy: Path, name: str) -> None:
    with pytest.raises(ConfigError, match="preset name"):
        load(config_copy, strategy=name)


def test_missing_preset_is_reported(config_copy: Path) -> None:
    with pytest.raises(ConfigError, match="universe/nope.yaml"):
        load(config_copy, universe="nope")


def test_preset_file_may_not_set_its_own_name(config_copy: Path) -> None:
    (config_copy / "strategies" / "test-preset.yaml").write_text("name: other\n" + STRATEGY_PRESET)
    with pytest.raises(ConfigError, match="must not set 'name'"):
        load(config_copy)


def test_bad_strategy_module_name_rejected(config_copy: Path) -> None:
    (config_copy / "strategies" / "test-preset.yaml").write_text("strategy: os.path\nparams: {}\n")
    with pytest.raises(ConfigError, match="strategy module name"):
        load(config_copy)


def test_missing_file_is_reported(config_copy: Path) -> None:
    (config_copy / "costs.yaml").unlink()
    with pytest.raises(ConfigError, match="costs.yaml"):
        load(config_copy)


def test_missing_value_is_an_error_not_a_default(config_copy: Path) -> None:
    (config_copy / "costs.yaml").write_text("slippage_bps: 5.0\n")
    with pytest.raises(ConfigError, match="commission_per_order"):
        load(config_copy)


def test_holdout_must_follow_in_sample_end(config_copy: Path) -> None:
    text = (config_copy / "dates.yaml").read_text()
    (config_copy / "dates.yaml").write_text(
        text.replace("holdout_start: 2026-04-01", "holdout_start: 2026-03-15")
    )
    with pytest.raises(ConfigError, match="holdout_start"):
        load(config_copy)


def test_iex_feed_rejected(config_copy: Path) -> None:
    text = (config_copy / "data.yaml").read_text()
    (config_copy / "data.yaml").write_text(text.replace("feed: sip", "feed: iex"))
    with pytest.raises(ConfigError, match="feed"):
        load(config_copy)


def test_duplicate_symbols_rejected(config_copy: Path) -> None:
    (config_copy / "universe" / "plumbing-10.yaml").write_text("symbols: [SPY, SPY]\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load(config_copy)


def test_default_config_dir_does_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harness.config import DEFAULT_CONFIG_DIR

    assert DEFAULT_CONFIG_DIR == REPO_CONFIG
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="strategies/test-preset.yaml"):
        load_recipe("test-preset", "plumbing-10")  # found config/, then the missing preset


def test_relative_data_paths_resolve_against_project_root(
    config_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    recipe = load(config_copy)
    root = config_copy.resolve().parent
    assert recipe.data.cache_dir == root / "data" / "bars"
    assert recipe.data.holdout_log == root / "backtests" / "holdout_log.csv"


def test_absolute_data_paths_kept(config_copy: Path, tmp_path: Path) -> None:
    text = (config_copy / "data.yaml").read_text()
    cache = tmp_path / "shared-cache"
    (config_copy / "data.yaml").write_text(
        text.replace("cache_dir: data/bars", f"cache_dir: {cache}")
    )
    assert load(config_copy).data.cache_dir == cache


@pytest.mark.parametrize("rate", ["4", "-0.1"])
def test_risk_free_rate_must_be_an_annual_decimal(config_copy: Path, rate: str) -> None:
    text = (config_copy / "backtest.yaml").read_text()
    (config_copy / "backtest.yaml").write_text(
        text.replace("risk_free_rate: 0.0", f"risk_free_rate: {rate}")
    )
    with pytest.raises(ConfigError, match="risk_free_rate"):
        load(config_copy)


def test_malformed_yaml_is_a_config_error_naming_the_file(config_copy: Path) -> None:
    (config_copy / "costs.yaml").write_text("slippage_bps: [5\n")
    with pytest.raises(ConfigError, match="costs.yaml is not valid YAML"):
        load(config_copy)


def test_empty_yaml_file_rejected(config_copy: Path) -> None:
    (config_copy / "costs.yaml").write_text("")
    with pytest.raises(ConfigError, match="costs.yaml must contain a mapping"):
        load(config_copy)


def test_non_json_param_rejected_so_saved_recipe_reloads_identically(config_copy: Path) -> None:
    preset = config_copy / "strategies" / "test-preset.yaml"
    preset.write_text("strategy: buy_and_hold\nparams: {start: 2020-01-01}\n")
    with pytest.raises(ConfigError, match="params"):
        load(config_copy)
    preset.write_text("strategy: buy_and_hold\nparams: {start: '2020-01-01', w: [1, 2.5]}\n")
    recipe = load(config_copy)
    assert Recipe.model_validate_json(recipe.model_dump_json()) == recipe


# --------------------------------------------------------------------------- risk presets


def test_recipe_without_a_risk_preset_has_no_risk_and_ignores_execution_yaml(
    config_copy: Path,
) -> None:
    (config_copy / "execution.yaml").unlink()  # plain backtests must not depend on it
    recipe = load(config_copy)
    assert recipe.risk is None and recipe.limits is None


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("conservative", (0.20, 0.80, 0.02, 0.10)),
        ("balanced", (0.25, 1.00, 0.03, 0.15)),
        ("aggressive", (0.30, 1.00, 0.05, 0.25)),
    ],
)
def test_repo_risk_presets_load(config_copy: Path, name: str, values: tuple[float, ...]) -> None:
    recipe = load_recipe("test-preset", "plumbing-10", config_copy, risk=name)
    risk = recipe.risk
    assert risk is not None and risk.name == name
    assert (
        risk.max_position_pct,
        risk.max_exposure_pct,
        risk.daily_loss_cap_pct,
        risk.max_drawdown_halt_pct,
    ) == values
    # A run with a risk preset also carries the order limits it will be held to.
    assert recipe.limits == load_execution(config_copy).limits
    assert Recipe.model_validate(recipe.model_dump(mode="json")) == recipe


def test_missing_or_unsafe_risk_preset_is_reported(config_copy: Path) -> None:
    with pytest.raises(ConfigError, match="reckless.yaml"):
        load_recipe("test-preset", "plumbing-10", config_copy, risk="reckless")
    with pytest.raises(ConfigError, match="invalid preset name"):
        load_recipe("test-preset", "plumbing-10", config_copy, risk="../balanced")


@pytest.mark.parametrize(
    ("line", "replacement", "match"),
    [
        ("max_exposure_pct: 1.00", "max_exposure_pct: 1.5", "max_exposure_pct"),  # leverage
        ("max_position_pct: 0.25", "max_position_pct: 25", "max_position_pct"),  # percent typo
        ("max_position_pct: 0.25", "max_position_pct: 0", "max_position_pct"),
        ("daily_loss_cap_pct: 0.03", "daily_loss_cap_pct: 3", "daily_loss_cap_pct"),
        ("max_drawdown_halt_pct: 0.15", "max_drawdown_halt_pct: 1.0", "max_drawdown_halt_pct"),
        ("daily_loss_cap_pct: 0.03", "", "daily_loss_cap_pct"),  # missing, not defaulted
        ("max_exposure_pct: 1.00", "max_exposure_pct: 0.2", "larger than max_exposure_pct"),
    ],
)
def test_bad_risk_values_rejected(
    config_copy: Path, line: str, replacement: str, match: str
) -> None:
    path = config_copy / "risk" / "balanced.yaml"
    text = path.read_text()
    assert line in text
    path.write_text(text.replace(line, replacement))
    with pytest.raises(ConfigError, match=match):
        load_recipe("test-preset", "plumbing-10", config_copy, risk="balanced")


def test_risk_and_limits_are_set_together() -> None:
    risk = RiskConfig(
        name="x",
        max_position_pct=0.2,
        max_exposure_pct=1.0,
        daily_loss_cap_pct=0.02,
        max_drawdown_halt_pct=0.1,
    )
    limits = OrderLimits(max_order_notional=1000.0, max_orders_per_session=5)
    base = load_recipe("example_buy_and_hold", "plumbing-10").model_dump()
    Recipe.model_validate({**base, "risk": risk, "limits": limits})
    for partial in ({"risk": risk}, {"limits": limits}):
        with pytest.raises(ValueError, match="set together"):
            Recipe.model_validate({**base, **partial})


# --------------------------------------------------------------------------- execution settings


def test_repo_execution_settings_load(config_copy: Path) -> None:
    cfg = load_execution(config_copy)
    assert cfg.entry_limit_pct == 0.02
    assert (cfg.approval_deadline, cfg.opg_cutoff) == (dt.time(9, 20), dt.time(9, 25))
    assert (cfg.exit_deadline, cfg.session_deadline) == (dt.time(9, 40), dt.time(9, 50))
    assert (cfg.poll_seconds, cfg.max_submit_attempts) == (5.0, 3)
    assert (cfg.max_requests_per_minute, cfg.cash_tolerance) == (150, 1.0)
    assert cfg.max_clock_skew_seconds == 30.0
    assert cfg.limits == OrderLimits(max_order_notional=5000.0, max_orders_per_session=25)
    assert ExecutionConfig.model_validate(cfg.model_dump(mode="json")) == cfg


def test_missing_execution_file_is_reported(config_copy: Path) -> None:
    (config_copy / "execution.yaml").unlink()
    with pytest.raises(ConfigError, match="execution.yaml"):
        load_execution(config_copy)
    with pytest.raises(ConfigError, match="execution.yaml"):
        load_recipe("test-preset", "plumbing-10", config_copy, risk="balanced")


@pytest.mark.parametrize(
    ("line", "replacement", "match"),
    [
        # Unquoted, YAML turns 10:30 into the number 630.
        ('session_deadline: "09:50"', "session_deadline: 10:30", "quoted text"),
        ('opg_cutoff: "09:25"', "opg_cutoff: 565", "quoted text"),
        ('exit_deadline: "09:40"', 'exit_deadline: "09:55"', "exit_deadline < session_deadline"),
        ('opg_cutoff: "09:25"', 'opg_cutoff: "09:45"', "opg_cutoff < exit_deadline"),
        ('approval_deadline: "09:20"', 'approval_deadline: "10:00"', "approval_deadline"),
        ('opg_cutoff: "09:25"', 'opg_cutoff: "09:35"', "before the 09:30 open"),
        ("max_requests_per_minute: 150", "max_requests_per_minute: 1500", "max_requests_per"),
        ("entry_limit_pct: 0.02", "entry_limit_pct: 2", "entry_limit_pct"),  # percent typo
        ("entry_limit_pct: 0.02", "entry_limit_pct: 0", "entry_limit_pct"),
        ("max_order_notional: 5000.0", "max_order_notional: 0", "max_order_notional"),
        ("max_orders_per_session: 25", "max_orders_per_session: 0", "max_orders_per_session"),
        ("max_submit_attempts: 3", "max_submit_attempts: 0", "max_submit_attempts"),
        ("poll_seconds: 5.0", "", "poll_seconds"),  # missing, not defaulted
        ("cash_tolerance: 1.00", "cash_tolerance: -1", "cash_tolerance"),
    ],
)
def test_bad_execution_values_rejected(
    config_copy: Path, line: str, replacement: str, match: str
) -> None:
    path = config_copy / "execution.yaml"
    text = path.read_text()
    assert line in text
    path.write_text(text.replace(line, replacement))
    with pytest.raises(ConfigError, match=match):
        load_execution(config_copy)


def test_execution_settings_reject_unknown_keys(config_copy: Path) -> None:
    path = config_copy / "execution.yaml"
    path.write_text(path.read_text() + "live: true\n")
    with pytest.raises(ConfigError, match="live"):
        load_execution(config_copy)
