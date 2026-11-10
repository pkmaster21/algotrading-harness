import datetime as dt
import shutil
from pathlib import Path

import pytest

from harness.config import ConfigError, Recipe, load_recipe

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
