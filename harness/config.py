"""Build a run's recipe from the YAML in `config/`.

`harness/config.py` is the only code that reads config files; every component receives
typed, frozen models. Every value is required in YAML; nothing here has a default, so
config values live in `config/` and are never hardcoded.

Layout:

    config/
      strategies/<name>.yaml   strategy presets, picked per run (UI-editable later)
      universe/<name>.yaml     symbol lists, picked per run
      dates.yaml  costs.yaml  construction.yaml  backtest.yaml  data.yaml
                               fixed settings, shared by every run (changed through review)

A run names its presets; `load_recipe` resolves them, together with the fixed settings,
into one `Recipe` holding every value the run uses. The run saves that resolved recipe,
so editing a preset later never changes what an old run means.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, JsonValue, ValidationError, model_validator

from harness.models import FrozenModel, Stock


class ConfigError(ValueError):
    """A config file is missing or invalid."""


# Preset names become file names and, later, are sent by the UI, so they are restricted
# to a safe character set (no path separators, no leading dot).
PRESET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_STRATEGY_MODULE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _check_preset_name(name: str) -> str:
    if not PRESET_NAME_RE.match(name):
        raise ValueError(f"invalid preset name {name!r}: use lowercase letters, digits, '-' or '_'")
    return name


class UniverseConfig(FrozenModel):
    name: str
    symbols: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_preset_name(self.name)
        for s in self.symbols:
            Stock(symbol=s)  # validates ticker format
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("universe contains duplicate symbols")
        return self


class StrategyPreset(FrozenModel):
    """A named strategy configuration: which strategy module runs, with which params.

    `params` is checked here only for shape: values must be JSON types (numbers, strings,
    booleans, null, lists, mappings), so the saved recipe reloads to an identical one.
    A YAML date, for example, is rejected; write it as a quoted string. The strategy's
    own `Params` model validates the values when the run entry point loads the strategy,
    before any data is loaded.
    """

    name: str
    strategy: str  # module name in strategies/, e.g. "rsi_pullback"
    params: dict[str, JsonValue]

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_preset_name(self.name)
        if not _STRATEGY_MODULE_RE.match(self.strategy):
            raise ValueError(f"invalid strategy module name {self.strategy!r}")
        return self


class DateConfig(FrozenModel):
    data_start: dt.date
    in_sample_end: dt.date
    holdout_start: dt.date
    holdout_end: dt.date | None  # None: latest completed session

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.data_start >= self.in_sample_end:
            raise ValueError("data_start must be before in_sample_end")
        if self.holdout_start != self.in_sample_end + dt.timedelta(days=1):
            raise ValueError("holdout_start must be the day after in_sample_end")
        if self.holdout_end is not None and self.holdout_end < self.holdout_start:
            raise ValueError("holdout_end must be on or after holdout_start")
        return self


class CostConfig(FrozenModel):
    slippage_bps: float = Field(ge=0, le=1000)  # per side; 5.0 = 0.05%
    commission_per_order: float = Field(ge=0)


class ConstructionConfig(FrozenModel):
    max_positions: int = Field(ge=1)
    # Force-close a held symbol after this many consecutive sessions with no bar.
    max_missing_sessions: int = Field(ge=1)


class BacktestConfig(FrozenModel):
    starting_cash: float = Field(gt=0)
    # Annual rate as a decimal (0.04 = 4%), used for Sharpe. The bounds catch percent typos.
    risk_free_rate: float = Field(ge=-0.05, le=0.20)
    benchmark: str
    # Run folders and backtests/index.csv; a relative path resolves like the data paths.
    output_dir: Path

    @model_validator(mode="after")
    def _check(self) -> Self:
        Stock(symbol=self.benchmark)
        return self


class DataConfig(FrozenModel):
    # Relative paths in data.yaml are resolved against the project root (the parent of
    # the config directory), never the working directory, so every run uses the same
    # cache and the same holdout log wherever it is launched from.
    cache_dir: Path
    holdout_log: Path
    # Fixed: fills use the open, and IEX-only OHLC is not the consolidated open.
    feed: Literal["sip"]
    adjustment: Literal["all"]  # splits and dividends
    batch_size: int = Field(ge=1)
    max_requests_per_minute: int = Field(ge=1)
    overlap_bars: int = Field(ge=1)
    full_refresh_days: int = Field(ge=1)


class Recipe(FrozenModel):
    """Every config value one run uses: the chosen presets plus the fixed settings.

    Risk presets join in slice 2; until then runs have no risk layer.
    """

    strategy: StrategyPreset
    universe: UniverseConfig
    dates: DateConfig
    costs: CostConfig
    construction: ConstructionConfig
    backtest: BacktestConfig
    data: DataConfig


_FIXED_SECTIONS = ("dates", "costs", "construction", "backtest", "data")
# Paths resolved against the project root, per section.
_PATH_FIELDS = {"data": ("cache_dir", "holdout_log"), "backtest": ("output_dir",)}

# config/ sits at the project root, next to harness/.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    try:
        with path.open() as fh:
            loaded = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML:\n{exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return loaded


def _read_preset(config_dir: Path, kind: str, name: str) -> dict[str, Any]:
    try:
        _check_preset_name(name)
    except ValueError as exc:
        raise ConfigError(f"{kind} preset: {exc}") from exc
    raw = _read_yaml(config_dir / kind / f"{name}.yaml")
    if "name" in raw:
        raise ConfigError(f"{kind}/{name}.yaml must not set 'name'; the file name is the name")
    return {"name": name, **raw}


def _resolve_paths(section: dict[str, Any], keys: tuple[str, ...], root: Path) -> dict[str, Any]:
    resolved = dict(section)
    for key in keys:
        value = resolved.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            resolved[key] = str(root / value)
    return resolved


def load_recipe(strategy: str, universe: str, config_dir: Path = DEFAULT_CONFIG_DIR) -> Recipe:
    """Resolve the named presets and the fixed settings in `config_dir` into one Recipe.

    Relative paths (data cache, holdout log, run outputs) come back absolute, resolved
    against `config_dir`'s parent.
    Raises ConfigError naming the offending file on any missing file, malformed YAML,
    bad preset name, or invalid value.
    """
    config_dir = config_dir.resolve()
    raw: dict[str, Any] = {
        "strategy": _read_preset(config_dir, "strategies", strategy),
        "universe": _read_preset(config_dir, "universe", universe),
    }
    for section in _FIXED_SECTIONS:
        raw[section] = _read_yaml(config_dir / f"{section}.yaml")
    for section, keys in _PATH_FIELDS.items():
        raw[section] = _resolve_paths(raw[section], keys, config_dir.parent)
    try:
        return Recipe.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {config_dir}:\n{exc}") from exc
