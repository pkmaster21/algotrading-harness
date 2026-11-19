"""Build a run's recipe from the YAML in `config/`.

`harness/config.py` is the only code that reads config files; every component receives
typed, frozen models. Every value is required in YAML; nothing here has a default, so
config values live in `config/` and are never hardcoded.

Layout:

    config/
      strategies/<name>.yaml   strategy presets, picked per run (UI-editable later)
      risk/<name>.yaml         risk presets (hard limits), picked per run
      universe/<name>.yaml     symbol lists or point-in-time membership, picked per run
      dates.yaml  costs.yaml  construction.yaml  backtest.yaml  data.yaml
                               fixed settings, shared by every run (changed through review)
      execution.yaml           fixed settings for sending orders to a venue (paper trading)

A run names its presets; `load_recipe` resolves them, together with the fixed settings,
into one `Recipe` holding every value the run uses. The run saves that resolved recipe,
so editing a preset later never changes what an old run means.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, JsonValue, ValidationError, field_validator, model_validator

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


class MembershipInterval(FrozenModel):
    """One stretch of index membership: `start` is the first session in, `end` the first
    session out (None: still a member)."""

    symbol: str
    start: dt.date
    end: dt.date | None

    @model_validator(mode="after")
    def _check(self) -> Self:
        Stock(symbol=self.symbol)  # validates ticker format
        if self.end is not None and not self.start < self.end:
            raise ValueError(f"{self.symbol}: membership start {self.start} not before end")
        return self


class Rename(FrozenModel):
    """`old`'s interval ending on `date` and `new`'s interval starting on it are one company."""

    old: str
    new: str
    date: dt.date


class Correction(FrozenModel):
    """A fix to one upstream membership row: replaced by (`new_start`, `new_end`), or
    dropped when both are None."""

    symbol: str
    start: dt.date
    end: dt.date | None
    new_start: dt.date | None
    new_end: dt.date | None
    reason: str = Field(min_length=1)


class BadBars(FrozenModel):
    """A window in which the data source's bars for `symbol` are not real prices of the
    member company (another security, or one stale price repeated): `start` inclusive,
    `end` exclusive, None meaning unbounded. The engine treats bars in the window as
    missing (docs/pit-universe-design.md)."""

    symbol: str
    start: dt.date | None
    end: dt.date | None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.start is not None and self.end is not None and not self.start < self.end:
            raise ValueError(f"{self.symbol}: bad-bars start {self.start} not before end")
        return self


class SourceFile(FrozenModel):
    name: str
    sha256: str


class Membership(FrozenModel):
    """Point-in-time index membership, resolved: corrections then renames applied,
    intervals that ended before the data start dropped, sorted by (symbol, start).

    The source files are identified by name and sha256, so a run's recipe pins exactly
    which membership it used. `corrections` and `renames` record what was applied; they
    are not reapplied. `bad_bars` is applied later, to the bars.
    """

    file: SourceFile
    corrections_file: SourceFile
    renames_file: SourceFile
    bad_bars_file: SourceFile
    corrections: tuple[Correction, ...]
    renames: tuple[Rename, ...]
    bad_bars: tuple[BadBars, ...]
    reviewed_multi_interval: tuple[str, ...]
    intervals: tuple[MembershipInterval, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        symbols = set(self.symbols)
        if unknown := sorted({b.symbol for b in self.bad_bars} - symbols):
            raise ValueError(f"bad bars listed for tickers that are not members: {unknown}")
        keys = [(iv.symbol, iv.start) for iv in self.intervals]
        if keys != sorted(keys):
            raise ValueError("membership intervals must be sorted by (symbol, start)")
        counts: dict[str, int] = {}
        for prev, cur in zip(self.intervals, self.intervals[1:], strict=False):
            if prev.symbol == cur.symbol and (prev.end is None or prev.end > cur.start):
                raise ValueError(f"{cur.symbol}: membership intervals overlap at {cur.start}")
        for iv in self.intervals:
            counts[iv.symbol] = counts.get(iv.symbol, 0) + 1
        multi = {s for s, n in counts.items() if n > 1}
        reviewed = set(self.reviewed_multi_interval)
        if unreviewed := sorted(multi - reviewed):
            raise ValueError(
                f"tickers with more than one membership interval need a review before use: "
                f"{unreviewed}; check that the bars between intervals are the same company "
                f"(or that the earlier one has none), then add them to reviewed_multi_interval"
            )
        if stale := sorted(reviewed - multi):
            raise ValueError(f"reviewed_multi_interval lists single-interval tickers: {stale}")
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every ticker with an interval, sorted."""
        return tuple(sorted({iv.symbol for iv in self.intervals}))


class UniverseConfig(FrozenModel):
    """A universe preset: a fixed symbol list, or point-in-time `membership`, whose
    tickers are then the symbols."""

    name: str
    symbols: tuple[str, ...] = Field(min_length=1)
    membership: Membership | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_preset_name(self.name)
        for s in self.symbols:
            Stock(symbol=s)  # validates ticker format
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("universe contains duplicate symbols")
        if self.membership is not None and self.symbols != self.membership.symbols:
            raise ValueError("a membership universe's symbols are its membership tickers")
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
    paper_access_log: Path  # one line per profile per day that paper trading read past the cutoff
    # Fixed: fills use the open, and IEX-only OHLC is not the consolidated open.
    feed: Literal["sip"]
    adjustment: Literal["all"]  # splits and dividends
    batch_size: int = Field(ge=1)
    max_requests_per_minute: int = Field(ge=1)
    overlap_bars: int = Field(ge=1)
    full_refresh_days: int = Field(ge=1)


class RiskConfig(FrozenModel):
    """A risk preset: the hard limits the risk manager enforces. Fractions, not percents."""

    name: str
    # Largest single position, as a fraction of equity, at the worst-case entry price.
    max_position_pct: float = Field(gt=0, le=1)
    # Long market value / equity. Capped at 1, so leverage cannot be configured.
    max_exposure_pct: float = Field(gt=0, le=1)
    # A close-to-close loss of this fraction blocks the next session's entries.
    daily_loss_cap_pct: float = Field(gt=0, lt=1)
    # A fall of this fraction from peak equity halts entries until an operator resets it.
    max_drawdown_halt_pct: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_preset_name(self.name)
        if self.max_position_pct > self.max_exposure_pct:
            raise ValueError("max_position_pct cannot be larger than max_exposure_pct")
        return self


MARKET_OPEN = dt.time(9, 30)  # New York; the regular open, also on half days


class OrderLimits(FrozenModel):
    """Fat-finger limits. Not risk appetite: they catch a bug, whatever preset is chosen."""

    max_order_notional: float = Field(gt=0)  # dollars, per order
    max_orders_per_session: int = Field(ge=1)  # venue orders, retries included


def _seconds(time_of_day: dt.time) -> float:
    return time_of_day.hour * 3600 + time_of_day.minute * 60 + time_of_day.second


class ExecutionConfig(FrozenModel):
    """How approved orders reach the venue. Times are New York wall-clock times."""

    # Entries are limit orders capped at the prior close times (1 + this). The upper
    # bound catches a percent typed as a fraction (2 instead of 0.02).
    entry_limit_pct: float = Field(gt=0, le=0.25)
    approval_deadline: dt.time  # entries still unapproved are canceled
    opg_cutoff: dt.time  # last moment an exit is sent into the opening auction
    exit_deadline: dt.time  # an unfilled auction exit is re-sent as a market order
    session_deadline: dt.time  # anything still working is canceled
    poll_seconds: float = Field(gt=0, le=60)
    max_submit_attempts: int = Field(ge=1, le=10)
    # How long one request to the venue may take. A request that takes longer counts as
    # "outcome unknown", like any other failure to get an answer.
    request_timeout_seconds: float = Field(gt=0, le=60)
    # How long a cancel may take to be confirmed by the venue before trading stops.
    cancel_confirm_seconds: float = Field(gt=0, le=300)
    max_requests_per_minute: int = Field(ge=1, le=200)  # Alpaca allows 200
    cash_tolerance: float = Field(ge=0)  # dollars
    max_clock_skew_seconds: float = Field(gt=0)
    limits: OrderLimits
    # Where frozen profiles and paper trading state live. Relative paths resolve against
    # the project root, like the data paths.
    profiles_dir: Path
    state_dir: Path

    @field_validator(
        "approval_deadline", "opg_cutoff", "exit_deadline", "session_deadline", mode="before"
    )
    @classmethod
    def _time_must_be_text(cls, value: object) -> object:
        # Unquoted, YAML reads 10:30 as the number 630, which would then be taken as
        # seconds after midnight. Only "HH:MM" text is accepted.
        if not isinstance(value, str):
            raise ValueError('write times as quoted text, for example "09:25"')
        return value

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.opg_cutoff < self.exit_deadline < self.session_deadline:
            raise ValueError("need opg_cutoff < exit_deadline < session_deadline")
        if not self.approval_deadline < self.session_deadline:
            raise ValueError("approval_deadline must be before session_deadline")
        # After the open no exit could go into the opening auction at all. The venue's
        # own auction cutoff is earlier still; the adapter checks against that.
        if not self.opg_cutoff < MARKET_OPEN:
            raise ValueError("opg_cutoff must be before the 09:30 open")
        # One order can take every submit attempt, each waiting the whole timeout. That
        # must fit between the auction cutoff and the exit deadline, or a run of
        # timeouts could carry an exit past the point where it is sent as a market order.
        worst = self.request_timeout_seconds * self.max_submit_attempts
        window = _seconds(self.exit_deadline) - _seconds(self.opg_cutoff)
        if not worst < window:
            raise ValueError(
                f"request_timeout_seconds * max_submit_attempts ({worst:g}s) must be less "
                f"than the time from opg_cutoff to exit_deadline ({window:g}s)"
            )
        return self


class Recipe(FrozenModel):
    """Every config value one run uses: the chosen presets plus the fixed settings.

    `risk` is None for a run without a risk preset. A run with one also carries the
    order limits, because the risk manager enforces both.
    """

    strategy: StrategyPreset
    universe: UniverseConfig
    dates: DateConfig
    costs: CostConfig
    construction: ConstructionConfig
    backtest: BacktestConfig
    data: DataConfig
    risk: RiskConfig | None = None
    limits: OrderLimits | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.risk is None) != (self.limits is None):
            raise ValueError("risk and limits are set together or not at all")
        return self


_FIXED_SECTIONS = ("dates", "costs", "construction", "backtest", "data")
# Paths resolved against the project root, per section.
_PATH_FIELDS = {
    "data": ("cache_dir", "holdout_log", "paper_access_log"),
    "backtest": ("output_dir",),
}
_EXECUTION_PATH_FIELDS = ("profiles_dir", "state_dir")

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


_RawInterval = tuple[str, dt.date, dt.date | None]
_MEMBERSHIP_HEADER = ["ticker", "start_date", "end_date"]
_CORRECTIONS_HEADER = [
    "ticker", "start_date", "end_date", "new_start_date", "new_end_date", "reason",
]  # fmt: skip
_RENAMES_HEADER = ["old", "new", "date", "note"]
_BAD_BARS_HEADER = ["ticker", "from_date", "to_date", "reason"]
# yaml key -> header of the CSV file it names
_MEMBERSHIP_FILES = {
    "file": _MEMBERSHIP_HEADER,
    "corrections": _CORRECTIONS_HEADER,
    "renames": _RENAMES_HEADER,
    "bad_bars": _BAD_BARS_HEADER,
}
_MEMBERSHIP_KEYS = {*_MEMBERSHIP_FILES, "reviewed_multi_interval"}


def _read_csv(path: Path, header: list[str]) -> list[list[str]]:
    """Rows of a CSV file whose first line must be exactly `header`."""
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] != header:
        raise ConfigError(f"{path}: the header must be {','.join(header)}")
    for n, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ConfigError(f"{path} line {n}: expected {len(header)} fields, got {len(row)}")
    return rows[1:]


def _date(value: str, path: Path, line: int) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{path} line {line}: {value!r} is not a YYYY-MM-DD date") from exc


def _opt_date(value: str, path: Path, line: int) -> dt.date | None:
    return _date(value, path, line) if value else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_corrections(
    intervals: list[_RawInterval], corrections: list[Correction]
) -> list[_RawInterval]:
    """Replace or drop upstream rows. Each correction must match exactly one row."""
    fixed = list(intervals)
    for c in corrections:
        row = (c.symbol, c.start, c.end)
        if fixed.count(row) != 1:
            raise ValueError(
                f"correction for {c.symbol} {c.start}..{c.end}: expected one matching row, "
                f"found {fixed.count(row)}"
            )
        i = fixed.index(row)
        if c.new_start is None and c.new_end is None:
            del fixed[i]
        elif c.new_start is None:
            raise ValueError(f"correction for {c.symbol} {c.start}: a replacement needs a start")
        else:
            fixed[i] = (c.symbol, c.new_start, c.new_end)
    return fixed


def apply_renames(intervals: list[_RawInterval], renames: list[Rename]) -> list[_RawInterval]:
    """Join each rename chain into one interval under its last ticker.

    A rename links the interval of `old` that ends on `date` to the interval of `new` that
    starts on it; each must exist exactly once, and an interval links at most once each
    way. Intervals not in any rename are returned unchanged.
    """
    nxt: dict[int, int] = {}
    prv: dict[int, int] = {}
    for r in renames:
        olds = [i for i, (s, _, end) in enumerate(intervals) if s == r.old and end == r.date]
        news = [i for i, (s, start, _) in enumerate(intervals) if s == r.new and start == r.date]
        if len(olds) != 1 or len(news) != 1:
            raise ValueError(
                f"rename {r.old} -> {r.new} on {r.date}: expected one {r.old} interval ending "
                f"and one {r.new} interval starting that day, found {len(olds)} and {len(news)}"
            )
        if olds[0] in nxt or news[0] in prv:
            raise ValueError(f"rename {r.old} -> {r.new} on {r.date}: interval already renamed")
        nxt[olds[0]], prv[news[0]] = news[0], olds[0]
    joined: list[_RawInterval] = []
    for i, (_, start, _) in enumerate(intervals):
        if i in prv:
            continue  # not the head of a chain
        last = i
        while last in nxt:
            last = nxt[last]
        joined.append((intervals[last][0], start, intervals[last][2]))
    return joined


def _resolve_membership(
    universe_dir: Path, raw: dict[str, Any], data_start: dt.date
) -> dict[str, Any]:
    """A point-in-time universe preset resolved into `UniverseConfig` fields.

    Order: upstream rows, then corrections (upstream errors), then renames, then intervals
    that ended before the data start are dropped. Bad-bar windows are only recorded here.
    """
    name = raw["name"]
    if set(raw) != {"name", "membership"}:
        raise ConfigError(f"universe/{name}.yaml: a membership universe sets only 'membership'")
    spec = raw["membership"]
    if not isinstance(spec, dict) or set(spec) != _MEMBERSHIP_KEYS:
        raise ConfigError(
            f"universe/{name}.yaml: membership needs exactly {sorted(_MEMBERSHIP_KEYS)}"
        )
    paths: dict[str, Path] = {}
    rows: dict[str, list[list[str]]] = {}
    for key, header in _MEMBERSHIP_FILES.items():
        value = spec[key]
        if not isinstance(value, str) or Path(value).name != value or not value.endswith(".csv"):
            raise ConfigError(f"universe/{name}.yaml: {key} must be a .csv file name in universe/")
        paths[key] = universe_dir / value
        rows[key] = _read_csv(paths[key], header)

    def parsed(key: str) -> list[tuple[int, list[str]]]:
        return list(enumerate(rows[key], start=2))

    p = paths["file"]
    intervals = [(t, _date(a, p, n), _opt_date(b, p, n)) for n, (t, a, b) in parsed("file")]
    p = paths["corrections"]
    corrections = [
        Correction(
            symbol=t,
            start=_date(a, p, n),
            end=_opt_date(b, p, n),
            new_start=_opt_date(na, p, n),
            new_end=_opt_date(nb, p, n),
            reason=why,
        )
        for n, (t, a, b, na, nb, why) in parsed("corrections")
    ]
    p = paths["renames"]
    renames = [Rename(old=o, new=w, date=_date(d, p, n)) for n, (o, w, d, _) in parsed("renames")]
    p = paths["bad_bars"]
    bad_bars = [
        {"symbol": t, "start": _opt_date(a, p, n), "end": _opt_date(b, p, n), "reason": why}
        for n, (t, a, b, why) in parsed("bad_bars")
    ]
    try:
        fixed = apply_corrections(intervals, corrections)
    except ValueError as exc:
        raise ConfigError(f"{paths['corrections']}: {exc}") from exc
    try:
        joined = apply_renames(fixed, renames)
    except ValueError as exc:
        raise ConfigError(f"{paths['renames']}: {exc}") from exc

    # Tickers that left before the data starts are dropped before validation, so the
    # older rows (unused, sometimes odd tickers) never have to pass it.
    kept = sorted(
        (iv for iv in joined if iv[2] is None or iv[2] > data_start), key=lambda iv: iv[:2]
    )
    membership = {
        "file": {"name": spec["file"], "sha256": _sha256(paths["file"])},
        "corrections_file": {"name": spec["corrections"], "sha256": _sha256(paths["corrections"])},
        "renames_file": {"name": spec["renames"], "sha256": _sha256(paths["renames"])},
        "bad_bars_file": {"name": spec["bad_bars"], "sha256": _sha256(paths["bad_bars"])},
        "corrections": [c.model_dump() for c in corrections],
        "renames": [r.model_dump() for r in renames],
        "bad_bars": bad_bars,
        "reviewed_multi_interval": spec["reviewed_multi_interval"],
        "intervals": [{"symbol": s, "start": a, "end": b} for s, a, b in kept],
    }
    symbols = sorted({iv[0] for iv in kept})
    return {"name": name, "symbols": symbols, "membership": membership}


def load_execution(config_dir: Path = DEFAULT_CONFIG_DIR) -> ExecutionConfig:
    """The fixed execution settings in `config_dir`/execution.yaml."""
    config_dir = config_dir.resolve()
    path = config_dir / "execution.yaml"
    raw = _resolve_paths(_read_yaml(path), _EXECUTION_PATH_FIELDS, config_dir.parent)
    try:
        return ExecutionConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{exc}") from exc


def load_recipe(
    strategy: str,
    universe: str,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    *,
    risk: str | None = None,
) -> Recipe:
    """Resolve the named presets and the fixed settings in `config_dir` into one Recipe.

    `risk` names a preset in `risk/`. Without it the recipe has no risk limits and
    execution.yaml is not read, so plain backtests do not depend on that file.

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
    if "membership" in raw["universe"]:
        try:
            data_start = DateConfig.model_validate(raw["dates"]).data_start
        except ValidationError as exc:
            raise ConfigError(f"invalid config in {config_dir / 'dates.yaml'}:\n{exc}") from exc
        raw["universe"] = _resolve_membership(config_dir / "universe", raw["universe"], data_start)
    for section, keys in _PATH_FIELDS.items():
        raw[section] = _resolve_paths(raw[section], keys, config_dir.parent)
    execution: ExecutionConfig | None = None
    if risk is not None:
        execution = load_execution(config_dir)
        raw["risk"] = _read_preset(config_dir, "risk", risk)
        raw["limits"] = execution.limits
    try:
        recipe = Recipe.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {config_dir}:\n{exc}") from exc
    if execution is not None:
        _check_risk_fits(recipe, execution.entry_limit_pct)
    return recipe


def _check_risk_fits(recipe: Recipe, entry_limit_pct: float) -> None:
    """A risk preset must leave room for one position at construction's size.

    Construction targets `equity / max_positions` per position. Risk judges an entry at
    its worst-case price: the close plus slippage in a backtest, the close plus the entry
    limit in paper trading. A preset tighter than that would shrink every entry, which is
    a mistake in the config, not a risk decision. The check uses the larger markup, so a
    recipe that loads for a backtest also fits in paper.
    """
    risk = recipe.risk
    if risk is None:
        return
    max_positions = recipe.construction.max_positions
    markup = max(entry_limit_pct, recipe.costs.slippage_bps / 10_000)
    needed = (1 / max_positions) * (1 + markup)
    if risk.max_position_pct < needed - 1e-9:  # 0.204 must pass for 0.2 * 1.02
        raise ConfigError(
            f"risk preset {risk.name!r}: max_position_pct {risk.max_position_pct} is too "
            f"small for max_positions {max_positions}. One equal-weight position is "
            f"{1 / max_positions:.4f} of equity, or {needed:.4f} at the worst-case entry "
            f"price ({markup:.2%} above the close). Raise max_position_pct to at least "
            f"{needed:.4f}, or raise max_positions."
        )
