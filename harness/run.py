"""Run a backtest: `run_backtest(recipe)` plus a thin CLI over it.

    uv run python -m harness.run --strategy <preset> [--universe <preset>] [--holdout]

`--strategy` names a strategy preset in `config/strategies/`; the preset names the
strategy module in `strategies/` and its params. The strategy is loaded and its params
validated before any data is loaded, so a bad preset fails immediately.

Strategy modules are checked before they are imported: every import must be on an
allowlist, and names that reach files, the network, the environment, or code execution
are rejected (builtins like `open`, and I/O functions of the allowed libraries such as
`pd.read_csv` or `np.load`). This catches accidental file, environment, network, or
broker access; it is not a sandbox. `harness.guard` backs it up at run time while
strategy code executes.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import hashlib
import importlib
import importlib.util
import inspect
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from harness.broker.sim import SimBroker
from harness.config import ConfigError, Recipe, load_recipe
from harness.data import (
    PROJECT_ROOT,
    DataError,
    DataLoader,
    HoldoutAccess,
    HoldoutLogError,
    PanelError,
)
from harness.engine import simulate
from harness.report import append_index, run_benchmark, write_run
from harness.risk import RiskManager, RiskState
from strategies.base import Strategy

# Top-level modules a strategy may import, and harness modules it may import from.
ALLOWED_IMPORTS = frozenset(
    {"__future__", "collections", "dataclasses", "enum", "functools", "itertools", "math",
     "statistics", "typing", "numpy", "pandas", "pydantic"}
)  # fmt: skip
ALLOWED_HARNESS_IMPORTS = frozenset(
    {"harness.models", "harness.view", "harness.indicators", "strategies.base"}
)
FORBIDDEN_CALLS = frozenset(
    {"exec", "eval", "compile", "__import__", "breakpoint", "input", "globals", "vars"}
)
# Rejected wherever they appear: as a bare name, an attribute (`typing.sys`,
# `pd.io.common.os`), or an imported name (`from statistics import sys`). Covers aliasing
# such as `o = open`, which a call-only check misses.
FORBIDDEN_NAMES = frozenset({"open", "__builtins__", "builtins", "sys", "os"})
# pydantic.v1 ships BaseSettings, which reads environment variables and .env files.
FORBIDDEN_MODULES = frozenset({"pydantic.v1"})
# Attribute and imported names that mean file or network I/O in the allowed libraries
# (pandas readers and writers, numpy load/save). Matched by name alone, on any object, so
# a strategy's own attribute with one of these names is rejected too: rename it. Only
# `read_` is a prefix (every pandas `read_*` is I/O); load and save are exact names, so
# `self.loaded` or `save_state` stay usable.
IO_NAME_PREFIXES = ("read_",)
IO_NAMES = frozenset(
    {"load", "loads", "loadtxt", "save", "savez", "savez_compressed", "savetxt",
     "fromfile", "tofile", "memmap", "open_memmap", "genfromtxt", "fromregex", "dump",
     "DataSource", "ExcelFile", "ExcelWriter", "HDFStore",
     "to_csv", "to_parquet", "to_pickle", "to_feather", "to_hdf", "to_excel", "to_json",
     "to_html", "to_xml", "to_latex", "to_stata", "to_orc", "to_sql", "to_gbq",
     "to_clipboard", "to_markdown", "to_string"}
)  # fmt: skip
DEFAULT_UNIVERSE = "plumbing-10"


class StrategyLoadError(RuntimeError):
    """A strategy module is missing, imports something disallowed, or has bad params."""


class HoldoutUnavailable(RuntimeError):
    """A holdout run was requested but there are no sessions on or after the holdout start."""


# Errors the CLI reports as one line. Anything else (including HoldoutViolation and
# LookAheadError, which only fire on a harness bug) keeps its full traceback.
USER_ERRORS = (ConfigError, StrategyLoadError, DataError, HoldoutLogError, HoldoutUnavailable,
               PanelError)  # fmt: skip


# --------------------------------------------------------------------------- loading


def _is_io_name(name: str) -> bool:
    return name in IO_NAMES or name.startswith(IO_NAME_PREFIXES)


def _name_problem(name: str) -> str | None:
    """Why `name` may not appear in a strategy (as attribute or imported name), or None."""
    if name in FORBIDDEN_NAMES:
        return f"{name} is not allowed"
    if _is_io_name(name):
        return f"{name} looks like file or network I/O, which strategies may not do"
    return None


def _module_forbidden(module: str) -> bool:
    return any(module == m or module.startswith(m + ".") for m in FORBIDDEN_MODULES)


def check_strategy_source(source: str, filename: str = "<strategy>") -> list[str]:
    """Return a list of problems with a strategy module's source (empty if allowed)."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems: set[tuple[int, str]] = set()  # (line, message); sorted by line at the end
    # Local name -> module it is bound to (`import pydantic as p` binds p to pydantic),
    # collected first so `p.v1` is recognized wherever the import sits in the file.
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                aliases[alias.asname or top] = alias.name if alias.asname else top
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_allowed(alias.name):
                    problems.add((node.lineno, f"import {alias.name} is not allowed"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                problems.add((node.lineno, "relative imports are not allowed"))
                continue
            if not _import_allowed(module):
                problems.add((node.lineno, f"from {module} import ... is not allowed"))
                continue
            for alias in node.names:
                if alias.name == "*":
                    problems.add((node.lineno, f"from {module} import * is not allowed"))
                elif _module_forbidden(f"{module}.{alias.name}"):
                    problems.add((node.lineno, f"import {module}.{alias.name} is not allowed"))
                elif problem := _name_problem(alias.name):
                    problems.add((node.lineno, f"from {module} import {alias.name}: {problem}"))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                problems.add((node.lineno, f"{node.id} is not allowed"))
        elif isinstance(node, ast.Attribute):
            if problem := _name_problem(node.attr):
                problems.add((node.lineno, f"attribute .{node.attr}: {problem}"))
            elif isinstance(node.value, ast.Name) and _module_forbidden(
                f"{aliases.get(node.value.id, '')}.{node.attr}"
            ):
                problems.add((node.lineno, f"{node.value.id}.{node.attr} is not allowed"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                problems.add((node.lineno, f"call to {node.func.id}() is not allowed"))
    return [f"line {line}: {message}" for line, message in sorted(problems)]


def _import_allowed(module: str) -> bool:
    if _module_forbidden(module):
        return False
    if module in ALLOWED_HARNESS_IMPORTS:
        return True
    top = module.split(".", 1)[0]
    return top in ALLOWED_IMPORTS


def load_strategy_class(module_name: str, package: str = "strategies") -> type[Strategy]:
    """Check, import, and return the single Strategy subclass defined in the module."""
    qualified = f"{package}.{module_name}"
    spec = importlib.util.find_spec(qualified)
    if spec is None or spec.origin is None:
        raise StrategyLoadError(f"no strategy module {qualified}")
    source = Path(spec.origin).read_text()
    problems = check_strategy_source(source, spec.origin)
    if problems:
        raise StrategyLoadError(f"{qualified} is not allowed:\n  " + "\n  ".join(problems))
    module = importlib.import_module(qualified)
    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, Strategy)
        and obj is not Strategy
        and obj.__module__ == qualified
    ]
    if len(classes) != 1:
        raise StrategyLoadError(
            f"{qualified} must define exactly one Strategy subclass, found {len(classes)}"
        )
    cls = classes[0]
    if not (isinstance(cls.Params, type) and issubclass(cls.Params, Strategy.Params)):
        raise StrategyLoadError(
            f"{qualified}: {cls.__name__}.Params must subclass Strategy.Params, which makes "
            "params immutable and rejects unknown preset keys"
        )
    return cls


def build_strategy(recipe: Recipe) -> Strategy:
    cls = load_strategy_class(recipe.strategy.strategy)
    try:
        return cls.from_params(recipe.strategy.params)
    except ValidationError as exc:
        raise StrategyLoadError(
            f"invalid params in strategy preset {recipe.strategy.name!r}:\n{exc}"
        ) from exc


def strategy_source_hash(strategy: Strategy) -> str:
    """sha256 of the strategy's module file, so a run records exactly which code ran."""
    source = inspect.getsourcefile(type(strategy))
    if source is None:
        raise StrategyLoadError(f"cannot locate source for {type(strategy).__qualname__}")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- running


def _git_commit(root: Path) -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def _reserve_run_dir(out: Path, run_id: str) -> Path:
    """Create and return a new, empty run folder: `run_id`, or `run_id-2`, `-3`, ... if
    that name is taken. Creating it is what reserves the name."""
    out.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        run_dir = out / (run_id if suffix == 1 else f"{run_id}-{suffix}")
        try:
            run_dir.mkdir()
        except FileExistsError:
            suffix += 1
        else:
            return run_dir


def build_risk(recipe: Recipe) -> RiskManager | None:
    """The risk manager for a backtest of `recipe`, or None if it names no risk preset.

    Entries are judged at the price construction sized them at: the close plus slippage.
    """
    if recipe.risk is None or recipe.limits is None:
        return None
    return RiskManager(
        recipe.risk,
        recipe.limits,
        entry_markup=recipe.costs.slippage_bps / 10_000,
        state=RiskState.initial(recipe.backtest.starting_cash),
    )


def run_backtest(
    recipe: Recipe,
    *,
    holdout: bool = False,
    loader: DataLoader | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    git_commit: Callable[[], str | None] = lambda: _git_commit(PROJECT_ROOT),
) -> dict[str, Any]:
    """Run one backtest end to end and write its outputs. Returns the summary, plus
    `run_dir`.

    In holdout mode, sessions before the holdout start are warm-up: the strategy's view
    holds them, but trading and the equity curve start at the first holdout session,
    with fresh capital.
    """
    strategy = build_strategy(recipe)  # fails before any data is touched
    started = now().astimezone(dt.UTC)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    out = recipe.backtest.output_dir
    # Reserved before the holdout log is written, so the logged run_id is always the
    # name of this run's folder (two runs in the same second get different folders).
    run_dir = _reserve_run_dir(out, f"{timestamp}_{recipe.strategy.name}")
    try:
        return _run_reserved(recipe, strategy, run_dir, started, holdout, loader, git_commit())
    except BaseException:
        with contextlib.suppress(OSError):
            run_dir.rmdir()  # only removes it if nothing was written
        raise


def _run_reserved(
    recipe: Recipe,
    strategy: Strategy,
    run_dir: Path,
    started: dt.datetime,
    holdout: bool,
    loader: DataLoader | None,
    commit: str | None,
) -> dict[str, Any]:
    run_id = run_dir.name
    access = (
        HoldoutAccess(run_id=run_id, strategy=recipe.strategy.name, git_commit=commit)
        if holdout
        else None
    )
    loader = loader or DataLoader(recipe.dates, recipe.data)
    symbols = list(dict.fromkeys([*recipe.universe.symbols, recipe.backtest.benchmark]))
    panel = loader.load(symbols, holdout=access)

    trade_from = 0
    if holdout:
        later = [i for i, d in enumerate(panel.dates) if d >= recipe.dates.holdout_start]
        if not later:
            raise HoldoutUnavailable("no holdout sessions available yet")
        trade_from = later[0]

    result = simulate(
        strategy,
        panel,
        SimBroker(recipe.backtest.starting_cash, recipe.costs),
        recipe.universe.symbols,
        recipe.construction,
        recipe.costs,
        trade_from,
        build_risk(recipe),
    )
    benchmark = run_benchmark(panel, recipe, trade_from)

    run_info = {
        "run_id": run_id,
        "timestamp_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": result.strategy,
        "strategy_class": f"{type(strategy).__module__}.{type(strategy).__qualname__}",
        # Strategies are gitignored (except the example), so the git commit alone does not
        # identify the code that ran; this hash does.
        "strategy_source_sha256": strategy_source_hash(strategy),
        "git_commit": commit,
        "python": sys.version.split()[0],
    }
    summary = write_run(run_dir, result, benchmark, recipe, holdout=holdout, run_info=run_info)
    append_index(run_dir.parent / "index.csv", summary, run_info["timestamp_utc"], run_dir)
    return {**summary, "run_dir": str(run_dir)}


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.run", description="Run a backtest.")
    parser.add_argument("--strategy", required=True, help="strategy preset in config/strategies/")
    parser.add_argument(
        "--universe", default=DEFAULT_UNIVERSE, help=f"universe preset (default {DEFAULT_UNIVERSE})"
    )
    parser.add_argument(
        "--risk", default=None, help="risk preset in config/risk/ (default: no risk limits)"
    )
    parser.add_argument(
        "--holdout", action="store_true", help="run on holdout data (logged in the holdout log)"
    )
    args = parser.parse_args(argv)
    try:
        recipe = load_recipe(args.strategy, args.universe, risk=args.risk)
        summary = run_backtest(recipe, holdout=args.holdout)
    except USER_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bench = summary["benchmark"]
    print(f"run folder: {summary['run_dir']}")
    print(f"{summary['start']} to {summary['end']}  holdout={summary['holdout']}")
    print(f"total return {_pct(summary['total_return'])}  (SPY {_pct(bench['total_return'])})")
    print(f"max drawdown {_pct(summary['max_drawdown'])}  (SPY {_pct(bench['max_drawdown'])})")
    print(f"sharpe {_num(summary['sharpe'])}  (SPY {_num(bench['sharpe'])})")
    print(f"trades {summary['trade_count']}  win rate {_pct(summary['win_rate'])}")
    return 0


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
