"""Run a backtest: `run_backtest(recipe)` plus a thin CLI over it.

    uv run python -m harness.run --strategy <preset> [--universe <preset>] [--holdout]

`--strategy` names a strategy preset in `config/strategies/`; the preset names the
strategy module in `strategies/` and its params. The strategy is loaded and its params
validated before any data is loaded, so a bad preset fails immediately.

Strategy modules are checked before they are imported: every import must be on an
allowlist, and calls to builtins that reach files or code execution are rejected. This
catches accidental file, environment, network, or broker access; it is not a sandbox.
"""

from __future__ import annotations

import argparse
import ast
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
    {"open", "exec", "eval", "compile", "__import__", "breakpoint", "input", "globals", "vars"}
)
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


def check_strategy_source(source: str, filename: str = "<strategy>") -> list[str]:
    """Return a list of problems with a strategy module's source (empty if allowed)."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems: list[tuple[int, str]] = []  # (line, message), sorted by line at the end
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_allowed(alias.name):
                    problems.append((node.lineno, f"import {alias.name} is not allowed"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                problems.append((node.lineno, "relative imports are not allowed"))
            elif not _import_allowed(module):
                problems.append((node.lineno, f"from {module} import ... is not allowed"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                problems.append((node.lineno, f"call to {node.func.id}() is not allowed"))
    return [f"line {line}: {message}" for line, message in sorted(problems)]


def _import_allowed(module: str) -> bool:
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
    return classes[0]


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
    run_id = f"{timestamp}_{recipe.strategy.name}"
    commit = git_commit()
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
    )
    benchmark = run_benchmark(panel, recipe, trade_from)

    out = recipe.backtest.output_dir
    run_dir = out / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = out / f"{run_id}-{suffix}"
    run_info = {
        "run_id": run_dir.name,
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
    append_index(out / "index.csv", summary, run_info["timestamp_utc"], run_dir)
    return {**summary, "run_dir": str(run_dir)}


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.run", description="Run a backtest.")
    parser.add_argument("--strategy", required=True, help="strategy preset in config/strategies/")
    parser.add_argument(
        "--universe", default=DEFAULT_UNIVERSE, help=f"universe preset (default {DEFAULT_UNIVERSE})"
    )
    parser.add_argument(
        "--holdout", action="store_true", help="run on holdout data (logged in the holdout log)"
    )
    args = parser.parse_args(argv)
    try:
        recipe = load_recipe(args.strategy, args.universe)
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
