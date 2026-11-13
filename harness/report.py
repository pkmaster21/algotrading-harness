"""Run outputs: trades, equity curve, metrics, the SPY benchmark, and the run index.

Every file has a fixed schema (the UI reads them later). `summary.json`, `trades.csv`,
`equity.csv`, and `events.csv` contain no timestamps or machine details, so two identical
runs write identical files; run identity (timestamp, git commit) lives in
`config_snapshot.yaml` and the index.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from harness.broker.sim import SimBroker
from harness.config import ConstructionConfig, CostConfig, Recipe
from harness.data import BarPanel
from harness.engine import BacktestResult, simulate
from harness.models import Fill, PortfolioSnapshot, Signal
from harness.view import MarketView
from strategies.base import Strategy

TRADE_COLUMNS = [
    "symbol", "entry_date", "exit_date", "quantity", "entry_price", "exit_price",
    "entry_cost", "exit_cost", "pnl", "return_pct", "holding_days", "exit_reason", "is_open",
]  # fmt: skip
EQUITY_COLUMNS = ["date", "cash", "market_value", "equity", "n_positions"]
EVENT_COLUMNS = ["date", "kind", "symbol", "detail"]
INDEX_COLUMNS = [
    "timestamp", "strategy", "strategy_preset", "risk_preset", "start", "end", "holdout",
    "total_return", "max_drawdown", "trade_count", "run_folder",
]  # fmt: skip
TRADING_DAYS = 252


class ReportError(RuntimeError):
    """Run results are inconsistent with what the report expects."""


# --------------------------------------------------------------------------- benchmark


class BenchmarkBuyAndHold(Strategy):
    """Buy the benchmark on the first traded session and hold. Harness-internal."""

    name = "benchmark_buy_and_hold"

    class Params(Strategy.Params):
        symbol: str

    params: Params

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        if portfolio.positions or view.bar(self.params.symbol) is None:
            return []
        return [Signal.enter(self.params.symbol, strength=1.0, reason="benchmark")]


def run_benchmark(
    panel: BarPanel, recipe: Recipe, trade_from: int, costs: CostConfig | None = None
) -> BacktestResult:
    """SPY buy-and-hold through the same engine, SimBroker, and costs as the strategy:
    signal on the first traded session, fill at the next open, whole shares."""
    symbol = recipe.backtest.benchmark
    return simulate(
        BenchmarkBuyAndHold(BenchmarkBuyAndHold.Params(symbol=symbol)),
        panel.select([symbol]),
        SimBroker(recipe.backtest.starting_cash, costs or recipe.costs),
        [symbol],
        ConstructionConfig(
            max_positions=1, max_missing_sessions=recipe.construction.max_missing_sessions
        ),
        costs or recipe.costs,
        trade_from,
    )


# --------------------------------------------------------------------------- trades


class _OpenTrade:
    """Fills of one position, from the first buy until it is flat again."""

    def __init__(self, entry_order_id: str) -> None:
        self.entry_order_id = entry_order_id
        self.buys: list[Fill] = []
        self.sells: list[Fill] = []

    @property
    def remaining(self) -> int:
        return sum(f.quantity for f in self.buys) - sum(f.quantity for f in self.sells)


def round_trips(result: BacktestResult) -> list[dict[str, Any]]:
    """One trade per position, from the first entry fill until the position is flat.

    An order can fill in several pieces, and an exit can take more than one order (a
    partly filled exit leaves a remainder that is sold later). All of those fills belong
    to one trade: its quantity is the shares bought, and its entry and exit prices are
    averages weighted by shares. Buys from a second order while the position is open are
    an error, because the harness never adds to a position.

    Positions still open at the end are reported with is_open=True: the shares left are
    marked at their last close with no exit cost, and averaged with any shares already
    sold. Holding days count trading sessions from the first entry fill to the fill that
    closed the position (or to the last session for open trades).
    """
    session_index = {p.date: i for i, p in enumerate(result.equity)}
    open_trades: dict[str, _OpenTrade] = {}
    trades: list[dict[str, Any]] = []
    for fill in result.fills:
        symbol = fill.instrument.key
        trade = open_trades.get(symbol)
        if fill.side == "buy":
            if trade is None:
                trade = open_trades[symbol] = _OpenTrade(fill.order_id)
            elif fill.order_id != trade.entry_order_id:
                raise ReportError(f"second entry for {symbol} while open")
            trade.buys.append(fill)
            continue
        if trade is None:
            raise ReportError(f"exit for {symbol} without an entry")
        if fill.quantity > trade.remaining:
            raise ReportError(f"exit for {symbol} sells more than the position holds")
        trade.sells.append(fill)
        if trade.remaining == 0:
            del open_trades[symbol]
            trades.append(
                _trade(
                    trade,
                    exit_date=fill.fill_date,
                    mark=None,
                    exit_reason="no_data" if fill.kind == "forced_close" else "signal",
                    session_index=session_index,
                )
            )

    last_date = result.equity[-1].date if result.equity else result.end
    for symbol, trade in open_trades.items():
        position = result.final_positions.get(symbol)
        if position is None:
            raise ReportError(f"{symbol} is open in fills but not in final positions")
        if position.quantity != trade.remaining:
            raise ReportError(
                f"{symbol}: fills leave {trade.remaining} shares but the final position "
                f"holds {position.quantity}"
            )
        trades.append(
            _trade(
                trade,
                exit_date=last_date,
                mark=result.final_marks[symbol],
                exit_reason="open",
                session_index=session_index,
            )
        )
    trades.sort(key=lambda t: (t["entry_date"], t["symbol"]))
    return trades


def _units(fill: Fill) -> float:
    return fill.quantity * fill.instrument.multiplier


def _average_price(lots: Sequence[tuple[float, float]]) -> float:
    """Average of (price, units) lots, weighted by units.

    When every lot has the same price that price is returned as is, so a single fill
    reports its exact fill price (no rounding from multiplying and dividing back).
    """
    first = lots[0][0]
    if all(price == first for price, _ in lots):
        return first
    return sum(price * units for price, units in lots) / sum(units for _, units in lots)


def _trade(
    trade: _OpenTrade,
    *,
    exit_date: dt.date,
    mark: float | None,
    exit_reason: str,
    session_index: Mapping[dt.date, int],
) -> dict[str, Any]:
    """Build one trades.csv row. `mark` values the shares not yet sold (open trades)."""
    buys, sells = trade.buys, trade.sells
    entry = buys[0]
    exit_lots = [(f.price, _units(f)) for f in sells]
    if mark is not None:
        exit_lots.append((mark, trade.remaining * entry.instrument.multiplier))
    cost_basis = sum(f.price * _units(f) for f in buys) + sum((f.commission for f in buys), 0.0)
    proceeds = sum(price * units for price, units in exit_lots)
    pnl = proceeds - sum((f.commission for f in sells), 0.0) - cost_basis
    start_i, end_i = session_index.get(entry.fill_date), session_index.get(exit_date)
    return {
        "symbol": entry.instrument.key,
        "entry_date": entry.fill_date.isoformat(),
        "exit_date": exit_date.isoformat(),
        "quantity": sum(f.quantity for f in buys),
        "entry_price": _average_price([(f.price, _units(f)) for f in buys]),
        "exit_price": _average_price(exit_lots),
        "entry_cost": sum((f.slippage_cost + f.commission for f in buys), 0.0),
        "exit_cost": sum((f.slippage_cost + f.commission for f in sells), 0.0),
        "pnl": pnl,
        "return_pct": pnl / cost_basis,
        "holding_days": end_i - start_i if start_i is not None and end_i is not None else None,
        "exit_reason": exit_reason,
        "is_open": exit_reason == "open",
    }


# --------------------------------------------------------------------------- metrics


def return_metrics(result: BacktestResult, risk_free_rate: float) -> dict[str, float | None]:
    """Total return, CAGR, max drawdown (negative fraction), annualized Sharpe."""
    equity = pd.Series([p.equity for p in result.equity], dtype="float64")
    if equity.empty:
        return {"total_return": None, "cagr": None, "max_drawdown": None, "sharpe": None}
    start_value = result.starting_cash
    final = float(equity.iloc[-1])
    total_return = final / start_value - 1
    days = (result.equity[-1].date - result.equity[0].date).days
    cagr = (final / start_value) ** (365.25 / days) - 1 if days > 0 and final > 0 else None
    curve = pd.concat([pd.Series([start_value]), equity], ignore_index=True)
    max_dd = float((curve / curve.cummax() - 1).min())
    returns = curve.pct_change().dropna().iloc[1:]  # first change is start -> first mark
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    std = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
    sharpe = float(excess.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else None
    return {"total_return": total_return, "cagr": cagr, "max_drawdown": max_dd, "sharpe": sharpe}


def summarize(
    result: BacktestResult,
    benchmark: BacktestResult,
    recipe: Recipe,
    *,
    holdout: bool,
    trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trades = round_trips(result) if trades is None else trades
    closed = [t for t in trades if not t["is_open"]]
    wins = [t for t in closed if t["pnl"] > 0]
    holding = [t["holding_days"] for t in closed if t["holding_days"] is not None]
    invested = [p.n_positions > 0 for p in result.equity]
    events: dict[str, int] = {}
    for e in result.events:
        events[e.kind] = events.get(e.kind, 0) + 1
    rf = recipe.backtest.risk_free_rate
    return {
        "strategy": result.strategy,
        "strategy_preset": recipe.strategy.name,
        "risk_preset": "none",
        "universe": recipe.universe.name,
        "holdout": holdout,
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "warmup_sessions": result.warmup_sessions,
        "sessions": len(result.equity),
        "starting_cash": result.starting_cash,
        "final_equity": result.equity[-1].equity if result.equity else result.starting_cash,
        **return_metrics(result, rf),
        "trade_count": len(trades),
        "closed_trades": len(closed),
        "open_trades": len(trades) - len(closed),
        "win_rate": len(wins) / len(closed) if closed else None,
        "avg_holding_days": sum(holding) / len(holding) if holding else None,
        "pct_time_invested": sum(invested) / len(invested) if invested else None,
        "benchmark": {
            "symbol": recipe.backtest.benchmark,
            **return_metrics(benchmark, rf),
        },
        "events": dict(sorted(events.items())),
        "data_fingerprint": result.data_fingerprint,
    }


# --------------------------------------------------------------------------- files


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in columns})


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def write_run(
    run_dir: Path,
    result: BacktestResult,
    benchmark: BacktestResult,
    recipe: Recipe,
    *,
    holdout: bool,
    run_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Write every run file into `run_dir` and return the summary.

    `run_dir` is created if missing. It may already exist only if it is empty (the run
    entry point reserves the folder up front); results are never overwritten.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run folder {run_dir} already has files")
    trades = round_trips(result)
    summary = summarize(result, benchmark, recipe, holdout=holdout, trades=trades)
    _write_csv(run_dir / "trades.csv", TRADE_COLUMNS, trades)
    _write_csv(run_dir / "equity.csv", EQUITY_COLUMNS, [p.model_dump() for p in result.equity])
    _write_csv(run_dir / "events.csv", EVENT_COLUMNS, [e.model_dump() for e in result.events])
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    snapshot = {
        "run": dict(run_info),
        "holdout": holdout,
        "data": {
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "warmup_sessions": result.warmup_sessions,
            "fingerprint": result.data_fingerprint,
        },
        "recipe": recipe.model_dump(mode="json"),
    }
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False))
    return summary


def append_index(
    index_path: Path, summary: Mapping[str, Any], timestamp: str, run_dir: Path
) -> None:
    """Append one line to backtests/index.csv, creating it with a header if missing."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    new = not index_path.exists()
    row = {
        "timestamp": timestamp,
        "strategy": summary["strategy"],
        "strategy_preset": summary["strategy_preset"],
        "risk_preset": summary["risk_preset"],
        "start": summary["start"],
        "end": summary["end"],
        "holdout": summary["holdout"],
        "total_return": summary["total_return"],
        "max_drawdown": summary["max_drawdown"],
        "trade_count": summary["trade_count"],
        "run_folder": run_dir.name,
    }
    with index_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow({c: _cell(v) for c, v in row.items()})
