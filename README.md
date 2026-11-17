# Trading Harness

A personal, backtest-first algorithmic trading harness for US stocks and ETFs. You write
a strategy as a small Python module that emits signals. The harness owns everything else:
data, portfolio construction, risk limits, costs, fills, position tracking and, for paper
trading, orders, reconciliation and a kill switch.

Status: backtesting (slice 1) and paper trading on Alpaca's paper account (slice 2) are
built. Daily bars, long-only. There is no live trading, and no UI yet.

This is a learning and portfolio project. Results are not investment advice and are
never presented as profits.

## Design in one paragraph

A strategy sees bars up to and including today's close and returns `Signal`s. Orders
from those signals fill at the next session's open, plus slippage and costs, never on the
signal bar. The same strategy, construction, risk and engine code runs in backtest and
paper. Strategies cannot place orders, read files, read keys or use the network: the
loader rejects such imports and names, and the access is blocked while `on_bar` runs.
Data from the holdout period is off-limits except to an explicitly logged holdout run
or to paper trading a frozen profile. Same data, config and strategy give identical
results.

More in [ARCHITECTURE.md](ARCHITECTURE.md) (pipeline, components, decision log, roadmap).
Diagrams of the components, data models and strategy lifecycle are in
[docs/diagrams.md](docs/diagrams.md).

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
uv sync
cp .env.example .env   # then fill in ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET_KEY
```

The keys are for an Alpaca paper account. Backtests use them to download daily bars
(cached in `data/`, so a fully cached run needs no keys); paper trading uses them to
send orders. Keys are read from `.env` only, never exported or logged.

## Quick start: run a backtest

```
uv run python -m harness.run --strategy example_buy_and_hold --risk balanced
```

| Option | Meaning |
|---|---|
| `--strategy` | Strategy preset in `config/strategies/` (required) |
| `--risk` | Risk preset in `config/risk/`: `conservative`, `balanced`, `aggressive` (default: no limits) |
| `--universe` | Symbol list in `config/universe/` (default `plumbing-10`) |
| `--holdout` | Run on holdout data. Logged. Use once, for the final check of a settled strategy |

Each run writes a folder under `backtests/` with the resolved recipe (every config value
used), the data range and the results, compared with SPY buy-and-hold. Read them
cheapest first:

1. `backtests/index.csv`: one line per run (strategy, presets, dates, return, drawdown, trades).
2. `backtests/<run>/summary.json`: full metrics for one run.
3. `backtests/<run>/trades.csv`, `equity.csv`: per-trade and per-day detail.

## Writing a strategy

1. Copy [strategies/example_buy_and_hold.py](strategies/example_buy_and_hold.py) to
   `strategies/<name>.py`. Its docstring lists the rules.
2. Implement `on_bar(view, portfolio)` and return `Signal.enter(...)` or `Signal.exit(...)`.
   The harness sizes positions and fills orders.
3. Declare every tunable value in the `Params` class, with a type, range and default.
4. Add a preset at `config/strategies/<name>.yaml`:

   ```yaml
   strategy: <name>
   params:
     some_param: 20
   ```

5. Run `uv run python -m harness.run --strategy <name>`.

Compute indicators from `view.history` / `view.series` each session rather than state
kept across calls. Strategy files and presets other than the example are gitignored.

## Paper trading

A backtest run is promoted to a frozen profile, the profile is bound to the Alpaca paper
account, and a runner trades it once each morning at the open. Paper trading is attended:
it is meant to be started and watched by a person.

```
uv run python -m harness.paper promote backtests/<run> <name>      # freeze; prints the profile id
uv run python -m harness.paper activate <profile id> --approval entries --operator <you>
uv run python -m harness.paper run                                 # each trading morning, before 09:20 NY
uv run python -m harness.cli status                                # views: status, positions, orders, pending, alerts, profiles
uv run python -m harness.cli kill --reason "<why>" --operator <you> # cancel open orders, stop sending
```

A real strategy needs a logged holdout run of the same recipe before it can be promoted;
`--plumbing` skips that for the example strategy only. Account settings, the morning
routine, approvals, the kill switch and what to do when a day goes wrong are in
[docs/paper-runbook.md](docs/paper-runbook.md). Read it before the first run.

## Configuration

| Path | What | Edited by |
|---|---|---|
| `config/strategies/` | Strategy presets (params) | You (later the UI) |
| `config/risk/` | Risk presets | You (later the UI) |
| `config/universe/` | Symbol lists | You |
| `config/*.yaml` | Fixed settings: costs, construction, dates (holdout cutoff), data, execution | Rarely, on purpose |

Config values are never hardcoded in code. `plumbing-10` is a temporary 10-symbol
universe with survivorship bias; treat results on it as plumbing checks, not evidence.

## Layout

```
harness/      engine, models, data, construction, risk, costs, broker adapters, paper runner, CLI
strategies/   one file per strategy (base class and example tracked)
config/       YAML presets and fixed settings
docs/         paper runbook, slice 2 design, UI design
tests/        pytest suite, including look-ahead, holdout, determinism and paper end-to-end tests
data/         local bar cache (gitignored)
backtests/    run outputs (gitignored)
profiles/     frozen profiles (gitignored)
state/        paper trading state and journal (gitignored)
```

## Development

```
uv run pytest
uv run ruff check harness strategies tests
uv run pyright
```

CI runs the same checks. Work happens on feature branches with one squashed commit per
pull request, using Conventional Commits. Contributor and agent rules are in
[CLAUDE.md](CLAUDE.md).

## Roadmap

Rule templates (strategies defined as data), then a UI (Python backend, React frontend,
read-only chat agent), then options. See "Slice roadmap" in
[ARCHITECTURE.md](ARCHITECTURE.md).
