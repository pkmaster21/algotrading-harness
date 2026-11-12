# Trading Harness

A personal, backtest-first algorithmic trading harness for US stocks (options later).
Strategies are pluggable modules; the harness owns everything else.

## Where to look (read only what the task needs)

| Need                         | Read                                       |
| ---------------------------- | ------------------------------------------ |
| Why a design choice was made | `ARCHITECTURE.md` → Decision log           |
| What a future slice adds     | `ARCHITECTURE.md` → Slice roadmap          |
| Which file owns what         | `ARCHITECTURE.md` → Components             |
| Unresolved decisions         | `ARCHITECTURE.md` → Open questions         |
| UI principles and contract   | `docs/ui-design.md`                        |
| Results of a past backtest   | `backtests/<run>/summary.json` (see below) |

Read `ARCHITECTURE.md` when planning a new slice or changing the harness,
not for routine strategy work.

## Architecture rules (never violate)

- **Strategies emit Signals only.** A strategy never calls a broker, never places orders,
  never reads API keys or environment variables, and never reads files directly.
- **The harness owns** data, portfolio construction, risk, execution, and position tracking.
- **One code path.** Backtest, paper, and live use the same engine. Only the Broker
  adapter changes (SimBroker for backtests, AlpacaBroker later).
- **Orders are lists of Legs.** A stock trade is a one-leg Signal. This keeps options
  (multi-leg) addable later without changing the data model.
- **No look-ahead.** At the close of bar t, a strategy sees bars up to and including t.
  Any resulting order fills at the open of bar t+1, plus slippage and costs.
  Never fill on the signal bar.
- **Determinism.** Same data + same config + same strategy = identical results.
- **Holdout discipline.** Data on or after the holdout start date is off-limits unless
  a backtest is explicitly run in holdout mode. Every holdout run is logged.

## Scope (current)

- Daily bars, US stocks/ETFs, long-only.
- Historical data from Alpaca (read-only data API). No order placement anywhere yet.

## Stack

Python 3.12, uv, pytest, pandas, pydantic, pyyaml, alpaca-py. Dev checks: ruff, pyright
(with pandas-stubs, matching what Pylance reports in the editor).

## Layout

```
harness/      # engine, models, data, construction, costs, broker adapters (locked after slice 1)
strategies/   # strategy modules only; one file per strategy
config/       # YAML, read only by harness/config.py
  strategies/ #   strategy presets (params, rule templates); UI-editable
  risk/       #   risk presets (conservative, balanced, aggressive)
  universe/   #   symbol lists
  *.yaml      #   fixed settings: costs, construction, dates, data (not UI-editable)
docs/         # detail docs (ui-design.md); entry-point docs stay at the root
tests/        # pytest; test strategies (buy-and-hold, look-ahead cheater) live here
data/         # local bar cache (gitignored)
backtests/    # run outputs (gitignored)
```

## Conventions

- Type hints everywhere; pydantic models for anything crossing a module boundary.
- Config values (costs, dates, limits) live in `config/`, never hardcoded.
- Every backtest run writes a reproducible snapshot: the resolved recipe (all config values
  used, not just preset names), strategy name, data range, results.
- Never commit to `main`. Work on a feature branch and open a pull request for review.
  Each PR has exactly one commit: squash work-in-progress commits before opening, and
  amend (then force-push the feature branch) when addressing review feedback.
  Use Conventional Commits (`feat:`, `fix:`, `test:`, ...).

## Pull requests

- **Small and focused:** one purpose per PR. If it grows, split it.
- **Title:** Conventional Commits style, e.g. `feat(engine): add next-open fill loop`.
- **Description:** fill in `.github/pull_request_template.md` (why, what, how to review,
  testing, risks). Say why the change is needed, not just what changed.
- **Guide the reviewer:** list the files to read first and the kind of feedback wanted.
- **Self-review first:** read the full diff for accidental changes, and run `uv run pytest`,
  `uv run ruff check harness strategies tests`, and `uv run pyright` before opening the PR
  (CI runs the same checks). Note any test that was skipped or not run.
- **Flag sensitive changes:** call out anything touching dependencies, `harness/`
  boundaries, data or holdout handling, credentials, or `.claude/settings.json`.
- **Link** the related issue if there is one.
- Open the PR, then stop and wait for review. Never merge your own PR.

## Reading backtest output (cheapest first)

1. `backtests/index.csv`: one line per run (timestamp, strategy, strategy preset,
   risk preset, date range, holdout flag, total return, max drawdown, trade count, run folder).
   The engine appends to it after every run.
2. `backtests/<run>/summary.json`: full metrics for one run.
3. `backtests/<run>/trades.csv` or `equity.csv`: only to investigate specific
   trades or periods. Filter with code; never load the whole file into context.
