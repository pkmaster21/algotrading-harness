# Architecture

Target end state of the trading harness, and why it looks this way.
`CLAUDE.md` holds the rules for every session; this doc holds the plan and the reasoning.
Update it when a decision changes, not on a schedule. Later slices are described only
at the level actually decided; details get filled in as earlier slices teach us things.

## Philosophy

- **Safety by architecture, not discipline.** An LLM can write strategies but is
  structurally unable to place orders, change risk limits, or read API keys.
- **Backtest-first.** Every strategy proves itself in backtest → holdout → paper
  before risking real money.
- **Small and opinionated.** A barebones framework with clean building blocks, not a
  feature-complete platform.
- **Personal use first.** Built to learn and practice algo trading; also a portfolio project.
  Demos use paper accounts only, and results are never pitched as profits.

## Target pipeline

```
Data feed ──► Strategy (alpha) ──► Construction ──► Risk (hard limits) ──► Order manager ──► Broker adapter ──► Alpaca | SimBroker
                                       ▲  ▲                ▲
                         Costs ────────┘  └──── Positions ─┘   (reconciled against broker)
Kill switch: cancels all orders and stops the engine, from anywhere.
```

Backtest, paper, and live run this exact pipeline. Only the broker adapter changes.

## Components

| File | Responsibility | Slice |
|---|---|---|
| `harness/models.py` | Shared data shapes: Bar, Stock, OptionContract, Leg, Signal, Fill, Position | 1 |
| `harness/config.py` | Build a run's recipe: resolve the chosen presets and fixed settings into validated pydantic models | 1 |
| `harness/data.py` | Fetch and cache bars; enforce the holdout cutoff | 1 |
| `harness/view.py` | Append-only bar buffer and the read-only `MarketView` strategies receive | 1 |
| `strategies/base.py` | `Strategy` interface; each strategy declares a pydantic `Params` class and receives it in its constructor | 1 |
| `strategies/*.py` | Emit entry and exit signals from a read-only view of past bars | 1+ |
| `harness/costs.py` | Estimate trade cost (flat % now) | 1 |
| `harness/construction.py` | Rank signals, size positions (equal weight, max N now) | 1 |
| `harness/engine.py` | Daily loop; hands strategies a past-only view; fills at next open | 1 |
| `harness/broker/base.py` | Broker interface shared by all adapters | 1 |
| `harness/broker/sim.py` | Simulated fills for backtests | 1 |
| `harness/run.py` | `run_backtest(recipe)` function plus a thin CLI over it; loads strategies by name, checks their imports | 1 |
| `harness/report.py` | Run outputs with a fixed schema, the resolved recipe per run, metrics, SPY benchmark, run index | 1 |
| `harness/indicators.py` | Indicator calculations (SMA, RSI, ...) from past bars only; each addition is reviewed with a look-ahead test | Rules |
| `strategies/rules.py` | Generic strategy that runs a rule template (indicators plus conditions, as data) | Rules |
| `harness/risk.py` | Hard limits; can shrink or reject; never blocks risk-reducing orders | 2 |
| `harness/orders.py` | Limit pricing, retries, rate limits, optional manual approval | 2 |
| `harness/positions.py` | Positions, cash, P&L, reconciliation against the broker | 2 |
| `harness/broker/alpaca.py` | Paper/live adapter; the only code holding trading keys | 2 |
| `harness/pricing.py` | Implied volatility and Greeks | Options |

Each component starts as the simplest rule that works and is upgraded only when
paper or live results show the simple version is the bottleneck. Any upgrade is
re-validated through backtest and paper trading.

## Boundaries and ownership

- **`harness/` is locked** after it's built: Claude Code settings deny edits.
  Changes are deliberate and reviewed. Locked means reviewed, not frozen. Code belongs in
  `harness/` when a bug in it would silently break every strategy or run.
- **`strategies/` is the sandbox:** the only place AI-written code lands day to day.
- **Keys live only in `.env`.** Only the data loader and broker adapters read them.
  Claude Code settings deny reading `.env`.
- **Paper and live keys use different variable names**, and live trading requires an
  explicit flag. Paper is the default everywhere.
- **Config is data, and only the loader reads it.** `harness/config.py` reads the YAML,
  validates it, and hands typed objects to each component. No component opens a config file.
- **UI forms write presets only:** strategy params and rule templates in
  `config/strategies/`, and the choice of risk preset and universe for a run. Costs,
  construction, and data settings are not editable from the UI; they change through review.
- **The UI chat is read-only:** it explains backtests, positions, and code. All code
  changes happen in Claude Code, inside the sandbox.

## Slice roadmap

**Slice 1: backtesting.** Models, data loader with holdout, strategy interface, costs,
construction, engine, SimBroker, report (vs SPY buy-and-hold), acceptance tests for
look-ahead, holdout, and determinism. Also the seams the UI needs later: strategies take
a `Params` object, `config/` holds presets per component, every run saves its resolved
recipe, and a run is a function the CLI (and later the backend) calls.

**Rule templates** (after slice 1). `harness/indicators.py` plus `strategies/rules.py`,
so a strategy can be defined as data (indicators and conditions) and edited in the UI.
First acceptance test: one strategy written both as Python and as a template produces
identical backtests.

**Slice 2: paper trading** (built once a strategy survives validation). Risk manager with
three risk presets (conservative, balanced, aggressive), order manager with an optional
manual-approval mode for early live use, positions and reconciliation, Alpaca paper
adapter, kill switch, logging and alerts. A run that passed validation is promoted to a
frozen profile, and that profile is what paper trading runs.

**Slice 3: UI.** Python backend API, React/TypeScript frontend, and a read-only chat agent
(Claude Agent SDK) that sees what's on screen. Principles and the harness-to-UI contract:
`docs/ui-design.md`. Page layout is designed after slice 1 is in use.

**Options phase.** Option chain data, `pricing.py`, Greek-exposure and max-loss risk
limits, multi-leg orders with a leg-by-leg close fallback, daily expiry sweep,
assignment detection, sim fills from historical option quotes. Each options strategy
goes through the full lifecycle on its own.

## Strategy lifecycle

1. Hypothesis: why should this edge exist, and who is on the other side?
2. Data: only what was knowable at the time.
3. Spec: exact rules, no judgment calls.
4. In-sample backtest with costs.
5. Validation: holdout (once), walk-forward, parameter sensitivity.
6. Paper trading: compare actual results to backtest expectations.
7. Live, small size.
8. Monitor and retire on predefined kill criteria.

Benchmark is SPY buy-and-hold, after costs and taxes, not zero.

## Model usage

| Work | Model |
|---|---|
| Slice 1 build | Opus, then a fresh Fable review of interfaces, `sim.py`, `engine.py`, `config.py`, `strategies/base.py`, `run.py` |
| Slice 2 build, chat-agent permissions, options harness changes | Fable |
| Strategy brainstorming, writing, debugging | Opus |
| Routine tasks: run backtests, summaries, tests, reports | Sonnet |
| Review before backtest → paper and paper → live | Fable, fresh session |

## Decision log

- **Custom harness over Lumibot/LEAN:** those frameworks let strategies place orders
  directly; this design makes the risk layer a protected boundary and keeps AI out of
  execution. Smaller scope, deliberately.
- **Strategies emit signals, never orders:** one shared layer sees the whole portfolio,
  and rewriting a strategy can't weaken risk checks.
- **Exit signals, not target positions:** natural fit for short-term swings with time
  and price exits.
- **Fill at next bar's open:** filling on the signal bar is the most common
  look-ahead bug in stock backtests.
- **Holdout enforced in the data loader:** one choke point instead of a rule each
  strategy must remember.
- **Orders are lists of legs from day one:** stocks are one-leg orders, so options
  are an addition, not a rewrite.
- **Risk-reducing orders always pass risk checks:** otherwise a loss cap could block
  closing a position.
- **Simple rules before models:** Narang's components start as a few lines each.
- **Python engine, TypeScript UI:** quant libraries and SDKs live in Python; the UI
  plays to existing frontend strengths.
- **Short-term swing (2–10 day holds), long-only stocks first:** fits a side project,
  free daily data, and LLM response times; options come after one stock strategy
  completes the full lifecycle.
- **Universe:** 10 names for plumbing only (survivorship-biased); index and sector ETFs
  for first real evaluation; historical index membership needed before stock-picking strategies.
- **Alpaca:** free data is fine for daily stock bars; real-time options data
  (OPRA) needs the paid data plan before options go live.

### Slice 1 design decisions (2026-11-06)

- **SIP feed only, not IEX:** fills use the open, and IEX-only OHLC is not the
  consolidated market open. The free plan allows historical SIP older than 15 minutes,
  enough for daily bars. The feed is fixed in code (not a config switch) and stored in
  each cache file's metadata, so a feed change forces a full refetch instead of mixing sources.
- **Whole shares:** quantities are integers. Conservative (a little cash drag), matches
  what a small account actually fills, and keeps position and determinism checks exact.
- **Size at signal close, fill at next open:** orders are sized from `close_t` plus
  estimated slippage (the only price known at decision time) and fill at `open_{t+1}`
  plus slippage. A buy that no longer fits in cash after a gap is reduced to the
  affordable share count (or canceled), so cash never goes negative.
- **Look-ahead barrier is an append-only buffer:** strategies read a `MarketView` over a
  buffer the engine fills one session at a time. Future rows are absent (unwritten NaN),
  not hidden, and returned data is a copy. This stops buggy code, not deliberate in-process
  introspection; a subprocess sandbox would be the only airtight option and is not worth it yet.
- **Strategy import allowlist:** the run entry point rejects strategy modules that import
  anything outside an allowlist (numeric libraries, `harness.models`, `harness.view`,
  `strategies.base`). Catches accidental file, env, network, or broker access. Test
  strategies are loaded directly by tests and are not subject to it.
- **Cache consistency via overlap check plus scheduled refresh:** when extending a symbol's
  cache, the loader refetches the last few cached bars and compares them. Any split or
  dividend after the cached end rescales those bars, so a mismatch triggers a full refetch
  of that symbol. Files are also fully refetched on a schedule (default 30 days) to catch
  corrections deep in history. Chosen over the corporate-actions API: no second API and no
  list of action types to keep complete.
- **Holdout cutoff applied to both fetch and read:** the loader never requests data on or
  after the holdout start without the holdout flag, and filters cached rows on read, so
  holdout rows already on disk from an earlier holdout run are never returned. The holdout
  log line is written before any holdout data is read.
- **Holdout runs start fresh capital at the holdout start:** in-sample bars are warm-up only.
  The run entry point converts the holdout start into a session index for the engine, so the
  engine still never compares dates. During warm-up the engine fills the view but does not
  call the strategy, so strategies must compute indicators from `view.history`, not from
  state accumulated across calls.
- **Delisted or missing-data positions are force-closed:** if a held symbol has no bar for N
  consecutive sessions (default 5, configurable), the engine closes it at its last close
  minus sell slippage and logs `position_closed_no_data`. The trade is marked
  `exit_reason=no_data` in `trades.csv`.
- **Open positions at the end of a run are marked, not liquidated:** they appear in
  `trades.csv` as open; win rate counts closed trades only.
- **Benchmark runs through the same engine:** SPY buy-and-hold uses the same SimBroker,
  costs, next-open fill, and whole shares as the strategy, so the comparison is like for like.

### Configuration and UI seams (2026-11-10)

- **Presets for per-run choices, single files for fixed settings:** `config/strategies/`,
  `config/risk/`, and `config/universe/` hold named presets picked for each run. Costs,
  construction, dates, and data settings stay single files. A run names its presets; the
  loader resolves them.
- **Every run stores its resolved recipe:** the run folder holds the full values used, not
  just preset names, so editing a preset later never changes what an old run means. The
  run index records the strategy, its preset, and the risk preset.
- **No profile management in the UI:** the run form picks strategy, risk preset, and
  universe, and the recipe is saved automatically. A named profile exists only once a
  validated run is frozen for paper or live trading.
- **Few UI knobs on purpose:** every editable setting is another way to fit noise. Costs
  especially stay out of the UI, since lowering them flatters every strategy.
- **Indicators live in `harness/`:** a look-ahead bug in one indicator would silently
  corrupt every template using it. `harness.indicators` joins the strategy import allowlist.
- **Rule grammar starts small:** compare an indicator to a number or another indicator,
  combined with all/any. A strategy that needs more is written in Python instead.
- **Risk presets set hard limits only:** max position size, max exposure, daily loss cap,
  max drawdown halt. Position count and sizing stay in construction. Revisit if the
  presets barely change results. No custom preset until a real need shows up.
- **Runs from before `risk.py` are labeled `risk: none`** and are not compared directly
  with runs that use a risk preset.

## Open questions

- Is the flat 0.05% cost assumption realistic once paper fills come in?
- Data source for historical index membership, if stock-picking strategies are pursued.
- Actual hard-limit values for each slice 2 risk preset (position size, exposure, daily
  loss cap, drawdown halt).
- Historical options data source for the options phase (ThetaData is the leading candidate).
- How long manual-approval mode stays on after going live.
- Back-adjusted prices are not the prices traded at the time. Returns are correct, but
  whole-share counts and any price-level rule (for example "price > $5") are slightly off
  historically. Fine for slice 1; revisit before strategies that filter on price level
  (raw prices plus a separate adjustment factor would fix it).
- Alpaca stock bars likely start around 2016-01, so the 2015-01-01 data start may be
  2016 in practice. Confirm on first download.
- Forced close at last close is optimistic for bankruptcies (the price collapses after
  delisting) but roughly right for acquisitions. Matters once the universe includes
  delisted names.
- How AlpacaBroker handles a forced close in slice 2: the base `Broker.force_close` is a
  sim concept; a real broker reports delistings itself, so this likely becomes reconciliation.
- A trading halt longer than N sessions also triggers a forced close. Rare; N is configurable.
