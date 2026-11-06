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

| File | Responsibility | Narang equivalent | Slice |
|---|---|---|---|
| `harness/models.py` | Shared data shapes: Bar, Stock, OptionContract, Leg, Signal, Fill, Position | — | 1 |
| `harness/data.py` | Fetch and cache bars; enforce the holdout cutoff | — | 1 |
| `strategies/*.py` | Emit entry and exit signals from a read-only view of past bars | Alpha model | 1+ |
| `harness/costs.py` | Estimate trade cost (flat % now) | Transaction cost model | 1 |
| `harness/construction.py` | Rank signals, size positions (equal weight, max N now) | Portfolio construction | 1 |
| `harness/engine.py` | Daily loop; hands strategies a past-only view; fills at next open | — | 1 |
| `harness/broker/base.py` | Broker interface shared by all adapters | Execution model (venue) | 1 |
| `harness/broker/sim.py` | Simulated fills for backtests | — | 1 |
| `harness/risk.py` | Hard limits; can shrink or reject; never blocks risk-reducing orders | Risk model | 2 |
| `harness/orders.py` | Limit pricing, retries, rate limits, optional manual approval | Execution model | 2 |
| `harness/positions.py` | Positions, cash, P&L, reconciliation against the broker | — | 2 |
| `harness/broker/alpaca.py` | Paper/live adapter; the only code holding trading keys | Execution model (venue) | 2 |
| `harness/pricing.py` | Implied volatility and Greeks | — | Options |

Each component starts as the simplest rule that works and is upgraded only when
paper or live results show the simple version is the bottleneck. Any upgrade is
re-validated through backtest and paper trading.

## Boundaries and ownership

- **`harness/` is locked** after it's built: Claude Code settings deny edits.
  Changes are deliberate and reviewed.
- **`strategies/` is the sandbox:** the only place AI-written code lands day to day.
- **Keys live only in `.env`.** Only the data loader and broker adapters read them.
  Claude Code settings deny reading `.env`.
- **Paper and live keys use different variable names**, and live trading requires an
  explicit flag. Paper is the default everywhere.
- **The UI chat is read-only:** it explains backtests, positions, and code. All file
  changes happen in Claude Code, inside the sandbox.

## Slice roadmap

**Slice 1: backtesting.** Models, data loader with holdout, strategy interface, costs,
construction, engine, SimBroker, report (vs SPY buy-and-hold), acceptance tests for
look-ahead, holdout, and determinism.

**Slice 2: paper trading** (built once a strategy survives validation). Risk manager,
order manager with an optional manual-approval mode for early live use, positions and
reconciliation, Alpaca paper adapter, kill switch, logging and alerts.

**Slice 3: UI.** Python backend API, React/TypeScript frontend, dashboard, and a
read-only chat agent (Claude Agent SDK) that sees what's on screen.

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
| Slice 1 build | Opus, then a fresh Fable review of interfaces, `sim.py`, `engine.py` |
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

## Open questions

- Is the flat 0.05% cost assumption realistic once paper fills come in?
- Data source for historical index membership, if stock-picking strategies are pursued.
- Actual hard-limit values for slice 2 (position size, daily loss cap).
- Historical options data source for the options phase (ThetaData is the leading candidate).
- How long manual-approval mode stays on after going live.
