# UI design (draft)

Slice 3 principles and the constraints the harness guarantees to any UI. Page layout is
deliberately not designed yet: it will be informed by using the slice 1 backtester from
the CLI. Architecture-level decisions live in `ARCHITECTURE.md`.

## Principles

- **Few knobs.** Every editable setting is a way to overfit. The UI exposes strategy
  params, risk preset, and universe only.
- **The UI edits config, never code.** New indicators or strategy logic go through Claude Code.
- **Runs save their recipe automatically.** There is no profile to create or manage; a
  named profile exists only once a validated run is frozen for paper or live trading.
- **Holdout is gated.** Running holdout is a separate, confirmed action, and the number of
  holdout runs per strategy is always visible.
- **Paper by default.** Live mode, once it exists, is visually distinct and needs an
  explicit toggle.
- **The chat is read-only.** It explains runs, positions, and code. It cannot edit files,
  change settings, or place orders.

## What the harness guarantees to a UI

**The UI may write:**
- Strategy presets and rule templates in `config/strategies/`.
- The choice of risk preset and universe for a run.

**The UI may not write:** costs, construction, dates, data settings, or any code.

**The UI reads:**
- `backtests/index.csv`: one row per run, including strategy, strategy preset, and risk preset.
- `backtests/<run>/`: the resolved recipe, `summary.json`, `trades.csv`, `equity.csv`,
  all with a fixed schema.
- Each strategy's `Params` schema (types, ranges, defaults), to generate forms.

**The UI runs backtests** through `run_backtest(recipe)`, the same function the CLI calls.

## Open

- Pages, navigation, and layout: design after slice 1 is in use.
