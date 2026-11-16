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
  explicit toggle. The mode is never something the UI passes with a request: it is a
  property of the active profile and the keys (`service.trading_mode()`, a constant
  while only paper exists).
- **The chat is read-only.** It explains runs, positions, and code. It cannot edit files,
  change settings, or place orders. It is given `harness.service` and nothing from
  `harness.control`, the kill switch included: an agent that can stop trading can be
  talked into stopping it at a bad moment.
- **Nothing is editable while a profile trades.** The risk preset is frozen into the
  profile at promotion and the approval mode is set at activation. Changing either
  means a new profile.

## What the harness guarantees to a UI

**The UI may write:**
- Strategy presets and rule templates in `config/strategies/`.
- The choice of risk preset and universe for a run.

**The UI may not write:** costs, construction, dates, data settings, or any code. It
writes nothing under `state/paper/` or `profiles/` either: those change only through the
commands below.

**The UI may trigger** (operator only, never the chat; each takes the operator's name
and leaves a record): the functions of `harness.control`.
- `engage_kill_switch(reason, operator)`: works at any time, also while a runner is hung.
- `approve_orders({order_id: content_hash}, operator)` and
  `reject_orders(order_ids, operator, reason)`: left in the inbox for the runner. An
  approval quotes the hash shown by `list_pending_approvals`.
- `reset_kill_switch(operator, confirm)`, `reset_risk_halt(operator, reason)`,
  `flatten(operator, confirm)`, `accept_broker_state(operator, note)`,
  `record_adjustment(adjustment)`, `promote_run(run_dir, name)`,
  `activate_profile(profile_id, approval, operator)`: refused while a runner is running.
  `confirm` is the active profile's id, typed by the operator. `flatten` blocks while
  its sells are worked (until the open, if started before it), so a backend calls it
  from a worker. It engages the kill switch first and leaves it engaged: after a
  flatten, trading starts again only through `reset_kill_switch`.
- A refusal is a `ControlError` whose message says what was and was not done.

**The UI reads:**
- `backtests/index.csv`: one row per run, including strategy, strategy preset, and risk preset.
- `backtests/<run>/`: the resolved recipe, `summary.json`, `trades.csv`, `equity.csv`,
  all with a fixed schema.
- Each strategy's `Params` schema (types, ranges, defaults), to generate forms.
- Paper trading, through `harness.service` (plain functions over state files; they never
  call the broker and never write): `get_status`, `get_positions`, `list_orders`,
  `list_pending_approvals`, `list_alerts`, `list_profiles`. Each result says when it is
  from. Positions are current to the last recorded fill; orders and pending approvals
  are the runner's snapshots (after the decision, and when its run ends).
- `state/paper/<profile>/`: `trades.csv`, `summary.json`, `equity.csv` and
  `config_snapshot.yaml`, with the schema of `backtests/<run>/` plus the paper columns.

**The UI runs backtests** through `run_backtest(recipe)`, the same function the CLI calls.

**The UI never runs a trading day.** The OS scheduler starts the runner in its own
process. The backend observes it and sends it commands.

## Open

- Pages, navigation, and layout: design after slice 1 is in use.
