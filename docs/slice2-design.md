# Slice 2 design: paper trading

Status: approved 2026-11-12, including D1 to D10 as recommended. Nothing here is built yet.

Written from reading slice 1 offline. No API calls were made and `.env` was not read, so
every statement about Alpaca's trading API is from memory and marked **(verify)**. Each
one is checked against the docs in the PR that depends on it, before code relies on it.

It folds in the notes from the session that reviewed slice 1 ("the slice 1 review"
below).

Contents: decisions, step 0 review, step 1 design, step 2 PR plan.

## Decisions (approved as recommended)

| # | Decision | Approved |
|---|---|---|
| D1 | How orders reach the open | Exits market-on-open before the open. Entries released after the exits have filled, as capped limit orders sized from actual cash. (Alternative: everything in the opening auction.) |
| D2 | When the daily decision runs | Morning of t+1 (about 08:00 New York), one process per day that ends after the open. (Alternative: evening of t.) |
| D3 | Where the order manager sits | `OrderManager` implements the engine-facing `Broker` and drives a thin `AlpacaBroker`. Backtests keep `SimBroker` alone. Needs the "only the Broker adapter changes" rule reworded. |
| D4 | What may paper trade before a strategy validates | Only a "plumbing" profile, allowed only for the tracked `example_buy_and_hold`. |
| D5 | What manual approval gates | Entries only. Exits never wait for a human. |
| D6 | What the kill switch does to positions | Nothing. It cancels orders and stops the engine. Flattening is a separate, confirmed operator command. |
| D7 | Risk preset values vs construction | Presets are backstops looser than construction, not a second sizing layer. Fat-finger limits live in fixed settings, not presets. |
| D8 | Alert channel | Log file plus `alerts.jsonl` now. A push channel (webhook) is a hard gate before any unattended run, not a nice-to-have: a laptop notification does not reach you when away. See "Gate before unattended use". |
| D9 | State store | Append-only JSONL journal plus derived snapshots. (Alternative: SQLite.) |
| D10 | End-to-end test | Buy-and-hold never sells, so the test ends with the operator flatten command to exercise the sell path. |

## Things I think are wrong or weaker than they look (flagged, not changed)

1. **"Only the Broker adapter changes" is not literally true** once slice 2 exists. The
   order manager, the ledger, and the clock-driven runner are live-only code. What is
   shared is strategy, construction, risk, and the engine's two phases. The rule in
   `CLAUDE.md` should say that, or it will be quietly violated. (D3) Proposed wording,
   applied in PR 1: "One code path: strategy, construction, risk and the engine are shared
   by backtest, paper and live. Live adds order management, reconciliation and a
   clock-driven runner around the broker."
2. **"`alpaca.py` is the only code holding trading keys" is already false for paper.**
   Alpaca's paper key pair is one pair for data and trading **(verify)**, and
   `harness/data.py` holds it today. Suggested wording: the data loader uses the paper
   pair forever, and live keys (later slice) are held only by the adapter.
3. **Risk presets as shipped can fight construction.** Construction targets
   `equity / max_positions` (20% at 5 positions). A "conservative" preset with a 10%
   position cap halves every position on every trade, so risk becomes sizing. (D7)
4. **A daily loss cap in a daily-bar system is an entry brake, not a stop.** Nothing
   watches prices intraday in this slice. It blocks new entries after a bad close. It
   cannot stop a loss during the day. The name oversells it.
5. **Paper trading before validation invites holdout leakage.** Paper runs on data past
   `holdout_start` by definition. Watching a candidate strategy trade on paper before its
   single holdout run spends the holdout informally. (D4 closes this.)
6. **Alpaca's paper simulator flatters fills (verify):** no market impact, no queue, and
   probably no real opening auction. Paper answers "does the plumbing work" and only
   partly answers the open question about the 0.05% cost assumption.
7. **A laptop cron job is a single point of failure.** If the machine is asleep at 08:00,
   nothing trades, including exits. Fail-safe for orders, not for positions. The runner
   detects and alerts on missed sessions the next time it starts; it cannot alert while
   it is not running. Tolerable on paper, not for unattended use: see "Gate before
   unattended use".
8. **The buy-and-hold end-to-end test does not exercise exits, partial fills, or
   rejections.** Those are covered only by the fake venue. (D10 covers exits.)

## Gate before unattended use (open decision, not solved in this slice)

Slice 2 is built for attended paper runs: you start or watch the morning job. Nothing
runs unattended, on paper or live, until both of these are done:

1. **Push alerts work** (D8): a critical alert reaches a phone.
2. **One answer to "the machine was off" is chosen:**

| Option | What it covers | What it costs or breaks |
|---|---|---|
| Always-on host (small cloud server) | The job runs. Exits, risk checks, and reconciliation all happen. | Trading keys live on a server; needs its own hardening and deploy path. |
| Broker-side protective stops held at Alpaca | A price floor per position even when nothing of ours runs. | Conflicts with this design as written: a resting stop is an open order outside the open workflow (reconciliation would flag it), it reserves the shares so a market-on-open exit for the same position can be rejected **(verify)**, and its fills arrive with no decision behind them. Adopting it means changing the order manager and the ledger, not adding a setting. |
| Heartbeat (missed-run alert that pages you) | You learn within minutes that the job did not run. | Covers detection only; you still have to act. Needs an outside service. |

The heartbeat is the cheapest and composes with either of the others. Not an option:
scheduled GitHub Actions holding trading keys on a public repository.

## Verify list (Alpaca facts stated from memory)

Checked against Alpaca's docs in PR 16, before the adapter relies on them. The paper
simulator items can only be settled by the end-to-end run (PR 22).

| Claim | Used by |
|---|---|
| Market-on-open (`opg`) orders are accepted until about 09:28 New York and rejected from then until the evening | `opg_cutoff`, D2 schedule |
| The paper simulator honors `opg` orders and fills them near the open | Exit path; fallback is a market day order after the open |
| The paper simulator fills against quotes with no market impact or queue | How far paper answers the cost question |
| One paper key pair serves both the data and trading APIs | Flag 2, key handling |
| Paper key ids start with `PK` | Live-impossible layer |
| `client_order_id` is at most 48 characters (possibly longer) and a duplicate is rejected | Idempotent submission |
| Account configuration offers `no_shorting`, `max_margin_multiplier`, `suspend_trade` | Broker-side backstops, kill switch second layer |
| Trading API rate limit is 200 requests per minute | `max_requests_per_minute` |
| An open sell order reserves the shares it covers | Protective-stop conflict above |
| Account activities expose per-execution fills | Ledger upgrade path |
| The full set of order status strings | Status mapping |

---

# Step 0: review of slice 1 for paper and live

## 0.1 Broker interface

| Member | Fits a real broker? | Proposed change | Breaks backtests or tests? |
|---|---|---|---|
| `submit(orders) -> ids` | Yes, if "queue" means "record the intent". Ids must exist before any network call so retries are idempotent. | Live returns deterministic client order ids (see 1.3). No signature change. | No |
| `on_session_open(session, opens)` | The name fits, `opens` is sim-only. Live gives it a new meaning: work today's queue until every order is terminal, then return. | Keep the signature (live ignores `opens`). Add to the contract: when it returns, every order submitted before it is terminal. Day orders only, as in the sim. | No (docstring and one new contract test) |
| `force_close` | No. A real broker cannot sell a delisted symbol at a stale price; it reports the cash-out itself. | Return `str \| None`. `SimBroker` is unchanged. A live broker returns `None`, the engine logs `position_stale_no_data` and alerts, and reconciliation resolves it. | No behavior change. One type annotation; engine handles `None`. |
| `poll_updates` | Yes, with incremental fills. | `OrderUpdate.fills` means fills since the previous update for that order. | No (sim emits one update per order) |
| `account()` | Yes, but live it must be the harness ledger (reconciled), not a raw broker read: `Position` needs `entry_date` and `entry_order_id`, which Alpaca's position list does not carry. | Docstring only. | No |
| `cancel_all(reason)` | Yes. Live must confirm, not assume. | New reasons (`kill_switch`, `session_deadline`). | No |

Missing from the interface, and deliberately kept out of it: order types, limit prices,
time in force, retries, approval. Those are execution policy and belong to the order
manager, below the engine.

## 0.2 "Fill at the next open" live, and sizing

Three sim behaviors have no auction equivalent (found in the slice 1
review): buys resized at the fill, sells funding buys in the same open, and the
position cap counting today's exits as free slots.

**Option P1, everything in the opening auction.** Exits market-on-open, entries
limit-on-open, all submitted before the open. Fills at the official open, the exact price
the backtest uses. But entries and exits fill at the same instant, so: a buy cannot be
resized after a gap; if an exit does not fill (halt), its replacement entry still does,
which means borrowed cash and `max_positions + 1` holdings; and the buy needs buying power
before the sell proceeds exist, so it only works on a margin account.

**Option P2, sequenced (recommended).** Exits go in the auction as market-on-open. After
the open, once exits are terminal, entries are re-checked against actual cash and actual
positions, resized, and sent as limit orders capped at `close_t * (1 + entry_limit_pct)`.
This is what `SimBroker` already does (sells first, then buys cut to the cash that is
really there), executed over a few minutes instead of one instant. Leverage and the
position-cap overshoot become impossible by construction, the account can run with margin
disabled, and no real-time quote is needed (the cap is the only price input, and a limit
above the market fills at the market). Cost: entries fill a few minutes after the open,
not at it. That difference is recorded, not assumed away: every live fill keeps its
`filled_at` timestamp, and once the day's bar exists the runner writes, per fill, the
official open, seconds after the open, and slippage in basis points against that open
(extra columns in the paper `trades.csv`, summarized in `summary.json`). This is the
measure of what D1 costs and the data the cost open question needs. Paper is compared
with its backtest by return and weight, not by share count, because backtest quantities
come from back-adjusted prices.

**Sizing from the prior close still holds for the target, not for affordability.**
Construction keeps sizing from `close_t` plus estimated slippage. Live adds one check at
release: `quantity * limit_price <= cash`, shrinking if not. The backtest analog is
`reduced_for_cash`. Known divergence: an entry that gaps above the cap does not fill live
(`limit_not_reached`), while the sim fills it reduced. It is logged per order so the
paper-vs-backtest comparison can count it. A matching optional rule in `SimBroker` is a
later, separate decision because it changes backtest results.

Exits that do not fill in the auction (or when the runner starts after the OPG cutoff)
are re-sent as market day orders for the remainder after the open, until the session
deadline. Exits are risk-reducing, so they get persistence that entries do not.

## 0.3 What drives the phases in real time; does one code path survive?

The engine never reads a clock, and that stays true. A new runner (1.6) reads the clock
and calls the same two methods `simulate` calls, in the same order:
`... on_session_close(t)`, `on_session_open(t+1)`, `on_session_close(t+1) ...`.

With D2, both calls happen in one morning process on t+1: `on_session_close(t)` at about
08:00 using bars through t, then `on_session_open(t+1)`, which blocks through the open
workflow. This removes two problems found in the slice 1 review: the loader's "yesterday" clamp is
already correct at 08:00 (no "today is complete" mode), and SIP data older than 15
minutes is all that is needed (free plan). It also means the engine is rebuilt from
scratch every day, so in-memory state must be derivable:

| Engine state | Live source |
|---|---|
| Bar buffer, `_last_close` | Replay history through a new `Engine.warm_up(session, row)` (buffer and closes only; no broker call, no force-close). Reusing `on_session_close(trade=False)` is unsafe live: with real positions it can call `force_close` during warm-up for a symbol with an old data gap and hit a missing `_last_close`. |
| `_missing` counters | Make them a pure function of the buffer and current holdings (trailing sessions with no close for each held symbol). Equivalent in the sim, because a position can only be opened on a session that has a bar. |
| `orders`, `updates`, equity curve | Journal and ledger (1.4). |
| Strategy instance state | None allowed. Already the rule since holdout warm-up; paper makes it mandatory. Promotion runs a restart-equivalence check (1.7). |

One code path survives for strategy, construction, risk, and the engine. It does not
cover execution policy (flag 1). Mitigation: a shared Broker contract test suite that
`SimBroker` and `OrderManager` over a fake venue must both pass.

## 0.4 Models

| Model | Gap | Change | Breaks? |
|---|---|---|---|
| `OrderStatus` | Only `filled`, `canceled`. | Add `partially_filled`, `rejected`, `expired`. Pending orders produce no update; they are order-manager state, visible through status (1.8). | Additive. `Engine._collect_updates` has an `else` that assumes a fill; it must branch on every status. |
| `OrderUpdateReason` | Sim reasons only. | Add `limit_not_reached`, `broker_rejected`, `risk_rejected`, `approval_denied`, `approval_timeout`, `session_deadline`, `kill_switch`, `unknown_outcome`. | Additive |
| `OrderUpdate` | "Canceled has no fills" conflicts with partial-then-canceled if fills are cumulative. | Define fills as incremental. Then a partial is a `partially_filled` update with fills, followed by a `canceled` update with none. Validators stay as they are, plus `rejected` and `expired` carry no fills. | No |
| `Fill` | `kind` is `next_open` or `forced_close`; a live fill is neither. No timestamp. `slippage_cost >= 0` cannot hold price improvement. | Add kind `venue` and `filled_at: datetime \| None` (required, timezone-aware, for `venue`; its New York date must equal `fill_date`). Same date rules as `next_open`: `fill_date > signal_date`, so a live fill on the signal session is still unconstructible. For `venue` fills `reference_price` is the decision close and `slippage_cost` is 0; realized slippage versus the official open is computed in the comparison report. | Additive |
| `Order` | No client id, no price or order type. | Leave it as the sized intent. The client id is the id `OrderManager.submit` returns. Price, type, and time in force live on a new `VenueOrder` below the engine. | No |
| `Position` | Fine if the ledger owns it. Partial entries average into `avg_price`; `entry_date` is the first fill. | None | No |
| `AccountState` | No buying power, account flags, or pending orders. | Leave it. Broker truth gets its own `VenueAccount` and `VenuePosition`, used only by reconciliation. | No |
| Broker-side corrections | Splits change quantity, dividends change cash, delistings remove positions, busts reverse fills. None arrive as fills. | Not modeled as fills. Reconciliation detects them, trading stops, and an operator records a `LedgerAdjustment` (1.4). Automation waits until one actually happens. | New model only |
| `EventKind` | | Add `order_rejected`, `order_expired`, `order_partial`, `risk_shrunk`, `risk_rejected`, `entries_halted`, `position_stale_no_data`. | Additive |

Outside the listed files, `report.round_trips` raises on a second buy fill for a symbol
and on a partial exit. It must group fills by order, and close a trade when the position
is flat (weighted average prices). Identical output for single-fill orders.

## 0.5 Decision log entries that paper or live contradicts

| Entry | Conflict | Proposal |
|---|---|---|
| "Fill at next bar's open" | Live entries fill shortly after the open (P2). | Reword to the invariant that matters: never fill on the signal session; orders are released for the next session's open. |
| "Size at signal close, fill at next open ... reduced to the affordable share count" | Cannot resize inside an auction. | Holds under P2 (resize at release). Would be dropped under P1. |
| "Delisted or missing-data positions are force-closed" | Sim-only. | Live: stale alert plus reconciliation (already an open question; this answers it). |
| "Holdout enforced in the data loader" | Paper must read past `holdout_start` every day; `HoldoutAccess` would flood the holdout log. | New `PaperAccess` proof object with its own log. Requires a frozen profile id, so there is still exactly one choke point. |
| "Keys live only in `.env`. Only the data loader and broker adapters read them." | `load_dotenv` exports keys into `os.environ` for the whole process, including strategy code. | Read with `dotenv_values` into locals. First code PR. |
| "`alpaca.py`: the only code holding trading keys" | False for paper (flag 2). | Reword. |
| "Determinism" | Live fills are not reproducible. | Scope the rule: decisions are deterministic given bars and ledger. Tested by replaying a paper day's inputs. |
| "Backtest → holdout → paper" (philosophy) | Slice order changed, lifecycle not. | Recorded in the Decision log (2026-11-12). D4 enforces it in code. |
| Open question: position cap exceeded when an exit is canceled | | Closed by P2 for live (entries release against actual positions). Still open in the sim. |
| Open question: back-adjusted prices | Limit caps come from adjusted closes. A late vendor adjustment after a split would misprice the cap. | Pre-trade reconciliation catches the quantity change first and stops the day. Both mispricing directions fail safe for buys (smaller fill or no fill). |

---

# Step 1: design

## 1.0 Pipeline and module map

```
08:00  runner ─► reconcile ─► load bars through t ─► Engine.on_session_close(t)
                                   strategy ─► construction ─► risk.review ─► OrderManager.submit   (journal only, nothing sent)
       approval window (optional, entries only)
09:25  OrderManager: exits ─► AlpacaBroker.place (market, OPG)
09:30  Engine.on_session_open(t+1) ─► OrderManager works the queue:
         exits terminal ─► ledger ─► risk.review_release(entries, actual cash) ─► place capped limits ─► terminal or canceled
~09:50 reconcile ─► snapshot, equity point, day report ─► exit
Kill latch: checked before every venue call and between every step. Trippable from any process.
```

| File | Role |
|---|---|
| `harness/broker/venue.py` | `Venue` protocol and wire-level models. No policy. |
| `harness/broker/alpaca.py` | `AlpacaBroker`: implements `Venue` for the paper account. The only trading network code. |
| `harness/risk.py` | `RiskManager`: hard limits. Runs in backtests too when a risk preset is chosen. |
| `harness/orders.py` | `OrderManager`: implements `Broker`. Staging, idempotent submission, the open workflow, approval. |
| `harness/positions.py` | `Ledger`, `reconcile`, P&L. |
| `harness/journal.py` | Append-only event log, the source of truth for restarts. |
| `harness/killswitch.py` | Kill latch. |
| `harness/alerts.py` | `Alerter` protocol and sinks. |
| `harness/profiles.py` | Frozen profiles and promotion. |
| `harness/paper.py` | `run_trading_day`: the clock-driven runner, plus a thin CLI. |
| `harness/service.py` | Functions the slice 3 backend calls (status, control). |
| `config/risk/*.yaml`, `config/execution.yaml` | Risk presets (per run) and execution settings (fixed, reviewed). |

State on disk (all gitignored):

```
state/paper/KILL                      kill latch, account-wide
state/paper/active.json               account id, active profile id, cash offset, approval mode
state/paper/runner.lock               one runner per account
state/paper/alerts.jsonl
state/paper/<profile_id>/journal.jsonl   single writer (the runner), fsync per record
state/paper/<profile_id>/ledger.json     derived snapshot, atomic replace, for readers
state/paper/<profile_id>/inbox/          control commands from other processes, one file each
state/paper/<profile_id>/{equity.csv,trades.csv,summary.json}   same schema as backtests
profiles/<profile_id>/{profile.json,strategy.py}
```

## 1.1 `harness/broker/alpaca.py` and the venue protocol

```python
# harness/broker/venue.py
class VenueOrder(FrozenModel):
    """One order as sent to a venue. Built only by the order manager."""
    client_order_id: str = Field(min_length=1, max_length=48)
    symbol: str
    side: Side
    quantity: int = Field(gt=0)                      # whole shares
    type: Literal["market", "limit"]
    limit_price: float | None = None                 # required for limit, forbidden for market; cents
    time_in_force: Literal["opg", "day"]

class VenueOrderState(FrozenModel):
    """A venue's current view of one order. Quantities are cumulative."""
    client_order_id: str
    venue_order_id: str
    status: Literal["pending", "partially_filled", "filled", "canceled", "rejected", "expired"]
    quantity: int
    filled_quantity: int = Field(ge=0)
    filled_avg_price: float | None
    updated_at: dt.datetime
    reject_reason: str = ""

class VenueAccount(FrozenModel):
    account_id: str
    cash: float
    buying_power: float
    equity: float
    trading_blocked: bool
    shorting_enabled: bool
    margin_multiplier: float

class VenuePosition(FrozenModel):
    symbol: str
    quantity: int            # signed; a negative value is a short and always an error here
    avg_entry_price: float

class MarketSession(FrozenModel):
    date: dt.date
    open: dt.datetime        # timezone-aware; half days differ only in `close`
    close: dt.datetime

class VenueRejected(Exception):
    """Definitive: the venue refused the order and it does not exist. Never retried."""

class VenueUnavailable(Exception):
    """Outcome unknown (timeout, connection error, 5xx). The order may or may not exist."""

class KillSwitchEngaged(Exception):
    """The kill latch is set (or unreadable). No order was sent."""

class Venue(Protocol):
    def now(self) -> dt.datetime: ...
    def sessions(self, start: dt.date, end: dt.date) -> list[MarketSession]: ...
    def account(self) -> VenueAccount: ...
    def positions(self) -> list[VenuePosition]: ...
    def open_orders(self) -> list[VenueOrderState]: ...
    def get_order(self, client_order_id: str) -> VenueOrderState | None:
        """The order with this client id, or None if the venue has never seen it."""
    def place(self, order: VenueOrder) -> VenueOrderState:
        """Send one order. Raises VenueRejected, VenueUnavailable, or KillSwitchEngaged.
        A duplicate client id is reported as the existing order, never as a second order."""
    def cancel(self, client_order_id: str) -> None:
        """Request cancellation. Success means requested, not canceled: poll to confirm."""
    def cancel_all(self) -> None: ...
```

```python
# harness/broker/alpaca.py
class AlpacaBroker:
    """`Venue` for the Alpaca PAPER account. There is no mode parameter and no live URL."""
    PAPER_URL: Final = "https://paper-api.alpaca.markets"

    def __init__(
        self,
        cfg: ExecutionConfig,
        kill: KillSwitch,
        limiter: RateLimiter,
        *,
        dotenv_path: Path = PROJECT_ROOT / ".env",
        client_factory: Callable[[str, str], TradingClientLike] | None = None,   # tests inject a fake
    ) -> None: ...
    def verify_account(self) -> VenueAccount:
        """Startup gate: paper account, not blocked, shorting disabled, margin multiplier 1.
        Raises VenueSetupError otherwise. Called by the runner before anything else."""
```

**Live stays impossible, in layers:** `paper=True` is a literal; only `ALPACA_PAPER_*`
names are read, with `dotenv_values`, into locals (never `os.environ`); the key id must
start with `PK` **(verify)**; the constructed client's base URL is asserted equal to
`PAPER_URL`; a test fails if the string `api.alpaca.markets` appears in `harness/` without
the `paper-` prefix or if any `ALPACA_LIVE` name appears; `FrozenProfile.mode` is
`Literal["paper"]`.

**Broker-side backstops (verify all three):** Alpaca account configuration is believed to
offer `no_shorting`, `max_margin_multiplier`, and `suspend_trade`. Setup sets the first
two on the paper account and `verify_account` refuses to trade without them, so an
oversized sell cannot open a short and a buy cannot borrow even if the harness is wrong.
`suspend_trade` is a candidate second layer for the kill switch.

**Status mapping:** Alpaca's in-flight statuses (`new`, `accepted`, `pending_new`,
`pending_cancel`, `held`, `done_for_day`, ...) map to `pending`. An unrecognized status
maps to `pending` and raises an alert: never assume terminal. A fractional quantity or a
`replaced` status is an error (the harness never replaces in place).

| Failure mode | Guard |
|---|---|
| Live order placed | Layers above. |
| Keys leak to strategy code or logs | `dotenv_values`; redaction test with sentinel keys over logs, journal, and exception text. |
| Order sent while killed | `place` reads the latch immediately before the network call; unreadable latch counts as set. |
| Timeout after the venue accepted the order | `place` raises `VenueUnavailable`; resolution by client id is the order manager's job (1.3). |
| Rate limit | Shared `RateLimiter` (moved out of `data.py`), set below Alpaca's 200/min **(verify)**; cancel calls from the kill path are not throttled. |
| Short sale or margin | Broker-side settings plus order-manager checks. |

**Tests (no network):** a fake `TradingClientLike` built from alpaca-py's own model
classes; mapping tests for every status; `verify_account` refusals; the live-impossible
tests; a suite-wide `conftest.py` fixture that makes any socket connection raise, so no
test in the repo can reach Alpaca even by mistake.

## 1.2 `harness/risk.py`

Presets hold the four limits the Decision log names. Placeholder values, to be replaced
(open question). Operational limits (max order notional, max orders per session) are not
risk appetite and live in `config/execution.yaml`.

```yaml
# config/risk/balanced.yaml  (placeholders)
max_position_pct: 0.25        # of equity, per position, at the worst-case price
max_exposure_pct: 1.00        # long market value / equity; 1.00 means no leverage
daily_loss_cap_pct: 0.03      # close-to-close loss that blocks the next session's entries
max_drawdown_halt_pct: 0.15   # from peak equity; halts entries until an operator resets
# conservative: 0.20 / 0.80 / 0.02 / 0.10      aggressive: 0.30 / 1.00 / 0.05 / 0.25
```

```python
RiskRule = Literal["max_position", "max_exposure", "daily_loss_cap", "drawdown_halt",
                   "no_leverage", "max_positions", "max_order_notional", "not_held"]

class RiskConfig(FrozenModel):
    name: str
    max_position_pct: float = Field(gt=0, le=1)
    max_exposure_pct: float = Field(gt=0, le=1)      # le=1: leverage is not expressible
    daily_loss_cap_pct: float = Field(gt=0, lt=1)
    max_drawdown_halt_pct: float = Field(gt=0, lt=1)

class RiskState(FrozenModel):
    """Everything risk remembers between sessions. In memory for backtests, in the ledger live."""
    peak_equity: float
    prior_equity: float | None
    entries_blocked_for: dt.date | None      # signal date whose entries the daily cap blocks
    halted: bool
    halt_reason: str = ""

class RiskDecision(FrozenModel):
    order: Order                 # as proposed
    approved_quantity: int       # 0 means rejected
    rule: RiskRule | None        # the limit that bound, if any
    detail: str = ""

class RiskReview(FrozenModel):
    decisions: tuple[RiskDecision, ...]      # input order preserved
    orders: tuple[Order, ...]                # approved, with shrunk quantities
    events: tuple[EngineEvent, ...]

class RiskManager:
    def __init__(self, cfg: RiskConfig, limits: OrderLimits, max_positions: int, state: RiskState) -> None: ...

    def on_mark(self, as_of: dt.date, equity: float) -> tuple[EngineEvent, ...]:
        """Update peak and prior equity; trip the daily cap or the drawdown halt. Once per session."""

    def review(self, orders: Sequence[Order], portfolio: PortfolioSnapshot,
               buy_prices: Mapping[str, float]) -> RiskReview:
        """Close-phase check. `buy_prices` is the worst-case price per entry symbol (estimated
        fill price in backtests, the limit cap live). Exits are credited: an entry is judged
        against the portfolio as it will be after today's approved exits."""

    def review_release(self, order: Order, account: AccountState, reserved_cash: float,
                       price: float, marks: Mapping[str, float]) -> RiskDecision:
        """Live only, just before an entry is sent: same limits against actual cash and
        positions after exits, plus no_leverage and the max_positions cap. `reserved_cash`
        is the worst-case cost of entries already released this session."""

    def reset_halt(self, operator: str, reason: str) -> None: ...
    @property
    def state(self) -> RiskState: ...
```

**Risk-reducing is defined by effect, not by label.** An order is risk-reducing if it is
a sell of at most the held quantity. Such an order is approved in full in every state:
halted, daily cap hit, over every limit. A sell larger than the position is cut to the
position (`not_held` if nothing is held), so it can never open a short. Buys are shrunk
to the largest whole-share quantity that satisfies every limit, or rejected at zero.

Engine hook: `Engine(..., risk: RiskManager | None = None)`. With `None` the code path is
today's, and runs are labeled `risk: none` as the Decision log already says.
`Recipe.risk` becomes `RiskConfig | None`.

| Failure mode | Guard |
|---|---|
| A limit blocks closing a position | Sells of held quantity bypass every limit; tested as a property over random states. |
| A mislabeled "exit" flips short | Decided by resulting position, not by `action`. |
| Float noise approves one share too many | Shrink with floor, then re-verify the result against the limit and step down. |
| Strategy or construction bug emits a huge order | `max_order_notional` and `max_orders_per_session` from fixed settings. |
| Leverage through sizing | `max_exposure_pct <= 1` by type; `no_leverage` at release against actual cash. |
| Halt forgotten across restarts | `RiskState` persisted with the ledger; resetting a halt is a journaled operator action. |
| Stale equity feeds the caps | `on_mark` runs only after a clean reconciliation. |

**Tests:** table-driven cases per rule; seeded random property tests (sell of held
quantity always passes; approved buys never breach any limit; review is deterministic and
order-stable); backtest with `risk=None` byte-identical to the golden run; a backtest with
a preset shows `risk_shrunk` and halt events at the expected sessions.

## 1.3 `harness/orders.py`

```python
class ExecutionConfig(FrozenModel):          # config/execution.yaml, fixed, placeholders
    entry_limit_pct: float                   # 0.02: entry cap above the prior close
    opg_cutoff: dt.time                      # 09:25 New York; later exits go as market day orders
    approval_deadline: dt.time               # 09:20
    exit_deadline: dt.time                   # 09:40: unfilled auction exits are re-sent as market
    session_deadline: dt.time                # 09:50: anything still working is canceled
    poll_seconds: float                      # 5
    max_submit_attempts: int                 # 3
    max_requests_per_minute: int             # 150
    limits: OrderLimits                      # max_order_notional, max_orders_per_session
    cash_tolerance: float                    # reconciliation, dollars
    max_clock_skew_seconds: float            # 30

OrderPhase = Literal["staged", "awaiting_approval", "approved", "submitting", "working",
                     "unknown", "filled", "canceled", "rejected", "expired"]

class WorkingOrder(FrozenModel):
    """Order-manager state for one intent. Rebuilt from the journal after a restart."""
    order_id: str                            # deterministic; what the engine sees
    order: Order
    phase: OrderPhase
    venue_orders: tuple[VenueOrder, ...]     # attempts, oldest first; at most one non-terminal
    filled_quantity: int
    reason: OrderUpdateReason = ""

class ApprovalRequest(FrozenModel):
    order_id: str
    content_hash: str                        # of symbol, side, quantity, limit; approval binds to it
    symbol: str
    side: Side
    quantity: int
    limit_price: float | None
    est_notional: float
    expires_at: dt.datetime

class OrderManager(Broker):
    """The engine-facing Broker for paper trading. Owns every decision about when and how
    an approved order reaches the venue. Never raises for venue errors: they become updates."""

    def __init__(self, venue: Venue, ledger: Ledger, journal: Journal, risk: RiskManager,
                 kill: KillSwitch, alerts: Alerter, cfg: ExecutionConfig, profile_id: str,
                 approval: Literal["off", "entries"], sleep: Callable[[float], None]) -> None: ...

    def submit(self, orders: Sequence[Order]) -> list[str]:
        """Journal each order as staged and return its id. Sends nothing. Calling it again
        for the same signal date, symbol, and action returns the existing id."""

    def on_session_open(self, session: dt.date, opens: Mapping[str, float]) -> None:
        """Work the staged orders to terminal states (see workflow). Blocks until done or
        `session_deadline`. Ignores `opens`."""

    def poll_updates(self) -> list[OrderUpdate]: ...
    def account(self) -> AccountState:
        """The ledger's cash and positions (reconciled), not a raw venue read."""
    def cancel_all(self, reason: OrderUpdateReason) -> list[str]:
        """Cancel everything non-terminal and confirm with the venue before returning."""
    def force_close(self, symbol: str, price: float, price_date: dt.date, session: dt.date) -> None:
        """Never fabricates a fill. Alerts `position_stale_no_data`; reconciliation resolves it."""

    def pending_approvals(self) -> list[ApprovalRequest]: ...
    def recover(self) -> None:
        """After a restart: rebuild from the journal, then resolve every `submitting` or
        `unknown` order by client id against the venue before anything new is sent."""
```

**Order ids.** `order_id = "<profile8>-<YYYYMMDD signal date>-<SYMBOL>-<e|x>"`, at most 48
characters **(verify Alpaca's limit)**. A venue attempt appends `-a1`, `-a2`. Because the
id is a function of the decision and not of time or a counter, the same decision can
never yield two different ids.

**Submission protocol (no duplicates):**
1. Journal `intent(client_order_id, order)` and fsync, before the network call.
2. `venue.place`. Success: journal the state. `VenueRejected`: terminal `rejected`, never
   retried.
3. `VenueUnavailable`: phase `unknown`. Call `get_order(client_order_id)`. Found: adopt
   it. Not found: place again with the **same** client id (Alpaca rejects a duplicate
   client id **(verify)**, so even a lost race cannot create a second order). Backoff
   between attempts. After `max_submit_attempts`: leave it `unknown`, block all further
   entries for the session, alert. Exits keep trying.
4. A new attempt (`-a2`) is created only after the previous attempt is confirmed terminal
   by the venue, with quantity `intent - filled`.

**Open workflow** inside `on_session_open`:
1. Approval (if `entries`): entries wait in `awaiting_approval`; commands arrive through
   the inbox. Unapproved at `approval_deadline`: `canceled / approval_timeout`. Exits skip
   this step.
2. Exits: market, `opg`, sent before `opg_cutoff`. If the runner is late, or the auction
   leaves a remainder at `exit_deadline`, the remainder goes as a market `day` order.
3. Wait for the open (venue clock). Poll exits until terminal or `exit_deadline`.
4. Apply exit fills to the ledger. For each entry in rank order: `risk.review_release`
   with actual cash minus cash reserved by earlier entries, then place a limit `day`
   order at `round_down_to_cent(close_t * (1 + entry_limit_pct))`.
5. Poll. At `session_deadline` cancel what is still working, confirm, and report partials
   as `partially_filled` then `canceled / limit_not_reached`.
6. Return only when every order is terminal, or raise `SessionIncomplete` (kill and
   alert) if the venue cannot confirm.

**Invariants** (asserted at run time and tested): at most one non-terminal venue order
per intent; filled quantity never exceeds the intent; no sell exceeds the ledger
position; every venue call has a journal record before it; no venue call while the latch
is set; at most `max_orders_per_session` venue orders per session.

| Failure mode | Guard |
|---|---|
| Duplicate order after a timeout or crash | Deterministic client id, write-ahead journal, lookup before resend, venue-side uniqueness. |
| Crash between decision and submission | Decisions are journaled once per (profile, signal date). A restart resumes them; it never re-runs the strategy for that date. |
| Fill arrives while canceling | Cancel is a request; the final state and filled quantity are read back after it. |
| Entry sent before its funding exit filled | Entries are released only after exits are terminal, against actual cash. |
| Approval applied to a different order | Approval carries the content hash; a mismatch is rejected. |
| Human away, exit blocked | Exits are never gated. |
| Runaway loop | Per-session order cap, rate limiter, bounded attempts. |
| Orders left resting | `on_session_open` cannot return with a non-terminal order; the day closes with a "no open orders" reconciliation. |
| Stock gaps past the cap | Not chased. `limit_not_reached`, logged for the comparison. |

**Tests:** `tests/fake_venue.py`, an in-memory venue with a controllable clock and a
script per order (fill, partial, reject, accept-then-timeout, timeout-before-accept, slow
fill, fill-during-cancel, duplicate id). Scenario tests per failure mode. A crash test
that injects an exception at every journal step of a scripted day, restarts, and asserts
the venue saw each client id once and the final ledger equals the uninterrupted run. The
shared Broker contract suite.

## 1.4 `harness/positions.py` and `harness/journal.py`

```python
class Journal:
    """Append-only JSONL. One writer. Each record has seq, UTC time, kind, payload."""
    def append(self, kind: JournalKind, payload: Mapping[str, JsonValue]) -> int:
        """Write and fsync before returning. A torn final line is dropped on read."""
    def read(self) -> Iterator[JournalRecord]: ...

class LedgerAdjustment(FrozenModel):
    """An operator-approved correction that did not come from a fill."""
    as_of: dt.date
    kind: Literal["split", "dividend", "fee", "delisting", "bust", "manual"]
    symbol: str | None
    cash_delta: float
    quantity_delta: int
    operator: str
    note: str = Field(min_length=1)

class PnL(FrozenModel):
    as_of: dt.date            # the close the marks come from
    cash: float
    market_value: float
    equity: float
    realized: float           # closed trades, after costs
    unrealized: float
    day_change: float | None

class Ledger:
    """Cash, positions, and realized P&L, derived only from fills and adjustments.
    Replaying the journal always rebuilds the same ledger."""
    @classmethod
    def replay(cls, journal: Journal, starting_cash: float) -> Ledger: ...
    def apply_fill(self, fill: Fill) -> None:
        """Idempotent per (order_id, cumulative quantity). Raises on a sell beyond the position."""
    def apply_adjustment(self, adj: LedgerAdjustment) -> None: ...
    def account(self) -> AccountState: ...
    def pnl(self, marks: Mapping[str, float], as_of: dt.date) -> PnL: ...

class PositionDiff(FrozenModel):
    symbol: str
    ledger_quantity: int
    venue_quantity: int

class ReconcileReport(FrozenModel):
    at: dt.datetime
    position_diffs: tuple[PositionDiff, ...]
    cash_diff: float                        # venue cash - cash offset - ledger cash
    unexpected_orders: tuple[str, ...]      # open at the venue, unknown to the journal
    missing_orders: tuple[str, ...]         # working in the journal, unknown to the venue
    severity: Literal["ok", "warn", "stop"]

def reconcile(ledger: AccountState, expected_open: Sequence[str], account: VenueAccount,
              positions: Sequence[VenuePosition], open_orders: Sequence[VenueOrderState],
              cash_offset: float, cash_tolerance: float) -> ReconcileReport:
    """Pure comparison. `stop` on any position difference, any short, any unexpected or
    missing order, or a cash difference beyond tolerance. `warn` within tolerance."""
```

The ledger, not the broker, is the book: `Position.entry_date` and `entry_order_id` exist
only there, and a time-based exit needs them. The broker is the check. The account must
be dedicated to one profile. Capital is the profile's `starting_cash`; if the paper
account holds more, the difference is recorded once as `cash_offset` at activation and
risk percentages use ledger equity.

Reconciliation runs before the decision, after the open workflow, and on demand. `stop`
trips the kill switch. It is cleared only by `accept_broker_state(operator, note)`, which
writes `LedgerAdjustment`s so the correction is on the record. Nothing auto-corrects in
this slice.

Fills are derived from each order's cumulative `filled_quantity` and `filled_avg_price`
(the increment's price is solved from the two averages). Alpaca's account activities feed
has per-execution records **(verify)** and is the upgrade path if averages prove too
coarse.

| Failure mode | Guard |
|---|---|
| Ledger drifts from the broker | Reconcile twice a day; any position difference stops trading. |
| Split, delisting, bust | Shows up as a difference; operator adjustment; no silent fix. |
| Same fill applied twice (poll overlap, restart) | Idempotent on cumulative quantity. |
| Snapshot corrupted or half-written | Snapshot is derived; the journal is the truth; atomic replace. |
| Someone trades the account by hand | `unexpected_orders` or a position difference: stop. |
| Dividend cash appears | Cash difference; `warn` within tolerance, `stop` beyond, adjust. |

**Tests:** replay equals incremental application; torn-line recovery; a reconcile table
(each difference type to its severity); idempotent fill application; P&L against
hand-computed cases, including partial exits.

## 1.5 Kill switch, logging, alerts

```python
class KillState(FrozenModel):
    engaged: bool
    at: dt.datetime | None = None
    reason: str = ""
    source: str = ""                    # "operator:cli", "auto:reconcile", ...
    cancel_confirmed: bool = False

class KillSwitch:
    """A latch file. Engaged means: no order may be sent, by anything, until an operator resets it."""
    def __init__(self, path: Path) -> None: ...
    def engaged(self) -> bool:
        """True if the latch exists or cannot be read. Fails closed."""
    def engage(self, reason: str, source: str) -> KillState:
        """Write the latch atomically. Idempotent. Does no network I/O."""
    def reset(self, operator: str, confirm: str) -> None:
        """Remove the latch. `confirm` must equal the active profile id."""

def kill(reason: str, source: str, *, venue: Venue | None = None) -> KillState:
    """Engage the latch first, then cancel every open order at the venue and confirm
    none remain. Safe to call from any process, including while the runner is hung.
    If the cancel cannot be confirmed, the latch stays set and the alert says so."""

class Alert(FrozenModel):
    at: dt.datetime
    severity: Literal["info", "warn", "critical"]
    kind: str
    message: str

class Alerter(Protocol):
    def send(self, alert: Alert) -> None:
        """Must not raise: a failing alert sink never blocks trading logic or the kill path."""
```

Three stop levels, weakest first:

| Level | Trigger | Effect | Cleared by |
|---|---|---|---|
| Entries blocked | Daily loss cap | No entries for the next session; exits continue | Automatically, next session |
| Entries halted | Drawdown halt | No entries; exits continue | Operator `reset_halt` |
| Killed | Operator; reconciliation `stop`; unknown order outcome; `SessionIncomplete`; clock skew; profile or strategy hash mismatch | Open orders canceled, runner exits, nothing sends | Operator `reset`, after a clean reconciliation |

The kill switch does not sell. A switch that market-sells everything during a data or
reconciliation glitch can itself be the loss. `flatten(operator, confirm)` is a separate
command: it works while killed, places risk-reducing sells only, and goes through the
same order manager and journal.

Logging: the journal is the audit trail (decisions, risk reviews, intents, venue
responses, fills, reconciliations, control commands, alerts). A human-readable log goes
to `logs/paper/<date>.log`. Keys never appear in either. Alerts go to the log and
`alerts.jsonl`; a webhook sink is a small later addition (D8).

**Tests:** latch engaged makes `place` raise without calling the client; unreadable latch
counts as engaged; `kill` from a second process while a fake runner is blocked; cancel
failure leaves the latch set and raises a critical alert; `flatten` produces only sells
of held quantity; an alert sink that raises does not break `kill`.

## 1.6 The runner and the market clock (`harness/paper.py`)

```python
class DayReport(FrozenModel):
    session: dt.date                      # t+1, the session traded
    decided_on: dt.date                   # t, the close the decision used
    outcome: Literal["traded", "no_session", "already_done", "skipped_stale_data", "killed"]
    orders: tuple[WorkingOrder, ...]
    reconcile_before: ReconcileReport | None
    reconcile_after: ReconcileReport | None
    pnl: PnL | None
    alerts: tuple[Alert, ...]

def run_trading_day(profile_id: str, *, venue: Venue | None = None,
                    loader: DataLoader | None = None,
                    sleep: Callable[[float], None] = time.sleep) -> DayReport:
    """Run one trading day for a frozen profile: decide on the last completed session,
    trade the next open, reconcile, exit. Safe to run twice: the second run resumes or
    reports `already_done`. Started by the OS scheduler, never by the backend."""
```

Steps, each journaled, with the kill latch checked between them:

1. Take `runner.lock`. Verify the profile hash and the strategy file hash. Latch clear.
2. `venue.verify_account()`. Compare local time with `venue.now()`; skew beyond the limit
   kills. Ask `venue.sessions` whether today is a session (holidays; half days only move
   the close, which this schedule never touches). Not a session: `no_session`.
3. Journal has `day_complete` for today: `already_done`. Journal has a partial day:
   `OrderManager.recover()`, then continue from the open workflow. The strategy is not
   called again.
4. Reconcile. `stop`: kill.
5. Load bars with `loader.load(symbols, paper=PaperAccess(profile_id))`. If the latest
   bar of the benchmark symbol is not session t: `skipped_stale_data`, alert, no orders.
   If earlier sessions were missed (calendar gap since the last `day_complete`): alert
   and carry on from the latest close.
6. Build a fresh `Engine` with `OrderManager` as its broker and the profile's
   `RiskManager`. `warm_up` every session before t, then `on_session_close(t, row)`.
   The decision record (orders after risk) is journaled.
7. `engine.on_session_open(t+1, {})`: the open workflow (1.3).
8. Reconcile again. Write the ledger snapshot, the equity point for t, `trades.csv`,
   `summary.json`, and `day_complete`. Release the lock.

Two checks the runner makes before step 6, because the engine raises on both (found in
the review of PR 5):

- Every held symbol is in the profile's universe and has at least one bar in the
  replayed history. Otherwise stop with a clear error (a manual trade in the paper
  account is the likely cause).
- A rebuilt engine starts with no `orders`. `OrderManager.poll_updates` must never
  report an update without fills for an order this engine did not submit (a late cancel
  or reject of an earlier day's order): those go to the journal, not to the engine.

The slice 1 review's thread warning is met by construction: strategy code runs in step 6 inside
`harness.guard`, venue I/O happens before and after it on the same thread, never during.

`PaperAccess` (in `harness/data.py`) is the loader's second proof object: it lifts the
holdout cutoff for a frozen profile and appends one line per day to a paper access log.
`HoldoutAccess` and the holdout log are untouched.

| Failure mode | Guard |
|---|---|
| Two runners | Lock file; deterministic ids as a second layer. |
| Run on a holiday or weekend | Venue calendar. |
| Wrong machine clock | Skew check; all scheduling uses venue time. |
| Yesterday's bar missing or late | Skip the day. No decision on stale data. |
| Crash anywhere | Resume from the journal; no re-decision; no duplicate orders. |
| Machine asleep | Not preventable here. Missed-session alert on the next run (flag 7). |
| Strategy raises | No orders for the day, critical alert, positions untouched. |

**Tests:** full days against `FakeVenue` with a fake clock; holiday; stale data; rerun
after completion; crash and resume at every step; a late start after the OPG cutoff;
decisions for a day equal a sim backtest's decisions for the same day given the same
positions.

## 1.7 Frozen profiles (`harness/profiles.py`)

```python
class FrozenProfile(FrozenModel):
    profile_id: str                        # "<name>-<first 8 of content_hash>"
    name: str
    mode: Literal["paper"]
    created_at: dt.datetime
    recipe: Recipe                         # fully resolved values, including risk
    execution: ExecutionConfig
    strategy_source_sha256: str
    harness_git_commit: str
    source_run_id: str | None              # the in-sample run it was promoted from
    holdout_run_id: str | None
    plumbing: bool                         # True only for the tracked example strategy
    content_hash: str                      # sha256 of the canonical JSON of everything above

def promote_run(run_dir: Path, name: str, *, plumbing: bool = False) -> FrozenProfile:
    """Freeze a validated run. Refuses unless: the run's strategy hash matches the file on
    disk; the run was made from a clean git tree; the run used a risk preset; a holdout run
    exists for the same recipe (or `plumbing` and the strategy is example_buy_and_hold);
    and the restart-equivalence check passes. Writes profile.json and a copy of the
    strategy source, read-only."""

def load_profile(profile_id: str) -> FrozenProfile:
    """Load and verify the content hash and the strategy file hash. Any mismatch raises
    ProfileTampered; the runner turns that into a kill."""

def check_restart_equivalence(recipe: Recipe, loader: DataLoader) -> None:
    """Run the backtest twice: once normally, once rebuilding the strategy and engine
    every session. Raises if signals differ, which means the strategy keeps state that a
    daily paper process would lose."""

def activate_profile(profile_id: str, approval: Literal["off", "entries"]) -> None:
    """Bind the paper account to this profile. Requires a flat account with no open
    orders. Records the account id and cash offset in active.json."""
```

A profile embeds values, not preset names, so editing a preset never changes what is
trading. Changing anything means a new profile with a new id and a new state folder.
Profiles are gitignored like strategies, except the plumbing example. A harness commit
that differs from `harness_git_commit` is a warning in the day report, not a refusal
(the harness keeps changing during this slice); worth revisiting before live.

**Tests:** each refusal; tamper detection (edit one byte of the profile or the strategy);
activation refuses a non-flat account; the equivalence check passes for the example
strategy and fails for a fixture strategy that counts `on_bar` calls.

## 1.8 UI seams (`harness/service.py`), no UI work

Plain functions over state files, the way `run_backtest(recipe)` works today. Read
functions never call the broker; they read the snapshot the runner wrote.

```python
# read-only: safe to hand to the chat agent
def trading_mode() -> Literal["paper"]: ...
def get_status() -> TradingStatus:          # profile, kill state, stop level, runner phase,
    ...                                     # last reconcile, last run outcome, as-of times
def get_positions() -> PositionsView: ...   # positions with marks, cash, PnL, as_of
def list_orders(session: dt.date | None = None) -> list[WorkingOrder]: ...
def list_pending_approvals() -> list[ApprovalRequest]: ...
def list_alerts(since: dt.datetime | None = None) -> list[Alert]: ...
def list_profiles() -> list[FrozenProfile]: ...

# control: operator only, each call journaled with the operator's name
def engage_kill_switch(reason: str, operator: str) -> KillState: ...
def reset_kill_switch(operator: str, confirm: str) -> None: ...
def approve_orders(approvals: Mapping[str, str], operator: str) -> None:   # order_id -> content_hash
    ...
def reject_orders(order_ids: Sequence[str], operator: str, reason: str) -> None: ...
def reset_risk_halt(operator: str, reason: str) -> None: ...
def flatten(operator: str, confirm: str) -> None: ...
def accept_broker_state(operator: str, note: str) -> ReconcileReport: ...
def promote_run(run_id: str, name: str) -> FrozenProfile: ...
def activate_profile(profile_id: str, approval: Literal["off", "entries"], operator: str) -> None: ...
```

Control functions do not write the journal directly (single writer). `engage_kill_switch`
writes the latch and cancels at the venue itself. The others drop a command file in the
inbox, which the runner ingests, or apply it immediately if no runner holds the lock.

**Conflicts with `docs/ui-design.md`:**

1. "The UI may write" lists presets and run choices only. Kill, approve, reject, reset,
   flatten, promote, and activate are new write actions. The contract needs a third list:
   "The UI may trigger".
2. "The chat is read-only." The read and control functions must be separate import
   surfaces, and the chat agent gets only the read one. That includes the kill switch:
   an agent that can stop trading can be talked into stopping it at a bad moment.
3. "Paper by default ... explicit toggle." Mode must not be a request parameter the UI
   passes. It is a property of the active profile and the key set. In this slice
   `trading_mode()` is a constant.
4. "Few knobs." Approval mode and the risk preset are new choices. The risk preset is
   frozen into the profile at promotion; approval mode is set at activation. Neither is
   editable while a profile is trading.
5. "The UI runs backtests through `run_backtest`." The UI must not run trading days. The
   scheduler does, in its own process. The backend observes and controls.
6. The "UI reads" list gains `state/paper/<profile>/` with the same file schemas as
   `backtests/<run>/`.

## 1.9 Changes to slice 1 code (summary)

| Change | File | Backtests | Existing tests |
|---|---|---|---|
| Keys via `dotenv_values` | `data.py` | Unchanged | Loader tests that patch the environment need a small edit |
| New statuses, reasons, `venue` fill kind, `filled_at` | `models.py` | Unchanged | Additive; new tests |
| `_collect_updates` branches on every status | `engine.py` | Unchanged | Pass |
| `force_close` may return `None` | `broker/base.py`, `engine.py` | Unchanged | Pass |
| `warm_up`; missing-bar counts derived from the buffer | `engine.py` | Must be byte-identical (golden run) | `test_missing_counter_resets_when_bars_return` should pass as is |
| Optional `risk` hook; `Recipe.risk` | `engine.py`, `config.py`, `run.py`, `report.py` | Unchanged with `risk=None`; saved recipe gains `risk: null` | Recipe round-trip tests gain a field |
| Round trips from multi-fill orders | `report.py` | Unchanged for single-fill orders | Pass; new tests |
| `PaperAccess` | `data.py` | Unchanged | Pass; new tests |
| `RateLimiter` moved to its own module | `data.py` | Unchanged | Import path |

A golden-run test (fixed synthetic data, expected `summary.json` and `trades.csv` checked
in) lands before any of these, so "unchanged" is proven per PR and not just asserted.
`test_determinism.py` only proves two runs agree with each other.

---

# Step 2: PR plan

One purpose and one commit each. Every PR runs pytest, ruff, and pyright, and every PR
that touches `harness/` shows the golden run unchanged. "Sensitive" follows the
`CLAUDE.md` list. Docs for a component are updated in the PR that adds it.

| # | Title | Contents | Sensitive |
|---|---|---|---|
| 1 | `docs: plan slice 2 ahead of strategy validation` | Decision log entry, this doc, the reworded one-code-path rule in `CLAUDE.md`, the unattended-use open question; `docs/slice2-handoff.md` is dropped (its content is folded into this doc) | `CLAUDE.md` rule change |
| 2 | `test: pin backtest outputs with a golden run` | Golden synthetic run; suite-wide socket block in `conftest.py` | |
| 3 | `fix(data): read API keys without exporting them` | `dotenv_values`; test that `os.environ` stays clean | Credentials |
| 4 | `feat(models): order states for partial fills, rejections, expiry` | 0.4 model changes; engine update handling | `harness/` boundary |
| 5 | `refactor(engine): derive missing-bar counts, add warm_up` | 0.3; `force_close -> str \| None` | `harness/` boundary |
| 6 | `test(broker): shared Broker contract suite` | Contract tests, run against `SimBroker` | |
| 7 | `feat(report): build round trips from multi-fill orders` | 0.4 | |
| 8 | `feat(config): risk presets and execution settings` | `config/risk/*.yaml`, `config/execution.yaml`, models, `Recipe.risk` | Config boundary |
| 9 | `feat(risk): hard limits` | `harness/risk.py`, engine hook, `risk: none` label | `harness/` boundary |
| 10 | `feat(broker): venue protocol and fake venue` | `venue.py`, `tests/fake_venue.py`, `RateLimiter` move | |
| 11 | `feat(positions): journal, ledger, reconciliation` | `journal.py`, `positions.py` | |
| 12 | `feat(killswitch): kill latch and alerts` | `killswitch.py`, `alerts.py` | |
| 13 | `feat(orders): idempotent submission and recovery` | `OrderManager` core: staging, ids, protocol, `recover`, crash tests | `harness/` boundary |
| 14 | `feat(orders): open workflow` | Exits, release with `review_release`, limit pricing, deadlines; contract suite now also runs against `OrderManager` | |
| 15 | `feat(orders): manual approval for entries` | Approval gate, inbox | |
| 16 | `feat(broker): Alpaca paper adapter` | `alpaca.py`; every **(verify)** item checked against the docs first and recorded in the PR | Credentials, dependencies if alpaca-py needs a bump |
| 17 | `feat(data): paper access path` | `PaperAccess`, paper access log | Holdout handling |
| 18 | `feat(profiles): frozen profiles and promotion` | `profiles.py`, restart-equivalence check, plumbing profile | |
| 19 | `feat(paper): daily runner` | `paper.py`, CLI, crash-and-resume tests | |
| 20 | `feat(service): status and control functions` | `service.py`; `ui-design.md` contract update | |
| 21 | `test(paper): end-to-end against the fake venue` | Multi-day buy-and-hold with injected faults; operator runbook | |
| 22 | End-to-end on the real paper account (run by you) | Checklist below; results recorded in `ARCHITECTURE.md` | Credentials |

PRs 13, 14, 16, 11, 9, 12, and 19 are the safety-critical ones, in that order of
priority if access runs short. PRs 1 to 8, 17, and 20 can be built by another model from
this doc.

**End-to-end paper checklist (PR 22), using `strategies/example_buy_and_hold.py`:**

1. Set the paper account: shorting off, margin multiplier 1, balance at or above
   `starting_cash`. `verify_account` passes.
2. `promote_run(..., plumbing=True)` and `activate_profile(..., approval="entries")`.
3. Day 1: approve the SPY entry; confirm the fill, the ledger, and a clean reconciliation.
   Record whether the paper simulator filled the limit near the open.
4. Day 2: no orders; reconciliation clean; equity point written.
5. Run the day twice: second run reports `already_done`, no new orders at Alpaca.
6. Kill the runner process mid-open on a day with an order (re-activate flat to get one);
   restart; one order at Alpaca, not two.
7. Engage the kill switch from a second terminal; confirm the next run refuses; reset.
8. Place a manual order in the Alpaca dashboard; confirm reconciliation stops trading;
   `accept_broker_state`.
9. `flatten`: confirms the market-on-open exit path and records whether paper honors OPG.
10. Compare paper fills with the official opens; note the slippage against 5 bps.
