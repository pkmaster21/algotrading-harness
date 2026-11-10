"""Portfolio construction: turn one session's signals into sized orders.

Slice 1 rule: long-only, equal weight of current equity, at most `max_positions` open
positions, competing entries ranked by signal strength.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from harness.config import ConstructionConfig, CostConfig
from harness.costs import apply_costs
from harness.models import (
    EngineEvent,
    EventKind,
    FrozenModel,
    Order,
    PortfolioSnapshot,
    Signal,
)

# Guards floor() against float noise such as 9.999999999 meaning 10. Any resulting
# overshoot is caught at fill time, where SimBroker caps buys at available cash.
_FLOOR_EPSILON = 1e-9


def _strength(sig: Signal) -> float:
    if sig.strength is None:  # unreachable: Signal requires strength on entries
        raise ValueError(f"entry signal for {sig.symbol} has no strength")
    return sig.strength


class ConstructionResult(FrozenModel):
    orders: tuple[Order, ...]  # exits first, then entries in rank order
    events: tuple[EngineEvent, ...]


def build_orders(
    signals: Sequence[Signal],
    portfolio: PortfolioSnapshot,
    closes: Mapping[str, float],
    cfg: ConstructionConfig,
    costs: CostConfig,
) -> ConstructionResult:
    """Turn signals emitted at the close of `portfolio.as_of` into sized orders.

    Expects one-leg stock signals (the engine validates this before calling).
    `closes` holds today's close for symbols that have a bar today.

    1. Exits: a held symbol is sold in full. An exit for a symbol not held is ignored
       and logged (`exit_ignored_not_held`).
    2. Entries: symbols already held are ignored (`entry_ignored_held`), including ones
       exiting today, so there is no same-day round trip. Entries without a close today
       are skipped (`entry_skipped_no_price`). Duplicate signals for a symbol keep the
       highest strength (first wins ties) and log `signal_duplicate`.
    3. Rank entries by (-strength, symbol); the symbol tiebreak keeps results deterministic.
    4. Open slots = max_positions minus positions remaining after today's exits. Entries
       beyond the slots are skipped (`entry_skipped_no_slot`).
    5. Each entry targets `equity / max_positions`, capped by the remaining budget
       (cash plus estimated exit proceeds). Shares = floor(allocation / estimated buy
       price), where the estimate is today's close plus slippage. Zero shares is logged
       as `entry_skipped_no_cash` and does not use a slot.
    """
    as_of = portfolio.as_of
    events: list[EngineEvent] = []
    orders: list[Order] = []

    def log(kind: EventKind, symbol: str, detail: str = "") -> None:
        events.append(EngineEvent(date=as_of, kind=kind, symbol=symbol, detail=detail))

    # 1. exits
    exiting: set[str] = set()
    seen_exits: set[str] = set()  # held or not, so a repeated exit logs one duplicate
    budget = portfolio.cash
    for sig in signals:
        if sig.action != "exit":
            continue
        sym = sig.symbol
        if sym in seen_exits:
            log("signal_duplicate", sym, "duplicate exit")
            continue
        seen_exits.add(sym)
        position = portfolio.positions.get(sym)
        if position is None:
            log("exit_ignored_not_held", sym, sig.reason)
            continue
        exiting.add(sym)
        orders.append(
            Order(
                created_on=as_of,
                action="exit",
                legs=sig.legs,
                quantity=position.quantity,
                reason=sig.reason,
            )
        )
        close = closes.get(sym)
        if close is not None and math.isfinite(close) and close > 0:
            est = apply_costs("sell", close, position.quantity, costs)
            budget += est.fill_price * position.quantity - est.commission
        # No close today: proceeds are not counted, so entries are sized conservatively.

    # 2. entry candidates, best signal per symbol
    best: dict[str, Signal] = {}
    for sig in signals:
        if sig.action != "enter":
            continue
        sym = sig.symbol
        if portfolio.holds(sym):
            detail = "exiting today" if sym in exiting else "already held"
            log("entry_ignored_held", sym, detail)
            continue
        current = best.get(sym)
        if current is not None:
            log("signal_duplicate", sym, "duplicate entry")
            if _strength(sig) > _strength(current):
                best[sym] = sig
            continue
        best[sym] = sig

    candidates: list[Signal] = []
    for sym, sig in best.items():
        close = closes.get(sym)
        if close is None or not math.isfinite(close) or close <= 0:
            log("entry_skipped_no_price", sym)
            continue
        candidates.append(sig)

    # 3. rank
    candidates.sort(key=lambda s: (-_strength(s), s.symbol))

    # 4-5. slots and sizing
    slots = max(0, cfg.max_positions - (len(portfolio.positions) - len(exiting)))
    target = portfolio.equity / cfg.max_positions
    for sig in candidates:
        sym = sig.symbol
        if slots == 0:
            log("entry_skipped_no_slot", sym, f"strength={sig.strength}")
            continue
        est = apply_costs("buy", closes[sym], 1, costs)
        allocation = min(target, budget) - est.commission
        quantity = 0
        if allocation > 0:
            quantity = math.floor(allocation / est.fill_price + _FLOOR_EPSILON)
        if quantity < 1:
            log("entry_skipped_no_cash", sym, f"allocation={allocation:.2f}")
            continue
        orders.append(
            Order(
                created_on=as_of,
                action="enter",
                legs=sig.legs,
                quantity=quantity,
                strength=sig.strength,
                reason=sig.reason,
            )
        )
        budget -= quantity * est.fill_price + est.commission
        slots -= 1

    return ConstructionResult(orders=tuple(orders), events=tuple(events))
