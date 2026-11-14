"""The ledger (the harness's own book of cash and positions) and reconciliation against
the venue.

The ledger, not the broker, is the book: it knows when and by which order each position
was opened, which a broker's position list does not carry. It changes only through fills
and operator adjustments, each written to the journal before it is applied, so replaying
the journal always rebuilds the same ledger.

The broker is the check. `reconcile` compares the two and never corrects anything: any
difference in positions or open orders, any short, and any cash difference beyond the
tolerance is `stop`. Clearing a stop is an operator decision recorded as adjustments.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from harness.broker.venue import Venue, VenueAccount, VenueOrderState, VenuePosition
from harness.journal import Journal
from harness.models import AccountState, Fill, FrozenModel, Position, Stock

_CENT = 0.005  # cash differences smaller than half a cent are float noise


class LedgerError(RuntimeError):
    """A fill or adjustment does not fit the ledger, or the journal does not replay."""


class LedgerAdjustment(FrozenModel):
    """An operator-approved correction that did not come from a fill.

    `quantity_delta` adds or removes shares of `symbol`. Shares that are added bring
    `cost_basis_delta` of cost with them. Shares that are removed take their share of the
    position's cost with them. `cash_delta` is the cash received or paid. Whatever cash
    does not match the cost moved is realized P&L (a dividend, a fee, a delisting cash-out
    above or below cost).

    A `split` is the exception: it changes the share count in either direction and leaves
    the position's total cost alone, so only its `cash_delta` (cash paid for a fractional
    leftover share) is realized.

    `adjustment_id` names the real-world event. The ledger applies each id once, so
    entering the same adjustment again (after a write error, say) changes nothing.
    """

    adjustment_id: str = Field(min_length=1)
    as_of: dt.date
    kind: Literal["split", "dividend", "fee", "delisting", "bust", "manual"]
    symbol: str | None = None
    cash_delta: float = 0.0
    quantity_delta: int = 0
    cost_basis_delta: float = Field(default=0.0, ge=0)
    operator: str = Field(min_length=1)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.symbol is not None:
            Stock(symbol=self.symbol)
        if self.quantity_delta and self.symbol is None:
            raise ValueError("a quantity change needs a symbol")
        if self.cost_basis_delta and self.quantity_delta <= 0:
            raise ValueError("cost_basis_delta only applies when shares are added")
        if not self.cash_delta and not self.quantity_delta:
            raise ValueError("an adjustment changes cash or a quantity")
        if self.kind == "split" and not self.quantity_delta:
            raise ValueError("a split changes a quantity")
        if self.kind == "split" and self.cost_basis_delta:
            raise ValueError("a split does not change the cost basis")
        return self


class PnL(FrozenModel):
    as_of: dt.date  # the close the marks come from
    cash: float
    market_value: float
    equity: float
    realized: float  # closed shares, dividends and fees, after costs
    unrealized: float
    day_change: float | None  # equity minus the prior equity given, if any


class LedgerSnapshot(FrozenModel):
    """The ledger's state for readers. Derived: the journal is the truth."""

    journal_seq: int
    starting_cash: float
    cash: float
    realized: float
    positions: dict[str, Position]
    cost_basis: dict[str, float]


class Ledger:
    """Cash, positions and realized P&L, built only from fills and adjustments."""

    def __init__(self, starting_cash: float) -> None:
        if not starting_cash > 0:
            raise ValueError("starting_cash must be positive")
        self._starting_cash = float(starting_cash)
        self._cash = float(starting_cash)
        self._realized = 0.0
        self._positions: dict[str, Position] = {}
        self._cost: dict[str, float] = {}  # total cost of the shares held, commissions included
        self._filled: dict[str, int] = {}  # order id -> cumulative quantity applied
        self._adjustments: dict[str, LedgerAdjustment] = {}  # by adjustment_id
        self._journal: Journal | None = None

    @classmethod
    def open(cls, journal: Journal, starting_cash: float) -> Ledger:
        """The ledger recorded in `journal`, ready to record more.

        An empty journal starts a new ledger with `starting_cash`. Otherwise the journal
        is replayed, and it must have been started with the same `starting_cash`.
        """
        ledger = cls(starting_cash)
        records = list(journal.read())
        if not any(r.kind == "ledger_open" for r in records):
            if any(r.kind in ("fill", "adjustment") for r in records):
                raise LedgerError("journal has fills or adjustments but no ledger_open record")
            journal.append("ledger_open", {"starting_cash": ledger._starting_cash})
        for record in records:
            try:
                if record.kind == "ledger_open":
                    recorded = record.payload["starting_cash"]
                    if recorded != ledger._starting_cash:
                        raise LedgerError(
                            f"journal was opened with starting cash {recorded}, "
                            f"not {ledger._starting_cash}"
                        )
                elif record.kind == "fill":
                    fill = Fill.model_validate(record.payload["fill"])
                    cumulative = record.payload["cumulative_quantity"]
                    if not isinstance(cumulative, int) or not ledger._check_fill(fill, cumulative):
                        raise LedgerError("fill was already applied")
                    ledger._apply_fill(fill, cumulative)
                elif record.kind == "adjustment":
                    adjustment = LedgerAdjustment.model_validate(record.payload)
                    if not ledger._check_adjustment(adjustment):
                        raise LedgerError("adjustment was already applied")
                    ledger._apply_adjustment(adjustment)
            except (LedgerError, KeyError, ValidationError) as exc:
                raise LedgerError(f"journal record {record.seq} does not replay: {exc}") from exc
        ledger._journal = journal
        return ledger

    # ------------------------------------------------------------------ changes

    def apply_fill(self, fill: Fill, cumulative_quantity: int) -> bool:
        """Record one fill. `cumulative_quantity` is the order's total filled quantity
        including this fill.

        Returns False, changing nothing, if the ledger already holds this fill (the same
        fill seen twice through a poll overlap or a restart). Raises LedgerError, also
        changing nothing, if fills were skipped, the sell exceeds the position, or the buy
        would add to a position opened by a different order.
        """
        if not self._check_fill(fill, cumulative_quantity):
            return False
        if self._journal is not None:
            self._journal.append(
                "fill",
                {"fill": fill.model_dump(mode="json"), "cumulative_quantity": cumulative_quantity},
            )
        self._apply_fill(fill, cumulative_quantity)
        return True

    def apply_adjustment(self, adjustment: LedgerAdjustment) -> bool:
        """Record an operator correction.

        Returns False, changing nothing, if an identical adjustment with this
        `adjustment_id` was already applied. Raises LedgerError, also changing nothing,
        if it does not fit or the id was used for a different adjustment.
        """
        if not self._check_adjustment(adjustment):
            return False
        if self._journal is not None:
            self._journal.append("adjustment", adjustment.model_dump(mode="json"))
        self._apply_adjustment(adjustment)
        return True

    # ------------------------------------------------------------------ views

    @property
    def realized(self) -> float:
        return self._realized

    def account(self) -> AccountState:
        return AccountState(cash=self._cash, positions=dict(self._positions))

    def pnl(
        self, marks: Mapping[str, float], as_of: dt.date, prior_equity: float | None = None
    ) -> PnL:
        """Value the ledger at `marks` (a close per held symbol)."""
        missing = sorted(s for s in self._positions if s not in marks)
        if missing:
            raise LedgerError(f"no mark for held symbols: {', '.join(missing)}")
        market_value = sum(p.quantity * marks[s] for s, p in self._positions.items())
        equity = self._cash + market_value
        return PnL(
            as_of=as_of,
            cash=self._cash,
            market_value=market_value,
            equity=equity,
            realized=self._realized,
            unrealized=market_value - sum(self._cost.values()),
            day_change=equity - prior_equity if prior_equity is not None else None,
        )

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(
            journal_seq=self._journal.last_seq if self._journal is not None else 0,
            starting_cash=self._starting_cash,
            cash=self._cash,
            realized=self._realized,
            positions=dict(self._positions),
            cost_basis=dict(self._cost),
        )

    def write_snapshot(self, path: Path) -> None:
        """Write the snapshot for readers. Atomic: a reader sees the old file or the new
        one, never part of one."""
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.snapshot().model_dump(mode="json"), indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ internals

    def _check_fill(self, fill: Fill, cumulative_quantity: int) -> bool:
        """True if the fill is new and fits. False if already applied. Raises if invalid."""
        if not isinstance(fill.instrument, Stock):
            raise LedgerError("the ledger holds stock positions only")
        symbol = fill.instrument.key
        applied = self._filled.get(fill.order_id, 0)
        if cumulative_quantity <= applied:
            return False
        if cumulative_quantity - fill.quantity != applied:
            raise LedgerError(
                f"order {fill.order_id}: fill of {fill.quantity} to a total of "
                f"{cumulative_quantity} does not follow the {applied} already applied"
            )
        held = self._positions.get(symbol)
        if fill.side == "sell":
            if held is None or fill.quantity > held.quantity:
                raise LedgerError(
                    f"sell of {fill.quantity} {symbol} exceeds the "
                    f"{held.quantity if held else 0} held"
                )
        elif held is not None and held.entry_order_id != fill.order_id:
            raise LedgerError(
                f"buy of {symbol} by order {fill.order_id} would add to the position "
                f"opened by {held.entry_order_id}"
            )
        return True

    def _apply_fill(self, fill: Fill, cumulative_quantity: int) -> None:
        symbol = fill.instrument.key
        self._filled[fill.order_id] = cumulative_quantity
        self._cash += fill.cash_delta
        held = self._positions.get(symbol)
        if fill.side == "buy":
            self._cost[symbol] = self._cost.get(symbol, 0.0) + fill.notional + fill.commission
            if held is None:
                self._positions[symbol] = Position(
                    instrument=fill.instrument,
                    quantity=fill.quantity,
                    avg_price=fill.price,
                    entry_date=fill.fill_date,
                    entry_order_id=fill.order_id,
                )
            else:
                quantity = held.quantity + fill.quantity
                value = held.avg_price * held.quantity + fill.price * fill.quantity
                self._positions[symbol] = held.model_copy(
                    update={"quantity": quantity, "avg_price": value / quantity}
                )
            return
        assert held is not None  # _check_fill
        cost_sold = self._release_cost(symbol, fill.quantity, held.quantity)
        self._realized += fill.notional - fill.commission - cost_sold
        self._set_quantity(symbol, held, held.quantity - fill.quantity)

    def _check_adjustment(self, adjustment: LedgerAdjustment) -> bool:
        """True if the adjustment is new and fits. False if already applied. Raises if
        invalid."""
        earlier = self._adjustments.get(adjustment.adjustment_id)
        if earlier is not None:
            if earlier != adjustment:
                raise LedgerError(
                    f"adjustment id {adjustment.adjustment_id} was already used for a "
                    f"different adjustment"
                )
            return False
        if not adjustment.quantity_delta:
            return True
        symbol = adjustment.symbol
        assert symbol is not None  # LedgerAdjustment
        held = self._positions.get(symbol)
        quantity = (held.quantity if held else 0) + adjustment.quantity_delta
        if quantity < 0:
            raise LedgerError(
                f"adjustment removes {-adjustment.quantity_delta} {symbol} but only "
                f"{held.quantity if held else 0} are held"
            )
        if adjustment.kind == "split":
            if held is None or quantity == 0:
                raise LedgerError(
                    f"a split of {symbol} needs a position before and after it; "
                    f"use a delisting or manual adjustment for a full cash-out"
                )
        elif held is None and not adjustment.cost_basis_delta > 0:
            raise LedgerError(f"a new {symbol} position needs a positive cost_basis_delta")
        return True

    def _apply_adjustment(self, adjustment: LedgerAdjustment) -> None:
        self._adjustments[adjustment.adjustment_id] = adjustment
        self._cash += adjustment.cash_delta
        cost_change = 0.0
        symbol = adjustment.symbol
        delta = adjustment.quantity_delta
        if delta and symbol is not None:
            held = self._positions.get(symbol)
            if adjustment.kind == "split":
                assert held is not None  # _check_adjustment
                quantity = held.quantity + delta
                self._positions[symbol] = held.model_copy(
                    update={
                        "quantity": quantity,
                        "avg_price": held.avg_price * held.quantity / quantity,
                    }
                )
            elif delta > 0:
                cost_change = adjustment.cost_basis_delta
                self._cost[symbol] = self._cost.get(symbol, 0.0) + cost_change
                quantity = (held.quantity if held else 0) + delta
                basis = held.avg_price * held.quantity if held else 0.0
                self._positions[symbol] = Position(
                    instrument=Stock(symbol=symbol),
                    quantity=quantity,
                    avg_price=(basis + cost_change) / quantity,
                    entry_date=held.entry_date if held else adjustment.as_of,
                    entry_order_id=held.entry_order_id if held else f"adjustment-{symbol}",
                )
            else:
                assert held is not None  # _check_adjustment
                cost_change = -self._release_cost(symbol, -delta, held.quantity)
                self._set_quantity(symbol, held, held.quantity + delta)
        self._realized += adjustment.cash_delta + cost_change

    def _release_cost(self, symbol: str, quantity: int, held_quantity: int) -> float:
        """Take `quantity` shares' part of the position's cost out of the book."""
        if quantity == held_quantity:
            return self._cost.pop(symbol)
        released = self._cost[symbol] * quantity / held_quantity
        self._cost[symbol] -= released
        return released

    def _set_quantity(self, symbol: str, held: Position, quantity: int) -> None:
        if quantity == 0:
            del self._positions[symbol]
            self._cost.pop(symbol, None)
        else:
            self._positions[symbol] = held.model_copy(update={"quantity": quantity})


# --------------------------------------------------------------------------- reconciliation

Severity = Literal["ok", "warn", "stop"]


class PositionDiff(FrozenModel):
    symbol: str
    ledger_quantity: int
    venue_quantity: int


class ReconcileReport(FrozenModel):
    at: dt.datetime
    position_diffs: tuple[PositionDiff, ...]
    cash_diff: float  # venue cash - cash offset - ledger cash
    unexpected_orders: tuple[str, ...]  # open at the venue, not expected by the harness
    missing_orders: tuple[str, ...]  # expected to be open, not open at the venue
    severity: Severity
    reasons: tuple[str, ...]  # one line per finding, for alerts


def reconcile(
    ledger: AccountState,
    expected_open: Sequence[str],
    account: VenueAccount,
    positions: Sequence[VenuePosition],
    open_orders: Sequence[VenueOrderState],
    cash_offset: float,
    cash_tolerance: float,
    at: dt.datetime,
) -> ReconcileReport:
    """Compare the ledger with what the venue reports. Pure: it changes nothing.

    `expected_open` are the client order ids the harness believes are working.
    `cash_offset` is venue cash that is not part of the profile's capital.

    `stop` on any position difference, any short, any unexpected or missing order, or a
    cash difference beyond `cash_tolerance`. `warn` for a cash difference within it.
    """
    reasons: list[str] = []
    venue_quantity: dict[str, int] = {}
    for position in positions:
        venue_quantity[position.symbol] = venue_quantity.get(position.symbol, 0) + position.quantity
    diffs = []
    for symbol in sorted(set(ledger.positions) | set(venue_quantity)):
        held = ledger.positions.get(symbol)
        ours, theirs = (held.quantity if held else 0), venue_quantity.get(symbol, 0)
        if theirs < 0:
            reasons.append(f"{symbol}: the venue reports a short of {-theirs}")
        if ours != theirs:
            diffs.append(PositionDiff(symbol=symbol, ledger_quantity=ours, venue_quantity=theirs))
            reasons.append(f"{symbol}: ledger holds {ours}, venue holds {theirs}")

    expected = set(expected_open)
    at_venue = {o.client_order_id for o in open_orders}
    unexpected = tuple(sorted(at_venue - expected))
    missing = tuple(sorted(expected - at_venue))
    reasons += [f"order {i} is open at the venue but unknown to the harness" for i in unexpected]
    reasons += [f"order {i} should be open but the venue does not list it" for i in missing]

    severity: Severity = "stop" if reasons else "ok"
    cash_diff = account.cash - cash_offset - ledger.cash
    if abs(cash_diff) > cash_tolerance:
        severity = "stop"
        reasons.append(f"cash differs by {cash_diff:+.2f} (tolerance {cash_tolerance:.2f})")
    elif abs(cash_diff) >= _CENT:
        if severity == "ok":
            severity = "warn"
        reasons.append(f"cash differs by {cash_diff:+.2f}, within tolerance")
    return ReconcileReport(
        at=at,
        position_diffs=tuple(diffs),
        cash_diff=cash_diff,
        unexpected_orders=unexpected,
        missing_orders=missing,
        severity=severity,
        reasons=tuple(reasons),
    )


def reconcile_venue(
    ledger: AccountState,
    expected_open: Sequence[str],
    venue: Venue,
    cash_offset: float,
    cash_tolerance: float,
) -> ReconcileReport:
    """Fetch the venue's state and reconcile against it.

    Venue data that fails validation (a zero average price, a number that is not finite)
    cannot be trusted, so it is reported as `stop`, not raised as a traceback.
    `VenueUnavailable` is not caught: the caller decides whether to retry.
    """
    at = venue.now()
    try:
        account = venue.account()
        positions = venue.positions()
        open_orders = venue.open_orders()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return ReconcileReport(
            at=at,
            position_diffs=(),
            cash_diff=0.0,
            unexpected_orders=(),
            missing_orders=(),
            severity="stop",
            reasons=(f"venue data failed validation, stopping: {exc.title}: {problems}",),
        )
    return reconcile(
        ledger, expected_open, account, positions, open_orders, cash_offset, cash_tolerance, at
    )
