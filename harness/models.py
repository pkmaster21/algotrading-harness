"""Shared data shapes for the harness.

Every object that crosses a module boundary is defined here as a frozen pydantic model.
Bulk bar data crosses as a validated array panel (see `harness.data.BarPanel`) for speed;
`Bar` is the single-bar shape used at the strategy boundary.

Orders are lists of legs: a stock trade is a one-leg order, so multi-leg options orders
can be added later without changing these shapes.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Side = Literal["buy", "sell"]
Action = Literal["enter", "exit"]
OptionRight = Literal["call", "put"]
FillKind = Literal["next_open", "forced_close"]
OrderStatus = Literal["filled", "canceled"]
OrderUpdateReason = Literal[
    "",
    "reduced_for_cash",
    "no_bar",
    "not_held",
    "insufficient_cash",
    "end_of_data",
    "no_data",
]
EventKind = Literal[
    "exit_ignored_not_held",
    "entry_ignored_held",
    "signal_duplicate",
    "entry_skipped_no_slot",
    "entry_skipped_no_cash",
    "entry_skipped_no_price",
    "order_canceled",
    "order_reduced",
    "order_unfilled_end",
    "position_closed_no_data",
]

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")
_PRICE_TOLERANCE = 1e-6  # relative slack for OHLC containment (vendor rounding)


class FrozenModel(BaseModel):
    """Base for harness models: immutable, no extra fields, no NaN or infinity."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _check_symbol(symbol: str) -> str:
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(
            f"invalid symbol {symbol!r}: expected uppercase ticker like 'AAPL' or 'BRK.B'"
        )
    return symbol


# --------------------------------------------------------------------------- bars


class Bar(FrozenModel):
    """One daily bar, adjusted for splits and dividends.

    Prices are positive, `low <= high`, and open and close lie within [low, high]
    (with a tiny relative tolerance for vendor rounding).
    """

    symbol: str
    date: dt.date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_symbol(self.symbol)
        slack = self.high * _PRICE_TOLERANCE
        if self.low > self.high + slack:
            raise ValueError(f"{self.symbol} {self.date}: low {self.low} > high {self.high}")
        for name in ("open", "close"):
            price = getattr(self, name)
            if not (self.low - slack <= price <= self.high + slack):
                raise ValueError(
                    f"{self.symbol} {self.date}: {name} {price} outside [{self.low}, {self.high}]"
                )
        return self


# --------------------------------------------------------------------------- instruments


class Stock(FrozenModel):
    """A US stock or ETF."""

    kind: Literal["stock"] = "stock"
    symbol: str

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_symbol(self.symbol)
        return self

    @property
    def key(self) -> str:
        """Unique instrument key, used to index positions."""
        return self.symbol

    @property
    def multiplier(self) -> int:
        return 1


class OptionContract(FrozenModel):
    """Stub for the options phase.

    Defined now so `Leg` and `Position` already accept options. The slice 1 engine
    rejects any Signal that contains one.
    """

    kind: Literal["option"] = "option"
    underlying: str
    expiry: dt.date
    strike: float = Field(gt=0)
    right: OptionRight
    multiplier: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_symbol(self.underlying)
        return self

    @property
    def key(self) -> str:
        """OCC-style symbol without padding, e.g. 'AAPL260417C00150000'."""
        strike = round(self.strike * 1000)
        return f"{self.underlying}{self.expiry:%y%m%d}{self.right[0].upper()}{strike:08d}"


Instrument = Annotated[Stock | OptionContract, Field(discriminator="kind")]


# --------------------------------------------------------------------------- signals and orders


class Leg(FrozenModel):
    """One instrument and side within a signal or order.

    `ratio` is the number of units of this leg per unit of the order. It is always 1
    for stock; multi-leg options orders use it for spreads like 1x2.
    """

    instrument: Instrument
    side: Side
    ratio: int = Field(default=1, ge=1)


def _check_strength(action: Action, strength: float | None) -> None:
    if action == "enter":
        if strength is None:
            raise ValueError("entry signals require a strength (higher ranks first)")
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
    elif strength is not None:
        raise ValueError("exit signals do not take a strength")


class Signal(FrozenModel):
    """A strategy's request to open or close exposure.

    A Signal carries no size, price, or date. The harness stamps it with the bar it was
    generated on, sizes it, and decides whether it becomes an order. Use `Signal.enter`
    and `Signal.exit` for one-leg stock signals.

    Entries require a finite `strength`; competing entries are ranked by it, highest first.
    Exits take no strength. An exit for a symbol not currently held is ignored and logged.
    """

    action: Action
    legs: tuple[Leg, ...] = Field(min_length=1)
    strength: float | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_strength(self.action, self.strength)
        return self

    @classmethod
    def enter(cls, symbol: str, strength: float, reason: str = "") -> Signal:
        """Buy `symbol` (sized by the harness)."""
        return cls(
            action="enter",
            legs=(Leg(instrument=Stock(symbol=symbol), side="buy"),),
            strength=strength,
            reason=reason,
        )

    @classmethod
    def exit(cls, symbol: str, reason: str = "") -> Signal:
        """Sell the full position in `symbol`."""
        return cls(
            action="exit",
            legs=(Leg(instrument=Stock(symbol=symbol), side="sell"),),
            reason=reason,
        )

    @property
    def symbol(self) -> str:
        """Instrument key of a one-leg signal. Raises for multi-leg signals."""
        if len(self.legs) != 1:
            raise ValueError("symbol is only defined for one-leg signals")
        return self.legs[0].instrument.key


class Order(FrozenModel):
    """A sized order built by construction from a Signal.

    `quantity` is in order units: shares for a one-leg stock order. Each leg trades
    `quantity * leg.ratio` units. `created_on` is the date of the bar whose close
    produced the signal; the order may only fill in a later session.
    """

    created_on: dt.date
    action: Action
    legs: tuple[Leg, ...] = Field(min_length=1)
    quantity: int = Field(gt=0)
    strength: float | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_strength(self.action, self.strength)
        return self

    @property
    def symbol(self) -> str:
        """Instrument key of a one-leg order. Raises for multi-leg orders."""
        if len(self.legs) != 1:
            raise ValueError("symbol is only defined for one-leg orders")
        return self.legs[0].instrument.key


# --------------------------------------------------------------------------- fills and positions


class Fill(FrozenModel):
    """One executed leg.

    `price_date` is the date of the bar whose price the fill used. The model itself
    enforces the timing rules, so no code path can construct a look-ahead fill:

    - `next_open`: price_date == fill_date and fill_date > signal_date
      (the open of a session after the signal bar).
    - `forced_close`: a sell with fill_date == signal_date and price_date < signal_date
      (the last close of a symbol that stopped trading; stale, never future).
    """

    order_id: str
    kind: FillKind = "next_open"
    signal_date: dt.date
    fill_date: dt.date
    price_date: dt.date
    instrument: Instrument
    side: Side
    quantity: int = Field(gt=0)
    reference_price: float = Field(gt=0)
    price: float = Field(gt=0)
    slippage_cost: float = Field(ge=0)
    commission: float = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.kind == "next_open":
            if self.price_date != self.fill_date:
                raise ValueError("next_open fill must use the fill session's own open")
            if self.fill_date <= self.signal_date:
                raise ValueError(
                    f"look-ahead: fill on {self.fill_date} for signal on {self.signal_date}"
                )
        else:
            if self.side != "sell":
                raise ValueError("forced_close fills must be sells")
            if self.fill_date != self.signal_date:
                raise ValueError("forced_close fills happen on the session that triggers them")
            if self.price_date >= self.signal_date:
                raise ValueError("forced_close must use a close from before the trigger session")
        return self

    @property
    def notional(self) -> float:
        """price * quantity * contract multiplier."""
        return self.price * self.quantity * self.instrument.multiplier

    @property
    def cash_delta(self) -> float:
        """Change in cash from this fill, including commission."""
        sign = -1.0 if self.side == "buy" else 1.0
        return sign * self.notional - self.commission


class OrderUpdate(FrozenModel):
    """Broker report on an order: filled (possibly reduced) or canceled, with a reason."""

    order_id: str
    status: OrderStatus
    fills: tuple[Fill, ...] = ()
    reason: OrderUpdateReason = ""

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.status == "filled" and not self.fills:
            raise ValueError("a filled update needs at least one fill")
        if self.status == "canceled" and self.fills:
            raise ValueError("a canceled update has no fills")
        if any(f.order_id != self.order_id for f in self.fills):
            raise ValueError("fill order_id does not match update")
        return self


class Position(FrozenModel):
    """An open long position. Slice 1 is long-only, so quantity is always positive."""

    instrument: Instrument
    quantity: int = Field(gt=0)
    avg_price: float = Field(gt=0)  # fill price including slippage
    entry_date: dt.date
    entry_order_id: str


def _check_position_keys(positions: dict[str, Position]) -> None:
    for key, pos in positions.items():
        if key != pos.instrument.key:
            raise ValueError(
                f"position key {key!r} does not match instrument {pos.instrument.key!r}"
            )


class AccountState(FrozenModel):
    """Cash and positions as the broker sees them."""

    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_position_keys(self.positions)
        return self


class PortfolioSnapshot(FrozenModel):
    """What a strategy sees about its holdings at the close of `as_of`.

    The engine builds a fresh snapshot every session, so mutating it has no effect.
    `equity` is cash plus positions marked at the latest available close.
    """

    as_of: dt.date
    cash: float
    equity: float
    positions: dict[str, Position] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        _check_position_keys(self.positions)
        return self

    def holds(self, symbol: str) -> bool:
        return symbol in self.positions


class EngineEvent(FrozenModel):
    """A harness decision that did not produce a normal fill: ignored, skipped, canceled."""

    date: dt.date
    kind: EventKind
    symbol: str
    detail: str = ""
