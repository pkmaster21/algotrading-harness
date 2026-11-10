import datetime as dt

import pytest
from pydantic import ValidationError

from harness.models import (
    AccountState,
    Bar,
    Fill,
    Leg,
    OptionContract,
    Order,
    OrderUpdate,
    PortfolioSnapshot,
    Position,
    Signal,
    Stock,
)

D1 = dt.date(2024, 1, 2)
D2 = dt.date(2024, 1, 3)
AAPL = Stock(symbol="AAPL")


def _fill(**overrides: object) -> Fill:
    base: dict[str, object] = dict(
        order_id="sim-000001",
        signal_date=D1,
        fill_date=D2,
        price_date=D2,
        instrument=AAPL,
        side="buy",
        quantity=10,
        reference_price=100.0,
        price=100.05,
        slippage_cost=0.5,
        commission=0.0,
    )
    base.update(overrides)
    return Fill(**base)  # type: ignore[arg-type]


class TestBar:
    def test_valid(self) -> None:
        Bar(symbol="SPY", date=D1, open=10, high=11, low=9, close=10.5, volume=100)

    @pytest.mark.parametrize(
        "o,h,lo,c",
        [(12, 11, 9, 10), (10, 11, 9, 8), (10, 9, 11, 10), (0, 11, 9, 10)],
    )
    def test_rejects_inconsistent_ohlc(self, o: float, h: float, lo: float, c: float) -> None:
        with pytest.raises(ValidationError):
            Bar(symbol="SPY", date=D1, open=o, high=h, low=lo, close=c, volume=1)

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValidationError):
            Bar(symbol="SPY", date=D1, open=float("nan"), high=11, low=9, close=10, volume=1)

    def test_rejects_bad_symbol(self) -> None:
        with pytest.raises(ValidationError):
            Bar(symbol="aapl", date=D1, open=10, high=11, low=9, close=10, volume=1)

    def test_frozen(self) -> None:
        bar = Bar(symbol="SPY", date=D1, open=10, high=11, low=9, close=10, volume=1)
        with pytest.raises(ValidationError):
            bar.close = 12  # type: ignore[misc]


class TestSignal:
    def test_enter_helper_is_one_leg_buy(self) -> None:
        s = Signal.enter("AAPL", strength=2.0, reason="breakout")
        assert s.action == "enter"
        assert s.symbol == "AAPL"
        assert s.legs == (Leg(instrument=AAPL, side="buy"),)

    def test_exit_helper_is_one_leg_sell(self) -> None:
        s = Signal.exit("AAPL")
        assert s.action == "exit"
        assert s.legs[0].side == "sell"
        assert s.strength is None

    def test_entry_requires_finite_strength(self) -> None:
        with pytest.raises(ValidationError):
            Signal(action="enter", legs=(Leg(instrument=AAPL, side="buy"),))
        with pytest.raises(ValidationError):
            Signal.enter("AAPL", strength=float("nan"))

    def test_exit_rejects_strength(self) -> None:
        with pytest.raises(ValidationError):
            Signal(action="exit", legs=(Leg(instrument=AAPL, side="sell"),), strength=1.0)

    def test_requires_a_leg(self) -> None:
        with pytest.raises(ValidationError):
            Signal(action="exit", legs=())

    def test_symbol_undefined_for_multi_leg(self) -> None:
        call = OptionContract(underlying="AAPL", expiry=D2, strike=150, right="call")
        s = Signal(
            action="enter",
            legs=(Leg(instrument=call, side="buy"), Leg(instrument=AAPL, side="sell")),
            strength=1.0,
        )
        with pytest.raises(ValueError):
            _ = s.symbol


class TestInstruments:
    def test_option_key_is_occ_style(self) -> None:
        put = OptionContract(
            underlying="SPY", expiry=dt.date(2026, 6, 19), strike=512.5, right="put"
        )
        assert put.key == "SPY260619P00512500"

    def test_leg_discriminates_instrument_from_dict(self) -> None:
        leg = Leg.model_validate({"instrument": {"kind": "stock", "symbol": "MSFT"}, "side": "buy"})
        assert isinstance(leg.instrument, Stock)


class TestOrder:
    def test_quantity_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Order(
                created_on=D1, action="exit", legs=(Leg(instrument=AAPL, side="sell"),), quantity=0
            )


class TestFill:
    def test_next_open_fill_valid(self) -> None:
        f = _fill()
        assert f.cash_delta == pytest.approx(-1000.5)

    def test_sell_cash_delta_subtracts_commission(self) -> None:
        f = _fill(side="sell", price=99.95, commission=1.0)
        assert f.cash_delta == pytest.approx(999.5 - 1.0)

    def test_rejects_fill_on_signal_bar(self) -> None:
        with pytest.raises(ValidationError, match="look-ahead"):
            _fill(fill_date=D1, price_date=D1)

    def test_rejects_next_open_using_other_bar_price(self) -> None:
        with pytest.raises(ValidationError):
            _fill(price_date=D1)

    def test_forced_close_uses_stale_close(self) -> None:
        later = dt.date(2024, 1, 10)
        f = _fill(
            kind="forced_close", side="sell", signal_date=later, fill_date=later, price_date=D1
        )
        assert f.cash_delta > 0

    def test_forced_close_cannot_use_trigger_or_future_price(self) -> None:
        later = dt.date(2024, 1, 10)
        with pytest.raises(ValidationError):
            _fill(
                kind="forced_close",
                side="sell",
                signal_date=later,
                fill_date=later,
                price_date=later,
            )

    def test_forced_close_must_be_sell(self) -> None:
        later = dt.date(2024, 1, 10)
        with pytest.raises(ValidationError):
            _fill(kind="forced_close", signal_date=later, fill_date=later, price_date=D1)

    def test_option_notional_uses_multiplier(self) -> None:
        call = OptionContract(
            underlying="AAPL", expiry=dt.date(2024, 2, 16), strike=150, right="call"
        )
        f = _fill(instrument=call, quantity=2, price=3.0, reference_price=3.0, slippage_cost=0.0)
        assert f.notional == pytest.approx(600.0)


class TestOrderUpdate:
    def test_filled_needs_fills(self) -> None:
        with pytest.raises(ValidationError):
            OrderUpdate(order_id="sim-000001", status="filled")

    def test_canceled_has_no_fills(self) -> None:
        with pytest.raises(ValidationError):
            OrderUpdate(order_id="sim-000001", status="canceled", fills=(_fill(),), reason="no_bar")

    def test_fill_ids_match(self) -> None:
        with pytest.raises(ValidationError):
            OrderUpdate(order_id="sim-000002", status="filled", fills=(_fill(),))


class TestPositions:
    def _pos(self) -> Position:
        return Position(
            instrument=AAPL, quantity=5, avg_price=100.0, entry_date=D2, entry_order_id="sim-1"
        )

    def test_long_only(self) -> None:
        with pytest.raises(ValidationError):
            Position(
                instrument=AAPL, quantity=-5, avg_price=100.0, entry_date=D2, entry_order_id="x"
            )

    def test_keys_must_match_instrument(self) -> None:
        with pytest.raises(ValidationError):
            AccountState(cash=0, positions={"MSFT": self._pos()})

    def test_snapshot_holds(self) -> None:
        snap = PortfolioSnapshot(as_of=D2, cash=500, equity=1000, positions={"AAPL": self._pos()})
        assert snap.holds("AAPL")
        assert not snap.holds("MSFT")
