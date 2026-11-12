"""Test strategies. They live in tests/, never in strategies/."""

from __future__ import annotations

from collections.abc import Sequence

from harness.models import FrozenModel, PortfolioSnapshot, Signal
from harness.view import MarketView
from strategies.base import Strategy


class BuyAndHold(Strategy):
    """Buy one symbol on the first bar it is called on, then hold forever.

    "First bar" is derived from the portfolio (nothing held), not from `view.n_bars` or
    state kept on self: after a warm-up the first call already sees many bars.
    """

    name = "test_buy_and_hold"

    class Params(Strategy.Params):
        symbol: str = "SPY"

    params: Params

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        symbol = self.params.symbol
        if portfolio.positions:
            return []
        return [Signal.enter(symbol, strength=1.0, reason="first bar")]


class Churn(Strategy):
    """Deterministic, data-independent churn across all symbols: lots of entries, exits,
    competing signals, and ignored exits. Used by the fill-timing and determinism tests."""

    name = "test_churn"

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        t = view.n_bars
        signals: list[Signal] = []
        for k, symbol in enumerate(view.symbols):
            score = (t * 7 + k * 13) % 11
            if portfolio.holds(symbol):
                if (t + k) % 4 == 0:
                    signals.append(Signal.exit(symbol, reason="time"))
            elif score < 4 and view.bar(symbol) is not None:
                signals.append(Signal.enter(symbol, strength=float(score), reason=f"s{score}"))
            if (t + k) % 9 == 0 and not portfolio.holds(symbol):
                signals.append(Signal.exit(symbol, reason="stray exit"))
        return signals


class Recorder(Strategy):
    """Records what it is shown each session; emits nothing."""

    name = "test_recorder"

    def __init__(self, params: FrozenModel | None = None) -> None:
        super().__init__(params)
        self.calls: list[tuple] = []

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        hist = view.history("close", lookback=10**6)
        self.calls.append((view.today, view.n_bars, hist.index[-1].date(), len(hist)))
        return []
