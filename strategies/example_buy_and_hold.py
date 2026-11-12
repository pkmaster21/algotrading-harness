"""Example strategy: buy one symbol on the first bar, then hold. Also a template.

To write a new strategy, copy this file to `strategies/<name>.py`, add a preset in
`config/strategies/<name>.yaml`, and run `uv run python -m harness.run --strategy <name>`.

Rules every strategy follows:

- Return Signals only (`Signal.enter`, `Signal.exit`). The harness sizes, fills, and
  tracks positions; orders fill at the next session's open.
- Declare every tunable value in `Params`, with a type, range, and default.
- Compute indicators from `view.history` / `view.series` each session, not from state
  accumulated across `on_bar` calls: during a holdout warm-up the engine fills the view
  without calling `on_bar`, so accumulated state would be missing those sessions.
- Import only numpy, pandas, math and similar, plus `harness.models`, `harness.view`,
  and `strategies.base`. The run entry point rejects anything else.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from harness.models import PortfolioSnapshot, Signal
from harness.view import MarketView
from strategies.base import Strategy


class ExampleBuyAndHold(Strategy):
    """Buy `symbol` once enough history is visible, then hold it forever."""

    class Params(Strategy.Params):
        symbol: str = "SPY"
        # Sessions with a bar required before buying; 1 means buy on the first bar.
        min_history: int = Field(default=1, ge=1, le=1000)

    params: Params

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        symbol = self.params.symbol
        if portfolio.holds(symbol):
            return []
        # Indicator from history, not from counting on_bar calls: works after warm-up.
        closes = view.series(symbol, "close", lookback=self.params.min_history)
        if closes.notna().sum() < self.params.min_history:
            return []
        return [Signal.enter(symbol, strength=1.0, reason="buy and hold")]
