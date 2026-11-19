"""Point-in-time membership enforced by the engine (docs/pit-universe-design.md, PR 2):
the member mask, entries for members only, forced exits on removal, and the masked view."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from harness.broker.sim import SimBroker
from harness.config import MembershipInterval
from harness.data import BarPanel
from harness.engine import BacktestResult, simulate
from harness.models import ForcedReason, PortfolioSnapshot, Signal
from harness.report import round_trips
from harness.view import MarketView, MemberMask
from strategies.base import Strategy
from tests.synthetic import COSTS, construction, make_panel, price_lookup

SYMBOLS = ["SPY", "AAA", "BBB"]
N = 30
SLIP = COSTS.slippage_bps / 10_000

Spans = Mapping[str, Sequence[tuple[int, int | None]]]  # symbol -> (start, end) session indexes


def _intervals(dates: Sequence[dt.date], spans: Spans) -> list[MembershipInterval]:
    return [
        MembershipInterval(symbol=s, start=dates[a], end=None if b is None else dates[b])
        for s, pairs in spans.items()
        for a, b in pairs
    ]


def _mask(panel: BarPanel, spans: Spans, symbols: Sequence[str] = SYMBOLS) -> MemberMask:
    return MemberMask(symbols, panel.dates, _intervals(panel.dates, spans))


class Script(Strategy):
    """Emits fixed signals by session index (view.n_bars - 1) and records each view."""

    def __init__(self, plan: Mapping[int, Sequence[Signal]] | None = None) -> None:
        super().__init__()
        self.plan = dict(plan or {})
        self.views: list[MarketView] = []

    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        self.views.append(view)
        return list(self.plan.get(view.n_bars - 1, []))


def _run(
    strategy: Strategy,
    spans: Spans,
    *,
    broker: SimBroker | None = None,
    **panel_kw: object,
) -> tuple[BarPanel, BacktestResult]:
    panel = make_panel(SYMBOLS, N, **panel_kw)  # type: ignore[arg-type]
    result = simulate(
        strategy, panel, broker or SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS,
        0, None, _mask(panel, spans),
    )  # fmt: skip
    return panel, result


ALWAYS: Spans = {"SPY": [(0, None)], "BBB": [(0, None)]}
BUY_AAA = {1: [Signal.enter("AAA", strength=1.0)]}  # fills at the open of session 2


def _kinds(result: BacktestResult, symbol: str) -> list[str]:
    return [e.kind for e in result.events if e.symbol == symbol]


# --------------------------------------------------------------------------- the mask


def test_mask_boundaries_start_inclusive_end_exclusive() -> None:
    panel = make_panel(SYMBOLS, 10)
    mask = _mask(panel, {"AAA": [(2, 5)], "BBB": [(0, 3), (6, None)]})
    d = panel.dates
    aaa = [bool(mask.members(x)[1]) for x in d]
    assert aaa == [False, False, True, True, True, False, False, False, False, False]
    assert [bool(mask.leaving(x)[1]) for x in d].index(True) == 4  # last member session
    assert sum(bool(mask.leaving(x)[1]) for x in d) == 1
    assert [bool(mask.hidden(x)[1]) for x in d] == [i >= 5 for i in range(10)]
    # BBB leaves and comes back: leaving on its first last-member session, never hidden.
    assert [bool(mask.members(x)[2]) for x in d] == [i < 3 or i >= 6 for i in range(10)]
    assert [bool(mask.leaving(x)[2]) for x in d] == [i == 2 for i in range(10)]
    assert not any(bool(mask.hidden(x)[2]) for x in d)
    # SPY has no interval: never a member, never hidden.
    assert not any(bool(mask.members(x)[0]) or bool(mask.hidden(x)[0]) for x in d)


def test_mask_last_session_is_never_leaving_and_unknown_sessions_raise() -> None:
    panel = make_panel(SYMBOLS, 10)
    mask = _mask(panel, {"AAA": [(0, None)]})
    assert not mask.leaving(panel.dates[-1]).any()
    with pytest.raises(KeyError, match="not in the member mask"):
        mask.members(dt.date(1999, 1, 4))


def test_engine_rejects_a_mask_for_other_symbols() -> None:
    panel = make_panel(SYMBOLS, 5)
    mask = _mask(panel, ALWAYS, symbols=["SPY", "BBB", "AAA"])
    with pytest.raises(ValueError, match="universe, in order"):
        simulate(Script(), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS,
                 0, None, mask)  # fmt: skip


# --------------------------------------------------------------------------- entries


def test_members_lists_what_may_be_entered_today() -> None:
    strategy = Script()
    _run(strategy, {**ALWAYS, "AAA": [(5, 10)]})
    members = [v.members() for v in strategy.views]
    assert members[4] == ("SPY", "BBB")
    assert members[5] == members[8] == ("SPY", "AAA", "BBB")
    assert members[9] == ("SPY", "BBB")  # last member session: an entry would fill after it
    assert members[10] == ("SPY", "BBB")


def test_entry_for_a_non_member_is_dropped_and_logged() -> None:
    _, result = _run(Script({3: [Signal.enter("AAA", strength=1.0, reason="r")]}),
                     {**ALWAYS, "AAA": [(5, 10)]})  # fmt: skip
    assert result.fills == ()
    (event,) = [e for e in result.events if e.kind == "entry_skipped_not_member"]
    assert (event.symbol, event.date, event.detail) == ("AAA", result.equity[3].date, "r")


def test_entry_on_the_last_member_session_is_dropped() -> None:
    # It would fill on the first session out and then never be force-exited.
    _, result = _run(Script({9: [Signal.enter("AAA", strength=1.0)]}),
                     {**ALWAYS, "AAA": [(5, 10)]})  # fmt: skip
    assert result.fills == ()
    assert _kinds(result, "AAA") == ["entry_skipped_not_member"]


def test_fixed_universe_is_unchanged_without_a_mask() -> None:
    panel = make_panel(SYMBOLS, N)
    strategy = Script(BUY_AAA)
    result = simulate(strategy, panel, SimBroker(10_000, COSTS), SYMBOLS, construction(), COSTS)
    assert strategy.views[0].members() == tuple(SYMBOLS)
    assert [f.side for f in result.fills] == ["buy"]


# --------------------------------------------------------------------------- forced exits


def _removal_fill(result: BacktestResult):  # type: ignore[no-untyped-def]
    (fill,) = [f for f in result.fills if f.side == "sell"]
    return fill


def test_removal_exit_fills_at_the_first_session_outs_real_open() -> None:
    panel, result = _run(Script(BUY_AAA), {**ALWAYS, "AAA": [(0, 10)]})
    price = price_lookup(panel)
    fill = _removal_fill(result)
    first_out = panel.dates[10]
    assert (fill.kind, fill.forced_reason) == ("next_open", "index_removal")
    assert (fill.signal_date, fill.fill_date, fill.price_date) == (
        panel.dates[9],
        first_out,
        first_out,
    )
    assert fill.reference_price == price("AAA", first_out, "open")
    order = result.orders[fill.order_id]
    assert (order.forced_reason, order.reason) == ("index_removal", "index_removal")
    (trade,) = round_trips(result)
    assert (trade["exit_reason"], trade["exit_date"]) == ("index_removal", first_out.isoformat())
    assert not result.final_positions


def test_strategy_exit_on_the_last_member_session_is_merged() -> None:
    plan = {**BUY_AAA, 9: [Signal.exit("AAA", reason="mine")]}
    _, result = _run(Script(plan), {**ALWAYS, "AAA": [(0, 10)]})
    sells = [o for o in result.orders.values() if o.action == "exit"]
    assert len(sells) == 1
    assert sells[0].forced_reason == "index_removal"
    assert "exit_ignored_not_held" not in _kinds(result, "AAA")


def test_removed_symbol_frees_its_slot_for_an_entry_the_same_day() -> None:
    panel = make_panel(SYMBOLS, N)
    plan = {**BUY_AAA, 9: [Signal.enter("BBB", strength=1.0)]}
    result = simulate(
        Script(plan), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(max_positions=1),
        COSTS, 0, None, _mask(panel, {**ALWAYS, "AAA": [(0, 10)]}),
    )  # fmt: skip
    bought = [f.instrument.key for f in result.fills if f.side == "buy"]
    assert bought == ["AAA", "BBB"]
    assert "entry_skipped_no_slot" not in _kinds(result, "BBB")


@pytest.mark.parametrize(
    "missing",
    [range(10, N), [10]],  # acquired: never trades again; halted on the first day out only
    ids=["no_bar_after_removal", "halted_on_first_day_out"],
)
def test_removal_without_a_bar_closes_on_the_first_session_out(missing: Sequence[int]) -> None:
    # Not after max_missing_sessions (5): on the first session out, at the last close.
    panel, result = _run(Script(BUY_AAA), {**ALWAYS, "AAA": [(0, 10)]},
                         missing={"AAA": missing})  # fmt: skip
    price = price_lookup(panel)
    fill = _removal_fill(result)
    assert (fill.kind, fill.forced_reason) == ("forced_close", "index_removal")
    assert (fill.fill_date, fill.price_date) == (panel.dates[10], panel.dates[9])
    last_close = price("AAA", panel.dates[9], "close")
    assert fill.reference_price == last_close
    assert fill.price == pytest.approx(last_close * (1 - SLIP))
    assert _kinds(result, "AAA") == ["order_canceled", "position_closed_index_removal"]
    (trade,) = round_trips(result)
    assert (trade["exit_reason"], trade["exit_date"]) == (
        "index_removal",
        panel.dates[10].isoformat(),
    )
    assert not result.final_positions
    assert [f.side for f in result.fills] == ["buy", "sell"]  # nothing after the halt


def test_proceeds_of_a_close_time_removal_do_not_fund_that_days_opening_buys() -> None:
    # Cash from the forced close arrives at the close of the first session out, so a buy
    # signaled the session before cannot use it: here BBB is cut for cash.
    panel = make_panel(SYMBOLS, N, missing={"AAA": range(10, N)})
    plan = {**BUY_AAA, 9: [Signal.enter("BBB", strength=1.0)]}
    result = simulate(
        Script(plan), panel, SimBroker(10_000, COSTS), SYMBOLS, construction(max_positions=1),
        COSTS, 0, None, _mask(panel, {**ALWAYS, "AAA": [(0, 10)]}),
    )  # fmt: skip
    bbb = [u for u in result.updates if u.fills and u.fills[0].instrument.key == "BBB"]
    canceled = [e for e in result.events if e.symbol == "BBB" and e.kind == "order_canceled"]
    assert not bbb and "insufficient_cash" in canceled[0].detail


def test_members_with_missing_bars_still_fall_to_the_no_data_rule() -> None:
    panel, result = _run(Script(BUY_AAA), {**ALWAYS, "AAA": [(0, None)]},
                         missing={"AAA": range(10, N)})  # fmt: skip
    fill = _removal_fill(result)
    assert (fill.kind, fill.forced_reason) == ("forced_close", "no_data")
    assert fill.fill_date == panel.dates[14]  # after max_missing_sessions (5)
    (trade,) = round_trips(result)
    assert trade["exit_reason"] == "no_data"


def test_member_that_stops_trading_on_its_last_session_closes_once() -> None:
    # Stale for 5 sessions on its last member session: the no-data rule closes it first,
    # and no removal exit is added for a position that is gone.
    panel, result = _run(Script(BUY_AAA), {**ALWAYS, "AAA": [(0, 15)]},
                         missing={"AAA": range(10, N)})  # fmt: skip
    fill = _removal_fill(result)
    assert (fill.forced_reason, fill.fill_date) == ("no_data", panel.dates[14])
    assert "exit_ignored_not_held" not in _kinds(result, "AAA")


def test_broker_that_cannot_force_close_a_removed_symbol_is_logged() -> None:
    class Venue(SimBroker):
        def force_close(  # type: ignore[override]
            self, symbol: str, price: float, price_date: dt.date, session: dt.date, *,
            reason: ForcedReason,
        ) -> None:  # fmt: skip
            return None

    _, result = _run(Script(BUY_AAA), {**ALWAYS, "AAA": [(0, 10)]},
                     broker=Venue(10_000, COSTS), missing={"AAA": range(10, N)})  # fmt: skip
    stale = [e for e in result.events if e.kind == "position_stale_no_data"]
    assert len(stale) == N - 10  # every session from the first one out
    assert "left the index" in stale[0].detail
    assert "AAA" in result.final_positions


def test_rejoining_symbol_can_be_entered_again() -> None:
    plan = {**BUY_AAA, 15: [Signal.enter("AAA", strength=1.0)]}
    _, result = _run(Script(plan), {**ALWAYS, "AAA": [(0, 10), (15, None)]})
    assert [(f.side, f.forced_reason) for f in result.fills] == [
        ("buy", None),
        ("sell", "index_removal"),
        ("buy", None),
    ]


# --------------------------------------------------------------------------- the view


def test_view_hides_bars_from_the_last_end_but_shows_history_before_joining() -> None:
    strategy = Script()
    panel, _ = _run(strategy, {**ALWAYS, "AAA": [(5, 10)]})
    closes = panel.field("close")[:, panel.column("AAA")]
    early = strategy.views[7].series("AAA", "close", 100)
    np.testing.assert_array_equal(early.to_numpy(), closes[:8])  # before joining: visible
    late = strategy.views[-1].series("AAA", "close", 100)
    np.testing.assert_array_equal(late.to_numpy()[:10], closes[:10])
    assert late.iloc[10:].isna().all()  # a reused ticker's bars are never seen
    assert strategy.views[12].bar("AAA") is None


def test_view_keeps_bars_between_two_intervals() -> None:
    strategy = Script()
    _run(strategy, {**ALWAYS, "AAA": [(0, 5), (10, None)]})
    assert strategy.views[7].bar("AAA") is not None


def test_hidden_bars_still_mark_and_fill() -> None:
    # The removal exit fills at the first session out, whose bar the strategy never sees.
    strategy = Script(BUY_AAA)
    panel, result = _run(strategy, {**ALWAYS, "AAA": [(0, 10)]})
    assert strategy.views[10].bar("AAA") is None
    assert _removal_fill(result).price_date == panel.dates[10]


def test_membership_runs_are_deterministic() -> None:
    def once() -> str:
        churn: Callable[[], Strategy] = lambda: Script(  # noqa: E731
            {i: [Signal.enter(s, strength=1.0) for s in SYMBOLS] for i in range(0, N, 3)}
            | {i: [Signal.exit(s) for s in SYMBOLS] for i in range(2, N, 7)}
        )
        spans: Spans = {"SPY": [(0, None)], "AAA": [(0, 8), (12, 20)], "BBB": [(4, 25)]}
        _, result = _run(churn(), spans, missing={"BBB": range(25, N)})
        return result.model_dump_json()

    first = once()
    assert first == once()
    assert '"index_removal"' in first
