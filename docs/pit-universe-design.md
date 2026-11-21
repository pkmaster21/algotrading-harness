# Point-in-time S&P 500 universe: design

Status: accepted (2026-11-19). Amended in PR 1 after its review: upstream corrections and
bad-bar windows (decisions 8 and 9), and an auditable same-day review of renames. Amended
in PR 2: how a removal exit fills when no bar follows (decision 6), what `members()`
returns (decision 4), and how the rename check measures coverage (decision 2). Amended
in PR 4: the summary field PR 3 shipped (decision 7) and the measured speed (Costs).
Implemented for backtests by #57, #59, #62 and #63 (2026-11-20).
Resolves the open question "Data source for historical index membership" in
`ARCHITECTURE.md` for backtests. Paper trading support is a later step with one hard
requirement fixed here.

## Why

Every stock universe so far is a list of today's names applied to all history. That
silently drops the companies that shrank, were acquired or failed, and keeps the ones that
grew into the index, so stock-picking backtests look better than reality. The decision log
already says historical membership is needed before stock-picking strategies; the
2026-11-19 strategy searches confirmed it: the remaining credible daily-bar strategies
(cross-sectional momentum, low volatility, 52-week high) are exactly the ones this bias
inflates most.

## What was checked first (2026-11-19)

- **Membership source.** `github.com/fja05680/sp500`, file `sp500_ticker_start_end.csv`:
  MIT license, S&P 500 membership since 1996 as one row per (ticker, start date, end date),
  using the ticker the stock traded under at the time (`FB` until 2022-06-09, then `META`).
  Built from Andreas Clenow's list (to 2019) plus changes the maintainer checks against
  Wikipedia and news. Last updated 2026-09-07 (commit a2430f2). 742 intervals overlap the
  data range (2016-01-04 to 2026-03-31); 239 of them end inside it.
- **Bars for names that left.** Alpaca serves bars for delisted, acquired and failed stocks
  up to their last trading day (TWTR, ATVI, SIVB, FRC, XLNX, CTXS, ...). 8 of the 239
  removals have no bars under their ticker: CCE, DOW (the pre-2017 Dow Chemical), DWDP, FI,
  SNDK (the pre-2016 SanDisk), STI, TE, WYND. Two of those (DWDP, FI) are renames whose
  bars Alpaca files under the later ticker (DD, FISV); the rest are acquisitions or mergers
  with no bars left anywhere.
- **Renames.** 18 removals are renames: the old ticker leaves and a new one joins within
  days, and their daily returns before the change correlate at 0.997 to 1.000 (FB -> META,
  UTX -> RTX, FLT -> CPAY, ANTM -> ELV, ...). Alpaca keeps the company's whole history under
  the new ticker as well, so without handling, one company appears as two columns with the
  same prices.
- **Tickers that left and rejoined.** 8 in range. Same company back again: EQT, FSLR, KDP,
  PCG, FISV. A different company reusing the ticker: SNDK (new SanDisk, 2025), DOW (Dow
  Inc., 2019); in both cases Alpaca has no bars for the earlier company, so the bars that
  exist are the new company's. DD runs through the DuPont, DowDuPont, DuPont chain.
- **Ticker reuse after leaving.** Alpaca's `FB` has Meta's bars until June 2022, nothing for
  three years, then a different security from 2025-06 (about $39).

## Decisions

1. **Pin the source in the repo.** Copy the file to `config/universe/sp500-membership.csv`
   with a header file (`sp500-membership.md`) recording the source URL, commit, date and
   the MIT notice. Updating it is a reviewed PR, never a fetch at run time: same data, same
   results. The run snapshot records the file's sha256.
2. **A reviewed rename map, applied before anything else.** `config/universe/
   sp500-renames.csv` lists `old, new, date` (FB, META, 2022-06-09). Each rename joins the
   old and new intervals into one membership under the new ticker, and the old ticker's
   column is dropped, so a renamed company is one column for the whole run and a position
   held across the date is untouched (no sell and rebuy, no forced exit). This works
   because Alpaca files the full history under the new ticker; every backtest checks that
   for every row (the surviving ticker must have bars for at least 95% of the panel
   sessions in the joined interval before the rename date, or the run fails). The check
   found one upstream interval that predates its ticker: Baker Hughes under `BHGE` from
   1996, while Alpaca has old Baker Hughes as a separate `BHI` series; a correction starts
   `BHGE` on 2017-07-05. Chains map to the last ticker (FISV -> FI -> FISV is one company under
   FISV). Reports name the company by its later ticker throughout, which is anachronistic
   but unambiguous. The map is built once by the detection above (same-week addition,
   return correlation above 0.98) and then checked by hand; the detection script goes in
   the PR description, not the harness.
3. **A universe preset can be point-in-time.** `config/universe/sp500.yaml` names the
   membership and rename files instead of listing symbols. `harness/config.py` reads them
   (config is data, only the loader reads it) into a validated `Membership` model:
   intervals per ticker, start inclusive, end exclusive (the source's end date is the first
   day out), no overlaps per ticker after renames are applied. The run's symbols are every
   ticker with an interval overlapping the run's dates.
4. **The harness enforces membership on both sides.** The run builds a boolean member mask
   (sessions x symbols) once, from the panel dates. Then:
   - `MarketView.members()` returns the symbols a strategy may enter today, for ranking:
     today's members minus any whose last member session is today.
   - Entries: the engine drops entries for a symbol that is not a member on the signal
     date, or whose last member session it is (the fill would come after it left), and
     logs `entry_skipped_not_member`.
   - Exits: at the close of a held symbol's last member session, the engine adds an exit
     for it (`exit_reason=index_removal`), which fills at the next open, the first session
     out, like an index fund. A strategy exit for the same symbol that day is merged into
     it. The exit goes through construction, so it frees its slot for that day's entries.
     Exits are never blocked by risk checks, as today.
   - `index_removal` means the ticker left the membership list, not always the company:
     the 7 same-company ticker changes Alpaca keeps as two series (`sp500-membership.md`)
     and VIAC, whose successor `PARA` has no usable bars, exit this way too. A strategy
     that buys the new ticker back pays a round trip in costs (8 events in 10 years).
   - The argument is the same for both: a strategy that forgets to filter must not trade
     a survivor-only universe, and must not keep holding a name that left it.
   - Membership is known at the signal close: index changes are announced days before
     they take effect. Removals that follow an acquisition closing can come at short
     notice, so the forced exit can fill a day or so later than a real index fund's; the
     no-data rule below covers the case where no bar follows at all.
5. **Masking is for the strategy's view only; the engine sees real bars.** The strategy's
   view hides a symbol's bars from its last membership end onward, so a reused ticker (the
   new `FB`) is never seen and never ranked. Fills and marks use the real bars, so the
   forced exit fills at the first day out's actual open. Bars before a ticker first joined
   stay visible (that history was public; momentum and volatility need it), and so do bars
   in the gap between two intervals of one ticker: in range, those are either the same
   company (EQT, FSLR, KDP, PCG, FISV) or a new company whose predecessor has no bars
   (SNDK, DOW), so nothing wrong is shown. PR 1 records that review in `sp500-membership.md`
   and fails the load if a future update adds a multi-interval ticker not on the reviewed
   list.
6. **A removal with no bar on the first day out closes that day at the last close.** When
   a removed ticker has no bar on its first session out, the removal exit is canceled at
   the open (no bar), and at that session's close the engine closes the position at its
   last close minus slippage (`force_close`, `exit_reason=index_removal`), without waiting
   for the 5-session no-data rule. This covers acquisitions whose target stopped trading
   before the first day out, a ticker halted that day, and 6 of the 7 split same-company
   ticker changes plus VIAC (their old ticker's bars end before the change day).
   - The price is the last member close, which for an acquisition is close to the deal
     price and for a bankruptcy is optimistic, as the decision log already says.
   - It misses one overnight move (where measurable: DISCA -1.4%, CDAY -0.2%). Filling at
     the successor's open instead was rejected: Alpaca's bars are dividend-adjusted, so a
     successor's early prices are scaled down by its later dividends (MYL to VTRS -23%,
     COG to CTRA -20% on the change day), which would book losses that never happened.
     Carrying the position over to the new ticker was rejected too: it needs a
     corporate-action path in every broker for 8 events.
   - The proceeds arrive at the close, so unlike a real open fill they cannot fund that
     session's opening buys.
   - The 5-session no-data rule now applies only to held symbols that lose their bars
     while still members.
7. **Missing bars are reported, not filled.** A point-in-time run's `summary.json` gets a
   `membership` block: the sha256 of the four source files, and `membership.coverage` over
   the traded sessions (warm-up left out): member-sessions, those with a bar, the share,
   those a bad-bars window removed (`bad_bar_sessions`), and `tickers_without_bars`, the
   members with no usable bar in any of their member sessions. Fixed universes write
   `membership: null`. Members without bars bias results slightly toward survivors; the
   number says how much.
   - `sp500`, 2016-01-04 to 2026-03-31, after #62: share 0.9959 (1,292,590 of 1,297,873
     member-sessions), 1,293 bad-bar sessions, no usable bar for AABA, PARA, STI, TE and
     WYND. PARA is masked as a different security throughout; STI and TE have bars only
     after they left (reused tickers).
   - A ticker is listed only when none of its member sessions has a bar. Old SanDisk
     (`SNDK`) and old Dow Chemical (`DOW`, bars marked bad) have none, but their tickers
     carry the later company's bars as members, so they lower the share without being
     listed. CCE, once on this list, was missing only because of the batching bug fixed
     in #62.

8. **Upstream errors are corrected in a reviewed file, not in the copy.**
   `sp500-corrections.csv` replaces or drops upstream rows (each must match one row
   exactly) before renames apply, so the pinned copy stays byte-identical to upstream. First
   use: upstream lists Michael Kors under both `KORS` and `CPRI` for 2016-2018, two members
   with identical bars.
9. **Bars that are not real prices are marked, not guessed around.** `sp500-bad-bars.csv`
   lists windows in which a member ticker's bars belong to a different security (`PARA`
   throughout, `CTRA` before 2021-10-04, `ARNC` from 2020-04-01, `DOW` before 2019-04-02) or
   repeat one stale price (`EXE`, `SMCI`). The engine treats bars in a
   window as missing for the strategy's view, fills and marks alike, and the coverage report
   counts the member sessions they remove. Masking after a ticker's last membership day
   (decision 5) already covers reuse after leaving, so those need no row.

## What stays the same

- Strategies still emit Signals only and never read files; the membership and rename files
  are read by `harness/config.py`, and strategies get members through the view.
- Construction, risk, costs and the benchmark (SPY) are unchanged.
- Static universes (`plumbing-10`, `large-cap-100`) keep working: no membership means every
  symbol is a member every session, no forced exits, no masking.
- One code path: the member mask, the entry filter and the forced exit live in the engine,
  so paper trading gets them too once it uses a point-in-time universe.

## Costs

- Data: about 730 symbols, about 40 MB of Parquet (the 239 removed names took about 14 MB); the
  first download is about 15 batches of 50 symbols, a few minutes at the rate limit. The
  panel is about 2,600 sessions x 730 symbols x 5 fields (75 MB in memory).
- Speed: measured in PR 3 with `speed_momentum` over the full range: 9.6 s from a warm
  cache, 95 s including a full cache rebuild. Fine for daily bars.

## PR plan (each small; harness PRs flagged as sensitive)

| # | PR | Files |
|---|---|---|
| 1 | `feat(config): point-in-time universe membership` | `config/universe/sp500-membership.{csv,md}` (source, commit, MIT notice, reviews, known issues), `sp500-corrections.csv`, `sp500-renames.csv`, `sp500-bad-bars.csv`, `sp500-same-day-review.csv` (audit, not read), `sp500.yaml`, `harness/config.py` (`Membership` model: corrections then renames applied, no overlaps, ticker format, reviewed multi-interval list, bad-bar windows recorded), tests |
| 2 | `feat(engine): enforce membership` | `harness/run.py` (member mask, symbols, rename coverage check against the bars), `harness/view.py` (`members()`, strategy-view masking after the last end), bad-bar windows as missing bars for view, fills and marks, `harness/engine.py` (`entry_skipped_not_member`, forced `index_removal` exit), `harness/models.py` (event kind, exit reason), tests: boundary days, non-member entry dropped, forced exit fills at the first day out's real open, strategy exit merged, rename held across the date untouched, reused ticker hidden from the view, acquisition falls to the no-data rule, look-ahead and determinism suites with a membership universe. To decide here: how a forced exit fills when the old ticker has no bar on the change day, which is the case for 6 of the 7 same-company changes that stay split on Alpaca (CBS, COG, DISCA, MYL, CDAY, ARNC) and for VIAC; for example at the new ticker's open, from a reviewed successor list |
| 3 | `feat(report): membership coverage and hash` | `harness/report.py` (summary fields, snapshot sha256), first full sp500 run, speed check, tests |
| 4 | `docs: point-in-time universe` | `ARCHITECTURE.md` (decision log entry, open question resolved for backtests, components table), `CLAUDE.md` layout line, example strategy note |

## Paper trading (later, own design; one requirement fixed now)

- **Hard requirement:** the paper runner refuses to trade a point-in-time profile when
  `sp500-membership.csv` is older than N days (N chosen in that design), because a stale
  list would buy removed names and miss forced exits.
- The membership and rename files are kept current by a monthly reviewed PR from the
  upstream file, checked against the current Wikipedia list.
- A frozen profile pins both files' hashes.
- Backtests do not need updates for now: the data ends 2026-03-31 and the source was
  updated 2026-09-07.

## Open questions

- None for backtests. (The source's thinner rows before 2001 are now noted in
  `sp500-membership.md`.)
