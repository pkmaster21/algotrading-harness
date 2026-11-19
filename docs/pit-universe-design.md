# Point-in-time S&P 500 universe: design

Status: proposed (2026-11-19), revised after the first review (forced exits, renames).
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
   because Alpaca files the full history under the new ticker; PR 2 checks that for every
   row (the new ticker must have bars for at least 95% of the old interval's sessions, or
   the load fails). Chains map to the last ticker (FISV -> FI -> FISV is one company under
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
   - `MarketView.members()` returns today's members, for ranking.
   - Entries: the engine drops entries for a symbol that is not a member on the signal
     date and logs `entry_skipped_not_member`.
   - Exits: at the close of a held symbol's last member session, the engine adds an exit
     for it (`exit_reason=index_removal`), which fills at the next open, the first session
     out, like an index fund. A strategy exit for the same symbol that day is merged into
     it. Exits are never blocked by risk checks, as today.
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
6. **Acquisitions with no bar after removal.** When the target stops trading before the
   first day out, the forced exit gets no bar and the existing rule applies: after 5
   sessions without a bar the position is closed at its last close minus slippage
   (`exit_reason=no_data`). That is close to the deal price for acquisitions and optimistic
   for bankruptcies, as the decision log already says.
7. **Missing bars are reported, not filled.** Each run's summary gets `membership_coverage`:
   member-sessions with a bar, as a share of all member-sessions, plus the tickers with
   none. The 6 removals with no bars anywhere (CCE, DOW, SNDK, STI, TE, WYND, all
   2016-2019) bias results slightly toward survivors; the number says how much.

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
- Speed: strategies that compute over all columns each session get slower; fine for daily
  bars, measured in PR 3.

## PR plan (each small; harness PRs flagged as sensitive)

| # | PR | Files |
|---|---|---|
| 1 | `feat(config): point-in-time universe membership` | `config/universe/sp500-membership.{csv,md}` (source, commit, MIT notice, multi-interval review), `sp500-renames.csv`, `sp500.yaml`, `harness/config.py` (`Membership` model: intervals, renames applied, no overlaps, ticker format, reviewed multi-interval list), tests |
| 2 | `feat(engine): enforce membership` | `harness/run.py` (member mask, symbols, rename coverage check against the bars), `harness/view.py` (`members()`, strategy-view masking after the last end), `harness/engine.py` (`entry_skipped_not_member`, forced `index_removal` exit), `harness/models.py` (event kind, exit reason), tests: boundary days, non-member entry dropped, forced exit fills at the first day out's real open, strategy exit merged, rename held across the date untouched, reused ticker hidden from the view, acquisition falls to the no-data rule, look-ahead and determinism suites with a membership universe |
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

- The source's rows before 2001 are thinner (fewer than 500 names); irrelevant while
  Alpaca bars start in 2016, but `sp500-membership.md` should say so.
