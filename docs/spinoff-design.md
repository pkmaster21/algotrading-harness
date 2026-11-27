# Spin-off adjustment: design

Status: built (2026-11-27); proposed 2026-11-24. Amended after its review: Form 10
filings as an audit source (decision 8), recovery after the guard fires (decision 6),
same-day dividends (decision 4), tickers (decision 2), several events on one symbol
(decision 4), and the restart check (decision 7). Second review: the data-fix holdout
rerun (decision 6) and Form 10 filings through today (decision 8). Fixes #55. Built in
#67, #68, #75 and #78; decisions 2 and 8 and the PR plan updated to match what was built.

## Why

Alpaca's bars (`adjustment: all`) do not adjust most spin-offs correctly. On a spin-off's
ex-date the parent's price drops by the value of the shares it handed out, and a backtest
books that drop as a loss a real holder never had. Any strategy that ranks on returns,
volatility or drawdown reads these as real moves, and the sp500 strategy searches are next.
This affects every universe, not only sp500: RTX is in `large-cap-100`.

## What was checked first (2026-11-20 to 11-24)

Scripts in `research/spinoffs_2026_11_20/` (gitignored).

- **Alpaca's spin-off feed is almost empty before 2021.** The corporate actions API returns
  1 spin-off in 2017, 1 in 2019, 5 in 2020, then 14 to 64 a year. It also lists warrants,
  rights, CVRs and when-issued tickers as spin-offs, and some rows are wrong (LH to FTRE at
  1:1, CR to CR). It is a lead to review, not a source of truth.
- **Three things happen to a spin-off in Alpaca's adjusted bars.** Measured as Alpaca's
  applied factor `a`, the ratio of adjusted to raw close the session before the ex-date over
  the same ratio on the ex-date:
  - Applied correctly (`a` equals the true factor): GE 2024-04-02, MMM 2024-04-01, and the
    listed post-2021 spin-offs checked so far.
  - Not applied (`a` = 1): YUM 2016-11-01, PNR 2018-05-01, CTXS 2017-02-01, JCI 2016-10-31
    (Adient). The cached series falls 10% to 33% that day.
  - Applied backwards, as a reverse split: RTX 2020-04-03 (`a` = 2), TGNA 2017-06-01 and
    APTV 2017-12-05 (`a` = 3). Prices before the ex-date are scaled up, so the cached series
    falls 72% to 78% that day, worse than no adjustment at all.
- **Price gaps do not find spin-offs on their own.** Among sp500 members while in the
  index, 151 one-day open gaps reach 20% (RTX, TGNA, APTV, YUM and PNR among them, but not
  CTXS at 18.6% or JCI at 10.3%). Lowering the bar to an 8% gap beyond SPY's that day
  catches both, and 1,321 others, mostly earnings: too many to review.
- **SEC Form 10 filings do.** Every US spin-off of a new listed company registers its
  shares on Form 10-12B. The SEC's quarterly form indexes (public, no key) list 120 first
  filings from 2016 to 2020, including Adient, Cars.com, nVent, Carrier, Otis, Arconic
  Rolled Products, Delphi Technologies (as Delphi Jersey Holdings), Yum China and Ingevity
  (both filed in 2015). Not covered: a spin-off merged at once into another company
  (CTXS's GetGo into LogMeIn), and a new share class of the same company (UA's class C).

## Decisions

1. **Scope: distributions of shares in another company.** Spin-offs, and share-class
   distributions that work the same way (UA's class C shares, 2016-04-08). Not mergers,
   cash acquisitions, tender offers or rights.
2. **Events are pinned, reviewed files.** A person lists each event in
   `config/corporate-actions/spinoff-events.csv` (parent, ex-date, children with their
   ratios as `CARR:1;OTIS:0.5`, any child closes given by hand, a source note), and the
   build (decision 8) resolves it into `spinoffs.csv`, one row per (parent, ex-date),
   applied to every universe. The list is kept apart from the built file so a rebuild
   never overwrites a person's input. Changing either is a reviewed PR, never a fetch at
   run time: same data, same results. `spinoffs.csv` adds the parent's and each child's
   raw ex-date close, the resolved factors (decision 4) and the cached step (decision 6).
   The ex-date is the first session the parent trades without the shares (regular way).
   **Tickers** are the ones Alpaca files the bars under today, which are the ones the
   universe files and the cache use: RTX (UTX on the day), TGNA (GCI), APTV (DLPH), the
   same as the `new` column of `sp500-renames.csv`. The note gives the ticker on the day.
   A child is looked up the same way; when Alpaca has no bars for it (acquired later, OTC
   only), the row gives its close by hand with the source.
3. **The true factor** is `f = P / (P + sum r_i * C_i)`: `P` the parent's raw close on the
   ex-date, `C_i` each child's raw close that day, `r_i` child shares per parent share.
   This is the convention CRSP and most vendors use.
4. **One multiplier per event, measured against Alpaca.** The harness applies
   `m = f * d / a` to the cached open, high, low and close of every bar before the ex-date.
   `d` is the factor of a cash dividend going ex the same day (1 when there is none; the
   build takes it from Alpaca's dividend feed and records it), so the dividend Alpaca
   folded into `a` is kept rather than cancelled. Correct in all three cases: `a = f * d`
   gives `m = 1` (the row stays, as a record that the event was checked), `a = d` gives
   `m = f`, and RTX gives about `0.5517 / 2 = 0.2759`. Volume is not changed (open
   question 1). **Several events on one symbol** (DD in 2019, UA) are separate rows whose
   multipliers compound: each scales every bar before its own ex-date. Two events on the
   same day are one row with all children listed.
5. **The recipe carries resolved values, not the file.** A new field
   `Recipe.spinoffs: tuple[SpinOff, ...]` holds `symbol, ex_date, multiplier, cached_step`
   for the recipe's own symbols (universe and benchmark), so an edit to an unrelated row
   leaves a recipe, and a frozen profile's hash, unchanged. It is listed in
   `HASH_OMITTED_AT` as `[]`, the JSON form a tuple dumps to (`()` would never match it):
   a profile frozen before this field loads and hashes exactly as before.
6. **A guard against Alpaca changing its mind.** `m` is only right while Alpaca's own
   adjustment of the event stays what it was when measured, and the cache refetches full
   history every 30 days. The ratio of the cached close on the ex-date to the cached close
   the session before is stable under any later split or dividend (both bars are rescaled
   alike), so the build records it as `cached_step`, and every load compares it with the
   bars as Alpaca served them, before any multiplier. A relative difference over 0.5%
   fails the load with a `DataError` naming the event. That is far above vendor rounding
   (Alpaca keeps four decimals on low adjusted prices) and well below any real change of
   adjustment (3% or more). Fail closed, so a fixed vendor adjustment never gets applied
   twice.
   **Recovery** is a rebuild of the row, which changes the multiplier. A frozen profile
   keeps the old values, so once its guard fires the paper runner keeps refusing, and the
   fix is a new profile from a new run. Reproducing an old backtest from its snapshot fails
   the same way: correct, since the bars it ran on are gone.
   **A data-fix rerun of the holdout** is allowed, once per data fix, so a correction does
   not end a validated strategy (decided by David, 2026-11-24). It reruns the original
   holdout run with the strategy file and parameters byte-identical and a recipe that
   differs only in `spinoffs`; the harness checks both against the original run's snapshot,
   and the holdout log line names the original run. The rerun must pass on its own: a fail
   ends the strategy, as any holdout fail does. The in-sample run (`source_run_id`) is
   rerun first under the same checks and must still pass the gates its pass card in
   `research/` recorded, checked by hand as at the first promotion (the harness does not
   know the gates), with any override the card recorded still holding; if it fails, the
   holdout is not touched and the strategy ends. The new profile's `source_run_id` and
   `holdout_run_id` are the two reruns, and each rerun's snapshot names its original.
   Nothing else may change, so it is not a new trial. Built when first needed (PR 6);
   until then a fired guard stops that profile.
7. **Applied once, in the shared path.** A pure function in a new `harness/adjust.py` (no
   API, keys or file access) takes a `BarPanel` and the recipe's events and returns the
   adjusted panel. Every place that loads bars for a recipe calls it straight after
   `DataLoader.load`: `run.py` (before its bad-bar windows), `paper.py`, and
   `check_restart_equivalence` in `profiles.py`. The last compares two runs on the same
   bars, so it would pass either way; it adjusts too so that it runs the strategy on what
   paper will, and so that a profile whose guard would fire is refused at promotion rather
   than on its first paper morning. An event whose ex-date bar or the session before is
   outside the panel is skipped: scaling a symbol's whole history changes no return.
   `raw_opens` (paper slippage) is untouched.
8. **Build and audit are a CLI, not the run path.** A new `harness/corporate_actions.py`:
   - `build` fetches raw and adjusted bars for each listed event's parent and children and
     fills in the closes, `f`, `d`, `a`, `m` and `cached_step`. It writes all or nothing:
     an event it cannot measure (no parent bar on the ex-date or the session before)
     fails the build, naming every such event, unless it is listed with a note in
     `spinoffs-reviewed.csv`. Events on or after the holdout start are resolved only with
     `--holdout`, logged in the holdout log like a holdout backtest; without it their
     current rows are kept as built. Its dividend request reaches 120 days past the
     event, since Alpaca selects dividends by process date, and drops any row dated in
     the holdout before anything reads it.
   - `audit` lists candidate events and fails until each one is either a row in
     `spinoffs.csv` or a row in `spinoffs-reviewed.csv` (ticker, date, note: "earnings",
     "parent not in any universe"). Candidates:
     - each company's first Form 10-12B filing in the SEC's quarterly form indexes, from
       a year before the data start (a Form 10 comes months before the distribution)
       through the latest index, read through today since an index holds no prices. Each
       is reviewed by hand in `form10-reviewed.csv`, keyed by CIK, as `listed` (the
       parent and ex-date of an event in `spinoffs.csv` or `spinoffs-reviewed.csv`),
       `none` (no event for these universes; the note says why) or `pending` (not yet
       distributed, or going ex in the holdout and not built with `--holdout`; printed on
       every audit, never failing it). From 2021 Alpaca's feed has the events and the
       ones checked are right, but the filings cost little and catch any the feed misses;
     - every row of Alpaca's spin-off feed for a symbol in any universe file, after
       dropping warrants, rights, CVRs and `.WI` tickers. The feed is requested from a
       year before the data start through today (process dates can lag the ex-date by
       years) and rows going ex on or after the holdout start are dropped before anything
       reads them. The audit is not a holdout route;
     - every step in Alpaca's adjusted-to-raw ratio over 3% that is not explained by a
       simple forward split in Alpaca's split feed (n for 1, or p for q with both at most
       10). Before 2021 that feed also files spin-offs as splits, so those stay
       candidates;
     - every open gap of 20% or more;
     - every sp500 addition whose first bar is within 5 sessions of its start date
       (spin-offs often join the index on day one, as CARR and OTIS did).
     The gap and addition rules catch what the filings miss (merged spin-offs, share
     classes) when the event is large.
   - The SEC asks automated clients for a contact in the User-Agent. The CLI reads it from
     a local setting and refuses to fetch without one; nothing personal is in the repo.
   - The thresholds live in the CLI. They decide what a person reviews, not what a run
     does, so they stay out of the recipe.

## What stays the same

- Splits and dividends: still Alpaca's adjustment.
- The cache format, the loader and holdout discipline in the run path.
- Strategies, construction, risk and the engine.
- Profiles frozen before this lands load and hash as before (decision 5).

## Costs

- Run time: one pass over the events of the panel's symbols, a few dozen at most. Nothing
  measurable.
- Build: two bar requests per parent and per child (raw and adjusted), a few hundred in
  all. Audit: raw bars for every symbol in every universe file, about 800 symbols, in
  batches, plus 48 SEC index files; a few minutes under the rate limit.
- Review, once: about 150 gap candidates and 220 Form 10 filings to mark, then only new
  ones each time the audit reruns.

## PR plan (each small; harness PRs flagged as sensitive)

1. This design.
2. `feat(data): apply spin-off adjustments`. `SpinOff` model and file loading in
   `config.py`, `Recipe.spinoffs` and its `HASH_OMITTED_AT` entry, `harness/adjust.py`
   with the guard, wiring in `run.py`, `paper.py` and `profiles.py`. Tests on synthetic
   bars for all three cases, a same-day dividend, two events on one symbol, the guard,
   skipped events, and an old profile's hash. The file ships with the header only.
3. `feat(data): spin-off build` (#68). `harness/corporate_actions.py build`,
   `spinoff-events.csv` and the first `spinoffs.csv`, with coverage limited to the listed
   events until the audit merged.
   3a. `feat(data): spin-off audit from Alpaca's data` (#75). The feed, step, gap and
   addition sources, the filled `spinoffs.csv` and `spinoffs-reviewed.csv`.
   3b. `feat(data): spin-off audit from SEC Form 10 filings` (#78). `harness/sec.py`
   and `form10-reviewed.csv`.
4. `docs: spin-off adjustment`. ARCHITECTURE.md decision log, components and open
   questions (volume, below), the CLAUDE.md holdout rule and layout, this plan, and how
   to rerun build and audit (below).
5. Paper refusal on a new spin-off (open question 2).
6. The data-fix holdout rerun (decision 6), when a guard first fires or PR 5 first
   refuses.

## Rerunning build and audit

Neither is part of a run, and a run reads only `spinoffs.csv`. Rerun:

- the **audit** after a change to any universe file or the data range, periodically
  for new Form 10 filings and feed rows (suggested: quarterly; `harness/sec.py` treats a
  quarter's index as final once it is fetched a week or more after the quarter ends), and
  before relying on a pending Form 10 row;
- the **build** after any change to `spinoff-events.csv`, and when a load fails the guard
  (decision 6: Alpaca changed its adjustment). A row the build prints as `CHANGED` needs
  a look before it is committed: Alpaca has served transient values (FTV, NEE, 2026-11-26)
  that were gone on the next fetch.

```
uv run python -m harness.corporate_actions audit --out <scratch>/open.csv --form10-out <scratch>/form10.csv
uv run python -m harness.corporate_actions build
```

The audit exits 0 when every candidate is listed or reviewed. Otherwise each open
(ticker, date) goes either into `spinoff-events.csv` (then build) or into
`spinoffs-reviewed.csv` with a note that says what was checked ("not checked against a
news source" when only the bars were read); each open Form 10 filing goes into
`form10-reviewed.csv`. `--out` and `--form10-out` write the open rows in those files'
formats. The audit needs `SEC_USER_AGENT` in `.env` (a name and contact email, set
locally, never committed). `build --holdout` is for an event dated in the holdout and is
logged in the holdout log. The updated files are a reviewed PR like any config change.

## Open questions

1. **Volume.** Alpaca's reverse-split treatment of RTX, TGNA and APTV seems to divide their
   volume before the ex-date by `a` too (cached monthly volume jumps 2.5x for RTX and 5.6x
   for TGNA in the ex-date's month; not yet checked against raw volume). Agreed to leave it:
   PR 4 records it in ARCHITECTURE.md's open questions, for the first strategy that reads
   volume to settle.
2. **Paper trading through a new spin-off.** A spin-off after a profile is frozen is not in
   its recipe, so the parent's next bars show the drop, and the paper account receives
   child shares the harness does not track. Agreed: the paper runner checks Alpaca's feed
   each morning and refuses to run (an alert, no orders) when a held or universe symbol
   has a spin-off since the profile was frozen. It counts only feed rows that pass the
   audit's filter (no warrants, rights, CVRs or `.WI` tickers), so feed noise does not stop
   trading. Recovery is a new profile, through a data-fix rerun of the holdout, as in
   decision 6.
