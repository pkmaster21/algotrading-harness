# S&P 500 membership files

Used by the `sp500` universe preset (`sp500.yaml`). Design: `docs/pit-universe-design.md`.
The CSVs named in `sp500.yaml` are read only by `harness/config.py`, in this order:
membership, corrections, renames, then intervals that ended before `data_start`
(2015-01-01) are dropped; bad-bar windows are applied to the bars. Change these files
through a reviewed PR, never by hand at run time.

Alpaca's bars start on 2016-01-04, a year after `data_start`, so intervals in 2015 are kept
but have no bars. Every check below uses in-sample bars only (to 2026-03-31), fetched one
symbol per request: in a multi-symbol request Alpaca returns no bars for an old ticker when
the ticker that now holds its history is in the same request (GEN with SYMC, DOC with HCP,
CPRI with KORS), so a batched cache can hide bars that exist.

## `sp500-membership.csv`

An unmodified copy of `sp500_ticker_start_end.csv` from
<https://github.com/fja05680/sp500>, commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`
(2026-09-07), retrieved 2026-11-19.
sha256 `b2f4fe2f2e4dce8eaf0eb4fbd3946a32a20c1ddfe69687dfca3b6b59a0b71f9f`.

One row per membership interval: `ticker,start_date,end_date`, the ticker the stock
traded under at the time. The harness reads `start_date` as the first session in the
index and `end_date` as the first session out (end exclusive); an empty `end_date` means
still a member. The upstream list comes from Andreas Clenow's data to 2019 plus changes the
maintainer checks against Wikipedia and news. Rows before 2001 are thinner (fewer than 500
names).

To update: copy the file from a newer upstream commit; update the commit, date and sha256
above; rerun the checks below (same-day review, multi-interval review, bar scan); and say
in the PR what changed.

## `sp500-corrections.csv`

Fixes to upstream rows, applied before renames; each must match one upstream row exactly.

- `KORS` and `CPRI`: upstream lists Michael Kors under `KORS` (2013-11-13 to 2018-09-19)
  and Capri under `CPRI` from 2016-01-04, so both tickers are members at once for nearly
  three years. They are one company, and Alpaca's `KORS` and `CPRI` bars are identical over
  682 sessions, so a strategy could buy it twice. Corrected to one `CPRI` interval from
  2013-11-13 to 2020-05-12.

## `sp500-renames.csv`

`old,new,date,note`: the old ticker's interval ending on `date` and the new ticker's
interval starting on `date` are one company. The harness joins each chain into a single
interval under its last ticker and drops the old tickers, because Alpaca files the whole
history under the later ticker. Every row must match one interval on each side exactly.

28 rows, built 2026-11-19:

- 18 found automatically: the removal and an addition within 5 days whose daily returns
  before the change correlate above 0.98 on Alpaca bars (all are 0.993 to 1.000). This needs
  bars under both tickers before the date, which Alpaca has for these 18.
- 3 after the in-sample end, found by the same-day review below on in-sample bars: `BK ->
  BNY` (2026-05-21, correlation 0.999), `SATS -> ECHO` (2026-06-24, 1.000), `EQR -> VMRK`
  (2026-08-18, 1.000). Backtests are unaffected (one column either way); paper trading
  needs them.
- 7 checked by hand, where the new ticker has no bars before the change (nothing to
  correlate) or the old ticker's bars are wrong:
  - `SYMC -> NLOK` and `HCP -> PEAK`: `NLOK` and `PEAK` have no bars before 2019-11-05;
    Symantec's history is under `GEN` (return correlation 1.000) and HCP's under `DOC`
    (0.993). Found in review.
  - `BBT -> TFC`: Alpaca's `BBT` is a different security (about $25 from 2016 to 2026);
    BB&T's history is under `TFC`.
  - `DD -> DWDP -> DD`: DowDuPont's bars are under `DD`, continuous through 2017-09-01
    (on a one-per-request cache `DWDP` has bars too, identical to `DD`'s: correlation 1.000).
  - `FISV -> FI -> FISV`: Fiserv's 2023-2025 bars are under `FISV`, continuous through
    2025-11-11.

Fetched one per request, `SYMC`, `NLOK`, `HCP`, `PEAK` and `KORS` all have bars; in a
request that also names `GEN`, `DOC` or `CPRI` they come back empty (see the top of this file).

## `sp500-same-day-review.csv`: every same-day removal and addition

Documentation only (not read by the harness). One row for every removal from 2015-01-01 to
the end of the membership file paired with each addition on the same day: 445 pairs.
Correlations use in-sample bars only, so rows after 2026-03-31 rest on the history before it. Columns: the return
correlation before the change on Alpaca bars and the overlap it rests on, whether the pair
is in the rename map, and the reason. A test checks that every row has a reason and that
the rows marked renamed are exactly `sp500-renames.csv`.

Reasons are assigned by rule where the bars decide it (correlation below 0.9 on at least 20
sessions: different companies; a 2015 date with no bars: before Alpaca bars begin) and by
hand otherwise (no overlap to correlate). Same-day ticker changes that stay split on Alpaca
are not renamed, because the new ticker has no bars before the change, so there is nothing
to join:

| Date | Old | New | Company |
|---|---|---|---|
| 2019-12-05 | CBS | VIAC | CBS absorbed Viacom as ViacomCBS |
| 2020-04-06 | ARNC | HWM | Arconic Inc. became Howmet |
| 2020-11-17 | MYL | VTRS | Mylan and Upjohn as Viatris |
| 2021-08-03 | LB | BBWI | L Brands became Bath & Body Works |
| 2021-10-04 | COG | CTRA | Cabot and Cimarex as Coterra |
| 2022-04-11 | DISCA | WBD | Discovery became Warner Bros. Discovery |
| 2024-02-01 | CDAY | DAY | Ceridian renamed Dayforce |

Side effects, once PR 2 enforces membership: a position held across one of these dates is
force-exited under the old ticker and can be re-entered under the new one, paying costs
twice (7 events in 10 years). In 6 of the 7 the old ticker has no bar on the change day
(CBS last bar 2019-12-04, COG 2021-10-01, DISCA 2022-04-08, MYL 2020-11-16, CDAY
2024-01-31; ARNC is marked bad from 2020-04-01), and the same holds for VIAC (last bar
2022-02-16, removed 2022-02-17 into the bad `PARA` bars). Those exits get no fill at a real
open and fall to the no-data rule (closed after 5 sessions at the last close). Only LB's
bars continue. PR 2 decides how these exits fill (for example, at the new ticker's open).

The script that builds the table is in the PR description (it reads a bar cache filled by
`DataLoader`); rerun it on every membership update.

## `sp500-bad-bars.csv`

Windows in which Alpaca's bars for a member ticker are not real prices of the member
company: another security, or one stale price repeated. PR 2 treats
bars in a window as missing, for the strategy's view and for the engine alike, so they are
never ranked, traded or used as a fill price; the coverage report counts the member
sessions they remove.

- `PARA`, all dates: Alpaca's `PARA` is not Paramount (adjusted closes from $578 to $1.1M).
  Paramount's 2022-2025 bars are not on Alpaca (`PSKY` starts 2025-08-07), so Paramount's
  membership from 2022-02-17 to 2025-08-08 has no bars.
- `CTRA`, before 2021-10-04: Contura Energy (return correlation with Cabot 0.264), not
  Coterra. Without this, strategies would see Contura's history as Coterra's.
- `ARNC`, from 2020-04-01: the Arconic Corp. spin-off, while membership says Howmet until
  2020-04-06. A forced exit then has no valid bar and falls to the no-data rule.
- `DOW`, before 2019-04-02: not Dow Chemical. To 2017-09-01 the series tracks DuPont
  (correlation 0.998 with `DD`), then repeats $44.18 for 396 sessions until Dow Inc.'s
  first real bar. Old Dow Chemical's membership to 2017-09-01 therefore has no bars.
- `EXE`, 2020-06-29 to 2021-02-10, and `SMCI`, 2018-08-23 to 2020-01-14: one stale price
  repeated (156 and 349 sessions) during Chesapeake's bankruptcy and SMCI's Nasdaq
  delisting. Both are before these tickers joined, but strategies would see them as history.

Removed tickers whose bars continue as another security after they leave (FB from 2025,
HCP after 2019, BBT) need no row: the strategy's view hides a ticker's bars after its last
membership day.

## Tickers with more than one interval (reviewed 2026-11-19)

After corrections and renames, these tickers have more than one interval that ends after
the data start. The bars between intervals stay visible to strategies, which is correct for
each:

| Ticker | Out | Back | Why visible gap bars are fine |
|---|---|---|---|
| DOW | 2017-09-01 | 2019-04-02 | Bars before 2019-04-02 are marked bad (not Dow Chemical, then a stale price), so the visible bars are Dow Inc. only |
| DXC | 2015-12-01 | 2017-04-04 | The first interval (Computer Sciences) ends before Alpaca's bars begin; the bars are DXC Technology only, from 2017-04-03 |
| EQT | 2018-11-13 | 2022-10-03 | Same company, dropped and re-added |
| FSLR | 2017-03-20 | 2022-12-19 | Same company |
| KDP | 2018-07-02 | 2022-06-21 | Dr Pepper Snapple then Keurig Dr Pepper; Alpaca's series is continuous |
| PCG | 2019-01-18 | 2022-10-03 | Same company (left during its bankruptcy) |
| SNDK | 2016-05-12 | 2025-11-28 | Old SanDisk has no Alpaca bars; the bars are the 2025 spin-off only |

`sp500.yaml` lists these; a new multi-interval ticker fails the load until it is reviewed
and added there.

## Bar scan (2026-11-19)

Every in-sample interval was scanned on a one-per-request cache for (1) two member tickers
with identical series at the same time, (2) impossible prices, (3) one-day moves beyond 40%
and (4) one price repeated for 10 or more sessions, in the window a strategy sees. Findings:
KORS/CPRI (corrected above), PARA, CTRA, DOW, EXE and SMCI (bad bars above), the spin-offs
below, and real events (FRC, SIVB, PCG, COVID March 2020, ...).

## Known data issues

- **Spin-offs are not adjusted** in Alpaca's bars: RTX -71% on 2020-04-03, ARNC -89%,
  APTV -72%, SW (WestRock's history) -84%, TGNA -78%, UA -49%, each on a spin-off date.
  A backtest books these as losses. Affects every universe; tracked in issue #55.
- `DD` jumps about +18% on 2019-06-03 (Corteva spin-off and DowDuPont's 1-for-3 reverse
  split): Alpaca's adjustment looks incomplete there.
- Removals since 2016 with no bars inside their interval: AABA, SNDK (old SanDisk), STI,
  TE, WYND, plus old Dow Chemical (bars marked bad) and every interval that ended in 2015,
  before Alpaca's bars begin. (STI and TE have bars, but only after they left: reused
  tickers.) Their member sessions have no bars, a small survivor bias the run summary will
  report as coverage.
- **Batched requests can drop an old ticker's bars** (top of this file): a harness data
  bug, tracked in issue #56. It does not affect `sp500` today, because renamed-away tickers
  are dropped before loading, but it can affect any universe that lists an old and a new
  ticker together.
- Additions after 2026-03-31 (for example RDDT) have no in-sample bars; they matter only
  for holdout and paper runs.

## License of `sp500-membership.csv`

```
MIT License

Copyright (c) 2019-2020 Farrell J. Aultman <fja0568@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
