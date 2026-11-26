"""SEC Form 10-12B filings, the spin-off audit's fifth source (docs/spinoff-design.md,
decision 8).

Every US spin-off of a new exchange-listed company registers its shares on Form 10-12B.
The SEC's quarterly form indexes (`full-index/<year>/QTR<n>/form.idx`, public, no key)
list every filing by form type. `Form10Index` fetches them, keeps only the 10-12B rows
(originals and amendments), and caches those rows per quarter under `data/sec/`:

- a quarter fetched a week or more after it ended is final and never fetched again;
- a later quarter (the current one) is fetched again once its cached copy is a day old.

The SEC asks automated clients for a name and contact email in the User-Agent. The
caller passes it (`harness.keys.read_sec_user_agent` reads it from `.env`); without one
nothing is fetched. Requests stay far under the SEC's 10 a second.

Nothing here is market data: the indexes say which companies filed, not any price, and
the run path never reads them.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from harness.models import FrozenModel
from harness.ratelimit import RateLimiter

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "sec"
INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"
FORMS = ("10-12B", "10-12B/A")
_FINAL_AFTER = dt.timedelta(days=7)  # the indexes are rebuilt nightly; a week is ample
_REFRESH_AFTER = dt.timedelta(days=1)
_REQUESTS_PER_MINUTE = 120  # the SEC allows 10 a second
_TIMEOUT = 60.0
# A data row: form type, company name, CIK, date filed, file name; columns are padded
# with two or more spaces (a company name may hold single spaces).
_ROW = re.compile(r"^(\S+(?: \S+)*)\s{2,}(.+?)\s{2,}(\d+)\s{2,}(\d{4}-\d{2}-\d{2})\s+(\S+)\s*$")


class SecError(RuntimeError):
    """An index could not be fetched or read."""


class Filing(FrozenModel):
    """One row of a form index."""

    cik: int
    company: str
    form: str
    filed: dt.date
    path: str  # under https://www.sec.gov/Archives/


Opener = Callable[[urllib.request.Request, float], bytes]


def _open(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https only)
        body = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


def parse_form_index(text: str) -> list[Filing]:
    """The 10-12B rows of a form.idx file. A file with no row at all (no dashed line
    under the header) is not an index and fails."""
    lines = text.splitlines()
    try:
        first = next(i for i, line in enumerate(lines) if line.startswith("-----")) + 1
    except StopIteration:
        raise SecError("not a form index: no header rule") from None
    out: list[Filing] = []
    for line in lines[first:]:
        if not line.startswith("10-12B"):
            continue
        m = _ROW.match(line)
        if m is None:
            raise SecError(f"unreadable form index row: {line!r}")
        form, company, cik, filed, path = m.groups()
        if form in FORMS:
            out.append(
                Filing(
                    cik=int(cik),
                    company=company.strip(),
                    form=form,
                    filed=dt.date.fromisoformat(filed),
                    path=path,
                )
            )
    return out


def quarters(start: dt.date, end: dt.date) -> Iterator[tuple[int, int]]:
    """Every (year, quarter) from the one holding `start` to the one holding `end`."""
    year, quarter = start.year, (start.month - 1) // 3 + 1
    last = (end.year, (end.month - 1) // 3 + 1)
    while (year, quarter) <= last:
        yield year, quarter
        year, quarter = (year, quarter + 1) if quarter < 4 else (year + 1, 1)


def _quarter_end(year: int, quarter: int) -> dt.date:
    if quarter == 4:
        return dt.date(year, 12, 31)
    return dt.date(year, 3 * quarter + 1, 1) - dt.timedelta(days=1)


class Form10Index:
    """Form 10-12B filings from the SEC's quarterly form indexes, cached per quarter."""

    def __init__(
        self,
        user_agent: str,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        *,
        opener: Opener = _open,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
        limiter: RateLimiter | None = None,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise SecError("the SEC asks for a name and contact email in the User-Agent")
        self._agent = user_agent
        self._cache_dir = cache_dir
        self._open = opener
        self._now = now
        self._limiter = limiter or RateLimiter(_REQUESTS_PER_MINUTE)

    def _path(self, year: int, quarter: int) -> Path:
        return self._cache_dir / f"form10-{year}Q{quarter}.json"

    def _cached(self, year: int, quarter: int) -> list[Filing] | None:
        path = self._path(year, quarter)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
            fetched = dt.datetime.fromisoformat(data["fetched"])
            rows = [Filing.model_validate(r) for r in data["filings"]]
        except (ValueError, KeyError, TypeError) as exc:  # ValidationError is a ValueError
            raise SecError(f"{path}: unreadable cache file ({exc}); delete it") from exc
        final = fetched.date() >= _quarter_end(year, quarter) + _FINAL_AFTER
        if final or self._now() - fetched < _REFRESH_AFTER:
            return rows
        return None

    def _store(self, year: int, quarter: int, rows: list[Filing], fetched: dt.datetime) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "fetched": fetched.isoformat(),
            "filings": [r.model_dump(mode="json") for r in rows],
        }
        fd, tmp = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(data, fh, indent=1)
            os.replace(tmp, self._path(year, quarter))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _fetch(self, year: int, quarter: int) -> list[Filing] | None:
        """The quarter's rows, or None when the SEC has no index for it yet (only the
        current quarter, in its first days)."""
        url = INDEX_URL.format(year=year, quarter=quarter)
        request = urllib.request.Request(
            url, headers={"User-Agent": self._agent, "Accept-Encoding": "gzip"}
        )
        self._limiter.acquire()
        try:
            body = self._open(request, _TIMEOUT)
        except urllib.error.HTTPError as exc:
            today = self.today()
            current = (today.year, (today.month - 1) // 3 + 1)
            if exc.code == 404 and (year, quarter) == current:
                return None
            raise SecError(f"{url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SecError(f"{url}: {exc}") from exc
        return parse_form_index(body.decode("latin-1"))

    def today(self) -> dt.date:
        return self._now().date()

    def filings(self, start: dt.date, end: dt.date) -> tuple[Filing, ...]:
        """Every 10-12B row filed from `start` to `end` (both inclusive), sorted by
        (filed, cik, form, path). `end` may not be after today."""
        today = self.today()
        if end > today:
            raise SecError(f"{end} is after today ({today})")
        out: list[Filing] = []
        for year, quarter in quarters(start, end):
            rows = self._cached(year, quarter)
            if rows is None:
                fetched = self._now()
                rows = self._fetch(year, quarter)
                if rows is None:
                    continue
                self._store(year, quarter, rows, fetched)
            out.extend(r for r in rows if start <= r.filed <= end)
        return tuple(sorted(out, key=lambda r: (r.filed, r.cik, r.form, r.path)))


def first_filings(filings: tuple[Filing, ...]) -> tuple[Filing, ...]:
    """Each company's first 10-12B row (original or amendment) among `filings`, one per
    CIK, sorted by (filed, cik)."""
    first: dict[int, Filing] = {}
    for f in sorted(filings, key=lambda r: (r.filed, r.form, r.path)):
        first.setdefault(f.cik, f)
    return tuple(sorted(first.values(), key=lambda r: (r.filed, r.cik)))
