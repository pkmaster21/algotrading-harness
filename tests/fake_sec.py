"""A stand-in for the SEC's quarterly form indexes: no network."""

from __future__ import annotations

import datetime as dt
import re
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

from harness.ratelimit import RateLimiter
from harness.sec import Form10Index

AGENT = "Test Harness test@example.invalid"

_HEADER = (
    "Description:           Daily Index of EDGAR Dissemination Feed by Form Type\n"
    "Last Data Received:    March 31, 2026\n"
    "Comments:              webmaster@sec.gov\n"
    "Anonymous FTP:         ftp://ftp.sec.gov/edgar/\n"
    "\n\n\n"
    "Form Type   Company Name                                                  CIK"
    "         Date Filed  File Name\n" + "-" * 140 + "\n"
)


def index_text(rows: list[tuple[str, str, int, dt.date]]) -> str:
    """A form.idx file holding `rows` of (form, company, cik, filed), laid out as the
    SEC lays it out (fixed-width columns)."""
    lines = [
        f"{form:<12}{company:<62}{cik:<12}{filed.isoformat():<12}edgar/data/{cik}/{n:04d}.txt"
        for n, (form, company, cik, filed) in enumerate(rows)
    ]
    return _HEADER + "".join(f"{line}\n" for line in lines)


class FakeSec:
    """Serves form.idx per (year, quarter) from `rows`; a quarter with no rows is an
    index with none. `missing` quarters answer 404. Every request is recorded."""

    def __init__(self) -> None:
        self.rows: dict[tuple[int, int], list[tuple[str, str, int, dt.date]]] = {}
        self.missing: set[tuple[int, int]] = set()
        self.requests: list[urllib.request.Request] = []

    def add(self, form: str, company: str, cik: int, filed: dt.date) -> None:
        key = (filed.year, (filed.month - 1) // 3 + 1)
        self.rows.setdefault(key, []).append((form, company, cik, filed))

    def fetched(self) -> list[tuple[int, int]]:
        out = []
        for r in self.requests:
            m = re.search(r"/(\d{4})/QTR(\d)/form\.idx$", r.full_url)
            assert m is not None, r.full_url
            out.append((int(m.group(1)), int(m.group(2))))
        return out

    def open(self, request: urllib.request.Request, timeout: float) -> bytes:
        self.requests.append(request)
        m = re.search(r"/(\d{4})/QTR(\d)/form\.idx$", request.full_url)
        assert m is not None, request.full_url
        key = (int(m.group(1)), int(m.group(2)))
        if key in self.missing:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), None)
        return index_text(self.rows.get(key, [])).encode("latin-1")

    def index(
        self, cache_dir: Path, now: dt.datetime | list[dt.datetime], agent: str = AGENT
    ) -> Form10Index:
        """A Form10Index on this fake. `now` may be a one-item list to move the clock."""
        clock = now if isinstance(now, list) else [now]
        return Form10Index(
            agent,
            cache_dir,
            opener=self.open,
            now=lambda: clock[0],
            limiter=RateLimiter(10**6),
        )
