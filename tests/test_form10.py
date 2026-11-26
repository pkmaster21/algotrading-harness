"""Form 10-12B filings as the spin-off audit's fifth source (docs/spinoff-design.md,
decision 8): reading the SEC's form indexes, caching them, the User-Agent the SEC asks
for, and how a filing is reviewed. No network: indexes come from tests/fake_sec.py."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from harness import corporate_actions
from harness.config import ConfigError
from harness.corporate_actions import FORM10_REVIEWED_FILE, audit, load_form10_reviewed
from harness.keys import KeysError, read_sec_user_agent
from harness.sec import Filing, SecError, first_filings, parse_form_index, quarters
from tests.fake_alpaca import NOW, REPO_CONFIG, spinoff_row
from tests.fake_sec import AGENT, FakeSec, index_text
from tests.test_spinoff_audit import AuditFetcher, _config, _loader

DAY = dt.date(2026, 1, 14)
FILED = dt.date(2025, 6, 2)  # inside the audit's filing range (from 2024-10-01)


# --------------------------------------------------------------------------- the index


def test_only_10_12b_rows_are_read() -> None:
    text = index_text(
        [
            ("10-12B", "Spinco Holdings Inc", 1001, FILED),
            ("10-12B/A", "Spinco  Holdings Inc", 1001, FILED),  # two spaces in a name
            ("10-12G", "Small Co", 1002, FILED),
            ("10-K", "Parent Corp", 1003, FILED),
            ("SC 13D", "Other Corp", 1004, FILED),
        ]
    )
    rows = parse_form_index(text)
    assert [(r.form, r.company, r.cik, r.filed) for r in rows] == [
        ("10-12B", "Spinco Holdings Inc", 1001, FILED),
        ("10-12B/A", "Spinco  Holdings Inc", 1001, FILED),
    ]
    assert rows[0].path == "edgar/data/1001/0000.txt"


def test_a_file_that_is_not_an_index_fails() -> None:
    with pytest.raises(SecError, match="not a form index"):
        parse_form_index("<html>Request Rate Threshold Exceeded</html>")
    with pytest.raises(SecError, match="unreadable"):
        parse_form_index(index_text([]) + "10-12B broken row\n")


def test_quarters_span_both_ends() -> None:
    assert list(quarters(dt.date(2024, 11, 5), dt.date(2025, 4, 1))) == [
        (2024, 4),
        (2025, 1),
        (2025, 2),
    ]


def test_each_company_counts_once_from_its_first_filing() -> None:
    def f(cik: int, form: str, day: dt.date) -> Filing:
        return Filing(cik=cik, company="X", form=form, filed=day, path=f"{cik}{form}{day}")

    later = FILED + dt.timedelta(days=30)
    rows = (f(1, "10-12B", later), f(1, "10-12B/A", FILED), f(2, "10-12B", later))
    assert [(r.cik, r.filed) for r in first_filings(rows)] == [(1, FILED), (2, later)]


def test_the_index_sends_the_user_agent_and_caches_finished_quarters(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.add("10-12B", "Spinco Inc", 1001, FILED)
    clock = [NOW]  # 2026-07-01: 2026Q3 is the current quarter
    index = sec.index(tmp_path, clock)
    start = dt.date(2025, 4, 1)
    assert [r.cik for r in index.filings(start, NOW.date())] == [1001]
    assert sec.fetched() == [(2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2), (2026, 3)]
    assert {r.get_header("User-agent") for r in sec.requests} == {AGENT}
    # Within a day nothing is fetched again.
    sec.requests.clear()
    index.filings(start, NOW.date())
    assert sec.fetched() == []
    # A day later, only quarters fetched less than a week after they ended: 2026Q2
    # (fetched on 2026-07-01) and the current one.
    clock[0] = NOW + dt.timedelta(days=1)
    index.filings(start, clock[0].date())
    assert sec.fetched() == [(2026, 2), (2026, 3)]
    sec.requests.clear()
    clock[0] = NOW + dt.timedelta(days=9)
    index.filings(start, clock[0].date())
    assert sec.fetched() == [(2026, 2), (2026, 3)]  # 2026Q2 fetched on 07-02: not final yet
    sec.requests.clear()
    clock[0] = NOW + dt.timedelta(days=11)
    index.filings(start, clock[0].date())
    assert sec.fetched() == [(2026, 3)]  # 2026Q2 fetched on 07-10: final


def test_the_cache_keeps_only_the_10_12b_rows(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.add("10-12B", "Spinco Inc", 1001, FILED)
    sec.add("10-K", "Parent Corp", 1003, FILED)
    sec.index(tmp_path, NOW).filings(FILED, FILED)
    data = json.loads((tmp_path / "form10-2025Q2.json").read_text())
    assert [r["cik"] for r in data["filings"]] == [1001]


def test_a_missing_index_fails_unless_it_is_the_current_quarter(tmp_path: Path) -> None:
    sec = FakeSec()
    sec.missing = {(2026, 3)}
    assert sec.index(tmp_path, NOW).filings(dt.date(2026, 4, 1), NOW.date()) == ()
    assert not (tmp_path / "form10-2026Q3.json").exists()  # asked for again next time
    sec.missing = {(2026, 1)}
    with pytest.raises(SecError, match="HTTP 404"):
        sec.index(tmp_path, NOW).filings(dt.date(2026, 1, 1), NOW.date())


def test_the_index_refuses_a_day_after_today(tmp_path: Path) -> None:
    with pytest.raises(SecError, match="after today"):
        FakeSec().index(tmp_path, NOW).filings(FILED, NOW.date() + dt.timedelta(days=1))


# --------------------------------------------------------------------------- the user agent


def test_the_user_agent_comes_from_dotenv_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    monkeypatch.setenv("SEC_USER_AGENT", "From Environment env@example.invalid")
    env.write_text("ALPACA_PAPER_API_KEY=x\n")
    with pytest.raises(KeysError, match="SEC_USER_AGENT must be set") as exc:
        read_sec_user_agent(env)
    assert "env@example" not in str(exc.value)
    env.write_text("SEC_USER_AGENT=Name only\n")
    with pytest.raises(KeysError) as exc:
        read_sec_user_agent(env)
    assert "Name only" not in str(exc.value)
    env.write_text("SEC_USER_AGENT=Jane Doe jane@example.invalid\n")
    assert read_sec_user_agent(env) == "Jane Doe jane@example.invalid"
    with pytest.raises(KeysError):
        read_sec_user_agent(tmp_path / "missing.env")


def test_the_audit_refuses_to_run_without_a_user_agent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    env = tmp_path / ".env"
    env.write_text("")
    fetcher = AuditFetcher()
    code = corporate_actions.main(
        ["audit"], config, loader=_loader(config, fetcher), dotenv_path=env
    )
    assert code == 2
    assert "SEC_USER_AGENT must be set" in capsys.readouterr().err
    assert fetcher.ends == []  # nothing fetched from Alpaca either


# --------------------------------------------------------------------------- the review


def _reviewed(config: Path, *rows: str) -> None:
    (config / FORM10_REVIEWED_FILE).write_text(
        "cik,company,filed,outcome,parent,ex_date,note\n" + "".join(f"{r}\n" for r in rows)
    )


def _run(config: Path, sec: FakeSec) -> corporate_actions.AuditResult:
    return audit(config, _loader(config, AuditFetcher()), "SPY", sec.index(config.parent, NOW))


def _filed(sec: FakeSec, cik: int = 1001, day: dt.date = FILED) -> None:
    sec.add("10-12B", f"Spinco {cik}", cik, day)


def test_a_filing_with_no_review_fails_the_audit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    sec = FakeSec()
    _filed(sec)
    sec.add("10-12B/A", "Spinco 1001", 1001, FILED + dt.timedelta(days=40))
    result = _run(config, sec)
    assert [(f.cik, f.filed) for f in result.filings_open] == [(1001, FILED)]
    assert result.failed and result.open == ()


def test_the_filing_range_runs_from_a_year_before_the_data_start_through_today(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)  # data start 2025-10-01, holdout start 2026-04-01
    sec = FakeSec()
    _filed(sec, 1, dt.date(2024, 9, 30))  # before the range
    _filed(sec, 2, dt.date(2024, 10, 1))
    _filed(sec, 3, dt.date(2026, 6, 30))  # in the holdout: an index is not market data
    result = _run(config, sec)
    assert (result.filed_from, result.filed_to) == (dt.date(2024, 10, 1), NOW.date())
    assert [f.cik for f in result.filings] == [2, 3]


def test_a_listed_filing_must_name_an_event_in_spinoffs_csv_or_reviewed(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        spinoffs=(spinoff_row("AAA", DAY, cached_step=0.8),),
        reviewed=(f"BBB,{DAY},applied correctly by Alpaca; same-day split",),
    )
    sec = FakeSec()
    for cik in (1, 2, 3):
        _filed(sec, cik)
    _reviewed(
        config,
        f"1,Spinco 1,{FILED},listed,AAA,{DAY},spin-off of AAA",
        f"2,Spinco 2,{FILED},listed,BBB,{DAY},spin-off of BBB",
        f"3,Spinco 3,{FILED},listed,CCC,{DAY},spin-off of CCC",
    )
    result = _run(config, sec)
    assert result.filings_open == ()
    assert [r.cik for r in result.filings_unmatched] == [3]
    assert result.failed
    assert result.unused == ()  # BBB's reviewed row is used by filing 2


def test_none_and_pending_pass_and_pending_is_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    sec = FakeSec()
    _filed(sec, 1)
    _filed(sec, 2)
    _reviewed(
        config,
        f"1,Spinco 1,{FILED},none,,,parent not in any universe",
        f"2,Spinco 2,{FILED},pending,,,not yet distributed (2026-07-01)",
        f"9,Gone Co,{FILED},none,,,withdrawn",
    )
    code = corporate_actions.main(
        ["audit"],
        config,
        loader=_loader(config, AuditFetcher()),
        form10=sec.index(config.parent, NOW),
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "pending: Spinco 2 (CIK 2" in out and "not yet distributed" in out
    assert "form 10 row Gone Co (CIK 9) matches no filing" in out


def test_the_cli_writes_open_filings_as_review_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path)
    sec = FakeSec()
    _filed(sec)
    sheet = tmp_path / "form10.csv"
    code = corporate_actions.main(
        ["audit", "--form10-out", str(sheet)],
        config,
        loader=_loader(config, AuditFetcher()),
        form10=sec.index(config.parent, NOW),
    )
    assert code == 1
    assert f"{FILED} CIK 1001" in capsys.readouterr().err
    lines = sheet.read_text().splitlines()
    assert lines == [
        "cik,company,filed,outcome,parent,ex_date,note",
        f"1001,Spinco 1001,{FILED},,,,10-12B",
    ]
    # Filled in, the rows pass.
    _reviewed(config, f"1001,Spinco 1001,{FILED},none,,,parent not in any universe")
    assert (
        corporate_actions.main(
            ["audit"],
            config,
            loader=_loader(config, AuditFetcher()),
            form10=sec.index(config.parent, NOW),
        )
        == 0
    )


@pytest.mark.parametrize(
    "row",
    [
        f"1,X,{FILED},listed,,,no parent",
        f"1,X,{FILED},listed,AAA,,no ex-date",
        f"1,X,{FILED},none,AAA,{DAY},parent given",
        f"1,X,{FILED},pending,,{DAY},ex-date given",
        f"1,X,{FILED},maybe,,,unknown outcome",
        f"1,X,{FILED},none,,,",
        f"0,X,{FILED},none,,,no CIK",
    ],
)
def test_bad_review_rows_fail_to_load(tmp_path: Path, row: str) -> None:
    config = _config(tmp_path)
    _reviewed(config, row)
    with pytest.raises(ConfigError, match="line 2"):
        load_form10_reviewed(config)


def test_a_cik_reviewed_twice_fails_to_load(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _reviewed(config, f"1,X,{FILED},none,,,a", f"1,X,{FILED},none,,,b")
    with pytest.raises(ConfigError, match="CIK 1 is listed twice"):
        load_form10_reviewed(config)


def test_every_shipped_form10_row_loads_and_names_a_known_event() -> None:
    rows = load_form10_reviewed(REPO_CONFIG)
    events = {(e.parent, e.ex_date) for e in corporate_actions.read_spinoffs(
        REPO_CONFIG / corporate_actions.SPINOFFS_FILE
    )} | set(corporate_actions.load_reviewed(REPO_CONFIG))  # fmt: skip
    listed = [r for r in rows.values() if r.outcome == "listed"]
    assert all((r.parent, r.ex_date) in events for r in listed)
