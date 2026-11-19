"""Point-in-time universe membership: loading, renames, validation (design PR 1)."""

import csv
import datetime as dt
import hashlib
import shutil
from pathlib import Path

import pytest

from harness.config import (
    ConfigError,
    Membership,
    Recipe,
    Rename,
    apply_renames,
    load_recipe,
)
from harness.run import run_backtest

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"
STRATEGY_PRESET = "strategy: buy_and_hold\nparams: {}\n"
D = dt.date


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    dest = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, dest)
    (dest / "strategies").mkdir(exist_ok=True)
    (dest / "strategies" / "test-preset.yaml").write_text(STRATEGY_PRESET)
    return dest


def write_universe(
    config_dir: Path,
    membership: str,
    renames: str = "",
    reviewed: str = "[]",
    corrections: str = "",
    bad_bars: str = "",
) -> None:
    u = config_dir / "universe"
    (u / "t-members.csv").write_text("ticker,start_date,end_date\n" + membership)
    (u / "t-corrections.csv").write_text(
        "ticker,start_date,end_date,new_start_date,new_end_date,reason\n" + corrections
    )
    (u / "t-renames.csv").write_text("old,new,date,note\n" + renames)
    (u / "t-bad.csv").write_text("ticker,from_date,to_date,reason\n" + bad_bars)
    (u / "t.yaml").write_text(MEMBERSHIP_YAML + f"  reviewed_multi_interval: {reviewed}\n")


MEMBERSHIP_YAML = (
    "membership:\n  file: t-members.csv\n  corrections: t-corrections.csv\n"
    "  renames: t-renames.csv\n  bad_bars: t-bad.csv\n"
)


def load(config_dir: Path, universe: str = "t") -> Recipe:
    return load_recipe("test-preset", universe, config_dir)


def intervals(recipe: Recipe) -> list[tuple[str, dt.date, dt.date | None]]:
    m = recipe.universe.membership
    assert m is not None
    return [(iv.symbol, iv.start, iv.end) for iv in m.intervals]


# ------------------------------------------------------------------ the repo's sp500 preset


def test_repo_sp500_loads_with_renames_applied(config_copy: Path) -> None:
    recipe = load(config_copy, "sp500")
    m = recipe.universe.membership
    assert m is not None
    symbols = set(recipe.universe.symbols)
    # Renamed-away tickers are gone; each company is one interval under its last ticker.
    assert not symbols & {"FB", "FI", "DWDP", "BBT", "HRS", "UTX", "ANTM"}
    by_symbol = {s: [iv for iv in m.intervals if iv.symbol == s] for s in ("META", "FISV", "TFC")}
    assert [(iv.start, iv.end) for iv in by_symbol["META"]] == [(D(2013, 12, 23), None)]
    assert [(iv.start, iv.end) for iv in by_symbol["FISV"]] == [(D(2001, 4, 2), None)]
    assert [(iv.start, iv.end) for iv in by_symbol["TFC"]] == [(D(1997, 12, 4), None)]
    # Symantec and HCP chains reach back to their first join; KORS folds into CPRI.
    assert not symbols & {"SYMC", "NLOK", "HCP", "PEAK", "KORS"}
    assert ("GEN", D(2003, 3, 31), None) in intervals(recipe)
    assert ("DOC", D(2008, 3, 31), None) in intervals(recipe)
    assert ("CPRI", D(2013, 11, 13), D(2020, 5, 12)) in intervals(recipe)
    assert {b.symbol for b in m.bad_bars} == {"ARNC", "CTRA", "DOW", "EXE", "PARA", "SMCI"}
    # Renames after the in-sample end are applied too (paper trading needs them).
    assert not symbols & {"BK", "SATS", "EQR"}
    assert {"BNY", "ECHO", "VMRK"} <= symbols
    # A removal stays a removal; a reused ticker keeps both of its intervals.
    assert ("SIVB", D(2018, 3, 19), D(2023, 3, 15)) in intervals(recipe)
    assert [iv.end for iv in m.intervals if iv.symbol == "SNDK"] == [D(2016, 5, 12), None]
    # Nothing that ended before the data start is kept.
    assert all(iv.end is None or iv.end > recipe.dates.data_start for iv in m.intervals)


def test_repo_sp500_pins_the_source_files(config_copy: Path) -> None:
    m = load(config_copy, "sp500").universe.membership
    assert m is not None
    members = (REPO_CONFIG / "universe" / "sp500-membership.csv").read_bytes()
    renames = (REPO_CONFIG / "universe" / "sp500-renames.csv").read_bytes()
    assert m.file.sha256 == hashlib.sha256(members).hexdigest()
    assert m.renames_file.sha256 == hashlib.sha256(renames).hexdigest()
    # The pinned copy is the upstream file named in sp500-membership.md.
    readme = (REPO_CONFIG / "universe" / "sp500-membership.md").read_text()
    assert m.file.sha256 in readme


def test_recipe_round_trips_through_its_snapshot(config_copy: Path) -> None:
    recipe = load(config_copy, "sp500")
    assert Recipe.model_validate(recipe.model_dump(mode="json")) == recipe


def test_backtests_refuse_a_membership_universe_until_enforced(config_copy: Path) -> None:
    with pytest.raises(ConfigError, match="point-in-time"):
        run_backtest(load(config_copy, "sp500"))


def test_static_universes_are_unchanged(config_copy: Path) -> None:
    recipe = load(config_copy, "plumbing-10")
    assert recipe.universe.membership is None
    assert recipe.universe.symbols[0] == "SPY"


# ------------------------------------------------------------------ renames


def test_rename_chain_joins_into_the_last_ticker() -> None:
    raw = [
        ("AAA", D(2010, 1, 1), D(2018, 1, 1)),
        ("BBB", D(2018, 1, 1), D(2021, 1, 1)),
        ("AAA", D(2021, 1, 1), None),
        ("CCC", D(2012, 1, 1), None),
    ]
    renames = [
        Rename(old="AAA", new="BBB", date=D(2018, 1, 1)),
        Rename(old="BBB", new="AAA", date=D(2021, 1, 1)),
    ]
    assert apply_renames(raw, renames) == [
        ("AAA", D(2010, 1, 1), None),
        ("CCC", D(2012, 1, 1), None),
    ]


def test_rename_links_only_the_matching_interval() -> None:
    # An older, unrelated AAA interval is left alone.
    raw = [
        ("AAA", D(2000, 1, 1), D(2005, 1, 1)),
        ("AAA", D(2010, 1, 1), D(2018, 1, 1)),
        ("BBB", D(2018, 1, 1), None),
    ]
    joined = apply_renames(raw, [Rename(old="AAA", new="BBB", date=D(2018, 1, 1))])
    assert joined == [("AAA", D(2000, 1, 1), D(2005, 1, 1)), ("BBB", D(2010, 1, 1), None)]


@pytest.mark.parametrize(
    "membership",
    [
        "AAA,2016-01-04,2018-01-02\nBBB,2018-01-03,\n",  # dates do not meet
        "AAA,2016-01-04,2018-01-02\n",  # no new interval
    ],
    ids=["gap", "missing new"],
)
def test_rename_that_matches_nothing_fails(config_copy: Path, membership: str) -> None:
    write_universe(config_copy, membership, "AAA,BBB,2018-01-02,test\n")
    with pytest.raises(ConfigError, match="rename AAA -> BBB"):
        load(config_copy)


def test_loaded_renames_are_recorded(config_copy: Path) -> None:
    write_universe(
        config_copy, "AAA,2016-01-04,2018-01-02\nBBB,2018-01-02,\n", "AAA,BBB,2018-01-02,x\n"
    )
    recipe = load(config_copy)
    assert intervals(recipe) == [("BBB", D(2016, 1, 4), None)]
    m = recipe.universe.membership
    assert m is not None and m.renames == (Rename(old="AAA", new="BBB", date=D(2018, 1, 2)),)


# ------------------------------------------------------------------ validation


def test_intervals_ended_before_the_data_start_are_dropped(config_copy: Path) -> None:
    # data_start is 2015-01-01 in the repo config; an odd old ticker is never validated.
    write_universe(
        config_copy, "old-1,1999-01-04,2010-01-04\nAAA,2014-01-02,2015-01-01\nBBB,2014-01-02,\n"
    )
    recipe = load(config_copy)
    assert recipe.universe.symbols == ("BBB",)


def test_overlapping_intervals_fail(config_copy: Path) -> None:
    write_universe(config_copy, "AAA,2016-01-04,2019-01-02\nAAA,2018-01-02,\n", reviewed="[AAA]")
    with pytest.raises(ConfigError, match="overlap"):
        load(config_copy)


def test_unreviewed_multi_interval_ticker_fails(config_copy: Path) -> None:
    write_universe(config_copy, "AAA,2016-01-04,2018-01-02\nAAA,2020-01-02,\n")
    with pytest.raises(ConfigError, match=r"need a review.*\['AAA'\]"):
        load(config_copy)
    write_universe(config_copy, "AAA,2016-01-04,2018-01-02\nAAA,2020-01-02,\n", reviewed="[AAA]")
    assert load(config_copy).universe.symbols == ("AAA",)


def test_reviewed_list_naming_a_single_interval_ticker_fails(config_copy: Path) -> None:
    write_universe(config_copy, "AAA,2016-01-04,\n", reviewed="[AAA]")
    with pytest.raises(ConfigError, match="single-interval"):
        load(config_copy)


@pytest.mark.parametrize(
    ("membership", "match"),
    [
        ("AAA,2016-01-04,2016-01-04\n", "not before end"),
        ("AAA,2016-13-01,\n", "not a YYYY-MM-DD date"),
        ("AAA,2016-01-04\n", "expected 3 fields"),
        ("aaa,2016-01-04,\n", "symbol"),
    ],
    ids=["empty interval", "bad date", "short row", "bad ticker"],
)
def test_bad_rows_fail(config_copy: Path, membership: str, match: str) -> None:
    write_universe(config_copy, membership)
    with pytest.raises(ConfigError, match=match):
        load(config_copy)


def test_bad_header_fails(config_copy: Path) -> None:
    write_universe(config_copy, "AAA,2016-01-04,\n")
    (config_copy / "universe" / "t-members.csv").write_text("symbol,start,end\nAAA,2016-01-04,\n")
    with pytest.raises(ConfigError, match="header"):
        load(config_copy)


@pytest.mark.parametrize(
    "yaml_text",
    [
        "symbols: [SPY]\n" + MEMBERSHIP_YAML + "  reviewed_multi_interval: []\n",
        MEMBERSHIP_YAML,
        MEMBERSHIP_YAML.replace("t-members.csv", "../t-members.csv")
        + "  reviewed_multi_interval: []\n",
    ],
    ids=["also symbols", "missing key", "path outside universe/"],
)
def test_malformed_membership_preset_fails(config_copy: Path, yaml_text: str) -> None:
    write_universe(config_copy, "AAA,2016-01-04,\n")
    (config_copy / "universe" / "t.yaml").write_text(yaml_text)
    with pytest.raises(ConfigError):
        load(config_copy)


def membership_fields(**overrides: object) -> dict[str, object]:
    source = {"name": "f.csv", "sha256": "0"}
    fields: dict[str, object] = {
        "file": source,
        "corrections_file": source,
        "renames_file": source,
        "bad_bars_file": source,
        "corrections": [],
        "renames": [],
        "bad_bars": [],
        "reviewed_multi_interval": [],
        "intervals": [{"symbol": "AAA", "start": D(2016, 1, 4), "end": None}],
    }
    return {**fields, **overrides}


def test_membership_must_be_sorted() -> None:
    unsorted = [
        {"symbol": "BBB", "start": D(2016, 1, 4), "end": None},
        {"symbol": "AAA", "start": D(2016, 1, 4), "end": None},
    ]
    with pytest.raises(ValueError, match="sorted"):
        Membership.model_validate(membership_fields(intervals=unsorted))


# ------------------------------------------------------------------ corrections


def test_correction_replaces_and_drops_upstream_rows(config_copy: Path) -> None:
    write_universe(
        config_copy,
        "OLD,2013-11-13,2018-09-19\nNEW,2016-01-04,2020-05-12\n",
        corrections="OLD,2013-11-13,2018-09-19,,,duplicate\n"
        "NEW,2016-01-04,2020-05-12,2013-11-13,2020-05-12,wrong start\n",
    )
    recipe = load(config_copy)
    assert intervals(recipe) == [("NEW", D(2013, 11, 13), D(2020, 5, 12))]
    m = recipe.universe.membership
    assert m is not None and [c.symbol for c in m.corrections] == ["OLD", "NEW"]


def test_correction_that_matches_no_row_fails(config_copy: Path) -> None:
    write_universe(
        config_copy, "AAA,2016-01-04,\n", corrections="AAA,2016-01-05,,,,typo in the start\n"
    )
    with pytest.raises(ConfigError, match="expected one matching row"):
        load(config_copy)


def test_corrections_apply_before_renames(config_copy: Path) -> None:
    # Upstream ends AAA a day early; the correction makes the rename line up.
    write_universe(
        config_copy,
        "AAA,2016-01-04,2018-01-01\nBBB,2018-01-02,\n",
        renames="AAA,BBB,2018-01-02,x\n",
        corrections="AAA,2016-01-04,2018-01-01,2016-01-04,2018-01-02,end off by one\n",
    )
    assert intervals(load(config_copy)) == [("BBB", D(2016, 1, 4), None)]


# ------------------------------------------------------------------ bad bars


def test_bad_bars_are_recorded(config_copy: Path) -> None:
    write_universe(config_copy, "AAA,2016-01-04,\n", bad_bars="AAA,,2018-01-02,another security\n")
    m = load(config_copy).universe.membership
    assert m is not None
    (bad,) = m.bad_bars
    assert (bad.symbol, bad.start, bad.end) == ("AAA", None, D(2018, 1, 2))


@pytest.mark.parametrize(
    ("bad_bars", "match"),
    [
        ("ZZZ,,,not a member\n", "not members"),
        ("AAA,2018-01-02,2018-01-02,empty\n", "not before end"),
        ("AAA,,,\n", "reason"),
    ],
    ids=["unknown ticker", "empty window", "no reason"],
)
def test_bad_bars_rows_are_validated(config_copy: Path, bad_bars: str, match: str) -> None:
    write_universe(config_copy, "AAA,2016-01-04,\n", bad_bars=bad_bars)
    with pytest.raises(ConfigError, match=match):
        load(config_copy)


# ------------------------------------------------------------------ the same-day review


def test_same_day_review_matches_the_rename_map() -> None:
    """Every same-day pair has a reason, and the pairs marked renamed are exactly the
    renames in the data range."""
    u = REPO_CONFIG / "universe"
    with (u / "sp500-same-day-review.csv").open(newline="") as fh:
        review = list(csv.DictReader(fh))
    with (u / "sp500-renames.csv").open(newline="") as fh:
        renames = {(r["old"], r["new"], r["date"]) for r in csv.DictReader(fh)}
    assert all(r["reason"] for r in review)
    marked = {(r["removed"], r["added"], r["date"]) for r in review if r["renamed"] == "yes"}
    assert marked == renames
