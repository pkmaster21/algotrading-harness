"""Key handling: keys come from the .env file only and never reach the environment."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from harness.keys import AlpacaPaperKeys, KeysError, read_alpaca_paper_keys

KEY_ID = "PKSENTINELKEYID"
SECRET = "sentinel-secret-value"


def _env_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_reads_both_keys_from_the_file(tmp_path: Path) -> None:
    path = _env_file(
        tmp_path,
        f"# comment\nALPACA_PAPER_API_KEY={KEY_ID}\nALPACA_PAPER_SECRET_KEY= {SECRET} \n",
    )
    keys = read_alpaca_paper_keys(path)
    assert (keys.key_id, keys.secret_key) == (KEY_ID, SECRET)


def test_environment_is_not_modified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_PAPER_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET_KEY", raising=False)
    path = _env_file(tmp_path, f"ALPACA_PAPER_API_KEY={KEY_ID}\nALPACA_PAPER_SECRET_KEY={SECRET}\n")
    before = dict(os.environ)
    read_alpaca_paper_keys(path)
    assert dict(os.environ) == before


def test_environment_is_not_a_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Keys exported in the shell are ignored: .env is the only source.
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", KEY_ID)
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", SECRET)
    with pytest.raises(KeysError):
        read_alpaca_paper_keys(tmp_path / "missing.env")
    with pytest.raises(KeysError):
        read_alpaca_paper_keys(_env_file(tmp_path, "OTHER=1\n"))


def test_environment_references_are_not_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME_SECRET", "from-environment")
    path = _env_file(
        tmp_path, "ALPACA_PAPER_API_KEY=${HOME_SECRET}\nALPACA_PAPER_SECRET_KEY=${HOME_SECRET}\n"
    )
    keys = read_alpaca_paper_keys(path)
    assert keys.key_id == "${HOME_SECRET}"


@pytest.mark.parametrize(
    "text",
    [
        "",
        f"ALPACA_PAPER_API_KEY={KEY_ID}\n",
        f"ALPACA_PAPER_SECRET_KEY={SECRET}\n",
        f"ALPACA_PAPER_API_KEY=\nALPACA_PAPER_SECRET_KEY={SECRET}\n",
        f"ALPACA_PAPER_API_KEY={KEY_ID}\nALPACA_PAPER_SECRET_KEY=   \n",
    ],
)
def test_missing_or_empty_keys_raise_without_leaking(tmp_path: Path, text: str) -> None:
    with pytest.raises(KeysError) as excinfo:
        read_alpaca_paper_keys(_env_file(tmp_path, text))
    message = str(excinfo.value)
    assert "ALPACA_PAPER_API_KEY" in message
    assert KEY_ID not in message and SECRET not in message


def test_repr_and_str_are_redacted() -> None:
    keys = AlpacaPaperKeys(key_id=KEY_ID, secret_key=SECRET)
    for text in (repr(keys), str(keys), f"{keys}", repr([keys])):
        assert KEY_ID not in text and SECRET not in text


def test_keys_are_immutable() -> None:
    keys = AlpacaPaperKeys(key_id=KEY_ID, secret_key=SECRET)
    with pytest.raises(AttributeError):
        keys.key_id = "other"  # type: ignore[misc]
