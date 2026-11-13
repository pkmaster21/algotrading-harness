"""Read API keys from `.env` without exporting them.

Keys are parsed from the file into a local object and handed to the one client that
needs them. They are never written to `os.environ`, so nothing else in the process
(strategy code included) can read them from the environment, and they are never taken
from the environment either: `.env` is the only source.

Only the data loader and the broker adapter call this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PAPER_KEY_ID = "ALPACA_PAPER_API_KEY"
PAPER_SECRET_KEY = "ALPACA_PAPER_SECRET_KEY"


class KeysError(RuntimeError):
    """The paper keys are missing from `.env`. The message never contains key material."""


@dataclass(frozen=True, slots=True, repr=False)
class AlpacaPaperKeys:
    """The Alpaca paper key pair. Its repr is redacted, so it cannot leak through a log
    line, an error message, or a traceback that prints local variables."""

    key_id: str
    secret_key: str

    def __repr__(self) -> str:
        return "AlpacaPaperKeys(<redacted>)"


def read_alpaca_paper_keys(dotenv_path: Path) -> AlpacaPaperKeys:
    """Parse `dotenv_path` and return the paper key pair.

    Raises KeysError if the file is missing or either value is empty. Does not read or
    modify `os.environ`, and does not expand `${VAR}` references from it.
    """
    values = dotenv_values(dotenv_path, interpolate=False)
    key_id = (values.get(PAPER_KEY_ID) or "").strip()
    secret_key = (values.get(PAPER_SECRET_KEY) or "").strip()
    if not key_id or not secret_key:
        raise KeysError(f"{PAPER_KEY_ID} and {PAPER_SECRET_KEY} must be set in {dotenv_path.name}")
    return AlpacaPaperKeys(key_id=key_id, secret_key=secret_key)
