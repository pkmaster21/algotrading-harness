"""Read API keys from `.env` without exporting them.

Keys are parsed from the file into a local object and handed to the one client that
needs them. They are never written to `os.environ`, so nothing else in the process
(strategy code included) can read them from the environment, and they are never taken
from the environment either: `.env` is the only source.

Only the data loader, the broker adapter and the spin-off audit (for the SEC contact,
`read_sec_user_agent`) call this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PAPER_KEY_ID = "ALPACA_PAPER_API_KEY"
PAPER_SECRET_KEY = "ALPACA_PAPER_SECRET_KEY"
SEC_USER_AGENT = "SEC_USER_AGENT"


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


def read_sec_user_agent(dotenv_path: Path) -> str:
    """The User-Agent the SEC asks automated clients to send: a name and a contact email
    (https://www.sec.gov/os/accessing-edgar-data). It is the owner's to set in `.env`;
    nothing in the repo holds one.

    Raises KeysError if it is missing, has no `@`, or holds a line break. The message never
    contains the value.
    """
    values = dotenv_values(dotenv_path, interpolate=False)
    agent = (values.get(SEC_USER_AGENT) or "").strip()
    if not agent or "@" not in agent or any(ch in agent for ch in "\r\n"):
        raise KeysError(
            f"{SEC_USER_AGENT} must be set in {dotenv_path.name} to a name and a contact "
            "email, as the SEC asks of automated clients"
        )
    return agent
