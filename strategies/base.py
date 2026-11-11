"""The interface every strategy implements.

A strategy turns market history into Signals. It never places orders, never reads files,
environment variables, or the network, and never imports broker or data modules: the run
entry point rejects strategy modules whose imports fall outside an allowlist.

Each strategy declares its parameters as a pydantic `Params` class (fields with types,
ranges, and defaults). The harness validates a preset's params against it and passes the
instance to the constructor; the UI later reads the same schema to build forms.

During holdout warm-up the engine fills the view without calling `on_bar`, so compute
indicators from `view.history`, not from state accumulated across `on_bar` calls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from harness.models import FrozenModel, PortfolioSnapshot, Signal
from harness.view import MarketView


class Strategy(ABC):
    """Base class for strategies. Subclasses override `Params` and implement `on_bar`."""

    class Params(FrozenModel):
        """No parameters. Subclasses replace this with their own frozen model."""

    #: Name used in reports. Defaults to the module name (e.g. "example_buy_and_hold").
    name: ClassVar[str] = ""

    def __init__(self, params: FrozenModel | None = None) -> None:
        params_cls = type(self).Params
        if params is None:
            params = params_cls()
        if not isinstance(params, params_cls):
            raise TypeError(
                f"{type(self).__name__} expects {params_cls.__qualname__}, "
                f"got {type(params).__qualname__}"
            )
        self.params = params

    @classmethod
    def from_params(cls, raw: Mapping[str, Any]) -> Self:
        """Validate raw preset params against `Params` and build the strategy."""
        return cls(cls.Params.model_validate(dict(raw)))

    @classmethod
    def strategy_name(cls) -> str:
        return cls.name or cls.__module__.rsplit(".", 1)[-1]

    def on_start(self, symbols: tuple[str, ...]) -> None:  # noqa: B027  (optional hook)
        """Called once before the first session with the universe symbols."""

    @abstractmethod
    def on_bar(self, view: MarketView, portfolio: PortfolioSnapshot) -> Sequence[Signal]:
        """Return this session's signals (possibly empty).

        `view` holds bars up to and including today; `portfolio` is cash, equity, and
        positions at today's close. Orders from these signals fill at the next open.
        """
