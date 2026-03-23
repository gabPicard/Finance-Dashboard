"""Strategy registry — maps strategy names to their class objects.

Usage
-----
    from src.registry import registry

    # Retrieve a class by name
    MarkClass = registry.get("Markowitz")

    # Register a custom strategy
    @registry.register_decorator
    class MyStrategy(AllocationStrategy):
        ...

    # Or imperatively
    registry.register("Custom", MyStrategy)

    # List all available strategies
    print(registry.list_strategies())
"""

from __future__ import annotations

from typing import Type


class StrategyRegistry:
    """Central registry mapping strategy names to their class objects.

    Attributes
    ----------
    _strategies : dict[str, type]
        Class-level mapping of name → strategy class.
    """

    _strategies: dict[str, type] = {}

    # ------------------------------------------------------------------

    @classmethod
    def register(cls, name: str, strategy_class: type) -> None:
        """Register a strategy class under *name*.

        Parameters
        ----------
        name:
            Human-readable identifier (e.g. ``"Markowitz"``).
        strategy_class:
            The strategy class itself (not an instance).
        """
        cls._strategies[name] = strategy_class

    @classmethod
    def get(cls, name: str) -> type:
        """Retrieve a strategy class by name.

        Parameters
        ----------
        name:
            Strategy name.

        Returns
        -------
        type
            The registered strategy class.

        Raises
        ------
        KeyError
            When *name* is not found in the registry.
        """
        if name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise KeyError(
                f"Strategy {name!r} not found. Available: {available}"
            )
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return a sorted list of registered strategy names.

        Returns
        -------
        list[str]
            Sorted strategy names.
        """
        return sorted(cls._strategies.keys())

    @classmethod
    def register_decorator(cls, strategy_class: type) -> type:
        """Class decorator that registers the class using its ``__name__``.

        Usage::

            @registry.register_decorator
            class MyStrategy(AllocationStrategy):
                ...

        Parameters
        ----------
        strategy_class:
            Class to register.

        Returns
        -------
        type
            The same class, unmodified.
        """
        cls.register(strategy_class.__name__, strategy_class)
        return strategy_class


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
registry = StrategyRegistry()

# Pre-register built-in strategies
from .strategies.Markowitz import MarkowitzStrategy  # noqa: E402
from .strategies.CAPM import CAPMStrategy  # noqa: E402
from .strategies.HRP import HRPStrategy  # noqa: E402
from .strategies.PairTrading import PairTradingStrategy  # noqa: E402

registry.register("Markowitz", MarkowitzStrategy)
registry.register("CAPM", CAPMStrategy)
registry.register("HRP", HRPStrategy)
registry.register("PairTrading", PairTradingStrategy)
