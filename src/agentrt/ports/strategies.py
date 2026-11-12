# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy registry port.

The signal / backtest tools and the planner need named strategies with the
BaseStrategy shape (``params``, ``preprocess_data``, ``generate_signals``,
``run``). By default this port binds to ``quantrisk.strategies`` when the
optional dependency quantrisk-core is installed; otherwise no strategy names
are offered and asking for one raises ``ValueError``, exactly as an unknown
name would.

    from agentrt.ports import strategies
    strategies.configure(my_provider)   # object with get_strategy(name, params=None) and names()
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol


class StrategyProvider(Protocol):
    def get_strategy(self, name: str, params: Optional[dict] = None) -> Any: ...
    def names(self) -> Iterable[str]: ...


class _QuantRiskProvider:
    def __init__(self, module: Any):
        self._m = module

    def get_strategy(self, name: str, params: Optional[dict] = None) -> Any:
        return self._m.get_strategy(name, params)

    def names(self) -> Iterable[str]:
        return [entry["name"] for entry in self._m.list_strategies()]


_provider: Optional[StrategyProvider] = None
_default_resolved = False
_default: Optional[StrategyProvider] = None


def configure(provider: Optional[StrategyProvider]) -> None:
    global _provider
    _provider = provider


def _resolve() -> Optional[StrategyProvider]:
    global _default_resolved, _default
    if _provider is not None:
        return _provider
    if not _default_resolved:
        _default_resolved = True
        try:
            import quantrisk.strategies as module  # optional dependency
        except ImportError:
            _default = None
        else:
            _default = _QuantRiskProvider(module)
    return _default


def get_strategy(name: str, params: Optional[dict] = None) -> Any:
    provider = _resolve()
    if provider is None:
        raise ValueError(
            f"unknown strategy '{name}': no strategy provider is configured "
            "(install agent-runtime-kit[quantrisk] or call agentrt.ports.strategies.configure)")
    return provider.get_strategy(name, params)


def strategy_names() -> list:
    provider = _resolve()
    return sorted(provider.names()) if provider is not None else []
