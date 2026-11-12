# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-driven backtest engine port (used only by the legacy BacktestAgent).

The original engine and its data-source manager were part of the host system
and are not included. Without a configured engine the agent fails the task with
a clear error instead of pretending to have run. The Agent Runtime's own
``backtest`` tool does not use this port; it runs ``BaseStrategy.run`` from the
strategy port.

    from agentrt.ports import backtest
    backtest.configure(engine_factory=..., monte_carlo_factory=..., history_fetcher=...)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, List, Optional

from . import PortNotConfigured

_engine_factory: Optional[Callable[[Dict[str, Any]], Any]] = None
_monte_carlo_factory: Optional[Callable[[int, List[float]], Any]] = None
_history_fetcher: Optional[Callable[[str, str, str], Awaitable[Any]]] = None


def configure(engine_factory=None, monte_carlo_factory=None, history_fetcher=None) -> None:
    global _engine_factory, _monte_carlo_factory, _history_fetcher
    _engine_factory, _monte_carlo_factory, _history_fetcher = engine_factory, monte_carlo_factory, history_fetcher


def make_engine(config: Dict[str, Any]) -> Any:
    if _engine_factory is None:
        raise PortNotConfigured("no backtest engine configured; call agentrt.ports.backtest.configure(engine_factory=...) "
                                "or use the runtime backtest tool (agentrt.tools.BacktestTool), which needs no engine")
    return _engine_factory(config)


def make_monte_carlo(n_simulations: int, confidence_levels: List[float]) -> Any:
    if _monte_carlo_factory is None:
        raise PortNotConfigured("no Monte Carlo simulator configured; call agentrt.ports.backtest.configure(monte_carlo_factory=...)")
    return _monte_carlo_factory(n_simulations, confidence_levels)


async def fetch_history(symbol: str, start_date: str, end_date: str) -> Any:
    if _history_fetcher is None:
        raise PortNotConfigured("no history fetcher configured; call agentrt.ports.backtest.configure(history_fetcher=...)")
    return await _history_fetcher(symbol, start_date, end_date)
