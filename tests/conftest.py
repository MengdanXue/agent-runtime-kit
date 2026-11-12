# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for agent tests."""

import pytest


class _StubAnalytics:
    """A PortfolioAnalyticsService stand-in returning real-shaped, fixed payloads.

    RiskAgent must never invent numbers, so its tests assert that it faithfully
    relays what the analytics service computed - including failures.
    """

    def __init__(self, overrides=None):
        self.calls = []
        self.overrides = overrides or {}

    def _payload(self, name, default):
        self.calls.append(name)
        return self.overrides.get(name, default)

    async def generate_portfolio_static(self, portfolio_id, window="1y"):
        return self._payload("portfolio_static", {
            "sharpe_ratio": 1.25,
            "volatility": 0.18,
            "max_drawdown": 0.22,
            "cumulative_return": 0.31,
        })

    async def calculate_portfolio_volatility(self, portfolio_id, **kwargs):
        return self._payload("volatility", {
            "portfolio_volatility": 0.011,
            "annualized_volatility": 0.18,
        })

    async def calculate_var(self, portfolio_id, confidence_level=0.05, **kwargs):
        # A 99% VaR (significance 0.01) is the larger loss of the two.
        key = "var_99" if confidence_level <= 0.01 else "var_95"
        amount = 8000.0 if key == "var_99" else 5000.0
        return self._payload(key, {
            "var_amount": amount,
            "var_absolute": amount,
            "var_pct": amount / 1000.0,
            "equity_var": amount,
            "equity_var_pct": amount / 1000.0,
            "abs_portfolio_value": 100000.0,
            "net_portfolio_value": 100000.0,
            "equity": 100000.0,
            "window": kwargs.get("window", "1y"),
            "var_window": 22,
            "sample_size": 240,
        })

    async def calculate_cvar(self, portfolio_id, confidence_level=0.05, **kwargs):
        return self._payload("cvar", {
            "cvar_amount": 6500.0,
            "cvar_pct": 6.5,
            "cvar_var_ratio": 1.3,
        })

    async def calculate_beta(self, portfolio_id, benchmark_symbol="SPY", window="1y"):
        return self._payload("beta", {
            "beta": 1.1,
            "alpha": 0.02,
            "correlation": 0.87,
        })

    async def calculate_marginal_risk_contribution(self, portfolio_id, **kwargs):
        return self._payload("risk_contribution", {
            "portfolio_volatility": 0.011,
            "positions": [
                {"symbol": "AAPL", "weight": 0.6, "mrc": 0.012, "risk_contribution_pct": 65.0},
                {"symbol": "MSFT", "weight": 0.4, "mrc": 0.009, "risk_contribution_pct": 35.0},
            ],
            "concentration_metrics": {"herfindahl_index": 0.52, "effective_n": 1.9},
        })

    async def calculate_correlation_matrix(self, portfolio_id, window="1y"):
        return self._payload("correlation", {
            "symbols": ["AAPL", "MSFT"],
            "correlation_matrix": [[1.0, 0.62], [0.62, 1.0]],
            "window": window,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        })


@pytest.fixture
def stub_analytics(monkeypatch):
    """Bind a RiskAgent to a stub analytics service instead of a live database."""
    from contextlib import asynccontextmanager

    def _install(agent, overrides=None):
        stub = _StubAnalytics(overrides)

        @asynccontextmanager
        async def _analytics():
            yield stub

        monkeypatch.setattr(agent, "_analytics", _analytics)
        return stub

    return _install

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_backtest_engine: exercises BacktestAgent against a host-provided event-driven engine "
        "(agentrt.ports.backtest); skipped when none is configured")
    config.addinivalue_line(
        "markers",
        "needs_strategies: uses named strategies from agentrt.ports.strategies; skipped when the port "
        "offers none (install quantrisk-core or configure a provider)")


def pytest_collection_modifyitems(config, items):
    from agentrt.ports import backtest as backtest_port
    from agentrt.ports.strategies import strategy_names

    no_engine = pytest.mark.skip(reason="no backtest engine configured (agentrt.ports.backtest); "
                                        "the host's event-driven engine is not part of agent-runtime-kit")
    no_strategies = pytest.mark.skip(reason="agentrt.ports.strategies offers no strategies "
                                            "(install quantrisk-core or configure a provider)")
    engine_missing = backtest_port._engine_factory is None
    strategies_missing = not strategy_names()
    for item in items:
        if engine_missing and "needs_backtest_engine" in item.keywords:
            item.add_marker(no_engine)
        if strategies_missing and "needs_strategies" in item.keywords:
            item.add_marker(no_strategies)
