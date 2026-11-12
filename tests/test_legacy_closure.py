# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy agent paths: REAL DATA ONLY by default.

The Agent Runtime (tool layer, loop, workflow) is the product path. The legacy
quant/orchestrator agents stay, but none of them may serve random numbers,
placeholder strategies or simulated fills unless ALLOW_MOCK_AGENT_DATA=true.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from agentrt.base import AgentStatus, AgentTask
from agentrt.quant.backtest_agent import BacktestAgent
from agentrt.quant.factor_agent import FactorAgent

AGENTS = pathlib.Path(__file__).resolve().parents[1] / "src" / "agentrt"


def _draws_random(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if not isinstance(n, ast.Attribute):
            continue
        if isinstance(n.value, ast.Name) and n.value.id == "random":
            return True  # random.uniform(...), random.choice(...)
        if (isinstance(n.value, ast.Attribute) and n.value.attr == "random"
                and isinstance(n.value.value, ast.Name) and n.value.value.id in ("np", "numpy")):
            return True  # np.random.normal(...)
    return False


def _calls_guard(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call)
               and getattr(n.func, "id", getattr(n.func, "attr", None)) == "require_real_data"
               for n in ast.walk(node))


def test_every_random_draw_in_the_agents_package_is_behind_the_mock_guard():
    offenders, guarded = [], []
    for path in sorted(AGENTS.rglob("*.py")):
        if "browser" in path.parts:  # browser automation, no market data
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _draws_random(node):
                where = f"{path.relative_to(AGENTS)}::{node.name}"
                (guarded if _calls_guard(node) else offenders).append(where)
    assert guarded, "scanner found nothing - check the path"
    assert offenders == [], f"random data without require_real_data: {offenders}"


@pytest.mark.needs_strategies
def test_backtest_agent_no_longer_substitutes_a_placeholder_strategy():
    agent = BacktestAgent()
    with pytest.raises(ValueError, match="cannot run in BacktestEngine"):
        agent._get_strategy_function("momentum", {})
    with pytest.raises(ValueError, match="Unknown strategy"):
        agent._get_strategy_function("no_such_strategy", {})

    def custom(event, portfolio):
        return None

    assert agent._get_strategy_function(custom, {}) is custom


@pytest.mark.asyncio
async def test_backtest_agent_reports_the_failure_instead_of_a_zero_trade_real_result():
    days = pd.date_range("2026-01-01", periods=60, freq="B")
    closes = 100 * np.cumprod(1 + np.full(60, 0.002))
    data = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                         "volume": 1_000_000}, index=days)
    result = await BacktestAgent().execute(AgentTask(
        agent_type="backtest", payload={"strategy": "momentum", "symbol": "AAPL", "data": data}))
    assert result.status == AgentStatus.FAILED
    assert "runtime" in result.error and result.result is None


@pytest.mark.asyncio
async def test_factor_attribution_on_a_sample_portfolio_is_refused(monkeypatch):
    monkeypatch.delenv("ALLOW_MOCK_AGENT_DATA", raising=False)
    result = await FactorAgent().execute(AgentTask(
        agent_type="factor", payload={"type": "attribution", "portfolio_id": 1}))
    assert result.status == AgentStatus.FAILED
    assert "not implemented" in result.error
