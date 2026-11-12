# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the synthetic-data guard on quant agents.

The regression these cover: agents used to answer with `random.uniform(...)`
whenever a real data source was missing. The response was a well-formed 200, so
no caller could tell the numbers were invented. Fabricating data must now be an
explicit, opt-in choice.
"""

import pytest

from agentrt.base import AgentTask, AgentStatus
from agentrt.quant.mock_guard import (
    MOCK_FLAG_ENV_VAR,
    MockDataDisabledError,
    mock_data_allowed,
    require_real_data,
)
from agentrt.quant.factor_agent import FactorAgent
from agentrt.quant.optimization_agent import OptimizationAgent
from agentrt.quant.backtest_agent import BacktestAgent
from agentrt.quant.risk_agent import RiskAgent


@pytest.fixture(autouse=True)
def forbid_mock_data(monkeypatch):
    """Default posture: synthetic data is not allowed."""
    monkeypatch.delenv(MOCK_FLAG_ENV_VAR, raising=False)


# ============================================================================
# The flag itself
# ============================================================================

class TestMockFlag:

    def test_disabled_by_default(self):
        assert mock_data_allowed() is False

    @pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on", " true "])
    def test_recognised_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv(MOCK_FLAG_ENV_VAR, value)
        assert mock_data_allowed() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "maybe"])
    def test_everything_else_is_off(self, monkeypatch, value):
        monkeypatch.setenv(MOCK_FLAG_ENV_VAR, value)
        assert mock_data_allowed() is False

    def test_require_real_data_raises_when_disabled(self):
        with pytest.raises(MockDataDisabledError) as exc:
            require_real_data("portfolio VaR", "Connect a database.")
        assert "portfolio VaR" in str(exc.value)
        assert MOCK_FLAG_ENV_VAR in str(exc.value)

    def test_require_real_data_passes_when_enabled(self, monkeypatch):
        monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
        require_real_data("portfolio VaR", "Connect a database.")  # does not raise


# ============================================================================
# Agents refuse rather than fabricate
# ============================================================================

class TestAgentsRefuseToFabricate:

    @pytest.mark.asyncio
    async def test_optimization_agent_refuses(self):
        """Its "best params" are the argmax of random scores, not backtests."""
        agent = OptimizationAgent()
        result = await agent.execute(AgentTask(
            agent_type="optimization",
            payload={
                "type": "parameter_optimization",
                "strategy": "momentum",
                "symbol": "AAPL",
                "param_ranges": {"window": [5, 50]},
            },
        ))

        assert result.status == AgentStatus.FAILED
        assert result.result is None
        assert "no optimizer wired up" in result.error

    @pytest.mark.asyncio
    async def test_factor_agent_refuses(self):
        """Its price history is a random walk, so every factor derived from it is too."""
        agent = FactorAgent()
        result = await agent.execute(AgentTask(
            agent_type="factor",
            payload={
                "type": "performance_analysis",
                "symbols": ["AAPL"],
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
        ))

        assert result.status == AgentStatus.FAILED
        assert result.result is None
        assert "price history" in result.error

    @pytest.mark.asyncio
    async def test_backtest_walk_forward_refuses(self):
        """Walk-forward folds were random draws, not out-of-sample backtests."""
        agent = BacktestAgent()
        result = await agent.execute(AgentTask(
            agent_type="backtest",
            payload={
                "type": "walk_forward",
                "strategy": "momentum",
                "symbol": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-06-30",
            },
        ))

        assert result.status == AgentStatus.FAILED
        assert result.result is None
        assert "walk-forward" in result.error

    @pytest.mark.asyncio
    async def test_enabling_the_flag_restores_the_synthetic_path(self, monkeypatch):
        """The escape hatch still works for local development."""
        monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
        agent = OptimizationAgent()
        result = await agent.execute(AgentTask(
            agent_type="optimization",
            payload={
                "type": "parameter_optimization",
                "strategy": "momentum",
                "symbol": "AAPL",
                "param_ranges": {"window": [5, 50]},
                "max_iterations": 3,
            },
        ))

        assert result.status == AgentStatus.COMPLETED


# ============================================================================
# RiskAgent has no synthetic path at all
# ============================================================================

class TestRiskAgentHasNoSyntheticPath:

    def test_source_contains_no_random_draws(self):
        """Regression: every risk metric used to be a random.uniform() call."""
        import inspect
        from agentrt.quant import risk_agent

        source = inspect.getsource(risk_agent)
        assert "random." not in source
        assert "_mock_" not in source

    @pytest.mark.asyncio
    async def test_analytics_failure_is_surfaced_not_substituted(self, monkeypatch):
        """If the analytics service cannot be reached, the task fails."""
        from contextlib import asynccontextmanager

        agent = RiskAgent()

        @asynccontextmanager
        async def _unavailable():
            raise RuntimeError("database unavailable")
            yield  # pragma: no cover

        monkeypatch.setattr(agent, "_analytics", _unavailable)

        result = await agent.execute(AgentTask(
            agent_type="risk",
            payload={"portfolio_id": 1, "type": "risk_analysis"},
        ))

        assert result.status == AgentStatus.FAILED
        assert result.result is None
        assert "database unavailable" in result.error


# ============================================================================
# DecisionAgent: a debate made of placeholder answers is not a decision
# ============================================================================

class _RealModelStub:
    """Stands in for a configured model: answers without the ``mock`` marker."""

    def __init__(self, name):
        self.name = name

    async def query(self, prompt, role, context=None, max_tokens=4096, temperature=0.7):
        from agentrt.debate import ModelResponse

        return ModelResponse(model=self.name, role=role, response=f"{self.name}: hold")

    def is_available(self):
        return True


def _decision_task(tier=2):
    return AgentTask(
        agent_type="decision",
        payload={
            "type": "trade",
            "context": {"question": "Should I add to AAPL?"},
            "force_debate": True,
            "debate_tier": tier,
        },
    )


class TestDecisionAgentRefusesPlaceholderDebate:
    """Regression: without model credentials, agent_routes.get_debate_engine() and
    create_adapter_set() swap every participant for MockAdapter. The UI's debate
    request then came back 200 with ``debate_used: true``, ``confidence: 0.9`` and
    rounds of "Mock architect response for: ..." - a fabricated decision."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier", [0, 1, 2])
    async def test_all_placeholder_participants_are_refused(self, tier):
        from agentrt.debate import DebateEngine, MockAdapter
        from agentrt.quant.decision_agent import DecisionAgent

        engine = DebateEngine(
            claude_adapter=MockAdapter("claude_mock"),
            gemini_adapter=MockAdapter("gemini_mock"),
            codex_adapter=MockAdapter("codex_mock"),
        )
        result = await DecisionAgent(debate_engine=engine).execute(_decision_task(tier))

        assert result.status == AgentStatus.FAILED
        assert result.result is None
        assert "multi-model debate decision" in result.error
        assert "claude_mock" in result.error

    @pytest.mark.asyncio
    async def test_one_placeholder_participant_is_enough_to_refuse(self):
        """Claude configured, Gemini and Codex not: the rounds still carry canned text."""
        from agentrt.debate import DebateEngine, MockAdapter
        from agentrt.quant.decision_agent import DecisionAgent

        engine = DebateEngine(
            claude_adapter=_RealModelStub("claude"),
            gemini_adapter=MockAdapter("gemini_mock"),
            codex_adapter=MockAdapter("codex_mock"),
        )
        result = await DecisionAgent(debate_engine=engine).execute(_decision_task())

        assert result.status == AgentStatus.FAILED
        assert "gemini_mock" in result.error
        assert "codex_mock" in result.error

    @pytest.mark.asyncio
    async def test_a_debate_between_real_models_is_unaffected(self):
        from agentrt.debate import DebateEngine
        from agentrt.quant.decision_agent import DecisionAgent

        engine = DebateEngine(
            claude_adapter=_RealModelStub("claude"),
            gemini_adapter=_RealModelStub("gemini"),
            codex_adapter=_RealModelStub("codex"),
        )
        result = await DecisionAgent(debate_engine=engine).execute(_decision_task())

        assert result.status == AgentStatus.COMPLETED
        assert result.result["debate_used"] is True
        assert result.result["debate_result"]["models_used"] == ["claude", "gemini", "codex"]

    @pytest.mark.asyncio
    async def test_enabling_the_flag_restores_the_placeholder_debate(self, monkeypatch):
        from agentrt.debate import DebateEngine, MockAdapter
        from agentrt.quant.decision_agent import DecisionAgent

        monkeypatch.setenv(MOCK_FLAG_ENV_VAR, "true")
        engine = DebateEngine(
            claude_adapter=MockAdapter("claude_mock"),
            gemini_adapter=MockAdapter("gemini_mock"),
            codex_adapter=MockAdapter("codex_mock"),
        )
        result = await DecisionAgent(debate_engine=engine).execute(_decision_task())

        assert result.status == AgentStatus.COMPLETED
        responses = [
            resp
            for rnd in result.result["debate_result"]["rounds"]
            for resp in rnd["responses"]
        ]
        assert responses and all(resp["metadata"].get("mock") for resp in responses)
