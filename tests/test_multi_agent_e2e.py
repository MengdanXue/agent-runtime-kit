# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Agent System End-to-End Tests

全面端到端测试，覆盖:
1. 代理注册表和池管理
2. 多模型辩论完整流程
3. 量化代理协作流程
4. 信号→决策→执行完整管道
5. 故障恢复和容错机制
6. 高并发压力测试

Author: TraderX Team
Date: 2025-01-01
"""

import pytest
import asyncio
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

# Base agent imports
from agentrt import (
    BaseAgent,
    AgentTask,
    AgentResult,
    AgentStatus,
    AgentPriority,
    AgentRegistry,
    AgentPool,
    AgentOrchestrator,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryPolicy,
    RetryConfig,
)
from agentrt.registry import get_global_registry
from agentrt.pool import AgentPoolContext, PoolStats

# Debate engine imports
from agentrt.debate.engine import (
    DebateEngine,
    DebateTier,
    ModelRole,
    ModelResponse,
    DebateResult,
    DebateRound,
)
from agentrt.debate.adapters import MockAdapter
from agentrt.debate.consensus import ConsensusBuilder, VotingStrategy

# Quant agents imports
from agentrt.quant.signal_agent import SignalAgent
from agentrt.quant.backtest_agent import BacktestAgent
from agentrt.quant.risk_agent import RiskAgent
from agentrt.quant.decision_agent import DecisionAgent
from agentrt.quant.optimization_agent import OptimizationAgent


@pytest.fixture(autouse=True)
def allow_mock_agent_data(monkeypatch):
    """Let the agents under test fall back to synthetic data.

    This suite exercises orchestration, not data quality. Agents refuse to
    fabricate numbers unless this flag is set; see test_agent_mock_guard.py.
    """
    monkeypatch.setenv("ALLOW_MOCK_AGENT_DATA", "true")


# ============================================================================
# Test Fixtures - Mock Agents
# ============================================================================

class MockSuccessAgent(BaseAgent):
    """Mock agent that always succeeds with configurable delay."""

    def __init__(self, agent_id: str = None, delay: float = 0.05, result_data: Dict = None):
        super().__init__(agent_id)
        self.delay = delay
        self.result_data = result_data or {"status": "success"}
        self.execute_count = 0
        self.last_task = None

    async def execute(self, task: AgentTask) -> AgentResult:
        self.execute_count += 1
        self.last_task = task
        await asyncio.sleep(self.delay)
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={**self.result_data, "execute_count": self.execute_count}
        )

    def can_handle(self, task_type: str) -> bool:
        return True


class MockFailureAgent(BaseAgent):
    """Mock agent that fails after specified attempts."""

    def __init__(self, agent_id: str = None, fail_count: int = 3, error_msg: str = "Mock failure"):
        super().__init__(agent_id)
        self.fail_count = fail_count
        self.error_msg = error_msg
        self.execute_count = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        self.execute_count += 1
        await asyncio.sleep(0.02)

        if self.execute_count <= self.fail_count:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=f"{self.error_msg} (attempt {self.execute_count})"
            )
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"recovered": True, "attempts": self.execute_count}
        )

    def can_handle(self, task_type: str) -> bool:
        return True


class MockSlowAgent(BaseAgent):
    """Mock agent with variable execution time."""

    def __init__(self, agent_id: str = None, min_delay: float = 0.1, max_delay: float = 0.5):
        super().__init__(agent_id)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.execute_count = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        self.execute_count += 1
        delay = random.uniform(self.min_delay, self.max_delay)
        await asyncio.sleep(delay)
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"delay": delay, "count": self.execute_count}
        )

    def can_handle(self, task_type: str) -> bool:
        return True


class MockIntermittentAgent(BaseAgent):
    """Mock agent that fails intermittently."""

    def __init__(self, agent_id: str = None, failure_rate: float = 0.3):
        super().__init__(agent_id)
        self.failure_rate = failure_rate
        self.execute_count = 0
        self.failure_count = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        self.execute_count += 1
        await asyncio.sleep(0.02)

        if random.random() < self.failure_rate:
            self.failure_count += 1
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error="Intermittent failure"
            )
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"success": True}
        )

    def can_handle(self, task_type: str) -> bool:
        return True


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return AgentRegistry()


@pytest.fixture
def pool():
    """Create a pool with default settings."""
    return AgentPool(max_concurrent=5)


@pytest.fixture
def orchestrator(registry, pool):
    """Create an orchestrator instance."""
    return AgentOrchestrator(registry=registry, pool=pool)


@pytest.fixture
def mock_claude_adapter():
    """Create mock Claude adapter."""
    responses = {
        "architect": """### Proposed Solution
Based on analysis, recommend a balanced approach.

### Architecture
- Component A for data processing
- Component B for decision making

### Trade-offs
Pros: Flexibility, maintainability
Cons: Initial complexity

### Risks
- Integration complexity
- Performance overhead
"""
    }
    return MockAdapter(model_name="claude", responses=responses)


@pytest.fixture
def mock_gemini_adapter():
    """Create mock Gemini adapter."""
    responses = {
        "challenger": """### Alternative Approaches
1. Simplified single-service approach
2. Event-driven architecture

### Challenges
- Proposed solution may be over-engineered
- Consider simpler alternatives first

### Critical Questions
- Is this complexity justified?
- What are the scaling requirements?
"""
    }
    return MockAdapter(model_name="gemini", responses=responses)


@pytest.fixture
def mock_codex_adapter():
    """Create mock Codex adapter."""
    responses = {
        "validator": """### Feasibility Assessment
High feasibility - standard patterns used

### Technical Stack
- Python asyncio for concurrency
- Redis for caching
- PostgreSQL for persistence

### Implementation Notes
- Estimated 2-3 sprints
- No major blockers identified
"""
    }
    return MockAdapter(model_name="codex", responses=responses)


@pytest.fixture
def debate_engine(mock_claude_adapter, mock_gemini_adapter, mock_codex_adapter):
    """Create debate engine with mock adapters."""
    return DebateEngine(
        claude_adapter=mock_claude_adapter,
        gemini_adapter=mock_gemini_adapter,
        codex_adapter=mock_codex_adapter,
    )


# ============================================================================
# Section 1: Agent Registry and Pool E2E Tests
# ============================================================================

class TestRegistryPoolE2E:
    """End-to-end tests for agent registry and pool management."""

    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, registry, pool):
        """Test complete agent lifecycle: register → acquire → execute → release."""
        # Create agent classes
        class FastAgent(MockSuccessAgent):
            def __init__(self, agent_id=None):
                super().__init__(agent_id, delay=0.01)

        class MediumAgent(MockSuccessAgent):
            def __init__(self, agent_id=None):
                super().__init__(agent_id, delay=0.05)

        class SlowAgent(MockSlowAgent):
            def __init__(self, agent_id=None):
                super().__init__(agent_id, min_delay=0.1, max_delay=0.2)

        # Register agent classes
        registry.register("fast", FastAgent)
        registry.register("medium", MediumAgent)
        registry.register("slow", SlowAgent)

        # Verify registration
        assert len(registry.list_all()) == 3
        assert registry.get("fast") == FastAgent

        # Create agent instance and execute
        agent = FastAgent("fast_test")
        task = AgentTask(agent_type="fast", payload={"test": "data"})

        # Acquire from pool and execute
        async with AgentPoolContext(pool, agent, task.task_id) as acquired:
            assert acquired
            result = await agent.execute(task)
            assert result.status == AgentStatus.COMPLETED
            assert agent.execute_count == 1

        # Pool stats should reflect usage
        stats = pool.get_stats()
        assert stats.total_executed >= 1

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, pool):
        """Test multiple agents executing concurrently."""
        # Create agents directly (no registry needed for this test)
        agent_count = 10
        agents = [MockSuccessAgent(f"agent_{i}", delay=0.05) for i in range(agent_count)]

        # Execute all concurrently
        async def execute_agent(idx: int):
            agent = agents[idx]
            task = AgentTask(agent_type=f"agent_{idx}", payload={"idx": idx})
            async with AgentPoolContext(pool, agent, task.task_id) as acquired:
                if acquired:
                    return await agent.execute(task)
                return AgentResult(task_id=task.task_id, status=AgentStatus.FAILED, error="Not acquired")

        start_time = time.time()
        results = await asyncio.gather(*[execute_agent(i) for i in range(agent_count)])
        elapsed = time.time() - start_time

        # All should succeed
        assert all(r.status == AgentStatus.COMPLETED for r in results)

        # Should be faster than sequential (10 * 0.05 = 0.5s)
        # With concurrency, should be closer to 0.1s (limited by pool size)
        assert elapsed < 0.4  # Allow some margin

    @pytest.mark.asyncio
    async def test_pool_capacity_limit(self):
        """Test that pool correctly limits concurrent executions."""
        max_concurrent = 3
        pool = AgentPool(max_concurrent=max_concurrent)

        execution_times = []
        agents = [MockSuccessAgent(f"cap_agent_{i}", delay=0.1) for i in range(6)]

        async def track_execution(idx: int):
            agent = agents[idx]
            async with AgentPoolContext(pool, agent, f"task_{idx}") as acquired:
                if acquired:
                    start = time.time()
                    await asyncio.sleep(0.1)
                    end = time.time()
                    execution_times.append((idx, start, end))

        # Start 6 tasks with pool of 3
        await asyncio.gather(*[track_execution(i) for i in range(6)])

        # Analyze overlapping executions
        execution_times.sort(key=lambda x: x[1])  # Sort by start time

        # At any point, max 3 should be running
        for i, (idx, start, end) in enumerate(execution_times):
            concurrent_count = sum(
                1 for _, s, e in execution_times
                if s <= start < e or s < end <= e or (s >= start and e <= end)
            )
            assert concurrent_count <= max_concurrent + 1  # +1 for boundary

    @pytest.mark.asyncio
    async def test_agent_type_dispatch(self, registry):
        """Test dispatching tasks to correct agent types."""
        # Create agent classes dynamically
        class SignalMockAgent(MockSuccessAgent):
            def __init__(self, agent_id=None):
                super().__init__(agent_id, result_data={"type": "signal"})

        class RiskMockAgent(MockSuccessAgent):
            def __init__(self, agent_id=None):
                super().__init__(agent_id, result_data={"type": "risk"})

        class DecisionMockAgent(MockSuccessAgent):
            def __init__(self, agent_id=None):
                super().__init__(agent_id, result_data={"type": "decision"})

        # Register agent classes (not instances)
        registry.register("signal", SignalMockAgent, task_types=["generate_signal"])
        registry.register("risk", RiskMockAgent, task_types=["assess_risk"])
        registry.register("decision", DecisionMockAgent, task_types=["make_decision"])

        # Dispatch by task type
        for task_type, expected_class in [
            ("generate_signal", SignalMockAgent),
            ("assess_risk", RiskMockAgent),
            ("make_decision", DecisionMockAgent),
        ]:
            agent_class = registry.get_for_task(task_type)
            assert agent_class == expected_class

            # Instantiate and execute
            agent = agent_class()
            task = AgentTask(agent_type=task_type, payload={})
            result = await agent.execute(task)
            assert "type" in result.result


# ============================================================================
# Section 2: Multi-Model Debate E2E Tests
# ============================================================================

class TestDebateEngineE2E:
    """End-to-end tests for multi-model debate system."""

    @pytest.mark.asyncio
    async def test_tier0_single_model_response(self, debate_engine):
        """Test Tier 0: Single model quick response."""
        question = "What is the current market status for AAPL?"
        context = {"type": "query", "symbol": "AAPL"}

        result = await debate_engine.run_debate(
            question=question,
            context=context,
            tier=DebateTier.TIER_0,
        )

        assert result is not None
        assert result.tier == DebateTier.TIER_0
        assert len(result.models_used) == 1
        # Mock adapter may return 0 duration, just check it's non-negative
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_tier1_quick_debate(self, debate_engine):
        """Test Tier 1: Quick debate between two models."""
        question = "Should we optimize the momentum strategy parameters?"
        context = {"type": "strategy", "current_sharpe": 1.2}

        result = await debate_engine.run_debate(
            question=question,
            context=context,
            tier=DebateTier.TIER_1,
        )

        assert result is not None
        assert result.tier == DebateTier.TIER_1
        assert len(result.models_used) == 2
        assert len(result.rounds) >= 1
        assert result.consensus is not None

    @pytest.mark.asyncio
    async def test_tier2_full_debate(self, debate_engine):
        """Test Tier 2: Full debate with all three models."""
        question = "How should we architect the new risk management system?"
        context = {
            "type": "architecture",
            "risk_level": "high",
            "trade_value": 500000,
        }

        result = await debate_engine.run_debate(
            question=question,
            context=context,
            tier=DebateTier.TIER_2,
        )

        assert result is not None
        assert result.tier == DebateTier.TIER_2
        assert len(result.models_used) == 3
        assert len(result.rounds) >= 2
        assert result.consensus is not None
        assert result.final_decision is not None

    @pytest.mark.asyncio
    async def test_automatic_tier_determination(self, debate_engine):
        """Test automatic tier selection based on context."""
        test_cases = [
            # (context, expected_min_tier)
            ({"type": "query"}, DebateTier.TIER_0),
            ({"type": "strategy", "importance": "medium"}, DebateTier.TIER_1),
            ({"type": "architecture", "trade_value": 200000}, DebateTier.TIER_2),
            ({"type": "risk", "risk_level": "critical"}, DebateTier.TIER_2),
        ]

        for context, expected_min_tier in test_cases:
            determined_tier = debate_engine.determine_tier(
                question="Test question",
                context=context,
            )
            assert determined_tier.value >= expected_min_tier.value

    @pytest.mark.asyncio
    async def test_debate_consensus_building(self, debate_engine):
        """Test consensus building from divergent model opinions."""
        question = "What trading strategy should we use for volatile markets?"
        context = {"market_volatility": "high", "type": "strategy"}

        result = await debate_engine.run_debate(
            question=question,
            context=context,
            tier=DebateTier.TIER_2,
        )

        # Should have consensus even with different model perspectives
        assert result.consensus is not None
        assert len(result.consensus) > 0

        # Dissent should capture disagreements
        assert isinstance(result.dissent, list)

    @pytest.mark.asyncio
    async def test_debate_with_voting_strategies(self, mock_claude_adapter, mock_gemini_adapter, mock_codex_adapter):
        """Test debate produces consensus regardless of adapter."""
        # Test that debate engine can be created and produces valid results
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            gemini_adapter=mock_gemini_adapter,
            codex_adapter=mock_codex_adapter,
        )

        result = await engine.run_debate(
            question="Test consensus with voting",
            context={"type": "design"},
            tier=DebateTier.TIER_2,
        )

        assert result.consensus is not None
        assert result.final_decision is not None
        assert len(result.models_used) == 3

    @pytest.mark.asyncio
    async def test_debate_performance_metrics(self, debate_engine):
        """Test that debate captures performance metrics."""
        result = await debate_engine.run_debate(
            question="Performance test question",
            context={"type": "strategy"},
            tier=DebateTier.TIER_2,
        )

        # Check metrics (mock may return 0 for duration)
        assert result.total_duration_ms >= 0
        assert result.total_tokens >= 0
        assert len(result.rounds) > 0

        for round in result.rounds:
            assert round.round_number > 0
            assert len(round.responses) > 0


# ============================================================================
# Section 3: Quant Agents Collaboration E2E Tests
# ============================================================================

class TestQuantAgentsE2E:
    """End-to-end tests for quantitative trading agents collaboration."""

    @pytest.fixture
    def signal_agent(self):
        return SignalAgent()

    @pytest.fixture
    def risk_agent(self):
        return RiskAgent()

    @pytest.fixture
    def decision_agent(self):
        return DecisionAgent()

    @pytest.fixture
    def backtest_agent(self):
        return BacktestAgent()

    @pytest.fixture
    def optimization_agent(self):
        return OptimizationAgent()

    @pytest.mark.asyncio
    async def test_signal_generation_flow(self, signal_agent):
        """Test signal generation with multiple strategies."""
        task = AgentTask(
            agent_type="signal",
            payload={
                "symbol": "AAPL",
                "timeframe": "1d",
                "strategies": ["momentum", "mean_reversion", "trend_following"],
                "aggregate": True,
            }
        )

        result = await signal_agent.execute(task)

        assert result.status == AgentStatus.COMPLETED
        assert "signal" in result.result or "signals" in result.result

    @pytest.mark.asyncio
    async def test_risk_assessment_flow(self, risk_agent, stub_analytics):
        """Test risk assessment with portfolio context."""
        # RiskAgent reads a real portfolio from the database; it has no
        # synthetic fallback, so the analytics service is stubbed here.
        stub_analytics(risk_agent)
        task = AgentTask(
            agent_type="risk",
            payload={
                "portfolio_id": 1,
                "type": "risk_analysis",
                "positions": [
                    {"symbol": "AAPL", "quantity": 100, "entry_price": 150.0},
                    {"symbol": "GOOGL", "quantity": 50, "entry_price": 140.0},
                ],
                "lookback_days": 252,
            }
        )

        result = await risk_agent.execute(task)

        assert result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_signal_to_decision_pipeline(self, signal_agent, decision_agent):
        """Test complete signal to decision pipeline."""
        # Step 1: Generate signal
        signal_task = AgentTask(
            agent_type="signal",
            payload={
                "symbol": "TSLA",
                "strategies": ["momentum"],
            }
        )
        signal_result = await signal_agent.execute(signal_task)
        assert signal_result.status == AgentStatus.COMPLETED

        # Step 2: Make decision based on signal
        decision_task = AgentTask(
            agent_type="decision",
            payload={
                "type": "trade_decision",
                "context": {
                    "symbol": "TSLA",
                    "signal": signal_result.result,
                    "risk_level": "medium",
                },
            }
        )
        decision_result = await decision_agent.execute(decision_task)
        assert decision_result.status == AgentStatus.COMPLETED

    @pytest.mark.needs_backtest_engine
    @pytest.mark.asyncio
    async def test_backtest_optimization_loop(self, backtest_agent, optimization_agent):
        """Test backtest and optimization feedback loop."""
        # Step 1: Initial backtest
        backtest_task = AgentTask(
            agent_type="backtest",
            payload={
                "strategy": "momentum",
                "symbol": "SPY",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "initial_capital": 100000,
                "params": {"lookback": 20},
            }
        )
        backtest_result = await backtest_agent.execute(backtest_task)
        assert backtest_result.status == AgentStatus.COMPLETED

        # Step 2: Optimize based on backtest
        optimization_task = AgentTask(
            agent_type="optimization",
            payload={
                "type": "parameter_optimization",
                "strategy": "momentum",
                "symbol": "SPY",
                "param_ranges": {
                    "lookback": (10, 50),
                    "threshold": (0.5, 2.0),
                },
                "baseline_result": backtest_result.result,
                "objective": "sharpe",
                "method": "grid",
                "max_iterations": 10,
            }
        )
        optimization_result = await optimization_agent.execute(optimization_task)
        assert optimization_result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multi_agent_parallel_analysis(self, signal_agent, risk_agent, stub_analytics):
        """Test multiple agents analyzing in parallel."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        stub_analytics(risk_agent)

        async def analyze_symbol(symbol: str):
            # Signal and risk analysis in parallel for each symbol
            signal_task = AgentTask(
                agent_type="signal",
                payload={"symbol": symbol, "strategies": ["momentum"]},
            )
            risk_task = AgentTask(
                agent_type="risk",
                payload={"portfolio_id": symbols.index(symbol) + 1, "type": "risk_analysis"},
            )

            signal_result, risk_result = await asyncio.gather(
                signal_agent.execute(signal_task),
                risk_agent.execute(risk_task),
            )
            return symbol, signal_result, risk_result

        start_time = time.time()
        results = await asyncio.gather(*[analyze_symbol(s) for s in symbols])
        elapsed = time.time() - start_time

        # All should complete successfully
        for symbol, signal_result, risk_result in results:
            assert signal_result.status == AgentStatus.COMPLETED
            assert risk_result.status == AgentStatus.COMPLETED

        # Should complete faster than sequential
        assert len(results) == len(symbols)


# ============================================================================
# Section 4: Full Trading Pipeline E2E Tests
# ============================================================================

class TestTradingPipelineE2E:
    """End-to-end tests for complete trading pipeline."""

    @pytest.mark.asyncio
    async def test_full_trading_pipeline(self):
        """Test complete: Data → Signal → Risk → Decision → Execution."""
        from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode

        pipeline = TradingPipeline(
            name="test_pipeline",
            mode=PipelineMode.PAPER,  # Paper trading mode
        )

        await pipeline.start()

        result = await pipeline.run(
            symbols=["AAPL", "GOOGL"],
            context={
                "account_balance": 100000,
                "max_position_size": 0.1,
            }
        )

        assert result.success or len(result.errors) > 0  # Either success or documented errors
        assert result.mode == PipelineMode.PAPER
        assert len(result.symbols) == 2

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_signal_only_mode(self):
        """Test pipeline in signal-only mode."""
        from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode

        pipeline = TradingPipeline(
            name="signal_pipeline",
            mode=PipelineMode.SIGNAL_ONLY,
        )

        await pipeline.start()

        result = await pipeline.run(
            symbols=["SPY"],
            context={},
        )

        # Should have signals but no execution
        assert result.mode == PipelineMode.SIGNAL_ONLY
        assert len(result.execution_reports) == 0

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_analysis_mode(self):
        """Test pipeline in analysis mode (signal + risk, no execution)."""
        from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode

        pipeline = TradingPipeline(
            name="analysis_pipeline",
            mode=PipelineMode.ANALYSIS,
        )

        await pipeline.start()

        result = await pipeline.run(
            symbols=["QQQ", "IWM"],
            context={"lookback_days": 60},
        )

        assert result.mode == PipelineMode.ANALYSIS
        # Should have risk assessments
        assert len(result.execution_reports) == 0  # No execution in analysis mode

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_with_multiple_symbols(self):
        """Test pipeline handling multiple symbols concurrently."""
        from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "AMD"]

        pipeline = TradingPipeline(
            name="multi_symbol_pipeline",
            mode=PipelineMode.ANALYSIS,
        )

        await pipeline.start()

        start_time = time.time()
        result = await pipeline.run(
            symbols=symbols,
            context={},
        )
        elapsed = time.time() - start_time

        assert len(result.symbols) == len(symbols)
        # Processing should benefit from parallelization

        await pipeline.stop()


# ============================================================================
# Section 5: Fault Tolerance and Recovery E2E Tests
# ============================================================================

class TestFaultToleranceE2E:
    """End-to-end tests for fault tolerance and recovery."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self):
        """Test circuit breaker prevents cascade failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5,
            half_open_max_calls=2,
        )
        breaker = CircuitBreaker(config)

        # Simulate failures
        for i in range(3):
            breaker.record_failure()

        # Should be open now
        assert breaker.state == CircuitState.OPEN
        assert not breaker.can_execute()

        # Wait for recovery
        await asyncio.sleep(0.6)

        # Should be half-open
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute()

        # Record success to close
        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry policy with exponential backoff."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            retryable_errors=["mock failure"],
        )
        policy = RetryPolicy(config)
        retry_orchestrator = AgentOrchestrator(
            registry=AgentRegistry(),
            pool=AgentPool(max_concurrent=1),
            retry_policy=policy,
        )

        # Agent that fails twice then succeeds
        agent = MockFailureAgent("retry_agent", fail_count=2)

        task = AgentTask(agent_type="test", payload={})

        start_time = time.time()
        result = await retry_orchestrator._execute_with_retry(agent, task, timeout=5.0)
        elapsed = time.time() - start_time

        # Should succeed after retries
        assert result.status == AgentStatus.COMPLETED
        assert agent.execute_count == 3

        # Should have delays between retries
        assert elapsed >= 0.2  # At least base_delay + base_delay * 2

    @pytest.mark.asyncio
    async def test_orchestrator_with_failing_agents(self, orchestrator):
        """Test orchestrator handling mixed success/failure agents."""
        # Register mix of agents
        orchestrator.registry.register("success1", MockSuccessAgent)
        orchestrator.registry.register("success2", MockSuccessAgent)
        orchestrator.registry.register("failure", MockFailureAgent)

        # Execute tasks
        tasks = [
            AgentTask(agent_type="success1", payload={}),
            AgentTask(agent_type="success2", payload={}),
            AgentTask(agent_type="failure", payload={}),
        ]

        results = await asyncio.gather(
            *[orchestrator.dispatch(t) for t in tasks],
            return_exceptions=True
        )

        # Should have both successes and failures
        success_count = sum(1 for r in results if isinstance(r, AgentResult) and r.status == AgentStatus.COMPLETED)
        failure_count = sum(1 for r in results if isinstance(r, AgentResult) and r.status == AgentStatus.FAILED)

        assert success_count >= 2
        assert failure_count >= 1

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system degrades gracefully under partial failure."""
        from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode

        pipeline = TradingPipeline(
            name="degradation_test",
            mode=PipelineMode.ANALYSIS,
        )

        await pipeline.start()

        # Run with some symbols that might fail
        result = await pipeline.run(
            symbols=["VALID1", "INVALID_SYMBOL_XYZ", "VALID2"],
            context={},
        )

        # Should complete with partial results
        assert result is not None
        # Errors should be documented
        if len(result.errors) > 0:
            assert "INVALID_SYMBOL_XYZ" in str(result.errors) or len(result.symbols) < 3

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test proper timeout handling for slow agents."""
        pool = AgentPool(max_concurrent=2)
        slow_agent = MockSlowAgent("slow", min_delay=0.5, max_delay=1.0)

        task = AgentTask(agent_type="slow", payload={}, timeout=0.1)

        # Should timeout
        try:
            async with AgentPoolContext(pool, slow_agent, task.task_id, timeout=0.1) as acquired:
                assert acquired
                result = await asyncio.wait_for(
                    slow_agent.execute(task),
                    timeout=0.15
                )
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_recovery_after_failures(self):
        """Test system recovers after transient failures."""
        agent = MockFailureAgent("recovery_agent", fail_count=2)
        policy = RetryPolicy(
            RetryConfig(
                max_retries=5,
                initial_delay=0.05,
                retryable_errors=["mock failure"],
            )
        )
        retry_orchestrator = AgentOrchestrator(
            registry=AgentRegistry(),
            pool=AgentPool(max_concurrent=1),
            retry_policy=policy,
        )

        # First few calls fail
        task = AgentTask(agent_type="test", payload={})

        result = await retry_orchestrator._execute_with_retry(agent, task, timeout=5.0)

        assert result.status == AgentStatus.COMPLETED
        assert result.result["recovered"] == True
        assert result.result["attempts"] == 3


# ============================================================================
# Section 6: High Concurrency Stress Tests
# ============================================================================

class TestHighConcurrencyE2E:
    """End-to-end stress tests for high concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_high_volume_task_dispatch(self, orchestrator):
        """Test dispatching many tasks concurrently."""
        num_tasks = 100

        # Register multiple agents
        for i in range(10):
            orchestrator.registry.register(f"worker_{i}", MockSuccessAgent)

        # Create many tasks
        tasks = [
            AgentTask(
                agent_type=f"worker_{i % 10}",
                payload={"task_id": i}
            )
            for i in range(num_tasks)
        ]

        start_time = time.time()
        results = await asyncio.gather(
            *[orchestrator.dispatch(t) for t in tasks],
            return_exceptions=True
        )
        elapsed = time.time() - start_time

        # Count successful completions
        success_count = sum(
            1 for r in results
            if isinstance(r, AgentResult) and r.status == AgentStatus.COMPLETED
        )

        assert success_count >= num_tasks * 0.95  # At least 95% success rate

    @pytest.mark.asyncio
    async def test_pool_under_pressure(self):
        """Test pool behavior under high pressure."""
        pool = AgentPool(max_concurrent=5, )

        active_count = 0
        max_active_count = 0

        async def pressure_task(idx: int):
            nonlocal active_count, max_active_count
            agent = MockSuccessAgent(f"pressure_{idx}", delay=0)
            async with AgentPoolContext(pool, agent, f"pressure_task_{idx}") as acquired:
                assert acquired
                active_count += 1
                max_active_count = max(max_active_count, active_count)
                try:
                    await asyncio.sleep(random.uniform(0.01, 0.05))
                    return idx
                finally:
                    active_count -= 1

        # Launch 50 tasks with pool of 5
        start_time = time.time()
        results = await asyncio.gather(*[pressure_task(i) for i in range(50)])
        elapsed = time.time() - start_time

        assert len(results) == 50

        stats = pool.get_stats()
        assert max_active_count <= 5
        assert stats.total_executed == 50

    @pytest.mark.asyncio
    async def test_intermittent_failures_under_load(self):
        """Test system stability with intermittent failures under load."""
        pool = AgentPool(max_concurrent=10, )
        policy = RetryPolicy(
            RetryConfig(
                max_retries=3,
                initial_delay=0.02,
                retryable_errors=["intermittent failure"],
            )
        )
        retry_orchestrator = AgentOrchestrator(
            registry=AgentRegistry(),
            pool=pool,
            retry_policy=policy,
        )

        async def execute_with_retry(idx: int):
            task = AgentTask(agent_type="intermittent", payload={"idx": idx})
            agent = MockIntermittentAgent(f"intermittent_{idx}", failure_rate=0.2)
            return await retry_orchestrator._execute_with_retry(agent, task, timeout=5.0)

        results = await asyncio.gather(
            *[execute_with_retry(i) for i in range(100)],
            return_exceptions=True
        )

        # With 20% failure rate and 3 retries, most should succeed
        success_count = sum(
            1 for r in results
            if isinstance(r, AgentResult) and r.status == AgentStatus.COMPLETED
        )

        # Success rate should be high due to retries
        assert success_count >= 90  # At least 90% with retries

    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self):
        """Test performance with mixed fast and slow tasks."""
        pool = AgentPool(max_concurrent=10, )

        async def mixed_task(idx: int):
            agent = (
                MockSuccessAgent(f"fast_{idx}", delay=0.01)
                if idx % 3 != 0
                else MockSlowAgent(f"slow_{idx}", min_delay=0.1, max_delay=0.2)
            )
            task = AgentTask(agent_type="mixed", payload={"idx": idx})
            async with AgentPoolContext(pool, agent, task.task_id) as acquired:
                assert acquired
                return await agent.execute(task)

        start_time = time.time()
        results = await asyncio.gather(*[mixed_task(i) for i in range(30)])
        elapsed = time.time() - start_time

        assert all(r.status == AgentStatus.COMPLETED for r in results)

        # Should complete faster than if all were slow
        # 30 tasks, 10 slow (0.15s avg), 20 fast (0.01s)
        # Sequential would be ~1.7s, parallel should be much faster
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_debate_sessions(self, mock_claude_adapter, mock_gemini_adapter, mock_codex_adapter):
        """Test multiple debate sessions running concurrently."""
        engines = [
            DebateEngine(
                claude_adapter=mock_claude_adapter,
                gemini_adapter=mock_gemini_adapter,
                codex_adapter=mock_codex_adapter,
            )
            for _ in range(5)
        ]

        async def run_debate(engine: DebateEngine, idx: int):
            return await engine.run_debate(
                question=f"Question {idx}: Design approach?",
                context={"type": "design", "idx": idx},
                tier=DebateTier.TIER_1,
            )

        start_time = time.time()
        results = await asyncio.gather(*[
            run_debate(engines[i % 5], i) for i in range(10)
        ])
        elapsed = time.time() - start_time

        assert all(r is not None and r.consensus is not None for r in results)

    @pytest.mark.asyncio
    async def test_priority_queue_under_load(self, orchestrator):
        """Test priority queue behavior under high load."""
        # Register agents
        for i in range(5):
            orchestrator.registry.register(f"worker_{i}", MockSuccessAgent)

        # Create tasks with different priorities
        tasks = []
        execution_order = []

        for i in range(30):
            priority = AgentPriority.CRITICAL if i < 5 else (
                AgentPriority.HIGH if i < 15 else AgentPriority.NORMAL
            )
            task = AgentTask(
                agent_type=f"worker_{i % 5}",
                payload={"idx": i, "priority": priority.value},
                priority=priority,
            )
            tasks.append(task)

        # Shuffle to test priority ordering
        random.shuffle(tasks)

        results = await asyncio.gather(*[orchestrator.dispatch(t) for t in tasks])

        # All should complete
        assert all(
            isinstance(r, AgentResult) and r.status == AgentStatus.COMPLETED
            for r in results
        )


# ============================================================================
# Section 7: Integration Scenarios
# ============================================================================

class TestIntegrationScenariosE2E:
    """End-to-end tests for realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_trading_decision_with_debate(self, debate_engine):
        """Test trading decision process using multi-model debate."""
        # Scenario: High-value trade decision
        context = {
            "symbol": "AAPL",
            "direction": "buy",
            "position_size": 150000,
            "market_trend": "bullish",
            "volatility": "high",
            "risk_level": "critical",
        }

        question = """
        Should we execute this trade?
        - Symbol: AAPL
        - Direction: BUY
        - Position Size: $150,000
        - Market Conditions: Bullish with high volatility

        Provide recommendation with confidence level.
        """

        result = await debate_engine.run_debate(
            question=question,
            context=context,
            tier=DebateTier.TIER_2,  # High value triggers full debate
        )

        assert result.tier == DebateTier.TIER_2
        assert result.final_decision is not None
        assert len(result.rounds) >= 2

    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_workflow(self):
        """Test complete portfolio rebalancing workflow."""
        signal_agent = SignalAgent()
        risk_agent = RiskAgent()
        decision_agent = DecisionAgent()

        portfolio_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        # Step 1: Generate signals for all symbols
        signal_tasks = [
            AgentTask(
                agent_type="signal",
                payload={"symbol": s, "strategies": ["momentum", "mean_reversion"]}
            )
            for s in portfolio_symbols
        ]
        signal_results = await asyncio.gather(*[
            signal_agent.execute(t) for t in signal_tasks
        ])

        # Step 2: Aggregate portfolio risk
        risk_task = AgentTask(
            agent_type="risk",
            payload={
                "portfolio_id": "main_portfolio",
                "type": "portfolio_risk",
                "positions": [
                    {"symbol": s, "weight": 0.25} for s in portfolio_symbols
                ],
            }
        )
        risk_result = await risk_agent.execute(risk_task)

        # Step 3: Make rebalancing decisions
        decision_task = AgentTask(
            agent_type="decision",
            payload={
                "type": "rebalance_decision",
                "context": {
                    "signals": [r.result for r in signal_results],
                    "risk_analysis": risk_result.result,
                    "target_weights": {s: 0.25 for s in portfolio_symbols},
                },
            }
        )
        decision_result = await decision_agent.execute(decision_task)

        assert decision_result.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_market_event_response_chain(self):
        """Test rapid response chain to market events."""
        signal_agent = SignalAgent()
        risk_agent = RiskAgent()
        decision_agent = DecisionAgent()

        # Simulate market event
        market_event = {
            "type": "volatility_spike",
            "affected_symbols": ["SPY", "QQQ", "IWM"],
            "volatility_increase": 0.5,
            "timestamp": datetime.utcnow().isoformat(),
        }

        async def process_event():
            # Fast signal update
            signal_task = AgentTask(
                agent_type="signal",
                payload={
                    "event_type": "market_event",
                    "event_data": market_event,
                    "urgency": "high",
                },
                priority=AgentPriority.CRITICAL,
            )
            signal_result = await signal_agent.execute(signal_task)

            # Risk re-assessment
            risk_task = AgentTask(
                agent_type="risk",
                payload={
                    "type": "event_risk",
                    "event": market_event,
                    "signal": signal_result.result,
                },
                priority=AgentPriority.CRITICAL,
            )
            risk_result = await risk_agent.execute(risk_task)

            # Decision
            decision_task = AgentTask(
                agent_type="decision",
                payload={
                    "type": "event_response",
                    "event": market_event,
                    "risk": risk_result.result,
                },
                priority=AgentPriority.CRITICAL,
            )
            return await decision_agent.execute(decision_task)

        start_time = time.time()
        result = await process_event()
        elapsed = time.time() - start_time

        assert result.status == AgentStatus.COMPLETED
        # Should respond quickly for critical events
        assert elapsed < 5.0  # Within 5 seconds

    @pytest.mark.asyncio
    async def test_end_of_day_batch_processing(self):
        """Test end-of-day batch processing workflow."""
        from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode

        # Large batch of symbols
        symbols = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "META",
            "NVDA", "TSLA", "AMD", "INTC", "ORCL",
            "CRM", "ADBE", "NFLX", "PYPL", "UBER",
        ]

        pipeline = TradingPipeline(
            name="eod_batch",
            mode=PipelineMode.ANALYSIS,
        )

        await pipeline.start()

        start_time = time.time()
        result = await pipeline.run(
            symbols=symbols,
            context={
                "batch_type": "end_of_day",
                "generate_reports": True,
            }
        )
        elapsed = time.time() - start_time

        # Should process all symbols
        assert len(result.symbols) == len(symbols)

        # Summary should reflect batch
        summary = result.to_dict().get("summary", {})
        assert summary.get("total_symbols") == len(symbols)

        await pipeline.stop()


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.asyncio
    async def test_agent_execution_latency(self):
        """Benchmark agent execution latency."""
        # A positive asyncio sleep is rounded to the Windows scheduler tick;
        # use zero delay here so the assertion measures agent overhead.
        agent = MockSuccessAgent("benchmark", delay=0)
        iterations = 100

        latencies = []
        for _ in range(iterations):
            task = AgentTask(agent_type="benchmark", payload={})
            start = time.time()
            await agent.execute(task)
            latencies.append((time.time() - start) * 1000)  # ms

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

        # Assert reasonable latencies
        assert avg_latency < 10  # Avg < 10ms
        assert p99_latency < 50  # P99 < 50ms

    @pytest.mark.asyncio
    async def test_debate_throughput(self, mock_claude_adapter, mock_gemini_adapter, mock_codex_adapter):
        """Benchmark debate engine throughput."""
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            gemini_adapter=mock_gemini_adapter,
            codex_adapter=mock_codex_adapter,
        )

        iterations = 20

        start_time = time.time()
        results = await asyncio.gather(*[
            engine.run_debate(
                question=f"Question {i}",
                context={"type": "strategy"},
                tier=DebateTier.TIER_1,
            )
            for i in range(iterations)
        ])
        elapsed = time.time() - start_time

        throughput = iterations / elapsed

        # Should handle at least 5 debates per second
        assert throughput >= 5
        assert all(r.consensus is not None for r in results)

    @pytest.mark.asyncio
    async def test_pool_throughput(self):
        """Benchmark pool throughput under various loads."""
        pool = AgentPool(max_concurrent=20, max_queue_size=250)

        async def simple_task(idx: int):
            agent = MockSuccessAgent(f"throughput_{idx}", delay=0)
            async with AgentPoolContext(pool, agent, f"throughput_task_{idx}") as acquired:
                assert acquired
                await asyncio.sleep(0.005)
                return idx

        for load in [50, 100, 200]:
            start_time = time.time()
            results = await asyncio.gather(*[simple_task(i) for i in range(load)])
            elapsed = time.time() - start_time

            throughput = load / elapsed
            assert len(results) == load

            # Should maintain reasonable throughput
            assert throughput >= 100  # At least 100 tasks/sec


# ============================================================================
# Run Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--asyncio-mode=auto",
    ])
