# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the Agent Framework (Orchestrator, Registry, Pool).

Tests cover:
1. Agent Registry - registration and retrieval
2. Agent Pool - concurrency control
3. Orchestrator - task dispatch
4. Circuit Breaker - failure protection
5. Retry Policy - exponential backoff

Author: TraderX Team
Date: 2025-12-23
"""

import pytest
import asyncio
from typing import Any, Dict
from datetime import datetime
import time

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


# ============================================================================
# Mock Agents for Testing
# ============================================================================

class MockSuccessAgent(BaseAgent):
    """Mock agent that always succeeds."""

    def __init__(self, agent_id: str = None, delay: float = 0.1):
        super().__init__(agent_id)
        self.delay = delay
        self.execute_count = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute task successfully after delay."""
        self.execute_count += 1
        await asyncio.sleep(self.delay)
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"message": "success", "data": task.payload}
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in ["test_task", "success_task"]


class MockFailureAgent(BaseAgent):
    """Mock agent that always fails."""

    def __init__(self, agent_id: str = None, error_message: str = "Mock failure"):
        super().__init__(agent_id)
        self.error_message = error_message
        self.execute_count = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute task with failure."""
        self.execute_count += 1
        await asyncio.sleep(0.05)
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.FAILED,
            error=self.error_message
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type in ["test_task", "fail_task"]


class MockTimeoutAgent(BaseAgent):
    """Mock agent that times out."""

    def __init__(self, agent_id: str = None, delay: float = 10.0):
        super().__init__(agent_id)
        self.delay = delay

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute task with long delay."""
        await asyncio.sleep(self.delay)
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"message": "should not reach here"}
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "timeout_task"


class MockRetryableAgent(BaseAgent):
    """Mock agent that fails N times then succeeds."""

    def __init__(self, agent_id: str = None, fail_count: int = 2):
        super().__init__(agent_id)
        self.fail_count = fail_count
        self.execute_count = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute with retryable failures."""
        self.execute_count += 1
        await asyncio.sleep(0.05)

        if self.execute_count <= self.fail_count:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error="temporary network error"
            )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"attempts": self.execute_count}
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "retryable_task"


class MockCalculationAgent(BaseAgent):
    """Mock agent for calculation tasks."""

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute calculation."""
        await asyncio.sleep(0.05)
        a = task.payload.get("a", 0)
        b = task.payload.get("b", 0)
        result = a + b
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={"sum": result}
        )

    def can_handle(self, task_type: str) -> bool:
        return task_type == "calculation"


# ============================================================================
# Test Agent Registry
# ============================================================================

class TestAgentRegistry:
    """Test suite for AgentRegistry."""

    def test_agent_registry_creation(self):
        """Test creating a new registry."""
        registry = AgentRegistry()
        assert len(registry) == 0
        assert registry.count() == 0
        assert registry.list_all() == []

    def test_register_agent(self):
        """Test registering an agent class."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent, task_types=["test_task"])

        assert "success" in registry
        assert registry.has("success")
        assert registry.count() == 1
        assert "success" in registry.list_all()

    def test_register_multiple_agents(self):
        """Test registering multiple agent types."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        registry.register("failure", MockFailureAgent)
        registry.register("calculation", MockCalculationAgent)

        assert registry.count() == 3
        assert set(registry.list_all()) == {"success", "failure", "calculation"}

    def test_get_agent_class(self):
        """Test retrieving agent class."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)

        agent_class = registry.get("success")
        assert agent_class == MockSuccessAgent

        # Test non-existent agent
        assert registry.get("nonexistent") is None

    def test_get_for_task_type(self):
        """Test finding agent by task type."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent, task_types=["test_task", "success_task"])

        agent_class = registry.get_for_task("test_task")
        assert agent_class == MockSuccessAgent

        agent_class = registry.get_for_task("success_task")
        assert agent_class == MockSuccessAgent

        # Test non-existent task type
        assert registry.get_for_task("unknown_task") is None

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        assert registry.count() == 1

        result = registry.unregister("success")
        assert result is True
        assert registry.count() == 0
        assert not registry.has("success")

        # Test unregistering non-existent agent
        result = registry.unregister("nonexistent")
        assert result is False

    def test_list_with_info(self):
        """Test getting detailed agent info."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent, task_types=["test_task"])
        registry.register("failure", MockFailureAgent, task_types=["fail_task"])

        info = registry.list_with_info()
        assert len(info) == 2

        success_info = next(i for i in info if i["agent_type"] == "success")
        assert success_info["class_name"] == "MockSuccessAgent"
        assert "test_task" in success_info["task_types"]

    def test_clear_registry(self):
        """Test clearing all registrations."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        registry.register("failure", MockFailureAgent)
        assert registry.count() == 2

        registry.clear()
        assert registry.count() == 0
        assert registry.list_all() == []

    def test_register_invalid_type(self):
        """Test registering invalid agent class."""
        registry = AgentRegistry()

        with pytest.raises(TypeError):
            registry.register("invalid", str)

        with pytest.raises(TypeError):
            registry.register("invalid", lambda x: x)

    def test_registry_repr(self):
        """Test registry string representation."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)

        repr_str = repr(registry)
        assert "AgentRegistry" in repr_str
        assert "success" in repr_str


# ============================================================================
# Test Agent Pool
# ============================================================================

class TestAgentPool:
    """Test suite for AgentPool."""

    @pytest.mark.asyncio
    async def test_pool_creation(self):
        """Test creating agent pool."""
        pool = AgentPool(max_concurrent=5, max_queue_size=10)
        assert pool.max_concurrent == 5
        assert pool.max_queue_size == 10
        assert pool.get_active_count() == 0
        assert pool.get_available_slots() == 5

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Test acquiring and releasing pool slots."""
        pool = AgentPool(max_concurrent=2)
        agent1 = MockSuccessAgent(agent_id="agent-1")
        agent2 = MockSuccessAgent(agent_id="agent-2")

        # Acquire first slot
        acquired1 = await pool.acquire(agent1, task_id="task-1")
        assert acquired1 is True
        assert pool.get_active_count() == 1
        assert pool.get_available_slots() == 1
        assert pool.is_active("agent-1")

        # Acquire second slot
        acquired2 = await pool.acquire(agent2, task_id="task-2")
        assert acquired2 is True
        assert pool.get_active_count() == 2
        assert pool.get_available_slots() == 0

        # Release first slot
        released1 = pool.release("agent-1", success=True)
        assert released1 is True
        assert pool.get_active_count() == 1
        assert pool.get_available_slots() == 1
        assert not pool.is_active("agent-1")

        # Release second slot
        released2 = pool.release("agent-2", success=True)
        assert released2 is True
        assert pool.get_active_count() == 0
        assert pool.get_available_slots() == 2

    @pytest.mark.asyncio
    async def test_pool_concurrency_limit(self):
        """Test pool enforces concurrency limit."""
        pool = AgentPool(max_concurrent=2)
        agent1 = MockSuccessAgent(agent_id="agent-1")
        agent2 = MockSuccessAgent(agent_id="agent-2")
        agent3 = MockSuccessAgent(agent_id="agent-3")

        # Acquire both slots
        await pool.acquire(agent1)
        await pool.acquire(agent2)
        assert pool.get_available_slots() == 0

        # Try to acquire third slot with timeout
        start_time = time.time()
        acquired3 = await pool.acquire(agent3, timeout=0.5)
        elapsed = time.time() - start_time

        assert acquired3 is False
        assert 0.4 <= elapsed <= 0.7  # Should timeout around 0.5s

    @pytest.mark.asyncio
    async def test_pool_context_manager(self):
        """Test AgentPoolContext context manager."""
        pool = AgentPool(max_concurrent=2)
        agent = MockSuccessAgent()

        assert pool.get_active_count() == 0

        async with AgentPoolContext(pool, agent, task_id="task-1") as acquired:
            assert acquired is True
            assert pool.get_active_count() == 1
            assert pool.is_active(agent.agent_id)

        # After context exits
        assert pool.get_active_count() == 0
        assert not pool.is_active(agent.agent_id)

    @pytest.mark.asyncio
    async def test_pool_context_with_exception(self):
        """Test context manager properly releases on exception."""
        pool = AgentPool(max_concurrent=2)
        agent = MockSuccessAgent()

        assert pool.get_active_count() == 0

        try:
            async with AgentPoolContext(pool, agent, task_id="task-1") as acquired:
                assert acquired is True
                assert pool.get_active_count() == 1
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Pool should still release the slot
        assert pool.get_active_count() == 0

    @pytest.mark.asyncio
    async def test_pool_stats(self):
        """Test pool statistics."""
        pool = AgentPool(max_concurrent=5)
        agent1 = MockSuccessAgent(agent_id="agent-1")
        agent2 = MockSuccessAgent(agent_id="agent-2")

        # Initial stats
        stats = pool.get_stats()
        assert stats.total_capacity == 5
        assert stats.active_count == 0
        assert stats.idle_count == 5
        assert stats.utilization == 0.0

        # Acquire slots
        await pool.acquire(agent1)
        await pool.acquire(agent2)

        stats = pool.get_stats()
        assert stats.active_count == 2
        assert stats.idle_count == 3
        assert stats.utilization == 40.0

        # Release and check execution count
        pool.release("agent-1", success=True)
        pool.release("agent-2", success=False)

        stats = pool.get_stats()
        assert stats.total_executed == 2
        assert stats.total_failed == 1

    @pytest.mark.asyncio
    async def test_get_slot_info(self):
        """Test getting slot information."""
        pool = AgentPool(max_concurrent=2)
        agent = MockSuccessAgent(agent_id="agent-1")

        await pool.acquire(agent, task_id="task-123")

        info = pool.get_slot_info("agent-1")
        assert info is not None
        assert info["agent_id"] == "agent-1"
        assert info["agent_name"] == "MockSuccessAgent"
        assert info["task_id"] == "task-123"
        assert "duration_seconds" in info

        # Non-existent agent
        info = pool.get_slot_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        """Test cancelling all active agents."""
        pool = AgentPool(max_concurrent=3)
        agents = [
            MockSuccessAgent(agent_id=f"agent-{i}")
            for i in range(3)
        ]

        # Acquire all slots and set them to running state
        for agent in agents:
            await pool.acquire(agent)
            agent.status = AgentStatus.RUNNING  # Set to running state for cancellation

        assert pool.get_active_count() == 3

        # Cancel all (returns count of agents that were in RUNNING state)
        cancelled = await pool.cancel_all()
        assert cancelled == 3

        # Verify all agents are cancelled
        for agent in agents:
            assert agent.status == AgentStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_queue_full_error(self):
        """Test queue full error."""
        pool = AgentPool(max_concurrent=1, max_queue_size=1)

        # Fill the pool
        agent1 = MockSuccessAgent(agent_id="agent-1")
        await pool.acquire(agent1)

        # Start a pending acquisition
        agent2 = MockSuccessAgent(agent_id="agent-2")
        acquire_task = asyncio.create_task(pool.acquire(agent2, timeout=1.0))
        await asyncio.sleep(0.1)  # Let it start waiting

        # Try to exceed queue size
        agent3 = MockSuccessAgent(agent_id="agent-3")
        with pytest.raises(RuntimeError, match="Pool queue full"):
            await pool.acquire(agent3)

        # Cleanup
        pool.release("agent-1")
        await acquire_task


# ============================================================================
# Test Orchestrator Dispatch
# ============================================================================

class TestOrchestratorDispatch:
    """Test suite for AgentOrchestrator task dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_success(self):
        """Test successful task dispatch."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        pool = AgentPool(max_concurrent=5)
        orchestrator = AgentOrchestrator(registry, pool)

        task = AgentTask(
            agent_type="success",
            payload={"value": 42}
        )

        result = await orchestrator.dispatch(task)

        assert result.is_success
        assert result.status == AgentStatus.COMPLETED
        assert result.result["message"] == "success"
        assert result.result["data"]["value"] == 42
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_dispatch_failure(self):
        """Test failed task dispatch."""
        registry = AgentRegistry()
        registry.register("failure", MockFailureAgent)
        pool = AgentPool(max_concurrent=5)
        orchestrator = AgentOrchestrator(registry, pool)

        task = AgentTask(
            agent_type="failure",
            payload={}
        )

        result = await orchestrator.dispatch(task)

        assert result.is_failure
        assert result.status == AgentStatus.FAILED
        assert "Mock failure" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_agent_not_found(self):
        """Test dispatch with unknown agent type."""
        registry = AgentRegistry()
        pool = AgentPool(max_concurrent=5)
        orchestrator = AgentOrchestrator(registry, pool)

        task = AgentTask(
            agent_type="nonexistent",
            payload={}
        )

        result = await orchestrator.dispatch(task)

        assert result.is_failure
        assert "No agent found" in result.error

    @pytest.mark.asyncio
    async def test_dispatch_timeout(self):
        """Test task timeout."""
        registry = AgentRegistry()
        registry.register("timeout", MockTimeoutAgent)
        pool = AgentPool(max_concurrent=5)
        orchestrator = AgentOrchestrator(registry, pool)

        task = AgentTask(
            agent_type="timeout",
            payload={},
            timeout=0.5  # 0.5 second timeout
        )

        result = await orchestrator.dispatch(task)

        assert result.is_failure
        assert result.status == AgentStatus.TIMEOUT
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_dispatch_parallel(self):
        """Test parallel task dispatch."""
        registry = AgentRegistry()
        registry.register("calculation", MockCalculationAgent)
        pool = AgentPool(max_concurrent=10)
        orchestrator = AgentOrchestrator(registry, pool)

        tasks = [
            AgentTask(agent_type="calculation", payload={"a": i, "b": i * 2})
            for i in range(5)
        ]

        start_time = time.time()
        results = await orchestrator.dispatch_parallel(tasks)
        elapsed = time.time() - start_time

        # All should succeed
        assert len(results) == 5
        assert all(r.is_success for r in results)

        # Check results
        for i, result in enumerate(results):
            expected_sum = i + (i * 2)
            assert result.result["sum"] == expected_sum

        # Should take ~0.05s (parallel) not 0.25s (sequential)
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_dispatch_parallel_with_limit(self):
        """Test parallel dispatch with concurrency limit."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        pool = AgentPool(max_concurrent=10)
        orchestrator = AgentOrchestrator(registry, pool)

        tasks = [
            AgentTask(agent_type="success", payload={"id": i})
            for i in range(10)
        ]

        results = await orchestrator.dispatch_parallel(tasks, max_concurrent=3)

        assert len(results) == 10
        assert all(r.is_success for r in results)

    @pytest.mark.asyncio
    async def test_dispatch_priority_queue(self):
        """Test priority-based dispatch."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        pool = AgentPool(max_concurrent=1)  # Force sequential execution
        orchestrator = AgentOrchestrator(registry, pool)

        tasks = [
            AgentTask(agent_type="success", payload={"id": 1}, priority=AgentPriority.LOW),
            AgentTask(agent_type="success", payload={"id": 2}, priority=AgentPriority.CRITICAL),
            AgentTask(agent_type="success", payload={"id": 3}, priority=AgentPriority.NORMAL),
            AgentTask(agent_type="success", payload={"id": 4}, priority=AgentPriority.HIGH),
        ]

        results = await orchestrator.dispatch_priority_queue(tasks)

        # Should execute in priority order: CRITICAL, HIGH, NORMAL, LOW
        assert results[0].result["data"]["id"] == 2  # CRITICAL
        assert results[1].result["data"]["id"] == 4  # HIGH
        assert results[2].result["data"]["id"] == 3  # NORMAL
        assert results[3].result["data"]["id"] == 1  # LOW

    @pytest.mark.asyncio
    async def test_orchestrator_metrics(self):
        """Test orchestrator metrics tracking."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        registry.register("failure", MockFailureAgent)
        pool = AgentPool(max_concurrent=5)
        orchestrator = AgentOrchestrator(registry, pool)

        # Execute some tasks
        await orchestrator.dispatch(AgentTask(agent_type="success", payload={}))
        await orchestrator.dispatch(AgentTask(agent_type="success", payload={}))
        await orchestrator.dispatch(AgentTask(agent_type="failure", payload={}))

        metrics = orchestrator.get_metrics()

        assert metrics["dispatch_count"] >= 3
        assert metrics["success_count"] >= 2
        assert metrics["failure_count"] >= 1
        assert 0 <= metrics["success_rate"] <= 100
        assert "circuit_breaker_state" in metrics
        assert "pool_stats" in metrics


# ============================================================================
# Test Circuit Breaker
# ============================================================================

class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute() is True

    def test_circuit_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=10.0
        )
        breaker = CircuitBreaker(config)

        # Record failures
        for i in range(3):
            breaker.record_failure()
            if i < 2:
                assert breaker.state == CircuitState.CLOSED
            else:
                assert breaker.state == CircuitState.OPEN

        assert not breaker.can_execute()

    def test_circuit_half_open_recovery(self):
        """Test circuit transitions to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,  # 0.5 second recovery
            half_open_max_calls=2
        )
        breaker = CircuitBreaker(config)

        # Trigger circuit open
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.6)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute() is True

    def test_circuit_closes_after_success(self):
        """Test circuit closes after successful calls in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            half_open_max_calls=2
        )
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

        # Record successful calls
        breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_reopens_on_half_open_failure(self):
        """Test circuit reopens if failure occurs in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5
        )
        breaker = CircuitBreaker(config)

        # Open circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for recovery
        time.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """Test successful call resets failure count."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        # Record some failures
        breaker.record_failure()
        breaker.record_failure()

        # Success should reset
        breaker.record_success()

        # More failures - should need 3 more to open
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_orchestrator(self):
        """Test circuit breaker integration with orchestrator."""
        registry = AgentRegistry()
        registry.register("failure", MockFailureAgent)
        pool = AgentPool(max_concurrent=5)

        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0
        )
        breaker = CircuitBreaker(circuit_config)
        retry_config = RetryConfig(max_retries=0)  # No retries for this test

        orchestrator = AgentOrchestrator(
            registry,
            pool,
            circuit_breaker=breaker,
            retry_policy=RetryPolicy(retry_config)
        )

        # Trigger circuit open
        for _ in range(3):
            result = await orchestrator.dispatch(
                AgentTask(agent_type="failure", payload={})
            )
            assert result.is_failure

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

        # Next request should be rejected
        result = await orchestrator.dispatch(
            AgentTask(agent_type="failure", payload={})
        )
        assert result.is_failure
        assert "Circuit breaker open" in result.error


# ============================================================================
# Test Retry Policy
# ============================================================================

class TestRetryPolicy:
    """Test suite for RetryPolicy."""

    def test_retry_policy_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0
        )
        policy = RetryPolicy(config)

        # Test exponential growth
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

        # Test max delay cap
        assert policy.get_delay(10) == 30.0

    def test_retry_policy_should_retry(self):
        """Test retry decision logic."""
        config = RetryConfig(
            max_retries=3,
            retryable_errors=["timeout", "temporary", "network"]
        )
        policy = RetryPolicy(config)

        # Retryable errors
        assert policy.should_retry("Connection timeout", attempt=0) is True
        assert policy.should_retry("Temporary failure", attempt=1) is True
        assert policy.should_retry("Network error", attempt=2) is True

        # Non-retryable errors
        assert policy.should_retry("Invalid request", attempt=0) is False
        assert policy.should_retry("Authentication failed", attempt=1) is False

        # Max retries exceeded
        assert policy.should_retry("Timeout", attempt=3) is False
        assert policy.should_retry("Timeout", attempt=4) is False

    @pytest.mark.asyncio
    async def test_retry_with_orchestrator(self):
        """Test retry policy integration with orchestrator."""
        registry = AgentRegistry()
        retryable_agent = MockRetryableAgent(fail_count=2)
        registry.register("retryable", MockRetryableAgent)
        pool = AgentPool(max_concurrent=5)

        retry_config = RetryConfig(
            max_retries=3,
            initial_delay=0.1,
            exponential_base=1.5,
            retryable_errors=["temporary", "network"]
        )

        orchestrator = AgentOrchestrator(
            registry,
            pool,
            retry_policy=RetryPolicy(retry_config)
        )

        # Use a consistent agent instance
        task = AgentTask(agent_type="retryable", payload={})

        # Manually create agent with fail_count=2
        agent = MockRetryableAgent(agent_id="retry-test", fail_count=2)

        # Track execution
        result = await orchestrator._execute_with_retry(agent, task, timeout=5.0)

        # Should succeed after retries
        assert result.is_success
        assert agent.execute_count == 3  # Failed 2 times, succeeded on 3rd

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test behavior when retries are exhausted."""
        registry = AgentRegistry()
        # Agent that always fails with retryable error
        registry.register("failure", MockFailureAgent, task_types=["fail_task"])
        pool = AgentPool(max_concurrent=5)

        # Make error retryable
        agent = MockFailureAgent(error_message="temporary network error")

        retry_config = RetryConfig(
            max_retries=2,
            initial_delay=0.05,
            retryable_errors=["temporary"]
        )

        orchestrator = AgentOrchestrator(
            registry,
            pool,
            retry_policy=RetryPolicy(retry_config)
        )

        task = AgentTask(agent_type="failure", payload={})

        result = await orchestrator._execute_with_retry(agent, task, timeout=5.0)

        assert result.is_failure
        # Either contains "Max retries exceeded" or the original error after retries
        assert ("Max retries exceeded" in result.error or
                "temporary network error" in result.error)
        assert agent.execute_count == 3  # Initial + 2 retries


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete agent framework."""

    @pytest.mark.asyncio
    async def test_full_workflow_success(self):
        """Test complete workflow with successful execution."""
        # Setup
        registry = AgentRegistry()
        registry.register("calculation", MockCalculationAgent)
        registry.register("success", MockSuccessAgent)

        pool = AgentPool(max_concurrent=10)
        orchestrator = AgentOrchestrator(registry, pool)

        # Execute multiple tasks
        tasks = [
            AgentTask(
                agent_type="calculation",
                payload={"a": i, "b": i * 2},
                priority=AgentPriority.NORMAL
            )
            for i in range(5)
        ]

        results = await orchestrator.dispatch_parallel(tasks)

        # Verify all succeeded
        assert len(results) == 5
        assert all(r.is_success for r in results)

        # Check metrics
        metrics = orchestrator.get_metrics()
        assert metrics["success_count"] == 5
        assert metrics["failure_count"] == 0
        assert metrics["success_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_full_workflow_with_failures(self):
        """Test workflow handling failures gracefully."""
        registry = AgentRegistry()
        registry.register("success", MockSuccessAgent)
        registry.register("failure", MockFailureAgent)

        pool = AgentPool(max_concurrent=5)

        circuit_config = CircuitBreakerConfig(failure_threshold=10)
        retry_config = RetryConfig(max_retries=0)

        orchestrator = AgentOrchestrator(
            registry,
            pool,
            circuit_breaker=CircuitBreaker(circuit_config),
            retry_policy=RetryPolicy(retry_config)
        )

        # Mix of success and failure tasks
        tasks = [
            AgentTask(agent_type="success", payload={"id": 1}),
            AgentTask(agent_type="failure", payload={"id": 2}),
            AgentTask(agent_type="success", payload={"id": 3}),
            AgentTask(agent_type="failure", payload={"id": 4}),
        ]

        results = await orchestrator.dispatch_parallel(tasks)

        # Check results
        assert len(results) == 4
        assert results[0].is_success
        assert results[1].is_failure
        assert results[2].is_success
        assert results[3].is_failure

        # Metrics
        metrics = orchestrator.get_metrics()
        assert metrics["success_count"] == 2
        assert metrics["failure_count"] == 2
        assert metrics["success_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """Stress test with high concurrency."""
        registry = AgentRegistry()
        registry.register("calculation", MockCalculationAgent)

        pool = AgentPool(max_concurrent=50)
        orchestrator = AgentOrchestrator(registry, pool)

        # Create 100 tasks
        tasks = [
            AgentTask(
                agent_type="calculation",
                payload={"a": i, "b": i + 1}
            )
            for i in range(100)
        ]

        start_time = time.time()
        results = await orchestrator.dispatch_parallel(tasks, max_concurrent=50)
        elapsed = time.time() - start_time

        # All should succeed
        assert len(results) == 100
        assert all(r.is_success for r in results)

        # Should complete reasonably fast (parallel execution)
        assert elapsed < 2.0  # Should be much faster than 5 seconds (100 * 0.05)

        # Verify results
        for i, result in enumerate(results):
            assert result.result["sum"] == i + (i + 1)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
