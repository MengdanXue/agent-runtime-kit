# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agent Orchestrator for task dispatch and coordination.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import time

from .base import AgentTask, AgentResult, AgentStatus, BaseAgent, AgentPriority
from .registry import AgentRegistry
from .pool import AgentPool, AgentPoolContext

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            if self._should_try_recovery():
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def _should_try_recovery(self) -> bool:
        """Check if recovery timeout has passed."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.config.recovery_timeout

    def record_success(self):
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker closed (recovered)")
        else:
            self._failure_count = 0

    def record_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker opened (half-open test failed)")
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened (failures: {self._failure_count})")

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        return self.state != CircuitState.OPEN


@dataclass
class RetryConfig:
    """Configuration for retry policy."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 30.0     # seconds
    exponential_base: float = 2.0
    retryable_errors: List[str] = field(default_factory=lambda: ["timeout", "temporary"])


class RetryPolicy:
    """
    Retry policy with exponential backoff.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.config.initial_delay * (self.config.exponential_base ** attempt)
        return min(delay, self.config.max_delay)

    def should_retry(self, error: str, attempt: int) -> bool:
        """Check if should retry based on error and attempt count."""
        if attempt >= self.config.max_retries:
            return False

        # Check if error is retryable
        error_lower = error.lower()
        for retryable in self.config.retryable_errors:
            if retryable in error_lower:
                return True

        return False


class AgentOrchestrator:
    """
    Central orchestrator for agent task dispatch and coordination.

    Features:
    - Task routing to appropriate agents
    - Parallel task execution
    - Circuit breaker protection
    - Retry policy
    - Priority-based scheduling

    Example:
        orchestrator = AgentOrchestrator(registry, pool)

        # Single task
        result = await orchestrator.dispatch(task)

        # Parallel tasks
        results = await orchestrator.dispatch_parallel([task1, task2, task3])
    """

    def __init__(
        self,
        registry: AgentRegistry,
        pool: AgentPool,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_policy: Optional[RetryPolicy] = None,
        redis_client: Any = None
    ):
        """
        Initialize the orchestrator.

        Args:
            registry: Agent registry for type lookups
            pool: Agent pool for concurrency control
            circuit_breaker: Optional circuit breaker
            retry_policy: Optional retry policy
            redis_client: Optional Redis client for pub/sub
        """
        self.registry = registry
        self.pool = pool
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.retry_policy = retry_policy or RetryPolicy()
        self.redis = redis_client

        # Metrics
        self._dispatch_count = 0
        self._success_count = 0
        self._failure_count = 0

    async def dispatch(
        self,
        task: AgentTask,
        timeout: Optional[float] = None
    ) -> AgentResult:
        """
        Dispatch a single task to an appropriate agent.

        Args:
            task: The task to dispatch
            timeout: Optional timeout override

        Returns:
            AgentResult with execution outcome
        """
        self._dispatch_count += 1
        timeout = timeout or task.timeout

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error="Circuit breaker open - service unavailable"
            )

        # Find appropriate agent class
        agent_class = self.registry.get(task.agent_type)
        if not agent_class:
            agent_class = self.registry.get_for_task(task.agent_type)

        if not agent_class:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=f"No agent found for type: {task.agent_type}"
            )

        # Create agent instance
        agent = agent_class(agent_id=f"{task.agent_type}_{task.task_id[:8]}")

        # Execute with retry
        return await self._execute_with_retry(agent, task, timeout)

    async def _execute_with_retry(
        self,
        agent: BaseAgent,
        task: AgentTask,
        timeout: float
    ) -> AgentResult:
        """Execute task with retry logic."""
        last_error = None
        attempts = 0

        while attempts <= self.retry_policy.config.max_retries:
            try:
                result = await self._execute_once(agent, task, timeout)

                if result.is_success:
                    self.circuit_breaker.record_success()
                    self._success_count += 1
                    return result

                # Check if should retry
                if result.error and self.retry_policy.should_retry(result.error, attempts):
                    last_error = result.error
                    delay = self.retry_policy.get_delay(attempts)
                    logger.info(f"Retrying task {task.task_id} after {delay}s (attempt {attempts + 1})")
                    await asyncio.sleep(delay)
                    attempts += 1
                    continue

                # Not retryable
                self.circuit_breaker.record_failure()
                self._failure_count += 1
                return result

            except Exception as e:
                last_error = str(e)
                if self.retry_policy.should_retry(str(e), attempts):
                    delay = self.retry_policy.get_delay(attempts)
                    await asyncio.sleep(delay)
                    attempts += 1
                    continue
                break

        # All retries exhausted
        self.circuit_breaker.record_failure()
        self._failure_count += 1
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.FAILED,
            error=f"Max retries exceeded. Last error: {last_error}"
        )

    async def _execute_once(
        self,
        agent: BaseAgent,
        task: AgentTask,
        timeout: float
    ) -> AgentResult:
        """Execute task once without retry."""
        async with AgentPoolContext(self.pool, agent, task.task_id, timeout=5.0) as acquired:
            if not acquired:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    error="Failed to acquire pool slot"
                )

            try:
                result = await asyncio.wait_for(
                    agent._execute_with_tracking(task),
                    timeout=timeout
                )
                return result
            except asyncio.TimeoutError:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.TIMEOUT,
                    error=f"Task timeout after {timeout}s"
                )

    async def dispatch_parallel(
        self,
        tasks: List[AgentTask],
        max_concurrent: Optional[int] = None
    ) -> List[AgentResult]:
        """
        Dispatch multiple tasks in parallel.

        Args:
            tasks: List of tasks to dispatch
            max_concurrent: Optional limit on concurrent tasks

        Returns:
            List of AgentResults in same order as input tasks
        """
        if not tasks:
            return []

        if max_concurrent:
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)

            async def limited_dispatch(task: AgentTask) -> AgentResult:
                async with semaphore:
                    return await self.dispatch(task)

            return await asyncio.gather(*[limited_dispatch(t) for t in tasks])
        else:
            return await asyncio.gather(*[self.dispatch(t) for t in tasks])

    async def dispatch_priority_queue(
        self,
        tasks: List[AgentTask]
    ) -> List[AgentResult]:
        """
        Dispatch tasks respecting priority order.

        Higher priority tasks are dispatched first.

        Args:
            tasks: List of tasks to dispatch

        Returns:
            List of AgentResults in priority order
        """
        # Sort by priority (higher priority first)
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t.priority.value if isinstance(t.priority, AgentPriority) else t.priority,
            reverse=True
        )

        results = []
        for task in sorted_tasks:
            result = await self.dispatch(task)
            results.append(result)

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "dispatch_count": self._dispatch_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": (
                self._success_count / self._dispatch_count * 100
                if self._dispatch_count > 0 else 0
            ),
            # 前端期望大写状态: CLOSED, OPEN, HALF_OPEN
            "circuit_breaker_state": self.circuit_breaker.state.name,
            "pool_stats": self.pool.get_stats().__dict__
        }

    def __repr__(self) -> str:
        metrics = self.get_metrics()
        return (
            f"AgentOrchestrator("
            f"dispatched={metrics['dispatch_count']}, "
            f"success_rate={metrics['success_rate']:.1f}%, "
            f"circuit={metrics['circuit_breaker_state']})"
        )
