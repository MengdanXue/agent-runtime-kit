# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agent Pool for managing concurrent agent execution.
"""

import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base import BaseAgent, AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class PoolStats:
    """Statistics about pool usage."""
    total_capacity: int
    active_count: int
    idle_count: int
    pending_count: int
    total_executed: int = 0
    total_failed: int = 0

    @property
    def utilization(self) -> float:
        """Get pool utilization as a percentage."""
        if self.total_capacity == 0:
            return 0.0
        return (self.active_count / self.total_capacity) * 100


@dataclass
class AgentSlot:
    """Represents a slot in the agent pool."""
    agent: BaseAgent
    acquired_at: datetime = field(default_factory=datetime.utcnow)
    task_id: Optional[str] = None


class AgentPool:
    """
    Pool for managing concurrent agent execution.

    Provides semaphore-based concurrency control and tracks
    active agents for monitoring and management.

    Example:
        pool = AgentPool(max_concurrent=10)

        async with pool.acquire(my_agent) as slot:
            result = await my_agent.execute(task)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        max_queue_size: int = 100
    ):
        """
        Initialize the agent pool.

        Args:
            max_concurrent: Maximum number of concurrent agents
            max_queue_size: Maximum number of pending acquisitions
        """
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_agents: Dict[str, AgentSlot] = {}
        self._pending_count = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._total_executed = 0
        self._total_failed = 0

    async def acquire(
        self,
        agent: BaseAgent,
        task_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire a slot in the pool for an agent.

        Args:
            agent: The agent to acquire a slot for
            task_id: Optional task ID for tracking
            timeout: Optional timeout in seconds

        Returns:
            True if slot was acquired, False on timeout

        Raises:
            RuntimeError: If queue is full
        """
        async with self._lock:
            if self._pending_count >= self.max_queue_size:
                raise RuntimeError(f"Pool queue full ({self.max_queue_size})")
            self._pending_count += 1

        try:
            if timeout:
                acquired = await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=timeout
                )
            else:
                await self._semaphore.acquire()
                acquired = True

            if acquired:
                async with self._lock:
                    self._active_agents[agent.agent_id] = AgentSlot(
                        agent=agent,
                        task_id=task_id
                    )
                logger.debug(f"Agent {agent.agent_id} acquired pool slot")
                return True

        except asyncio.TimeoutError:
            logger.warning(f"Timeout acquiring pool slot for {agent.agent_id}")
            return False
        finally:
            async with self._lock:
                self._pending_count -= 1

        return False

    def release(self, agent_id: str, success: bool = True) -> bool:
        """
        Release a pool slot.

        Args:
            agent_id: ID of the agent to release
            success: Whether the task completed successfully

        Returns:
            True if slot was released, False if not found
        """
        if agent_id in self._active_agents:
            del self._active_agents[agent_id]
            self._semaphore.release()

            # Update statistics
            self._total_executed += 1
            if not success:
                self._total_failed += 1

            logger.debug(f"Agent {agent_id} released pool slot")
            return True
        return False

    def is_active(self, agent_id: str) -> bool:
        """Check if an agent is currently active in the pool."""
        return agent_id in self._active_agents

    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        return list(self._active_agents.keys())

    def get_active_count(self) -> int:
        """Get number of currently active agents."""
        return len(self._active_agents)

    def get_available_slots(self) -> int:
        """Get number of available slots."""
        return self.max_concurrent - len(self._active_agents)

    def get_stats(self) -> PoolStats:
        """Get current pool statistics."""
        return PoolStats(
            total_capacity=self.max_concurrent,
            active_count=len(self._active_agents),
            idle_count=self.get_available_slots(),
            pending_count=self._pending_count,
            total_executed=self._total_executed,
            total_failed=self._total_failed
        )

    def get_slot_info(self, agent_id: str) -> Optional[Dict]:
        """Get information about a specific slot."""
        slot = self._active_agents.get(agent_id)
        if slot:
            return {
                "agent_id": agent_id,
                "agent_name": slot.agent.name,
                "status": slot.agent.status.value,
                "task_id": slot.task_id,
                "acquired_at": slot.acquired_at.isoformat(),
                "duration_seconds": (datetime.utcnow() - slot.acquired_at).total_seconds()
            }
        return None

    async def cancel_all(self) -> int:
        """
        Cancel all active agents.

        Returns:
            Number of agents cancelled
        """
        cancelled = 0
        for agent_id, slot in list(self._active_agents.items()):
            if slot.agent.cancel():
                cancelled += 1
        return cancelled

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"AgentPool(active={stats.active_count}/{stats.total_capacity}, "
            f"pending={stats.pending_count}, "
            f"utilization={stats.utilization:.1f}%)"
        )


class AgentPoolContext:
    """
    Context manager for pool slot acquisition.

    Example:
        async with AgentPoolContext(pool, agent, task_id) as acquired:
            if acquired:
                result = await agent.execute(task)
    """

    def __init__(
        self,
        pool: AgentPool,
        agent: BaseAgent,
        task_id: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        self.pool = pool
        self.agent = agent
        self.task_id = task_id
        self.timeout = timeout
        self._acquired = False
        self._success = True

    async def __aenter__(self) -> bool:
        self._acquired = await self.pool.acquire(
            self.agent,
            task_id=self.task_id,
            timeout=self.timeout
        )
        return self._acquired

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            self._success = exc_type is None
            self.pool.release(self.agent.agent_id, success=self._success)
        return False  # Don't suppress exceptions
