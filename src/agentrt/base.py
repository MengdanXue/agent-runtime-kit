# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Base classes for the agent framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
import uuid


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentTask:
    """
    Represents a task to be executed by an agent.

    Attributes:
        task_id: Unique identifier for this task
        agent_type: Type of agent required to handle this task
        payload: Task-specific data and parameters
        priority: Task priority level
        timeout: Maximum execution time in seconds
        retry_count: Number of retries on failure
        metadata: Additional task metadata
    """
    agent_type: str
    payload: Dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: AgentPriority = AgentPriority.NORMAL
    timeout: int = 300  # 5 minutes default
    retry_count: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if isinstance(self.priority, int):
            self.priority = AgentPriority(self.priority)


@dataclass
class AgentResult:
    """
    Result of an agent task execution.

    Attributes:
        task_id: ID of the executed task
        status: Final execution status
        result: Task result data (if successful)
        error: Error message (if failed)
        duration_ms: Execution duration in milliseconds
        metadata: Additional result metadata
    """
    task_id: str
    status: AgentStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == AgentStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        """Check if task failed."""
        return self.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT, AgentStatus.CANCELLED)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    All agent implementations must inherit from this class and implement
    the execute() and can_handle() methods.

    Example:
        class MyAgent(BaseAgent):
            async def execute(self, task: AgentTask) -> AgentResult:
                # Implementation here
                pass

            def can_handle(self, task_type: str) -> bool:
                return task_type == "my_task_type"
    """

    def __init__(self, agent_id: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent instance
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self._current_task: Optional[AgentTask] = None
        self._start_time: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Get the agent's class name."""
        return self.__class__.__name__

    @property
    def is_busy(self) -> bool:
        """Check if agent is currently executing a task."""
        return self.status == AgentStatus.RUNNING

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute the given task.

        This method must be implemented by all agent subclasses.

        Args:
            task: The task to execute

        Returns:
            AgentResult containing the execution outcome
        """
        pass

    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """
        Check if this agent can handle the given task type.

        Args:
            task_type: The type of task to check

        Returns:
            True if this agent can handle the task type
        """
        pass

    async def _execute_with_tracking(self, task: AgentTask) -> AgentResult:
        """
        Execute task with status tracking and timing.

        This is an internal method that wraps execute() with
        status management and duration tracking.
        """
        import time

        self._current_task = task
        self._start_time = datetime.utcnow()
        self.status = AgentStatus.RUNNING
        start_ms = time.time() * 1000

        try:
            result = await self.execute(task)
            result.duration_ms = int(time.time() * 1000 - start_ms)
            return result
        except Exception as e:
            duration_ms = int(time.time() * 1000 - start_ms)
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=str(e),
                duration_ms=duration_ms
            )
        finally:
            self._current_task = None
            self._start_time = None
            if self.status == AgentStatus.RUNNING:
                self.status = AgentStatus.IDLE

    def cancel(self) -> bool:
        """
        Request cancellation of the current task.

        Returns:
            True if cancellation was requested successfully
        """
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.CANCELLED
            return True
        return False

    def get_info(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "is_busy": self.is_busy,
            "current_task": self._current_task.task_id if self._current_task else None,
            "start_time": self._start_time.isoformat() if self._start_time else None
        }

    def __repr__(self) -> str:
        return f"{self.name}(id={self.agent_id[:8]}, status={self.status.value})"
