# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agent Registry for managing agent type registrations.
"""

from typing import Dict, List, Optional, Type, Set
from .base import BaseAgent
import logging

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for agent types.

    Manages registration and discovery of agent classes. Supports
    multiple task types per agent and automatic capability discovery.

    Example:
        registry = AgentRegistry()
        registry.register("signal", SignalAgent)
        registry.register("backtest", BacktestAgent)

        agent_class = registry.get("signal")
        agent = agent_class(agent_id="my-agent")
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._agents: Dict[str, Type[BaseAgent]] = {}
        self._task_types: Dict[str, Set[str]] = {}  # agent_type -> task_types

    def register(
        self,
        agent_type: str,
        agent_class: Type[BaseAgent],
        task_types: Optional[List[str]] = None
    ) -> None:
        """
        Register an agent class.

        Args:
            agent_type: Unique identifier for this agent type
            agent_class: The agent class to register
            task_types: List of task types this agent handles (optional)

        Raises:
            ValueError: If agent_type is already registered
            TypeError: If agent_class is not a BaseAgent subclass
        """
        if not isinstance(agent_class, type) or not issubclass(agent_class, BaseAgent):
            raise TypeError(f"{agent_class} must be a BaseAgent subclass")

        if agent_type in self._agents:
            logger.warning(f"Overwriting existing agent type: {agent_type}")

        self._agents[agent_type] = agent_class
        self._task_types[agent_type] = set(task_types or [])

        logger.info(f"Registered agent: {agent_type} -> {agent_class.__name__}")

    def unregister(self, agent_type: str) -> bool:
        """
        Unregister an agent type.

        Args:
            agent_type: The agent type to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_type in self._agents:
            del self._agents[agent_type]
            del self._task_types[agent_type]
            logger.info(f"Unregistered agent: {agent_type}")
            return True
        return False

    def get(self, agent_type: str) -> Optional[Type[BaseAgent]]:
        """
        Get an agent class by type.

        Args:
            agent_type: The agent type to retrieve

        Returns:
            The agent class, or None if not found
        """
        return self._agents.get(agent_type)

    def get_for_task(self, task_type: str) -> Optional[Type[BaseAgent]]:
        """
        Find an agent class that can handle the given task type.

        Args:
            task_type: The task type to find an agent for

        Returns:
            An agent class that can handle the task, or None
        """
        # First check explicit task type mappings
        for agent_type, task_types in self._task_types.items():
            if task_type in task_types:
                return self._agents[agent_type]

        # Fall back to checking each agent's can_handle method
        for agent_type, agent_class in self._agents.items():
            try:
                # Create a temporary instance to check capability
                temp_agent = agent_class(agent_id="capability-check")
                if temp_agent.can_handle(task_type):
                    return agent_class
            except Exception:
                continue

        return None

    def list_all(self) -> List[str]:
        """
        Get all registered agent types.

        Returns:
            List of registered agent type names
        """
        return list(self._agents.keys())

    def list_with_info(self) -> List[Dict]:
        """
        Get detailed info about all registered agents.

        Returns:
            List of dicts containing agent type, class name, and task types
        """
        return [
            {
                "agent_type": agent_type,
                "class_name": agent_class.__name__,
                "task_types": list(self._task_types.get(agent_type, set()))
            }
            for agent_type, agent_class in self._agents.items()
        ]

    def has(self, agent_type: str) -> bool:
        """Check if an agent type is registered."""
        return agent_type in self._agents

    def count(self) -> int:
        """Get the number of registered agent types."""
        return len(self._agents)

    def clear(self) -> None:
        """Clear all registrations."""
        self._agents.clear()
        self._task_types.clear()
        logger.info("Registry cleared")

    def __contains__(self, agent_type: str) -> bool:
        return self.has(agent_type)

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={self.list_all()})"


# Global registry instance
_global_registry: Optional[AgentRegistry] = None


def get_global_registry() -> AgentRegistry:
    """Get the global agent registry (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def register_agent(
    agent_type: str,
    task_types: Optional[List[str]] = None
):
    """
    Decorator for registering agent classes.

    Example:
        @register_agent("signal", task_types=["generate_signal", "multi_signal"])
        class SignalAgent(BaseAgent):
            ...
    """
    def decorator(cls: Type[BaseAgent]) -> Type[BaseAgent]:
        get_global_registry().register(agent_type, cls, task_types)
        return cls
    return decorator
