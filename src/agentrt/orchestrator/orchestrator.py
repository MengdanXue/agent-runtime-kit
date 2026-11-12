# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agent Orchestrator - 多代理协调器
负责代理的注册、调度、消息路由和生命周期管理

设计特点：
1. 支持并行和串行执行模式
2. 基于优先级的调度
3. 依赖解析和DAG执行
4. 内置容错和重试机制
5. 实时监控和指标收集
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Coroutine
from uuid import uuid4
import logging
import traceback

from .agent_base import (
    BaseAgent,
    AgentMessage,
    AgentResult,
    AgentState,
    AgentPriority,
    MessageType,
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"   # 串行执行
    PARALLEL = "parallel"       # 并行执行
    DAG = "dag"                 # 依赖图执行


@dataclass
class ExecutionPlan:
    """执行计划"""
    id: str = field(default_factory=lambda: str(uuid4()))
    mode: ExecutionMode = ExecutionMode.PARALLEL
    agents: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # agent -> [dependencies]
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 120.0
    retry_count: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionResult:
    """执行结果"""
    plan_id: str
    success: bool
    results: Dict[str, AgentResult] = field(default_factory=dict)
    failed_agents: List[str] = field(default_factory=list)
    total_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AgentOrchestrator:
    """
    多代理协调器

    核心功能：
    1. 代理注册和生命周期管理
    2. 消息路由和分发
    3. 执行计划调度
    4. 依赖解析和DAG执行
    5. 监控和指标收集
    """

    def __init__(self, name: str = "orchestrator"):
        self.name = name
        self.id = str(uuid4())

        # 代理注册表
        self._agents: Dict[str, BaseAgent] = {}

        # 主题订阅
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)

        # 消息总线
        self._message_bus: asyncio.Queue[AgentMessage] = asyncio.Queue()

        # 执行历史
        self._execution_history: List[ExecutionResult] = []

        # 运行状态
        self._running = False
        self._message_router_task: Optional[asyncio.Task] = None

        # 事件回调
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # 指标
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_messages_routed": 0,
            "agents_registered": 0,
        }

        logger.info(f"Orchestrator {self.name} ({self.id}) initialized")

    # === 代理管理 ===

    def register_agent(self, agent: BaseAgent) -> None:
        """注册代理"""
        if agent.name in self._agents:
            raise ValueError(f"Agent {agent.name} already registered")

        agent._orchestrator = self
        self._agents[agent.name] = agent

        # 注册代理的订阅
        for topic in agent._subscriptions:
            self._topic_subscribers[topic].add(agent.name)

        self._metrics["agents_registered"] = len(self._agents)
        logger.info(f"Registered agent: {agent.name}")

    def unregister_agent(self, agent_name: str) -> None:
        """注销代理"""
        if agent_name not in self._agents:
            return

        agent = self._agents.pop(agent_name)

        # 移除订阅
        for topic in agent._subscriptions:
            self._topic_subscribers[topic].discard(agent_name)

        self._metrics["agents_registered"] = len(self._agents)
        logger.info(f"Unregistered agent: {agent_name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """获取代理"""
        return self._agents.get(name)

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有代理"""
        return [
            {
                "name": agent.name,
                "id": agent.id,
                "state": agent.state.name,
                "priority": agent.priority.name,
                "metrics": agent.metrics,
            }
            for agent in self._agents.values()
        ]

    # === 消息路由 ===

    async def start(self) -> None:
        """启动协调器"""
        if self._running:
            return

        self._running = True

        # 启动所有代理
        start_tasks = [agent.start() for agent in self._agents.values()]
        await asyncio.gather(*start_tasks, return_exceptions=True)

        # 启动消息路由器
        self._message_router_task = asyncio.create_task(self._message_router())

        logger.info(f"Orchestrator {self.name} started")

    async def stop(self) -> None:
        """停止协调器"""
        if not self._running:
            return

        self._running = False

        # 停止消息路由器
        if self._message_router_task:
            self._message_router_task.cancel()
            try:
                await self._message_router_task
            except asyncio.CancelledError:
                pass

        # 停止所有代理
        stop_tasks = [agent.stop() for agent in self._agents.values()]
        await asyncio.gather(*stop_tasks, return_exceptions=True)

        logger.info(f"Orchestrator {self.name} stopped")

    async def _message_router(self) -> None:
        """消息路由器主循环"""
        while self._running:
            try:
                # 从消息总线获取消息
                message = await asyncio.wait_for(
                    self._message_bus.get(),
                    timeout=1.0
                )

                await self._route_message(message)
                self._metrics["total_messages_routed"] += 1

            except asyncio.TimeoutError:
                # 检查代理的发件箱
                for agent in self._agents.values():
                    try:
                        while not agent._outbox.empty():
                            msg = agent._outbox.get_nowait()
                            await self._route_message(msg)
                            self._metrics["total_messages_routed"] += 1
                    except asyncio.QueueEmpty:
                        pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message router: {e}")

    async def _route_message(self, message: AgentMessage) -> None:
        """路由消息到目标代理"""
        # 检查消息TTL
        age = (datetime.utcnow() - message.timestamp).total_seconds()
        if age > message.ttl:
            logger.warning(f"Message {message.id} expired (age={age}s, ttl={message.ttl}s)")
            return

        # 广播消息
        if not message.receiver:
            # 发送给所有订阅该主题的代理
            subscribers = self._topic_subscribers.get(message.topic, set())
            for agent_name in subscribers:
                if agent_name != message.sender and agent_name in self._agents:
                    agent = self._agents[agent_name]
                    await agent.receive_message(message)
        else:
            # 点对点消息
            if message.receiver in self._agents:
                agent = self._agents[message.receiver]
                await agent.receive_message(message)
            else:
                logger.warning(f"Target agent {message.receiver} not found")

    async def send_message(self, message: AgentMessage) -> None:
        """发送消息到消息总线"""
        await self._message_bus.put(message)

    # === 执行调度 ===

    async def execute_plan(self, plan: ExecutionPlan) -> ExecutionResult:
        """执行计划"""
        start_time = datetime.utcnow()
        self._metrics["total_executions"] += 1

        logger.info(f"Executing plan {plan.id} with mode {plan.mode.value}")

        try:
            if plan.mode == ExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential(plan)
            elif plan.mode == ExecutionMode.PARALLEL:
                results = await self._execute_parallel(plan)
            elif plan.mode == ExecutionMode.DAG:
                results = await self._execute_dag(plan)
            else:
                raise ValueError(f"Unknown execution mode: {plan.mode}")

            # 检查失败的代理
            failed_agents = [
                name for name, result in results.items()
                if not result.success
            ]

            success = len(failed_agents) == 0
            if success:
                self._metrics["successful_executions"] += 1
            else:
                self._metrics["failed_executions"] += 1

            execution_result = ExecutionResult(
                plan_id=plan.id,
                success=success,
                results=results,
                failed_agents=failed_agents,
                total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            logger.exception(f"Plan execution failed: {e}")
            self._metrics["failed_executions"] += 1
            execution_result = ExecutionResult(
                plan_id=plan.id,
                success=False,
                results={},
                failed_agents=plan.agents,
                total_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

        # 记录执行历史
        self._execution_history.append(execution_result)
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

        # 触发事件
        await self._emit_event("execution_complete", execution_result)

        return execution_result

    async def _execute_sequential(self, plan: ExecutionPlan) -> Dict[str, AgentResult]:
        """串行执行"""
        results = {}
        context = plan.context.copy()

        for agent_name in plan.agents:
            if agent_name not in self._agents:
                results[agent_name] = AgentResult(
                    success=False,
                    error=f"Agent {agent_name} not found"
                )
                continue

            agent = self._agents[agent_name]

            # 将前一个代理的结果加入上下文
            context["previous_results"] = results.copy()

            result = await self._execute_agent_with_retry(
                agent, context, plan.retry_count
            )
            results[agent_name] = result

            # 如果失败，可以选择中断或继续
            if not result.success:
                logger.warning(f"Agent {agent_name} failed in sequential execution")

        return results

    async def _execute_parallel(self, plan: ExecutionPlan) -> Dict[str, AgentResult]:
        """并行执行"""
        tasks = {}

        for agent_name in plan.agents:
            if agent_name not in self._agents:
                continue

            agent = self._agents[agent_name]
            task = asyncio.create_task(
                self._execute_agent_with_retry(agent, plan.context, plan.retry_count)
            )
            tasks[agent_name] = task

        # 等待所有任务完成
        results = {}
        for agent_name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=plan.timeout)
                results[agent_name] = result
            except asyncio.TimeoutError:
                results[agent_name] = AgentResult(
                    success=False,
                    error=f"Agent {agent_name} timed out"
                )
            except Exception as e:
                results[agent_name] = AgentResult(
                    success=False,
                    error=str(e)
                )

        return results

    async def _execute_dag(self, plan: ExecutionPlan) -> Dict[str, AgentResult]:
        """DAG依赖图执行"""
        results = {}
        completed: Set[str] = set()
        in_flight: Dict[str, asyncio.Task] = {}

        # 构建依赖关系
        dependencies = plan.dependencies.copy()
        dependents: Dict[str, Set[str]] = defaultdict(set)
        for agent, deps in dependencies.items():
            for dep in deps:
                dependents[dep].add(agent)

        # 找出没有依赖的代理
        ready = [
            name for name in plan.agents
            if not dependencies.get(name)
        ]

        context = plan.context.copy()

        while ready or in_flight:
            # 启动所有就绪的代理
            for agent_name in ready:
                if agent_name in self._agents:
                    agent = self._agents[agent_name]
                    # 添加已完成代理的结果到上下文
                    agent_context = context.copy()
                    agent_context["dependency_results"] = {
                        dep: results[dep] for dep in dependencies.get(agent_name, [])
                        if dep in results
                    }
                    task = asyncio.create_task(
                        self._execute_agent_with_retry(agent, agent_context, plan.retry_count)
                    )
                    in_flight[agent_name] = task
            ready.clear()

            if not in_flight:
                break

            # 等待任意一个任务完成
            done, _ = await asyncio.wait(
                in_flight.values(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=plan.timeout
            )

            # 处理完成的任务
            for task in done:
                # 找到对应的代理名
                agent_name = None
                for name, t in in_flight.items():
                    if t == task:
                        agent_name = name
                        break

                if agent_name:
                    del in_flight[agent_name]
                    try:
                        result = task.result()
                        results[agent_name] = result
                    except Exception as e:
                        results[agent_name] = AgentResult(
                            success=False,
                            error=str(e)
                        )

                    completed.add(agent_name)

                    # 检查依赖于此代理的其他代理是否可以启动
                    for dependent in dependents.get(agent_name, set()):
                        deps = dependencies.get(dependent, [])
                        if all(d in completed for d in deps):
                            if dependent not in completed and dependent not in in_flight:
                                ready.append(dependent)

        # 检查未完成的代理
        for agent_name in plan.agents:
            if agent_name not in results:
                results[agent_name] = AgentResult(
                    success=False,
                    error="Agent was not executed (possibly due to dependency failure)"
                )

        return results

    async def _execute_agent_with_retry(
        self,
        agent: BaseAgent,
        context: Dict[str, Any],
        retry_count: int
    ) -> AgentResult:
        """带重试的代理执行"""
        last_error = None

        for attempt in range(retry_count):
            try:
                result = await agent.run(context)
                if result.success:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Agent {agent.name} attempt {attempt + 1} failed: {e}")

            # 重试间隔
            if attempt < retry_count - 1:
                await asyncio.sleep(0.5 * (attempt + 1))

        return AgentResult(
            success=False,
            error=f"Failed after {retry_count} attempts: {last_error}"
        )

    # === 便捷方法 ===

    async def run_agents(
        self,
        agent_names: List[str],
        context: Optional[Dict[str, Any]] = None,
        mode: ExecutionMode = ExecutionMode.PARALLEL,
        timeout: float = 120.0,
    ) -> ExecutionResult:
        """便捷方法：运行指定代理"""
        plan = ExecutionPlan(
            mode=mode,
            agents=agent_names,
            context=context or {},
            timeout=timeout,
        )
        return await self.execute_plan(plan)

    async def run_single(
        self,
        agent_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """便捷方法：运行单个代理"""
        result = await self.run_agents([agent_name], context)
        return result.results.get(agent_name, AgentResult(
            success=False,
            error=f"Agent {agent_name} not found"
        ))

    # === 事件系统 ===

    def on_event(self, event_name: str, handler: Callable) -> None:
        """注册事件处理器"""
        self._event_handlers[event_name].append(handler)

    async def _emit_event(self, event_name: str, data: Any) -> None:
        """触发事件"""
        for handler in self._event_handlers.get(event_name, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    # === 监控和指标 ===

    @property
    def metrics(self) -> Dict[str, Any]:
        """获取协调器指标"""
        return {
            **self._metrics,
            "running": self._running,
            "message_queue_size": self._message_bus.qsize(),
            "execution_history_size": len(self._execution_history),
        }

    def get_execution_history(
        self,
        limit: int = 10
    ) -> List[ExecutionResult]:
        """获取执行历史"""
        return self._execution_history[-limit:]

    def __repr__(self) -> str:
        return f"<AgentOrchestrator name={self.name} agents={len(self._agents)} running={self._running}>"
