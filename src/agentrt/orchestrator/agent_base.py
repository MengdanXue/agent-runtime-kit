# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Base Agent Classes for Multi-Agent System
多代理系统基础类

设计原则：
1. 异步优先 - 所有代理方法都是异步的
2. 消息驱动 - 代理间通过消息通信
3. 状态管理 - 每个代理有明确的生命周期状态
4. 可观测性 - 内置指标收集和日志记录
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """代理状态"""
    IDLE = auto()        # 空闲
    RUNNING = auto()     # 运行中
    WAITING = auto()     # 等待其他代理
    COMPLETED = auto()   # 已完成
    ERROR = auto()       # 错误
    CANCELLED = auto()   # 已取消


class AgentPriority(Enum):
    """代理优先级"""
    CRITICAL = 1   # 关键 - 风险控制、止损
    HIGH = 2       # 高 - 信号生成
    NORMAL = 3     # 普通 - 数据获取
    LOW = 4        # 低 - 分析报告


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"           # 请求
    RESPONSE = "response"         # 响应
    BROADCAST = "broadcast"       # 广播
    COMMAND = "command"           # 命令
    EVENT = "event"               # 事件
    ERROR = "error"               # 错误


@dataclass
class AgentMessage:
    """代理间消息"""
    id: str = field(default_factory=lambda: str(uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    receiver: str = ""           # 空字符串表示广播
    topic: str = ""              # 消息主题
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None  # 关联ID，用于请求-响应配对
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: AgentPriority = AgentPriority.NORMAL
    ttl: int = 60               # 消息生存时间（秒）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "topic": self.topic,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(
            id=data.get("id", str(uuid4())),
            type=MessageType(data.get("type", "request")),
            sender=data.get("sender", ""),
            receiver=data.get("receiver", ""),
            topic=data.get("topic", ""),
            payload=data.get("payload", {}),
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            priority=AgentPriority(data.get("priority", 3)),
            ttl=data.get("ttl", 60),
        )


@dataclass
class AgentResult:
    """代理执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


T = TypeVar('T')


class BaseAgent(ABC, Generic[T]):
    """
    代理基类

    所有代理都继承此类，实现以下核心功能：
    1. 消息发送和接收
    2. 状态管理
    3. 生命周期钩子
    4. 错误处理
    """

    def __init__(
        self,
        name: str,
        priority: AgentPriority = AgentPriority.NORMAL,
        timeout: float = 30.0,
    ):
        self.id = str(uuid4())
        self.name = name
        self.priority = priority
        self.timeout = timeout
        self.state = AgentState.IDLE

        # 消息队列
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._outbox: asyncio.Queue[AgentMessage] = asyncio.Queue()

        # 消息处理器
        self._message_handlers: Dict[str, Callable] = {}

        # 等待响应的请求
        self._pending_requests: Dict[str, asyncio.Future] = {}

        # 指标
        self._metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time_ms": 0.0,
        }

        # 订阅的主题
        self._subscriptions: set[str] = set()

        # 协调器引用（运行时设置）
        self._orchestrator: Optional[Any] = None

        logger.info(f"Agent {self.name} ({self.id}) initialized")

    @property
    def metrics(self) -> Dict[str, Any]:
        """获取代理指标"""
        return {
            **self._metrics,
            "state": self.state.name,
            "pending_requests": len(self._pending_requests),
            "inbox_size": self._inbox.qsize(),
        }

    # === 生命周期方法 ===

    async def start(self) -> None:
        """启动代理"""
        logger.info(f"Starting agent {self.name}")
        self.state = AgentState.IDLE
        await self.on_start()

    async def stop(self) -> None:
        """停止代理"""
        logger.info(f"Stopping agent {self.name}")
        self.state = AgentState.CANCELLED

        # 取消所有待处理请求
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        await self.on_stop()

    async def on_start(self) -> None:
        """启动时钩子（子类可覆盖）"""
        pass

    async def on_stop(self) -> None:
        """停止时钩子（子类可覆盖）"""
        pass

    # === 消息处理 ===

    def subscribe(self, topic: str) -> None:
        """订阅主题"""
        self._subscriptions.add(topic)
        logger.debug(f"Agent {self.name} subscribed to topic: {topic}")

    def unsubscribe(self, topic: str) -> None:
        """取消订阅"""
        self._subscriptions.discard(topic)

    def register_handler(self, topic: str, handler: Callable) -> None:
        """注册消息处理器"""
        self._message_handlers[topic] = handler
        self.subscribe(topic)

    async def send_message(self, message: AgentMessage) -> None:
        """发送消息"""
        message.sender = self.name
        if self._orchestrator is not None:
            await self._orchestrator.send_message(message)
        else:
            await self._outbox.put(message)
        self._metrics["messages_sent"] += 1
        logger.debug(f"Agent {self.name} sent message to {message.receiver or 'broadcast'}: {message.topic}")

    async def receive_message(self, message: AgentMessage) -> None:
        """接收消息"""
        self._metrics["messages_received"] += 1

        # 检查是否是响应消息
        if message.type == MessageType.RESPONSE and message.correlation_id:
            if message.correlation_id in self._pending_requests:
                future = self._pending_requests.pop(message.correlation_id)
                if not future.done():
                    future.set_result(message)
                return

        # 检查是否有处理器
        if message.topic in self._message_handlers:
            handler = self._message_handlers[message.topic]
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.topic}: {e}")
        else:
            # 放入收件箱等待处理
            await self._inbox.put(message)

    async def request(
        self,
        receiver: str,
        topic: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> AgentMessage:
        """发送请求并等待响应"""
        request_id = str(uuid4())
        message = AgentMessage(
            type=MessageType.REQUEST,
            sender=self.name,
            receiver=receiver,
            topic=topic,
            payload=payload,
            correlation_id=request_id,
            priority=self.priority,
        )

        # 创建Future等待响应
        loop = asyncio.get_event_loop()
        future: asyncio.Future[AgentMessage] = loop.create_future()
        self._pending_requests[request_id] = future

        # 发送请求
        await self.send_message(message)

        # 等待响应
        try:
            response = await asyncio.wait_for(future, timeout=timeout or self.timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"Request to {receiver} timed out after {timeout or self.timeout}s")

    async def respond(self, original: AgentMessage, payload: Dict[str, Any]) -> None:
        """响应请求"""
        response = AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.name,
            receiver=original.sender,
            topic=f"{original.topic}.response",
            payload=payload,
            correlation_id=original.correlation_id,
            priority=self.priority,
        )
        await self.send_message(response)

    async def broadcast(self, topic: str, payload: Dict[str, Any]) -> None:
        """广播消息"""
        message = AgentMessage(
            type=MessageType.BROADCAST,
            sender=self.name,
            receiver="",  # 空表示广播
            topic=topic,
            payload=payload,
            priority=self.priority,
        )
        await self.send_message(message)

    # === 核心执行方法 ===

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行代理任务（子类必须实现）

        Args:
            context: 执行上下文，包含输入数据和配置

        Returns:
            AgentResult: 执行结果
        """
        pass

    async def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        运行代理（带状态管理和错误处理）
        """
        start_time = datetime.utcnow()
        self.state = AgentState.RUNNING

        try:
            result = await asyncio.wait_for(
                self.execute(context),
                timeout=self.timeout
            )

            if result.success:
                self._metrics["tasks_completed"] += 1
            else:
                self._metrics["tasks_failed"] += 1

            self.state = AgentState.COMPLETED

        except asyncio.TimeoutError:
            result = AgentResult(
                success=False,
                error=f"Agent {self.name} timed out after {self.timeout}s"
            )
            self._metrics["tasks_failed"] += 1
            self.state = AgentState.ERROR

        except Exception as e:
            logger.exception(f"Agent {self.name} execution failed")
            result = AgentResult(
                success=False,
                error=str(e)
            )
            self._metrics["tasks_failed"] += 1
            self.state = AgentState.ERROR

        # 计算执行时间
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.execution_time_ms = execution_time
        self._metrics["total_execution_time_ms"] += execution_time

        return result

    # === 工具方法 ===

    async def wait_for_agents(
        self,
        agent_names: List[str],
        topic: str = "status",
        timeout: Optional[float] = None,
    ) -> Dict[str, AgentMessage]:
        """等待多个代理完成"""
        self.state = AgentState.WAITING

        tasks = []
        for name in agent_names:
            task = self.request(name, topic, {"action": "get_status"}, timeout)
            tasks.append((name, task))

        results = {}
        for name, task in tasks:
            try:
                response = await task
                results[name] = response
            except Exception as e:
                logger.error(f"Failed to get status from {name}: {e}")

        self.state = AgentState.RUNNING
        return results

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} state={self.state.name}>"
