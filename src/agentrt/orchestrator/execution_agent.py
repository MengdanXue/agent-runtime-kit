# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Execution Agent - 执行代理
负责将交易信号转化为实际订单执行

功能：
1. 订单拆分和TWAP/VWAP执行
2. 智能订单路由
3. 滑点控制
4. 执行质量监控
5. 订单状态追踪
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from uuid import uuid4
import logging
import random

from agentrt.guards.mock_guard import require_real_data

from .agent_base import BaseAgent, AgentResult, AgentPriority

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"               # 时间加权
    VWAP = "vwap"               # 成交量加权
    ICEBERG = "iceberg"         # 冰山订单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionAlgorithm(Enum):
    """执行算法"""
    IMMEDIATE = "immediate"     # 立即执行
    TWAP = "twap"               # 时间加权平均
    VWAP = "vwap"               # 成交量加权平均
    POV = "pov"                 # 市场参与率
    ICEBERG = "iceberg"         # 冰山单
    ADAPTIVE = "adaptive"       # 自适应


@dataclass
class Order:
    """订单"""
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    exchange: str = ""
    parent_id: Optional[str] = None      # 父订单ID（拆分订单用）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "exchange": self.exchange,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionReport:
    """执行报告"""
    order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    requested_quantity: float
    filled_quantity: float
    avg_price: float
    slippage: float              # 滑点
    execution_time_ms: float
    num_child_orders: int = 0
    child_orders: List[Order] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "avg_price": self.avg_price,
            "slippage": self.slippage,
            "execution_time_ms": self.execution_time_ms,
            "num_child_orders": self.num_child_orders,
            "fill_rate": self.filled_quantity / self.requested_quantity if self.requested_quantity > 0 else 0,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionConfig:
    """执行配置"""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.IMMEDIATE
    max_slippage_pct: float = 0.005      # 最大滑点 0.5%
    time_horizon_minutes: int = 30       # TWAP/VWAP执行时间窗口
    participation_rate: float = 0.1      # POV参与率
    min_fill_rate: float = 0.8           # 最低成交率
    iceberg_display_pct: float = 0.1     # 冰山单显示比例
    retry_count: int = 3                 # 重试次数
    order_timeout_seconds: int = 60      # 订单超时


class ExecutionAgent(BaseAgent[ExecutionReport]):
    """
    执行代理

    核心功能：
    1. 将交易信号转换为订单
    2. 执行算法选择
    3. 订单拆分和执行
    4. 滑点控制和监控
    5. 执行质量报告
    """

    def __init__(
        self,
        name: str = "execution_agent",
        timeout: float = 120.0,
        default_config: Optional[ExecutionConfig] = None,
    ):
        super().__init__(name, AgentPriority.HIGH, timeout)

        self.default_config = default_config or ExecutionConfig()

        # 订单追踪
        self._active_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []

        # 交易所适配器
        self._exchange_adapters: Dict[str, Any] = {}

        # 执行统计
        self._execution_stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "total_volume": 0.0,
            "avg_slippage": 0.0,
        }

        # 订阅主题
        self.subscribe("order.submit")
        self.subscribe("order.cancel")
        self.subscribe("signal.aggregated")

        # 注册消息处理器
        self.register_handler("signal.aggregated", self._handle_aggregated_signal)

        logger.info(f"ExecutionAgent {name} initialized")

    # PAPER ONLY: this agent simulates fills and must never reach an exchange.
    PAPER_ONLY = True

    def register_exchange(self, exchange_name: str, adapter: Any) -> None:
        """交易所适配器注册被禁止：ExecutionAgent 只做纸面模拟。"""
        raise RuntimeError(
            f"ExecutionAgent is PAPER ONLY; refusing to register exchange adapter '{exchange_name}'")

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行订单

        context 参数:
        - symbol: str - 交易标的
        - side: str - 交易方向 (buy/sell)
        - quantity: float - 交易数量
        - price: Optional[float] - 限价（市价单不需要）
        - order_type: str - 订单类型
        - exchange: str - 交易所
        - config: Optional[Dict] - 执行配置
        """
        symbol = context.get("symbol")
        side_str = context.get("side", "buy")
        quantity = context.get("quantity", 0)
        price = context.get("price")
        order_type_str = context.get("order_type", "market")
        exchange = context.get("exchange", "binance")
        config_dict = context.get("config", {})

        if not symbol or quantity <= 0:
            return AgentResult(
                success=False,
                error="Invalid order parameters: symbol and quantity required"
            )
        if context.get("live") or str(context.get("mode", "paper")).lower() != "paper":
            return AgentResult(
                success=False,
                error="ExecutionAgent is PAPER ONLY; live execution is not supported"
            )

        try:
            # 解析参数
            side = OrderSide(side_str)
            order_type = OrderType(order_type_str)

            # 创建执行配置
            config = ExecutionConfig(
                algorithm=ExecutionAlgorithm(config_dict.get("algorithm", self.default_config.algorithm.value)),
                max_slippage_pct=config_dict.get("max_slippage", self.default_config.max_slippage_pct),
                time_horizon_minutes=config_dict.get("time_horizon", self.default_config.time_horizon_minutes),
            )

            # 创建主订单
            order = Order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                exchange=exchange,
            )

            self._active_orders[order.id] = order
            self._execution_stats["total_orders"] += 1

            # 根据执行算法执行
            start_time = datetime.utcnow()

            if config.algorithm == ExecutionAlgorithm.IMMEDIATE:
                report = await self._execute_immediate(order, config)
            elif config.algorithm == ExecutionAlgorithm.TWAP:
                report = await self._execute_twap(order, config)
            elif config.algorithm == ExecutionAlgorithm.VWAP:
                report = await self._execute_vwap(order, config)
            elif config.algorithm == ExecutionAlgorithm.ICEBERG:
                report = await self._execute_iceberg(order, config)
            else:
                report = await self._execute_immediate(order, config)

            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            report.execution_time_ms = execution_time

            # 更新统计
            if report.status == OrderStatus.FILLED:
                self._execution_stats["filled_orders"] += 1
                self._execution_stats["total_volume"] += report.filled_quantity * report.avg_price
                # 更新平均滑点
                n = self._execution_stats["filled_orders"]
                old_avg = self._execution_stats["avg_slippage"]
                self._execution_stats["avg_slippage"] = (old_avg * (n-1) + report.slippage) / n

            # 移到历史
            del self._active_orders[order.id]
            self._order_history.append(order)
            if len(self._order_history) > 1000:
                self._order_history = self._order_history[-1000:]

            # 广播执行结果
            await self.broadcast("order.executed", {
                "order_id": order.id,
                "symbol": symbol,
                "side": side.value,
                "status": report.status.value,
                "filled_quantity": report.filled_quantity,
                "avg_price": report.avg_price,
                "slippage": report.slippage,
            })

            return AgentResult(
                success=report.status in [OrderStatus.FILLED, OrderStatus.PARTIAL],
                data=report,
                metadata={
                    "order_id": order.id,
                    "algorithm": config.algorithm.value,
                    "execution_time_ms": execution_time,
                }
            )

        except Exception as e:
            logger.exception(f"Order execution failed: {e}")
            self._execution_stats["rejected_orders"] += 1
            return AgentResult(
                success=False,
                error=str(e)
            )

    async def _execute_immediate(
        self,
        order: Order,
        config: ExecutionConfig
    ) -> ExecutionReport:
        """立即执行"""
        # 模拟执行（实际应调用交易所API）
        fill_result = await self._simulate_fill(order, 1.0)

        order.status = fill_result["status"]
        order.filled_quantity = fill_result["filled_quantity"]
        order.avg_fill_price = fill_result["avg_price"]
        order.updated_at = datetime.utcnow()

        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            status=order.status,
            requested_quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            avg_price=order.avg_fill_price,
            slippage=fill_result["slippage"],
            execution_time_ms=0,
        )

    async def _execute_twap(
        self,
        order: Order,
        config: ExecutionConfig
    ) -> ExecutionReport:
        """时间加权平均执行"""
        # 将订单拆分为多个子订单
        num_slices = min(10, max(3, config.time_horizon_minutes // 3))
        slice_quantity = order.quantity / num_slices
        slice_interval = (config.time_horizon_minutes * 60) / num_slices  # 秒

        child_orders: List[Order] = []
        total_filled = 0.0
        total_value = 0.0

        for i in range(num_slices):
            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=slice_quantity,
                exchange=order.exchange,
                parent_id=order.id,
            )

            # 执行子订单
            fill_result = await self._simulate_fill(child_order, 1.0)

            child_order.status = fill_result["status"]
            child_order.filled_quantity = fill_result["filled_quantity"]
            child_order.avg_fill_price = fill_result["avg_price"]

            child_orders.append(child_order)

            if fill_result["status"] == OrderStatus.FILLED:
                total_filled += fill_result["filled_quantity"]
                total_value += fill_result["filled_quantity"] * fill_result["avg_price"]

            # TWAP间隔等待（模拟）
            if i < num_slices - 1:
                await asyncio.sleep(0.1)  # 模拟用短间隔

        avg_price = total_value / total_filled if total_filled > 0 else 0
        reference_price = order.price or avg_price

        order.status = OrderStatus.FILLED if total_filled >= order.quantity * config.min_fill_rate else OrderStatus.PARTIAL
        order.filled_quantity = total_filled
        order.avg_fill_price = avg_price

        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            status=order.status,
            requested_quantity=order.quantity,
            filled_quantity=total_filled,
            avg_price=avg_price,
            slippage=(avg_price - reference_price) / reference_price if reference_price > 0 else 0,
            execution_time_ms=0,
            num_child_orders=len(child_orders),
            child_orders=child_orders,
        )

    async def _execute_vwap(
        self,
        order: Order,
        config: ExecutionConfig
    ) -> ExecutionReport:
        """成交量加权平均执行"""
        # VWAP需要历史成交量分布数据
        # 简化实现：按典型的日内成交量分布拆分
        volume_profile = [0.08, 0.12, 0.15, 0.10, 0.08, 0.07, 0.08, 0.10, 0.12, 0.10]
        num_slices = len(volume_profile)

        child_orders: List[Order] = []
        total_filled = 0.0
        total_value = 0.0

        for i, weight in enumerate(volume_profile):
            slice_quantity = order.quantity * weight

            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                quantity=slice_quantity,
                exchange=order.exchange,
                parent_id=order.id,
            )

            fill_result = await self._simulate_fill(child_order, 1.0)

            child_order.status = fill_result["status"]
            child_order.filled_quantity = fill_result["filled_quantity"]
            child_order.avg_fill_price = fill_result["avg_price"]

            child_orders.append(child_order)

            if fill_result["status"] == OrderStatus.FILLED:
                total_filled += fill_result["filled_quantity"]
                total_value += fill_result["filled_quantity"] * fill_result["avg_price"]

            await asyncio.sleep(0.05)  # 模拟

        avg_price = total_value / total_filled if total_filled > 0 else 0
        reference_price = order.price or avg_price

        order.status = OrderStatus.FILLED if total_filled >= order.quantity * config.min_fill_rate else OrderStatus.PARTIAL
        order.filled_quantity = total_filled
        order.avg_fill_price = avg_price

        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            status=order.status,
            requested_quantity=order.quantity,
            filled_quantity=total_filled,
            avg_price=avg_price,
            slippage=(avg_price - reference_price) / reference_price if reference_price > 0 else 0,
            execution_time_ms=0,
            num_child_orders=len(child_orders),
            child_orders=child_orders,
        )

    async def _execute_iceberg(
        self,
        order: Order,
        config: ExecutionConfig
    ) -> ExecutionReport:
        """冰山订单执行"""
        display_quantity = order.quantity * config.iceberg_display_pct
        remaining = order.quantity

        child_orders: List[Order] = []
        total_filled = 0.0
        total_value = 0.0

        while remaining > 0:
            slice_quantity = min(display_quantity, remaining)

            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=slice_quantity,
                price=order.price,
                exchange=order.exchange,
                parent_id=order.id,
            )

            fill_result = await self._simulate_fill(child_order, 0.9)  # 限价单成交率略低

            child_order.status = fill_result["status"]
            child_order.filled_quantity = fill_result["filled_quantity"]
            child_order.avg_fill_price = fill_result["avg_price"]

            child_orders.append(child_order)

            filled = fill_result["filled_quantity"]
            total_filled += filled
            total_value += filled * fill_result["avg_price"]
            remaining -= slice_quantity

            await asyncio.sleep(0.05)

            # 如果成交率太低，提前终止
            if len(child_orders) > 5 and total_filled / order.quantity < 0.3:
                break

        avg_price = total_value / total_filled if total_filled > 0 else 0
        reference_price = order.price or avg_price

        order.status = OrderStatus.FILLED if total_filled >= order.quantity * config.min_fill_rate else OrderStatus.PARTIAL
        order.filled_quantity = total_filled
        order.avg_fill_price = avg_price

        return ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            status=order.status,
            requested_quantity=order.quantity,
            filled_quantity=total_filled,
            avg_price=avg_price,
            slippage=(avg_price - reference_price) / reference_price if reference_price > 0 else 0,
            execution_time_ms=0,
            num_child_orders=len(child_orders),
            child_orders=child_orders,
        )

    async def _simulate_fill(
        self,
        order: Order,
        fill_probability: float
    ) -> Dict[str, Any]:
        """模拟订单成交（随机成交率与滑点，仅限显式开启 mock 数据的本地开发）"""
        # Random fill rate, random slippage and a default price are fabricated
        # execution data. The product's paper path is the runtime ExecutionTool,
        # which fills at the observed EOD close after a human approves a proposal.
        require_real_data(
            f"paper fill for {order.side.value} {order.quantity} {order.symbol}",
            "ExecutionAgent simulates fills with random numbers; use the Agent Runtime "
            "(proposal -> human approval -> ExecutionTool, PAPER ONLY) instead.",
        )

        # 模拟成交率
        if random.random() > fill_probability:
            return {
                "status": OrderStatus.REJECTED,
                "filled_quantity": 0,
                "avg_price": 0,
                "slippage": 0,
            }

        # 模拟价格滑点
        base_price = order.price or 100.0  # 默认价格
        slippage = random.uniform(-0.002, 0.005)  # -0.2% 到 0.5% 滑点

        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + slippage)
        else:
            fill_price = base_price * (1 - slippage)

        # 模拟部分成交
        fill_rate = random.uniform(0.9, 1.0) if fill_probability > 0.8 else random.uniform(0.5, 0.9)
        filled_quantity = order.quantity * fill_rate

        return {
            "status": OrderStatus.FILLED if fill_rate >= 0.99 else OrderStatus.PARTIAL,
            "filled_quantity": filled_quantity,
            "avg_price": fill_price,
            "slippage": slippage,
        }

    async def _handle_aggregated_signal(self, message: Any) -> None:
        """处理聚合信号并生成订单"""
        payload = message.payload
        signals = payload.get("aggregated_signals", {})

        for symbol, signal_data in signals.items():
            # 根据信号生成订单
            signal_type = signal_data.get("signal_type", "hold")

            if signal_type in ["buy", "strong_buy"]:
                side = "buy"
            elif signal_type in ["sell", "strong_sell"]:
                side = "sell"
            else:
                continue  # HOLD信号不生成订单

            # 创建执行上下文（实际应包含更多参数）
            context = {
                "symbol": symbol,
                "side": side,
                "quantity": 100,  # 示例数量
                "order_type": "market",
                "config": {
                    "algorithm": "immediate",
                }
            }

            # 发送订单执行请求
            await self.broadcast("order.request", context)

    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id not in self._active_orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self._active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow()

        # 广播取消事件
        await self.broadcast("order.cancelled", {
            "order_id": order_id,
            "symbol": order.symbol,
        })

        return True

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """获取活跃订单"""
        return [order.to_dict() for order in self._active_orders.values()]

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            **self._execution_stats,
            "fill_rate": self._execution_stats["filled_orders"] / self._execution_stats["total_orders"]
                        if self._execution_stats["total_orders"] > 0 else 0,
        }
