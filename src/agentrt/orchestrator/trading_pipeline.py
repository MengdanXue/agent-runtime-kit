# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Trading Pipeline - 交易流水线
将多个代理组合成完整的交易决策和执行流程

流程：
1. DataSourceAgent: 获取市场数据
2. SignalAggregatorAgent: 生成聚合信号
3. RiskAssessmentAgent: 风险评估
4. ExecutionAgent: 订单执行

支持的模式：
- 完整流程: 数据 -> 信号 -> 风险 -> 执行
- 信号模式: 数据 -> 信号 (不执行)
- 分析模式: 数据 -> 信号 -> 风险 (不执行)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import logging

from .orchestrator import AgentOrchestrator, ExecutionPlan, ExecutionMode, ExecutionResult
from .data_source_agent import DataSourceAgent, MarketData
from .signal_aggregator_agent import SignalAggregatorAgent, AggregatedSignal, TradingSignal, SignalType
from .risk_assessment_agent import RiskAssessmentAgent, TradeRiskAssessment, RiskLevel
from .execution_agent import ExecutionAgent, ExecutionReport

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """流水线模式"""
    FULL = "full"           # 完整流程（包含执行）
    SIGNAL_ONLY = "signal"  # 仅生成信号
    ANALYSIS = "analysis"   # 分析模式（信号+风险）
    PAPER = "paper"         # 模拟交易


@dataclass
class PipelineResult:
    """流水线结果"""
    mode: PipelineMode
    success: bool
    symbols: List[str]
    market_data: Dict[str, MarketData] = field(default_factory=dict)
    signals: Dict[str, AggregatedSignal] = field(default_factory=dict)
    risk_assessments: Dict[str, TradeRiskAssessment] = field(default_factory=dict)
    execution_reports: Dict[str, ExecutionReport] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    total_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "success": self.success,
            "symbols": self.symbols,
            "market_data": {s: d.to_dict() for s, d in self.market_data.items()},
            "signals": {s: sig.to_dict() for s, sig in self.signals.items()},
            "risk_assessments": {s: r.to_dict() for s, r in self.risk_assessments.items()},
            "execution_reports": {s: e.to_dict() for s, e in self.execution_reports.items()},
            "errors": self.errors,
            "total_time_ms": self.total_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要"""
        buy_signals = [s for s in self.signals.values() if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in self.signals.values() if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]
        approved = [r for r in self.risk_assessments.values() if r.approved]
        rejected = [r for r in self.risk_assessments.values() if not r.approved]

        return {
            "total_symbols": len(self.symbols),
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "risk_approved": len(approved),
            "risk_rejected": len(rejected),
            "orders_executed": len(self.execution_reports),
            "errors_count": len(self.errors),
        }


class TradingPipeline:
    """
    交易流水线

    集成所有代理，提供端到端的交易决策流程
    """

    def __init__(
        self,
        name: str = "trading_pipeline",
        mode: PipelineMode = PipelineMode.ANALYSIS,
    ):
        self.name = name
        self.mode = mode

        # 创建协调器
        self.orchestrator = AgentOrchestrator(f"{name}_orchestrator")

        # 创建代理
        self.data_agent = DataSourceAgent(f"{name}_data")
        self.signal_agent = SignalAggregatorAgent(f"{name}_signal")
        self.risk_agent = RiskAssessmentAgent(f"{name}_risk")
        self.execution_agent = ExecutionAgent(f"{name}_execution")

        # 注册代理
        self.orchestrator.register_agent(self.data_agent)
        self.orchestrator.register_agent(self.signal_agent)
        self.orchestrator.register_agent(self.risk_agent)
        self.orchestrator.register_agent(self.execution_agent)

        # 配置默认数据源
        self.data_agent.configure_default_sources()

        # 注册默认策略
        self._register_default_strategies()

        logger.info(f"TradingPipeline {name} initialized in {mode.value} mode")

    def _register_default_strategies(self) -> None:
        """注册默认策略"""
        strategies = [
            ("momentum", 1.0),
            ("mean_reversion", 0.8),
            ("trend_following", 1.2),
            ("rsi", 0.7),
            ("macd", 0.9),
        ]
        for name, weight in strategies:
            self.signal_agent.register_strategy(name, weight)

    async def start(self) -> None:
        """启动流水线"""
        await self.orchestrator.start()
        logger.info(f"TradingPipeline {self.name} started")

    async def stop(self) -> None:
        """停止流水线"""
        await self.orchestrator.stop()
        logger.info(f"TradingPipeline {self.name} stopped")

    async def run(
        self,
        symbols: List[str],
        signals: Optional[List[Dict[str, Any]]] = None,
        mode: Optional[PipelineMode] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        运行交易流水线

        Args:
            symbols: 要处理的标的列表
            signals: 可选的输入信号（如果不提供，将生成模拟信号）
            mode: 运行模式（覆盖默认模式）
            context: 额外的上下文参数

        Returns:
            PipelineResult: 流水线执行结果
        """
        start_time = datetime.utcnow()
        run_mode = mode or self.mode
        ctx = context or {}

        result = PipelineResult(
            mode=run_mode,
            success=True,
            symbols=symbols,
        )

        try:
            # 阶段1: 获取市场数据
            logger.info(f"Stage 1: Fetching market data for {len(symbols)} symbols")
            data_result = await self.data_agent.run({
                "symbols": symbols,
                "data_type": "quote",
                **ctx.get("data_config", {}),
            })

            if data_result.success:
                result.market_data = data_result.data.get("merged", {})
            else:
                result.errors["data_fetch"] = data_result.error or "Failed to fetch data"

            # 阶段2: 生成/聚合信号
            logger.info("Stage 2: Generating and aggregating signals")

            if signals:
                # 使用提供的信号
                input_signals = signals
            else:
                # 生成模拟信号
                input_signals = self._generate_mock_signals(symbols, result.market_data)

            signal_result = await self.signal_agent.run({
                "signals": input_signals,
                "symbols": symbols,
                **ctx.get("signal_config", {}),
            })

            if signal_result.success:
                result.signals = signal_result.data.get("aggregated", {})
            else:
                result.errors["signal_aggregation"] = signal_result.error or "Failed to aggregate signals"

            # 如果是信号模式，到此结束
            if run_mode == PipelineMode.SIGNAL_ONLY:
                result.total_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return result

            # 阶段3: 风险评估
            logger.info("Stage 3: Risk assessment")

            for symbol, agg_signal in result.signals.items():
                if agg_signal.signal_type == SignalType.HOLD:
                    continue

                market_data_for_symbol = result.market_data.get(symbol)
                price = market_data_for_symbol.last if market_data_for_symbol else 100.0

                risk_result = await self.risk_agent.run({
                    "symbol": symbol,
                    "action": "buy" if agg_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else "sell",
                    "quantity": 100,  # 示例数量
                    "price": price,
                    "market_data": {
                        "volatility": 0.2,
                        "price": price,
                    },
                    **ctx.get("risk_config", {}),
                })

                if risk_result.success:
                    result.risk_assessments[symbol] = risk_result.data
                else:
                    result.errors[f"risk_{symbol}"] = risk_result.error or "Risk assessment failed"

            # 如果是分析模式，到此结束
            if run_mode in [PipelineMode.ANALYSIS, PipelineMode.SIGNAL_ONLY]:
                result.total_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return result

            # 阶段4: 订单执行
            logger.info("Stage 4: Order execution")

            for symbol, risk_assessment in result.risk_assessments.items():
                if not risk_assessment.approved:
                    logger.info(f"Skipping {symbol}: risk not approved")
                    continue

                agg_signal = result.signals.get(symbol)
                if not agg_signal:
                    continue

                market_data_for_symbol = result.market_data.get(symbol)
                price = market_data_for_symbol.last if market_data_for_symbol else 100.0

                exec_result = await self.execution_agent.run({
                    "symbol": symbol,
                    "side": "buy" if agg_signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY] else "sell",
                    "quantity": min(100, risk_assessment.max_position_size / price) if price > 0 else 100,
                    "price": price,
                    "order_type": "market",
                    "config": ctx.get("execution_config", {}),
                })

                if exec_result.success:
                    result.execution_reports[symbol] = exec_result.data
                else:
                    result.errors[f"execution_{symbol}"] = exec_result.error or "Execution failed"

        except Exception as e:
            logger.exception(f"Pipeline execution failed: {e}")
            result.success = False
            result.errors["pipeline"] = str(e)

        result.total_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.success = len(result.errors) == 0

        return result

    def _generate_mock_signals(
        self,
        symbols: List[str],
        market_data: Dict[str, MarketData]
    ) -> List[Dict[str, Any]]:
        """生成模拟信号（仅限显式开启 mock 数据的本地开发 / 演示）"""
        from agentrt.guards.mock_guard import require_real_data

        require_real_data(
            "trading signals for " + ", ".join(symbols),
            "TradingPipeline is a demo; pass real signals in, or use "
            "IntegratedTradingPipeline.analyze(), which computes them with SignalTool.",
        )
        import random

        signals = []
        strategies = ["momentum", "mean_reversion", "trend_following", "rsi", "macd"]

        for symbol in symbols:
            for strategy in strategies:
                signal_type = random.choice(["buy", "sell", "hold", "hold", "hold"])  # 倾向HOLD
                signals.append({
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "strength": random.uniform(0.3, 0.9),
                    "confidence": random.uniform(0.4, 0.9),
                    "strategy": strategy,
                    "timestamp": datetime.utcnow().isoformat(),
                })

        return signals

    async def analyze_symbol(
        self,
        symbol: str,
        signals: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """分析单个标的"""
        result = await self.run([symbol], signals, PipelineMode.ANALYSIS)
        return result.to_dict()

    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """执行单个交易"""
        # 先进行风险评估
        risk_result = await self.risk_agent.run({
            "symbol": symbol,
            "action": side,
            "quantity": quantity,
            "price": price or 100,
        })

        if not risk_result.success or (risk_result.data and not risk_result.data.approved):
            return {
                "success": False,
                "error": "Risk assessment rejected the trade",
                "risk_assessment": risk_result.data.to_dict() if risk_result.data else None,
            }

        # 执行交易
        exec_result = await self.execution_agent.run({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": "market",
        })

        return {
            "success": exec_result.success,
            "execution_report": exec_result.data.to_dict() if exec_result.data else None,
            "error": exec_result.error,
        }

    def get_status(self) -> Dict[str, Any]:
        """获取流水线状态"""
        return {
            "name": self.name,
            "mode": self.mode.value,
            "orchestrator": self.orchestrator.metrics,
            "agents": {
                "data_source": self.data_agent.metrics,
                "signal_aggregator": self.signal_agent.metrics,
                "risk_assessment": self.risk_agent.metrics,
                "execution": self.execution_agent.metrics,
            },
            "data_sources": self.data_agent.get_source_status(),
            "strategy_performance": self.signal_agent.get_strategy_performance(),
            "execution_stats": self.execution_agent.get_execution_stats(),
        }


# 使用示例
async def demo():
    """演示如何使用交易流水线"""
    # 创建流水线
    pipeline = TradingPipeline("demo_pipeline", PipelineMode.ANALYSIS)

    # 启动
    await pipeline.start()

    try:
        # 运行分析
        symbols = ["BTCUSDT", "ETHUSDT", "AAPL", "GOOGL"]

        print("\n=== 运行交易流水线 ===\n")

        result = await pipeline.run(symbols)

        print(f"执行时间: {result.total_time_ms:.2f}ms")
        print(f"成功: {result.success}")
        print(f"\n--- 摘要 ---")
        summary = result._generate_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print(f"\n--- 信号 ---")
        for symbol, signal in result.signals.items():
            print(f"  {symbol}: {signal.signal_type.value} (强度={signal.strength:.2f}, 一致性={signal.consensus_score:.2f})")

        print(f"\n--- 风险评估 ---")
        for symbol, risk in result.risk_assessments.items():
            status = "✓ 批准" if risk.approved else "✗ 拒绝"
            print(f"  {symbol}: {status} (风险等级={risk.risk_level.value}, 得分={risk.risk_score:.1f})")

        if result.errors:
            print(f"\n--- 错误 ---")
            for key, error in result.errors.items():
                print(f"  {key}: {error}")

        # 获取状态
        print(f"\n--- 系统状态 ---")
        status = pipeline.get_status()
        print(f"  协调器: {status['orchestrator']}")

    finally:
        await pipeline.stop()


if __name__ == "__main__":
    asyncio.run(demo())
