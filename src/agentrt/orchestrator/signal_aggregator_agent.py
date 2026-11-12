# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Signal Aggregator Agent - 信号聚合代理
负责从多个策略收集信号并进行加权聚合

功能：
1. 多策略信号收集
2. 信号权重计算
3. 冲突解决
4. 信号质量评估
5. 历史表现追踪
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import statistics

from .agent_base import BaseAgent, AgentResult, AgentPriority

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class ConflictResolution(Enum):
    """冲突解决策略"""
    WEIGHTED_AVERAGE = "weighted_average"   # 加权平均
    MAJORITY_VOTE = "majority_vote"         # 多数投票
    HIGHEST_CONFIDENCE = "highest_conf"     # 最高置信度
    CONSERVATIVE = "conservative"           # 保守（倾向HOLD）
    AGGRESSIVE = "aggressive"               # 激进（倾向交易）


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    strength: float           # 0-1 信号强度
    confidence: float         # 0-1 置信度
    strategy: str             # 来源策略
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1d"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "timestamp": self.timestamp.isoformat(),
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timeframe": self.timeframe,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedSignal:
    """聚合后的信号"""
    symbol: str
    signal_type: SignalType
    strength: float
    confidence: float
    contributing_signals: List[TradingSignal]
    consensus_score: float    # 策略一致性得分
    timestamp: datetime
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "consensus_score": self.consensus_score,
            "contributing_strategies": [s.strategy for s in self.contributing_signals],
            "num_signals": len(self.contributing_signals),
            "timestamp": self.timestamp.isoformat(),
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


@dataclass
class StrategyPerformance:
    """策略表现追踪"""
    strategy: str
    total_signals: int = 0
    correct_signals: int = 0
    accuracy: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    weight: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class SignalAggregatorAgent(BaseAgent[Dict[str, AggregatedSignal]]):
    """
    信号聚合代理

    核心功能：
    1. 收集多策略信号
    2. 评估信号质量
    3. 解决信号冲突
    4. 生成聚合信号
    5. 追踪策略表现
    """

    def __init__(
        self,
        name: str = "signal_aggregator_agent",
        timeout: float = 30.0,
        conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_AVERAGE,
        min_consensus: float = 0.6,      # 最小一致性要求
        min_confidence: float = 0.5,      # 最小置信度要求
    ):
        super().__init__(name, AgentPriority.HIGH, timeout)

        self.conflict_resolution = conflict_resolution
        self.min_consensus = min_consensus
        self.min_confidence = min_confidence

        # 策略权重
        self._strategy_weights: Dict[str, float] = {}

        # 策略表现
        self._strategy_performance: Dict[str, StrategyPerformance] = {}

        # 信号历史
        self._signal_history: List[TradingSignal] = []
        self._max_history = 1000

        # 待处理信号
        self._pending_signals: Dict[str, List[TradingSignal]] = {}  # symbol -> signals

        # 订阅主题
        self.subscribe("signal.generated")
        self.subscribe("signal.request")

        # 注册消息处理器
        self.register_handler("signal.generated", self._handle_signal)

        logger.info(f"SignalAggregatorAgent {name} initialized")

    def register_strategy(
        self,
        strategy_name: str,
        initial_weight: float = 1.0,
    ) -> None:
        """注册策略"""
        self._strategy_weights[strategy_name] = initial_weight
        self._strategy_performance[strategy_name] = StrategyPerformance(
            strategy=strategy_name,
            weight=initial_weight,
        )
        logger.info(f"Registered strategy: {strategy_name} with weight {initial_weight}")

    def update_strategy_weight(
        self,
        strategy_name: str,
        weight: float,
    ) -> None:
        """更新策略权重"""
        if strategy_name in self._strategy_weights:
            self._strategy_weights[strategy_name] = weight
            if strategy_name in self._strategy_performance:
                self._strategy_performance[strategy_name].weight = weight

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行信号聚合

        context 参数:
        - signals: List[Dict] - 输入的交易信号列表
        - symbols: Optional[List[str]] - 要聚合的标的（如果不指定则聚合所有）
        - conflict_resolution: Optional[str] - 冲突解决策略
        - min_consensus: Optional[float] - 最小一致性要求
        """
        raw_signals = context.get("signals", [])
        target_symbols = context.get("symbols")
        resolution = ConflictResolution(
            context.get("conflict_resolution", self.conflict_resolution.value)
        )
        min_consensus = context.get("min_consensus", self.min_consensus)

        # 转换为TradingSignal对象
        signals = self._parse_signals(raw_signals)

        if not signals:
            return AgentResult(
                success=False,
                error="No valid signals provided"
            )

        # 按标的分组
        signals_by_symbol: Dict[str, List[TradingSignal]] = {}
        for signal in signals:
            if target_symbols and signal.symbol not in target_symbols:
                continue
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)

        # 聚合每个标的的信号
        aggregated: Dict[str, AggregatedSignal] = {}
        rejected: Dict[str, str] = {}

        for symbol, symbol_signals in signals_by_symbol.items():
            try:
                agg_signal = self._aggregate_signals(symbol, symbol_signals, resolution)

                # 检查一致性要求
                if agg_signal.consensus_score < min_consensus:
                    rejected[symbol] = f"Low consensus: {agg_signal.consensus_score:.2f}"
                    continue

                # 检查置信度要求
                if agg_signal.confidence < self.min_confidence:
                    rejected[symbol] = f"Low confidence: {agg_signal.confidence:.2f}"
                    continue

                aggregated[symbol] = agg_signal

            except Exception as e:
                logger.error(f"Error aggregating signals for {symbol}: {e}")
                rejected[symbol] = str(e)

        # 记录历史
        self._signal_history.extend(signals)
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]

        # 广播聚合结果
        if aggregated:
            await self.broadcast("signal.aggregated", {
                "aggregated_signals": {s: a.to_dict() for s, a in aggregated.items()},
                "rejected": rejected,
            })

        return AgentResult(
            success=True,
            data={
                "aggregated": aggregated,
                "rejected": rejected,
            },
            metadata={
                "total_signals": len(signals),
                "symbols_processed": len(signals_by_symbol),
                "signals_accepted": len(aggregated),
                "signals_rejected": len(rejected),
                "resolution_method": resolution.value,
            }
        )

    def _parse_signals(self, raw_signals: List[Dict]) -> List[TradingSignal]:
        """解析原始信号"""
        signals = []
        for raw in raw_signals:
            try:
                signal = TradingSignal(
                    symbol=raw["symbol"],
                    signal_type=SignalType(raw.get("signal_type", "hold")),
                    strength=float(raw.get("strength", 0.5)),
                    confidence=float(raw.get("confidence", 0.5)),
                    strategy=raw.get("strategy", "unknown"),
                    timestamp=datetime.fromisoformat(raw["timestamp"]) if "timestamp" in raw else datetime.utcnow(),
                    price_target=raw.get("price_target"),
                    stop_loss=raw.get("stop_loss"),
                    take_profit=raw.get("take_profit"),
                    timeframe=raw.get("timeframe", "1d"),
                    metadata=raw.get("metadata", {}),
                )
                signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to parse signal: {e}")
        return signals

    def _aggregate_signals(
        self,
        symbol: str,
        signals: List[TradingSignal],
        resolution: ConflictResolution,
    ) -> AggregatedSignal:
        """聚合信号"""
        if not signals:
            raise ValueError("No signals to aggregate")

        if len(signals) == 1:
            signal = signals[0]
            return AggregatedSignal(
                symbol=symbol,
                signal_type=signal.signal_type,
                strength=signal.strength,
                confidence=signal.confidence,
                contributing_signals=signals,
                consensus_score=1.0,
                timestamp=datetime.utcnow(),
                price_target=signal.price_target,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

        # 根据冲突解决策略聚合
        if resolution == ConflictResolution.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(symbol, signals)
        elif resolution == ConflictResolution.MAJORITY_VOTE:
            return self._majority_vote_aggregation(symbol, signals)
        elif resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            return self._highest_confidence_aggregation(symbol, signals)
        elif resolution == ConflictResolution.CONSERVATIVE:
            return self._conservative_aggregation(symbol, signals)
        elif resolution == ConflictResolution.AGGRESSIVE:
            return self._aggressive_aggregation(symbol, signals)
        else:
            return self._weighted_average_aggregation(symbol, signals)

    def _weighted_average_aggregation(
        self,
        symbol: str,
        signals: List[TradingSignal]
    ) -> AggregatedSignal:
        """加权平均聚合"""
        # 计算每个信号的权重
        weighted_scores: Dict[SignalType, float] = {t: 0.0 for t in SignalType}
        total_weight = 0.0

        for signal in signals:
            strategy_weight = self._strategy_weights.get(signal.strategy, 1.0)
            signal_weight = strategy_weight * signal.confidence
            weighted_scores[signal.signal_type] += signal_weight * signal.strength
            total_weight += signal_weight

        # 找出最强的信号类型
        if total_weight > 0:
            for signal_type in weighted_scores:
                weighted_scores[signal_type] /= total_weight

        best_signal_type = max(weighted_scores, key=weighted_scores.get)
        strength = weighted_scores[best_signal_type]

        # 计算一致性得分
        directional_weight = sum(
            self._strategy_weights.get(signal.strategy, 1.0)
            * signal.confidence
            * signal.strength
            for signal in signals
            if self._signals_same_direction(signal.signal_type, best_signal_type)
        )
        total_directional_weight = sum(
            self._strategy_weights.get(signal.strategy, 1.0)
            * signal.confidence
            * signal.strength
            for signal in signals
        )
        consensus_score = (
            directional_weight / total_directional_weight
            if total_directional_weight > 0
            else 0.0
        )

        # 计算平均置信度
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        # 聚合止损/止盈
        price_targets = [s.price_target for s in signals if s.price_target]
        stop_losses = [s.stop_loss for s in signals if s.stop_loss]
        take_profits = [s.take_profit for s in signals if s.take_profit]

        return AggregatedSignal(
            symbol=symbol,
            signal_type=best_signal_type,
            strength=strength,
            confidence=avg_confidence * consensus_score,
            contributing_signals=signals,
            consensus_score=consensus_score,
            timestamp=datetime.utcnow(),
            price_target=statistics.mean(price_targets) if price_targets else None,
            stop_loss=statistics.mean(stop_losses) if stop_losses else None,
            take_profit=statistics.mean(take_profits) if take_profits else None,
        )

    def _majority_vote_aggregation(
        self,
        symbol: str,
        signals: List[TradingSignal]
    ) -> AggregatedSignal:
        """多数投票聚合"""
        votes: Dict[SignalType, int] = {t: 0 for t in SignalType}

        for signal in signals:
            votes[signal.signal_type] += 1

        winner = max(votes, key=votes.get)
        winner_count = votes[winner]

        # 筛选同方向的信号
        winning_signals = [s for s in signals if s.signal_type == winner]

        consensus_score = winner_count / len(signals)
        avg_strength = sum(s.strength for s in winning_signals) / len(winning_signals)
        avg_confidence = sum(s.confidence for s in winning_signals) / len(winning_signals)

        return AggregatedSignal(
            symbol=symbol,
            signal_type=winner,
            strength=avg_strength,
            confidence=avg_confidence,
            contributing_signals=signals,
            consensus_score=consensus_score,
            timestamp=datetime.utcnow(),
        )

    def _highest_confidence_aggregation(
        self,
        symbol: str,
        signals: List[TradingSignal]
    ) -> AggregatedSignal:
        """最高置信度聚合"""
        best_signal = max(signals, key=lambda s: s.confidence)

        # 计算一致性
        same_direction = sum(
            1 for s in signals
            if self._signals_same_direction(s.signal_type, best_signal.signal_type)
        )
        consensus_score = same_direction / len(signals)

        return AggregatedSignal(
            symbol=symbol,
            signal_type=best_signal.signal_type,
            strength=best_signal.strength,
            confidence=best_signal.confidence,
            contributing_signals=signals,
            consensus_score=consensus_score,
            timestamp=datetime.utcnow(),
            price_target=best_signal.price_target,
            stop_loss=best_signal.stop_loss,
            take_profit=best_signal.take_profit,
        )

    def _conservative_aggregation(
        self,
        symbol: str,
        signals: List[TradingSignal]
    ) -> AggregatedSignal:
        """保守聚合 - 倾向HOLD"""
        # 检查是否有强一致性
        buy_signals = [s for s in signals if s.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]]
        sell_signals = [s for s in signals if s.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]]

        buy_strength = sum(s.strength * s.confidence for s in buy_signals) if buy_signals else 0
        sell_strength = sum(s.strength * s.confidence for s in sell_signals) if sell_signals else 0

        # 需要显著优势才产生交易信号
        threshold = 0.7 * len(signals)

        if len(buy_signals) >= threshold and buy_strength > sell_strength * 1.5:
            signal_type = SignalType.BUY
            relevant_signals = buy_signals
        elif len(sell_signals) >= threshold and sell_strength > buy_strength * 1.5:
            signal_type = SignalType.SELL
            relevant_signals = sell_signals
        else:
            signal_type = SignalType.HOLD
            relevant_signals = signals

        consensus_score = len(relevant_signals) / len(signals) if signal_type != SignalType.HOLD else 1.0

        return AggregatedSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=sum(s.strength for s in relevant_signals) / len(relevant_signals),
            confidence=sum(s.confidence for s in relevant_signals) / len(relevant_signals) * 0.8,  # 保守降低置信度
            contributing_signals=signals,
            consensus_score=consensus_score,
            timestamp=datetime.utcnow(),
        )

    def _aggressive_aggregation(
        self,
        symbol: str,
        signals: List[TradingSignal]
    ) -> AggregatedSignal:
        """激进聚合 - 倾向交易"""
        # 找出最强的非HOLD信号
        trading_signals = [s for s in signals if s.signal_type != SignalType.HOLD]

        if trading_signals:
            # 选择最强的信号
            best_signal = max(trading_signals, key=lambda s: s.strength * s.confidence)
            signal_type = best_signal.signal_type

            same_direction = sum(
                1 for s in signals
                if self._signals_same_direction(s.signal_type, signal_type)
            )
        else:
            signal_type = SignalType.HOLD
            same_direction = len(signals)

        consensus_score = same_direction / len(signals)

        return AggregatedSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=max(s.strength for s in signals),
            confidence=max(s.confidence for s in signals) * 1.1,  # 激进提高置信度
            contributing_signals=signals,
            consensus_score=consensus_score,
            timestamp=datetime.utcnow(),
        )

    def _signals_same_direction(
        self,
        signal1: SignalType,
        signal2: SignalType
    ) -> bool:
        """判断两个信号是否同方向"""
        buy_types = {SignalType.BUY, SignalType.STRONG_BUY}
        sell_types = {SignalType.SELL, SignalType.STRONG_SELL}

        if signal1 in buy_types and signal2 in buy_types:
            return True
        if signal1 in sell_types and signal2 in sell_types:
            return True
        if signal1 == SignalType.HOLD and signal2 == SignalType.HOLD:
            return True
        return False

    async def _handle_signal(self, message: Any) -> None:
        """处理收到的信号"""
        payload = message.payload
        signal_data = payload.get("signal")

        if signal_data:
            signals = self._parse_signals([signal_data])
            for signal in signals:
                if signal.symbol not in self._pending_signals:
                    self._pending_signals[signal.symbol] = []
                self._pending_signals[signal.symbol].append(signal)

    def update_strategy_performance(
        self,
        strategy: str,
        was_correct: bool,
        return_pct: float,
    ) -> None:
        """更新策略表现"""
        if strategy not in self._strategy_performance:
            self._strategy_performance[strategy] = StrategyPerformance(strategy=strategy)

        perf = self._strategy_performance[strategy]
        perf.total_signals += 1
        if was_correct:
            perf.correct_signals += 1
        perf.accuracy = perf.correct_signals / perf.total_signals
        perf.avg_return = (perf.avg_return * (perf.total_signals - 1) + return_pct) / perf.total_signals
        perf.last_updated = datetime.utcnow()

        # 动态调整权重
        new_weight = 0.5 + perf.accuracy * 0.5 + max(0, perf.avg_return) * 0.1
        self._strategy_weights[strategy] = new_weight
        perf.weight = new_weight

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """获取策略表现统计"""
        return {
            name: {
                "total_signals": perf.total_signals,
                "correct_signals": perf.correct_signals,
                "accuracy": perf.accuracy,
                "avg_return": perf.avg_return,
                "weight": perf.weight,
                "last_updated": perf.last_updated.isoformat(),
            }
            for name, perf in self._strategy_performance.items()
        }
