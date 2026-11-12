# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Risk Assessment Agent - 风险评估代理
负责交易前风险评估和持续风险监控

功能：
1. 交易前风险评估
2. 投资组合风险计算
3. 仓位限制检查
4. 风险预警
5. VaR/CVaR计算
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging
import math

from .agent_base import BaseAgent, AgentResult, AgentPriority

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """风险类型"""
    POSITION_SIZE = "position_size"       # 仓位规模
    CONCENTRATION = "concentration"       # 集中度
    CORRELATION = "correlation"           # 相关性
    VOLATILITY = "volatility"             # 波动率
    DRAWDOWN = "drawdown"                 # 回撤
    LIQUIDITY = "liquidity"               # 流动性
    MARKET = "market"                     # 市场风险


@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float = 0.0              # 95% VaR
    var_99: float = 0.0              # 99% VaR
    cvar_95: float = 0.0             # 95% CVaR (Expected Shortfall)
    max_drawdown: float = 0.0        # 最大回撤
    current_drawdown: float = 0.0    # 当前回撤
    volatility: float = 0.0          # 波动率
    sharpe_ratio: float = 0.0        # 夏普比率
    sortino_ratio: float = 0.0       # 索提诺比率
    beta: float = 0.0                # Beta系数
    correlation_avg: float = 0.0     # 平均相关性

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "beta": self.beta,
            "correlation_avg": self.correlation_avg,
        }


@dataclass
class RiskWarning:
    """风险预警"""
    type: RiskType
    level: RiskLevel
    message: str
    symbol: Optional[str] = None
    value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "level": self.level.value,
            "message": self.message,
            "symbol": self.symbol,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeRiskAssessment:
    """交易风险评估结果"""
    symbol: str
    approved: bool
    risk_level: RiskLevel
    risk_score: float                    # 0-100 风险得分
    warnings: List[RiskWarning]
    max_position_size: float             # 建议最大仓位
    suggested_stop_loss: float           # 建议止损
    position_sizing: Dict[str, float]    # 仓位建议
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "approved": self.approved,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "warnings": [w.to_dict() for w in self.warnings],
            "max_position_size": self.max_position_size,
            "suggested_stop_loss": self.suggested_stop_loss,
            "position_sizing": self.position_sizing,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskLimits:
    """风险限制"""
    max_position_pct: float = 0.1        # 单一仓位最大占比
    max_sector_pct: float = 0.3          # 单一板块最大占比
    max_correlation: float = 0.7         # 最大相关性
    max_volatility: float = 0.5          # 最大波动率
    max_drawdown: float = 0.2            # 最大回撤
    max_var_daily: float = 0.05          # 每日最大VaR
    min_liquidity_ratio: float = 3.0     # 最小流动性比率
    max_leverage: float = 1.0            # 最大杠杆


class RiskAssessmentAgent(BaseAgent[TradeRiskAssessment]):
    """
    风险评估代理

    核心功能：
    1. 交易前风险评估
    2. 投资组合风险监控
    3. 仓位规模计算
    4. 风险预警生成
    """

    def __init__(
        self,
        name: str = "risk_assessment_agent",
        timeout: float = 30.0,
        risk_limits: Optional[RiskLimits] = None,
    ):
        super().__init__(name, AgentPriority.CRITICAL, timeout)

        self.risk_limits = risk_limits or RiskLimits()

        # 投资组合状态
        self._portfolio: Dict[str, float] = {}      # symbol -> value
        self._total_value: float = 0.0

        # 市场数据缓存
        self._volatility_cache: Dict[str, float] = {}
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}

        # 历史风险记录
        self._risk_history: List[RiskMetrics] = []
        self._warning_history: List[RiskWarning] = []

        # 订阅主题
        self.subscribe("risk.assess")
        self.subscribe("portfolio.updated")

        # 注册消息处理器
        self.register_handler("portfolio.updated", self._handle_portfolio_update)

        logger.info(f"RiskAssessmentAgent {name} initialized")

    def set_portfolio(self, positions: Dict[str, float], total_value: float) -> None:
        """设置投资组合"""
        self._portfolio = positions.copy()
        self._total_value = total_value

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行风险评估

        context 参数:
        - symbol: str - 要评估的标的
        - action: str - 交易动作 (buy, sell)
        - quantity: float - 交易数量
        - price: float - 交易价格
        - portfolio: Optional[Dict] - 当前持仓
        - market_data: Optional[Dict] - 市场数据
        """
        symbol = context.get("symbol")
        action = context.get("action", "buy")
        quantity = context.get("quantity", 0)
        price = context.get("price", 0)
        portfolio = context.get("portfolio", self._portfolio)
        market_data = context.get("market_data", {})

        if not symbol:
            return AgentResult(
                success=False,
                error="Symbol is required for risk assessment"
            )

        try:
            # 计算交易价值
            trade_value = quantity * price

            # 收集所有风险警告
            warnings: List[RiskWarning] = []

            # 1. 检查仓位规模
            position_warning = self._check_position_size(symbol, trade_value, action)
            if position_warning:
                warnings.append(position_warning)

            # 2. 检查集中度
            concentration_warning = self._check_concentration(symbol, trade_value, action, portfolio)
            if concentration_warning:
                warnings.append(concentration_warning)

            # 3. 检查波动率
            volatility_warning = self._check_volatility(symbol, market_data)
            if volatility_warning:
                warnings.append(volatility_warning)

            # 4. 检查相关性
            correlation_warning = self._check_correlation(symbol, portfolio)
            if correlation_warning:
                warnings.append(correlation_warning)

            # 5. 检查流动性
            liquidity_warning = self._check_liquidity(symbol, trade_value, market_data)
            if liquidity_warning:
                warnings.append(liquidity_warning)

            # 计算风险得分
            risk_score = self._calculate_risk_score(warnings)

            # 确定风险等级
            risk_level = self._determine_risk_level(risk_score, warnings)

            # 判断是否批准交易
            approved = self._should_approve_trade(risk_level, warnings)

            # 计算建议仓位
            max_position = self._calculate_max_position(
                symbol, trade_value, risk_score, market_data
            )

            # 计算建议止损
            stop_loss = self._calculate_stop_loss(symbol, price, market_data)

            # 仓位建议
            position_sizing = self._calculate_position_sizing(
                symbol, price, risk_score, market_data
            )

            assessment = TradeRiskAssessment(
                symbol=symbol,
                approved=approved,
                risk_level=risk_level,
                risk_score=risk_score,
                warnings=warnings,
                max_position_size=max_position,
                suggested_stop_loss=stop_loss,
                position_sizing=position_sizing,
            )

            # 如果有高风险警告，广播预警
            if any(w.level in [RiskLevel.HIGH, RiskLevel.CRITICAL] for w in warnings):
                await self.broadcast("risk.warning", {
                    "symbol": symbol,
                    "warnings": [w.to_dict() for w in warnings],
                    "risk_level": risk_level.value,
                })

            return AgentResult(
                success=True,
                data=assessment,
                metadata={
                    "risk_score": risk_score,
                    "risk_level": risk_level.value,
                    "warnings_count": len(warnings),
                    "approved": approved,
                }
            )

        except Exception as e:
            logger.exception(f"Risk assessment failed: {e}")
            return AgentResult(
                success=False,
                error=str(e)
            )

    def _check_position_size(
        self,
        symbol: str,
        trade_value: float,
        action: str
    ) -> Optional[RiskWarning]:
        """检查仓位规模"""
        if self._total_value <= 0:
            return None

        current_position = self._portfolio.get(symbol, 0)

        if action == "buy":
            new_position = current_position + trade_value
        else:
            new_position = current_position - trade_value

        position_pct = abs(new_position) / self._total_value

        if position_pct > self.risk_limits.max_position_pct:
            level = RiskLevel.CRITICAL if position_pct > self.risk_limits.max_position_pct * 1.5 else RiskLevel.HIGH
            return RiskWarning(
                type=RiskType.POSITION_SIZE,
                level=level,
                message=f"Position size {position_pct:.1%} exceeds limit {self.risk_limits.max_position_pct:.1%}",
                symbol=symbol,
                value=position_pct,
                threshold=self.risk_limits.max_position_pct,
            )

        return None

    def _check_concentration(
        self,
        symbol: str,
        trade_value: float,
        action: str,
        portfolio: Dict[str, float]
    ) -> Optional[RiskWarning]:
        """检查集中度"""
        if self._total_value <= 0:
            return None

        # 简化：检查单一资产占比
        # 实际应该按板块/资产类别分组
        total_position = sum(abs(v) for v in portfolio.values())
        if total_position <= 0:
            return None

        top_positions = sorted(portfolio.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_concentration = sum(abs(v) for _, v in top_positions) / self._total_value

        if top_concentration > 0.6:  # 前3大持仓占比超过60%
            return RiskWarning(
                type=RiskType.CONCENTRATION,
                level=RiskLevel.MEDIUM,
                message=f"Portfolio too concentrated: top 3 positions = {top_concentration:.1%}",
                value=top_concentration,
                threshold=0.6,
            )

        return None

    def _check_volatility(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[RiskWarning]:
        """检查波动率"""
        volatility = market_data.get("volatility", self._volatility_cache.get(symbol, 0))

        if volatility > self.risk_limits.max_volatility:
            level = RiskLevel.HIGH if volatility > self.risk_limits.max_volatility * 1.5 else RiskLevel.MEDIUM
            return RiskWarning(
                type=RiskType.VOLATILITY,
                level=level,
                message=f"High volatility: {volatility:.1%} > {self.risk_limits.max_volatility:.1%}",
                symbol=symbol,
                value=volatility,
                threshold=self.risk_limits.max_volatility,
            )

        return None

    def _check_correlation(
        self,
        symbol: str,
        portfolio: Dict[str, float]
    ) -> Optional[RiskWarning]:
        """检查相关性"""
        if symbol not in self._correlation_matrix:
            return None

        correlations = self._correlation_matrix.get(symbol, {})
        high_correlations = [
            (s, c) for s, c in correlations.items()
            if s in portfolio and c > self.risk_limits.max_correlation
        ]

        if high_correlations:
            highest = max(high_correlations, key=lambda x: x[1])
            return RiskWarning(
                type=RiskType.CORRELATION,
                level=RiskLevel.MEDIUM,
                message=f"High correlation with {highest[0]}: {highest[1]:.2f}",
                symbol=symbol,
                value=highest[1],
                threshold=self.risk_limits.max_correlation,
            )

        return None

    def _check_liquidity(
        self,
        symbol: str,
        trade_value: float,
        market_data: Dict[str, Any]
    ) -> Optional[RiskWarning]:
        """检查流动性"""
        avg_volume = market_data.get("avg_volume", 0)
        avg_price = market_data.get("price", 1)

        if avg_volume <= 0:
            return None

        daily_liquidity = avg_volume * avg_price
        liquidity_ratio = daily_liquidity / trade_value if trade_value > 0 else float('inf')

        if liquidity_ratio < self.risk_limits.min_liquidity_ratio:
            return RiskWarning(
                type=RiskType.LIQUIDITY,
                level=RiskLevel.HIGH,
                message=f"Low liquidity: trade is {1/liquidity_ratio:.1%} of daily volume",
                symbol=symbol,
                value=liquidity_ratio,
                threshold=self.risk_limits.min_liquidity_ratio,
            )

        return None

    def _calculate_risk_score(self, warnings: List[RiskWarning]) -> float:
        """计算风险得分 (0-100)"""
        if not warnings:
            return 0.0

        # 根据警告级别计算得分
        level_scores = {
            RiskLevel.LOW: 10,
            RiskLevel.MEDIUM: 30,
            RiskLevel.HIGH: 60,
            RiskLevel.CRITICAL: 100,
        }

        total_score = sum(level_scores[w.level] for w in warnings)

        # 归一化到0-100
        return min(100, total_score)

    def _determine_risk_level(
        self,
        risk_score: float,
        warnings: List[RiskWarning]
    ) -> RiskLevel:
        """确定整体风险等级"""
        # 如果有CRITICAL警告，直接返回CRITICAL
        if any(w.level == RiskLevel.CRITICAL for w in warnings):
            return RiskLevel.CRITICAL

        if risk_score >= 60:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _should_approve_trade(
        self,
        risk_level: RiskLevel,
        warnings: List[RiskWarning]
    ) -> bool:
        """判断是否批准交易"""
        # CRITICAL风险不批准
        if risk_level == RiskLevel.CRITICAL:
            return False

        # 多个HIGH级别警告不批准
        high_warnings = [w for w in warnings if w.level == RiskLevel.HIGH]
        if len(high_warnings) >= 2:
            return False

        return True

    def _calculate_max_position(
        self,
        symbol: str,
        trade_value: float,
        risk_score: float,
        market_data: Dict[str, Any]
    ) -> float:
        """计算建议的最大仓位"""
        base_max = self._total_value * self.risk_limits.max_position_pct

        # 根据风险得分调整
        risk_factor = 1 - (risk_score / 100) * 0.5  # 风险越高，仓位越小

        # 根据波动率调整
        volatility = market_data.get("volatility", 0.2)
        vol_factor = max(0.5, 1 - volatility)

        return base_max * risk_factor * vol_factor

    def _calculate_stop_loss(
        self,
        symbol: str,
        price: float,
        market_data: Dict[str, Any]
    ) -> float:
        """计算建议止损价"""
        volatility = market_data.get("volatility", 0.2)

        # 基于ATR或波动率计算止损
        atr = market_data.get("atr", price * volatility * 0.1)

        # 止损设在2倍ATR
        stop_loss = price - (2 * atr)

        return max(0, stop_loss)

    def _calculate_position_sizing(
        self,
        symbol: str,
        price: float,
        risk_score: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算仓位建议"""
        volatility = market_data.get("volatility", 0.2)

        # Kelly公式简化版
        win_rate = market_data.get("win_rate", 0.5)
        avg_win = market_data.get("avg_win", 0.02)
        avg_loss = market_data.get("avg_loss", 0.01)

        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # 限制Kelly比例
        else:
            kelly_fraction = 0.1

        # 固定风险比例法
        risk_per_trade = 0.02  # 每笔交易风险2%
        stop_distance = volatility * price * 2  # 2倍波动率止损
        if stop_distance > 0:
            fixed_risk_size = (self._total_value * risk_per_trade) / stop_distance
        else:
            fixed_risk_size = 0

        return {
            "kelly_fraction": kelly_fraction,
            "kelly_position": self._total_value * kelly_fraction,
            "fixed_risk_size": fixed_risk_size,
            "conservative_size": min(kelly_fraction, 0.05) * self._total_value,
            "aggressive_size": kelly_fraction * 1.5 * self._total_value,
        }

    async def _handle_portfolio_update(self, message: Any) -> None:
        """处理投资组合更新"""
        payload = message.payload
        self._portfolio = payload.get("positions", {})
        self._total_value = payload.get("total_value", 0)

    def set_volatility(self, symbol: str, volatility: float) -> None:
        """设置标的波动率"""
        self._volatility_cache[symbol] = volatility

    def set_correlation(self, symbol1: str, symbol2: str, correlation: float) -> None:
        """设置相关性"""
        if symbol1 not in self._correlation_matrix:
            self._correlation_matrix[symbol1] = {}
        if symbol2 not in self._correlation_matrix:
            self._correlation_matrix[symbol2] = {}

        self._correlation_matrix[symbol1][symbol2] = correlation
        self._correlation_matrix[symbol2][symbol1] = correlation

    async def calculate_portfolio_risk(self) -> RiskMetrics:
        """计算投资组合风险指标"""
        # 简化实现 - 实际应使用历史数据计算
        metrics = RiskMetrics()

        if not self._portfolio or self._total_value <= 0:
            return metrics

        # 计算加权波动率
        weighted_vol = 0.0
        for symbol, value in self._portfolio.items():
            weight = abs(value) / self._total_value
            vol = self._volatility_cache.get(symbol, 0.2)
            weighted_vol += weight * vol

        metrics.volatility = weighted_vol

        # 简化VaR计算 (假设正态分布)
        metrics.var_95 = self._total_value * weighted_vol * 1.645 / math.sqrt(252)
        metrics.var_99 = self._total_value * weighted_vol * 2.326 / math.sqrt(252)
        metrics.cvar_95 = metrics.var_95 * 1.2  # 简化CVaR

        return metrics
