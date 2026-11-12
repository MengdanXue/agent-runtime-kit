# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Signal Agent - Generates trading signals using multiple strategies.
"""

from typing import Any, Dict, List, Optional
import asyncio
import logging

from ..base import BaseAgent, AgentTask, AgentResult, AgentStatus
from ..registry import register_agent
from .mock_guard import require_real_data

logger = logging.getLogger(__name__)


@register_agent("signal", task_types=["generate_signal", "multi_signal", "signal_fusion"])
class SignalAgent(BaseAgent):
    """
    Agent for generating trading signals.

    Supports multiple strategies and signal aggregation.

    Task Payload:
        symbol: str - Trading symbol (e.g., "AAPL")
        timeframe: str - Timeframe (e.g., "1d", "1h")
        strategies: List[str] - Strategy names to use (optional)
        aggregate: bool - Whether to aggregate signals (default: True)

    Example:
        task = AgentTask(
            agent_type="signal",
            payload={
                "symbol": "AAPL",
                "timeframe": "1d",
                "strategies": ["momentum", "mean_reversion"],
                "aggregate": True
            }
        )
        result = await orchestrator.dispatch(task)
    """

    # Supported task types
    SUPPORTED_TASKS = {"generate_signal", "multi_signal", "signal_fusion"}

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id)
        self._strategy_cache = {}

    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle the task type."""
        return task_type in self.SUPPORTED_TASKS

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute signal generation task."""
        self.status = AgentStatus.RUNNING

        try:
            symbol = task.payload.get("symbol")
            if not symbol:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    error="Missing required field: symbol"
                )

            timeframe = task.payload.get("timeframe", "1d")
            strategies = task.payload.get("strategies", [])
            aggregate = task.payload.get("aggregate", True)

            # Generate signals
            signals = await self._generate_signals(symbol, timeframe, strategies)

            # Aggregate if requested
            aggregated = None
            if aggregate and len(signals) > 1:
                aggregated = self._aggregate_signals(signals)

            self.status = AgentStatus.COMPLETED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                result={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "signals": signals,
                    "aggregated": aggregated,
                    "strategies_used": [s["strategy"] for s in signals]
                }
            )

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            self.status = AgentStatus.FAILED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=str(e)
            )

    async def _generate_signals(
        self,
        symbol: str,
        timeframe: str,
        strategies: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate signals from strategies."""
        signals = []

        # Import strategy modules dynamically
        try:
            from ...strategies import get_strategy_by_name
        except ImportError as exc:
            require_real_data(
                f"trading signals for {symbol}",
                f"The strategy package could not be imported ({exc}); install it so that "
                f"signals come from real strategies instead of random draws.",
            )
            return self._generate_mock_signals(symbol, timeframe, strategies)

        for strategy_name in strategies:
            try:
                strategy = get_strategy_by_name(strategy_name)
                if strategy:
                    signal = await self._run_strategy(strategy, symbol, timeframe)
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                signals.append({
                    "strategy": strategy_name,
                    "signal": "ERROR",
                    "confidence": 0.0,
                    "error": str(e)
                })

        return signals

    async def _run_strategy(
        self,
        strategy,
        symbol: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Run a single strategy."""
        # Execute strategy in thread pool if it's CPU-bound
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: strategy.generate_signal(symbol, timeframe)
        )

        return {
            "strategy": strategy.name,
            "signal": result.signal.value if hasattr(result.signal, 'value') else str(result.signal),
            "confidence": result.confidence if hasattr(result, 'confidence') else 0.8,
            "metadata": result.metadata if hasattr(result, 'metadata') else {}
        }

    def _generate_mock_signals(
        self,
        symbol: str,
        timeframe: str,
        strategies: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate mock signals for testing."""
        require_real_data("random draws in _generate_mock_signals",
                          "synthetic data for local development only (ALLOW_MOCK_AGENT_DATA=true)")
        import random

        signal_values = ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]

        signals = []
        for strategy in strategies or ["default"]:
            signals.append({
                "strategy": strategy,
                "signal": random.choice(signal_values),
                "confidence": random.uniform(0.5, 0.95),
                "metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "mock": True
                }
            })

        return signals

    def _aggregate_signals(
        self,
        signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate multiple signals into a consensus."""
        if not signals:
            return None

        # Score mapping
        score_map = {
            "STRONG_BUY": 2,
            "BUY": 1,
            "HOLD": 0,
            "SELL": -1,
            "STRONG_SELL": -2,
        }

        # Calculate weighted average
        total_score = 0
        total_weight = 0

        for sig in signals:
            signal = sig.get("signal", "HOLD")
            confidence = sig.get("confidence", 0.5)

            if signal in score_map:
                total_score += score_map[signal] * confidence
                total_weight += confidence

        if total_weight == 0:
            return {"signal": "HOLD", "confidence": 0.0}

        avg_score = total_score / total_weight

        # Map back to signal
        if avg_score >= 1.5:
            final_signal = "STRONG_BUY"
        elif avg_score >= 0.5:
            final_signal = "BUY"
        elif avg_score <= -1.5:
            final_signal = "STRONG_SELL"
        elif avg_score <= -0.5:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        return {
            "signal": final_signal,
            "score": avg_score,
            "confidence": total_weight / len(signals),
            "contributing_signals": len(signals)
        }
