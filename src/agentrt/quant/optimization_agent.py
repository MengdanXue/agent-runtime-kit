# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Optimization Agent - Strategy parameter optimization.
"""

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging

from ..base import BaseAgent, AgentTask, AgentResult, AgentStatus
from ..registry import register_agent
from .mock_guard import require_real_data

logger = logging.getLogger(__name__)


@register_agent("optimization", task_types=["parameter_optimization", "portfolio_optimization", "strategy_tuning"])
class OptimizationAgent(BaseAgent):
    """
    Agent for strategy and portfolio optimization.

    Supports parameter tuning, portfolio allocation optimization,
    and strategy selection.

    Task Payload:
        type: str - Optimization type
        strategy: str - Strategy to optimize
        symbol: str - Symbol to optimize for
        param_ranges: Dict - Parameter ranges to search
        objective: str - Optimization objective (sharpe, return, risk)
        method: str - Optimization method (grid, random, bayesian)
        max_iterations: int - Maximum iterations
    """

    SUPPORTED_TASKS = {"parameter_optimization", "portfolio_optimization", "strategy_tuning"}

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute optimization task."""
        self.status = AgentStatus.RUNNING

        try:
            opt_type = task.payload.get("type", "parameter_optimization")

            # Every path below scores candidates with random draws rather than
            # backtested objective values, so "best_params" is the argmax of noise.
            require_real_data(
                f"{opt_type} results",
                "OptimizationAgent has no optimizer wired up. Connect it to the backtest "
                "engine (src.backtest) so candidates are scored on real performance.",
            )

            if opt_type == "portfolio_optimization":
                result = await self._optimize_portfolio(task)
            elif opt_type == "strategy_tuning":
                result = await self._tune_strategy(task)
            else:
                result = await self._optimize_parameters(task)

            self.status = AgentStatus.COMPLETED
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.status = AgentStatus.FAILED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=str(e)
            )

    async def _optimize_parameters(self, task: AgentTask) -> AgentResult:
        """Optimize strategy parameters."""
        strategy = task.payload.get("strategy")
        symbol = task.payload.get("symbol")
        param_ranges = task.payload.get("param_ranges", {})
        objective = task.payload.get("objective", "sharpe")
        method = task.payload.get("method", "grid")
        max_iterations = task.payload.get("max_iterations", 100)

        # Run optimization
        if method == "grid":
            best_params, best_score, history = await self._grid_search(
                strategy, symbol, param_ranges, objective, max_iterations
            )
        elif method == "random":
            best_params, best_score, history = await self._random_search(
                strategy, symbol, param_ranges, objective, max_iterations
            )
        else:
            best_params, best_score, history = await self._bayesian_search(
                strategy, symbol, param_ranges, objective, max_iterations
            )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "optimization_type": "parameter_optimization",
                "strategy": strategy,
                "symbol": symbol,
                "objective": objective,
                "method": method,
                "best_params": best_params,
                "best_score": best_score,
                "iterations": len(history),
                "optimization_history": history[-10:]  # Last 10 iterations
            }
        )

    async def _optimize_portfolio(self, task: AgentTask) -> AgentResult:
        """Optimize portfolio allocation."""
        symbols = task.payload.get("symbols", [])
        objective = task.payload.get("objective", "sharpe")
        constraints = task.payload.get("constraints", {})

        # Mock optimization - in production would use scipy.optimize or cvxpy
        optimal_weights = self._mock_portfolio_optimization(
            symbols, objective, constraints
        )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "optimization_type": "portfolio_optimization",
                "symbols": symbols,
                "objective": objective,
                "optimal_weights": optimal_weights,
                "expected_return": sum(w * 0.1 for w in optimal_weights.values()),
                "expected_volatility": 0.15,
                "expected_sharpe": 0.67
            }
        )

    async def _tune_strategy(self, task: AgentTask) -> AgentResult:
        """Tune strategy for specific market conditions."""
        strategy = task.payload.get("strategy")
        market_regime = task.payload.get("market_regime", "neutral")
        lookback_period = task.payload.get("lookback_period", 252)

        # Suggest tuned parameters based on market regime
        tuned_params = self._suggest_regime_params(strategy, market_regime)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "optimization_type": "strategy_tuning",
                "strategy": strategy,
                "market_regime": market_regime,
                "tuned_params": tuned_params,
                "regime_fit_score": 0.85,
                "recommendations": self._get_regime_recommendations(market_regime)
            }
        )

    async def _grid_search(
        self,
        strategy: str,
        symbol: str,
        param_ranges: Dict,
        objective: str,
        max_iterations: int
    ) -> Tuple[Dict, float, List]:
        """Grid search optimization."""
        require_real_data("random draws in _grid_search",
                          "synthetic data for local development only (ALLOW_MOCK_AGENT_DATA=true)")
        import random

        history = []
        best_params = {}
        best_score = float("-inf")

        # Generate grid
        for i in range(min(max_iterations, 50)):
            # Random params within ranges (simplified grid)
            params = {}
            for key, (low, high) in param_ranges.items():
                if isinstance(low, int):
                    params[key] = random.randint(low, high)
                else:
                    params[key] = random.uniform(low, high)

            # Evaluate (mock)
            score = random.uniform(0.5, 2.0)

            history.append({"params": params, "score": score})

            if score > best_score:
                best_score = score
                best_params = params

            # Simulate some computation time
            await asyncio.sleep(0.01)

        return best_params, best_score, history

    async def _random_search(
        self,
        strategy: str,
        symbol: str,
        param_ranges: Dict,
        objective: str,
        max_iterations: int
    ) -> Tuple[Dict, float, List]:
        """Random search optimization."""
        # Similar to grid but truly random
        return await self._grid_search(strategy, symbol, param_ranges, objective, max_iterations)

    async def _bayesian_search(
        self,
        strategy: str,
        symbol: str,
        param_ranges: Dict,
        objective: str,
        max_iterations: int
    ) -> Tuple[Dict, float, List]:
        """Bayesian optimization (mock)."""
        require_real_data("random draws in _bayesian_search",
                          "synthetic data for local development only (ALLOW_MOCK_AGENT_DATA=true)")
        import random

        history = []
        best_params = {}
        best_score = float("-inf")

        # Bayesian-style: starts random, then focuses on promising regions
        for i in range(min(max_iterations, 30)):
            params = {}
            for key, (low, high) in param_ranges.items():
                if i < 10:
                    # Exploration phase
                    if isinstance(low, int):
                        params[key] = random.randint(low, high)
                    else:
                        params[key] = random.uniform(low, high)
                else:
                    # Exploitation phase - sample near best
                    if best_params:
                        center = best_params.get(key, (low + high) / 2)
                        spread = (high - low) * 0.2  # 20% of range
                        if isinstance(low, int):
                            params[key] = int(max(low, min(high, random.gauss(center, spread))))
                        else:
                            params[key] = max(low, min(high, random.gauss(center, spread)))
                    else:
                        params[key] = random.uniform(low, high)

            # Evaluate (mock) - Bayesian tends to improve over time
            base_score = random.uniform(0.5, 2.0)
            score = base_score + i * 0.02  # Slight improvement trend

            history.append({"params": params, "score": score})

            if score > best_score:
                best_score = score
                best_params = params

            await asyncio.sleep(0.01)

        return best_params, best_score, history

    def _mock_portfolio_optimization(
        self,
        symbols: List[str],
        objective: str,
        constraints: Dict
    ) -> Dict[str, float]:
        """Mock portfolio optimization."""
        require_real_data("random draws in _mock_portfolio_optimization",
                          "synthetic data for local development only (ALLOW_MOCK_AGENT_DATA=true)")
        import random

        n = len(symbols)
        if n == 0:
            return {}

        # Generate random weights that sum to 1
        raw_weights = [random.random() for _ in range(n)]
        total = sum(raw_weights)
        weights = {s: w / total for s, w in zip(symbols, raw_weights)}

        # Apply constraints
        max_weight = constraints.get("max_weight", 0.4)
        min_weight = constraints.get("min_weight", 0.05)

        for s in weights:
            weights[s] = max(min_weight, min(max_weight, weights[s]))

        # Renormalize
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}

        return weights

    def _suggest_regime_params(
        self,
        strategy: str,
        market_regime: str
    ) -> Dict[str, Any]:
        """Suggest parameters based on market regime."""
        # Default params
        params = {
            "lookback_period": 20,
            "entry_threshold": 1.5,
            "exit_threshold": 0.5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

        # Adjust for regime
        if market_regime == "bullish":
            params["entry_threshold"] = 1.0  # More aggressive entry
            params["take_profit_pct"] = 0.06  # Higher targets
        elif market_regime == "bearish":
            params["entry_threshold"] = 2.0  # More conservative
            params["stop_loss_pct"] = 0.015  # Tighter stops
        elif market_regime == "volatile":
            params["lookback_period"] = 10  # Shorter lookback
            params["stop_loss_pct"] = 0.03  # Wider stops
        elif market_regime == "low_volatility":
            params["lookback_period"] = 30  # Longer lookback
            params["entry_threshold"] = 1.2  # Slightly aggressive

        return params

    def _get_regime_recommendations(self, market_regime: str) -> List[str]:
        """Get strategy recommendations for market regime."""
        recommendations = {
            "bullish": [
                "Consider momentum strategies",
                "Reduce hedging",
                "Increase position sizes",
            ],
            "bearish": [
                "Consider mean reversion strategies",
                "Increase hedging",
                "Reduce position sizes",
                "Consider short strategies",
            ],
            "volatile": [
                "Consider volatility strategies",
                "Use wider stops",
                "Reduce leverage",
            ],
            "low_volatility": [
                "Consider range-bound strategies",
                "May increase position sizes",
                "Consider selling options",
            ],
            "neutral": [
                "Maintain balanced approach",
                "Monitor for regime change",
            ],
        }
        return recommendations.get(market_regime, recommendations["neutral"])
