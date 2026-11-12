# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Risk Agent - Analyzes portfolio risk metrics.

Every number this agent returns is computed by
:class:`~src.services.portfolio_analytics.PortfolioAnalyticsService` from the
portfolio's real positions and real price history. When a metric cannot be
computed it is reported as unavailable with the reason - it is never replaced by
a plausible-looking substitute.
"""

from typing import Any, Awaitable, Dict, List, Optional
from contextlib import asynccontextmanager
import logging

from ..base import BaseAgent, AgentTask, AgentResult, AgentStatus
from ..registry import register_agent

logger = logging.getLogger(__name__)


@register_agent("risk", task_types=["risk_analysis", "var_calculation", "stress_test", "correlation"])
class RiskAgent(BaseAgent):
    """
    Agent for portfolio risk analysis.

    Calculates VaR, correlations, stress tests, and other risk metrics against
    the portfolio's stored positions.

    Task Payload:
        portfolio_id: int - Portfolio identifier (required)
        type: str - Analysis type (risk_analysis, var_calculation, stress_test, correlation)
        confidence_level: float - VaR confidence level, e.g. 0.95 (default: 0.95)
        window: str - Historical lookback window, e.g. '1m'/'3m'/'1y'/'3y' (default: '1y')
        benchmark: str - Benchmark symbol for beta/correlation (default: 'SPY')
    """

    SUPPORTED_TASKS = {"risk_analysis", "var_calculation", "stress_test", "correlation"}

    DEFAULT_WINDOW = "1y"
    DEFAULT_BENCHMARK = "SPY"

    # VaR as a share of portfolio exposure, in percent.
    HIGH_RISK_VAR_PCT = 5.0
    MEDIUM_RISK_VAR_PCT = 2.0

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute risk analysis task."""
        self.status = AgentStatus.RUNNING

        try:
            analysis_type = task.payload.get("type", "risk_analysis")

            if analysis_type == "var_calculation":
                result = await self._calculate_var(task)
            elif analysis_type == "stress_test":
                result = await self._run_stress_test(task)
            elif analysis_type == "correlation":
                result = await self._calculate_correlations(task)
            else:
                result = await self._full_risk_analysis(task)

            self.status = AgentStatus.COMPLETED
            return result

        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            self.status = AgentStatus.FAILED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=str(e)
            )

    # ------------------------------------------------------------------
    # Analytics plumbing
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _analytics(self):
        """Yield a PortfolioAnalyticsService bound to a fresh async DB session.

        Imports are deferred: opening a session through ``agentrt.ports.db`` may build engines
        and requires DB credentials, which must not be a side effect of importing
        the agent package.
        """
        from agentrt.ports.db import open_session
        from agentrt.ports.portfolio import provider

        async with open_session() as session:
            yield provider().analytics(session)

    @staticmethod
    def _parse_portfolio_id(raw: Any) -> int:
        """Portfolio ids are integers in the database; reject anything else."""
        if raw is None or raw == "":
            raise ValueError("portfolio_id is required for risk analysis")
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"portfolio_id must be an integer, got {raw!r}")

    @staticmethod
    def _to_significance(confidence_level: float) -> float:
        """Convert a confidence level (0.95) to a significance level (0.05).

        PortfolioAnalyticsService takes the tail probability. Confidence levels
        below 0.5 are rejected rather than guessed at, so that a caller passing
        0.05 gets an error instead of a silently inverted 95% tail.
        """
        try:
            confidence = float(confidence_level)
        except (TypeError, ValueError):
            raise ValueError(f"confidence_level must be a number, got {confidence_level!r}")
        if not 0.5 < confidence < 1.0:
            raise ValueError(
                f"confidence_level must be a confidence between 0.5 and 1.0 "
                f"(e.g. 0.95 or 0.99), got {confidence}"
            )
        return round(1.0 - confidence, 10)

    @staticmethod
    def _unwrap(label: str, payload: Any) -> Dict:
        """Turn the service's ``{'error': ...}`` convention into a real failure."""
        if not isinstance(payload, dict):
            raise RuntimeError(f"{label} returned an unexpected payload: {payload!r}")
        if "error" in payload:
            raise RuntimeError(f"{label} failed: {payload['error']}")
        return payload

    async def _collect(
        self,
        label: str,
        awaitable: Awaitable[Any],
        unavailable: Dict[str, str],
    ) -> Optional[Dict]:
        """Await one metric, recording why it is missing instead of inventing it."""
        try:
            return self._unwrap(label, await awaitable)
        except Exception as exc:
            logger.warning("Risk metric '%s' unavailable: %s", label, exc)
            unavailable[label] = str(exc)
            return None

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    async def _full_risk_analysis(self, task: AgentTask) -> AgentResult:
        """Run comprehensive risk analysis from stored positions and price history."""
        portfolio_id = self._parse_portfolio_id(task.payload.get("portfolio_id"))
        window = task.payload.get("window", self.DEFAULT_WINDOW)
        benchmark = task.payload.get("benchmark", self.DEFAULT_BENCHMARK)

        metrics: Dict[str, Any] = {}
        unavailable: Dict[str, str] = {}

        async with self._analytics() as service:
            static = await self._collect(
                "portfolio_static",
                service.generate_portfolio_static(portfolio_id, window),
                unavailable,
            )
            if static:
                metrics["sharpe_ratio"] = static.get("sharpe_ratio")
                metrics["max_drawdown"] = static.get("max_drawdown")
                metrics["cumulative_return"] = static.get("cumulative_return")

            volatility = await self._collect(
                "volatility",
                service.calculate_portfolio_volatility(portfolio_id, window=window),
                unavailable,
            )
            if volatility:
                metrics["volatility"] = volatility.get("annualized_volatility")
                metrics["daily_volatility"] = volatility.get("portfolio_volatility")

            var_95 = await self._collect(
                "var_95",
                service.calculate_var(portfolio_id, confidence_level=0.05, window=window),
                unavailable,
            )
            if var_95:
                metrics["var_95"] = var_95.get("var_amount")
                metrics["var_95_pct"] = var_95.get("var_pct")
                if var_95.get("equity") is not None:  # the base the *_pct figures divide by
                    metrics["var_base_value"] = var_95.get("equity")

            var_99 = await self._collect(
                "var_99",
                service.calculate_var(portfolio_id, confidence_level=0.01, window=window),
                unavailable,
            )
            if var_99:
                metrics["var_99"] = var_99.get("var_amount")
                metrics["var_99_pct"] = var_99.get("var_pct")

            cvar_95 = await self._collect(
                "cvar_95",
                service.calculate_cvar(portfolio_id, confidence_level=0.05, window=window),
                unavailable,
            )
            if cvar_95:
                metrics["cvar_95"] = cvar_95.get("cvar_amount")
                metrics["cvar_95_pct"] = cvar_95.get("cvar_pct")

            beta = await self._collect(
                "beta",
                service.calculate_beta(portfolio_id, benchmark, window),
                unavailable,
            )
            if beta:
                metrics["beta"] = beta.get("beta")
                metrics["alpha"] = beta.get("alpha")
                metrics[f"correlation_to_{benchmark.lower()}"] = beta.get("correlation")

            contribution = await self._collect(
                "risk_contribution",
                service.calculate_marginal_risk_contribution(portfolio_id, window=window),
                unavailable,
            )

        if not metrics:
            raise RuntimeError(
                f"No risk metric could be computed for portfolio {portfolio_id}: "
                + "; ".join(f"{k}: {v}" for k, v in unavailable.items())
            )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "portfolio_id": portfolio_id,
                "analysis_type": "full_risk_analysis",
                "window": window,
                "benchmark": benchmark,
                "metrics": metrics,
                "risk_contribution": (contribution or {}).get("positions", []),
                "concentration": (contribution or {}).get("concentration_metrics"),
                "risk_level": self._assess_risk_level(metrics),
                # Metrics that could not be computed, and why. Never silently null.
                "unavailable": unavailable,
            }
        )

    async def _calculate_var(self, task: AgentTask) -> AgentResult:
        """Calculate Value at Risk by historical simulation."""
        portfolio_id = self._parse_portfolio_id(task.payload.get("portfolio_id"))
        significance = self._to_significance(task.payload.get("confidence_level", 0.95))
        window = task.payload.get("window", self.DEFAULT_WINDOW)
        var_window = task.payload.get("var_window")

        async with self._analytics() as service:
            var_result = self._unwrap(
                "var",
                await service.calculate_var(
                    portfolio_id,
                    confidence_level=significance,
                    window=window,
                    var_window=var_window,
                ),
            )
            cvar_result = self._unwrap(
                "cvar",
                await service.calculate_cvar(
                    portfolio_id,
                    confidence_level=significance,
                    window=window,
                    var_window=var_window,
                ),
            )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "portfolio_id": portfolio_id,
                "analysis_type": "var_calculation",
                "var_results": {
                    "method": "historical_simulation",
                    "var_amount": var_result.get("var_amount"),
                    "var_pct": var_result.get("var_pct"),
                    "equity_var": var_result.get("equity_var"),
                    "equity_var_pct": var_result.get("equity_var_pct"),
                    "cvar_amount": cvar_result.get("cvar_amount"),
                    "cvar_pct": cvar_result.get("cvar_pct"),
                    "cvar_var_ratio": cvar_result.get("cvar_var_ratio"),
                    "confidence_level": 1.0 - significance,
                    "significance_level": significance,
                    "window": var_result.get("window"),
                    "var_window": var_result.get("var_window"),
                    "sample_size": var_result.get("sample_size"),
                    "portfolio_value": var_result.get("abs_portfolio_value"),
                    "equity": var_result.get("equity"),
                },
            }
        )

    async def _run_stress_test(self, task: AgentTask) -> AgentResult:
        """Run scenario stress tests.

        Model: a single-factor shock. Each scenario's market move is translated to
        the portfolio through its measured beta to the benchmark, applied to the
        real net exposure:

            P&L = net_exposure * beta * shock

        This is a first-order approximation, not a full revaluation - it assumes a
        constant beta and ignores convexity and idiosyncratic moves. The response
        says so in ``model`` so the number is not mistaken for a repricing.
        """
        portfolio_id = self._parse_portfolio_id(task.payload.get("portfolio_id"))
        window = task.payload.get("window", self.DEFAULT_WINDOW)
        benchmark = task.payload.get("benchmark", self.DEFAULT_BENCHMARK)
        scenarios = task.payload.get("scenarios") or self._default_scenarios()

        async with self._analytics() as service:
            exposure = self._unwrap(
                "portfolio exposure",
                await service.calculate_var(portfolio_id, confidence_level=0.05, window=window),
            )
            beta_result = self._unwrap(
                "beta",
                await service.calculate_beta(portfolio_id, benchmark, window),
            )

        net_exposure = exposure.get("net_portfolio_value") or 0.0
        equity = exposure.get("equity") or 0.0
        beta = beta_result.get("beta")
        if beta is None:
            raise RuntimeError("Beta is required for a stress test but was not returned")

        results = []
        for scenario in scenarios:
            shock = scenario.get("shock")
            if shock is None:
                raise ValueError(f"Scenario {scenario.get('name')!r} is missing a 'shock' value")
            pnl = net_exposure * beta * float(shock)
            results.append({
                "scenario": scenario.get("name"),
                "description": scenario.get("description", ""),
                "market_shock_pct": float(shock) * 100,
                "portfolio_pnl": pnl,
                "portfolio_impact_pct": (pnl / net_exposure * 100) if net_exposure else None,
                "equity_impact_pct": (pnl / equity * 100) if equity else None,
            })

        results.sort(key=lambda r: r["portfolio_pnl"])

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "portfolio_id": portfolio_id,
                "analysis_type": "stress_test",
                "model": "single_factor_beta_shock",
                "benchmark": benchmark,
                "beta": beta,
                "net_exposure": net_exposure,
                "equity": equity,
                "scenarios": results,
                "worst_case": results[0] if results else None,
            }
        )

    async def _calculate_correlations(self, task: AgentTask) -> AgentResult:
        """Calculate the holdings correlation matrix from real price history.

        Returns the same shape as ``/portfolios/{id}/analytics/correlation``:
        ``symbols`` plus a row-major ``correlation_matrix``.
        """
        portfolio_id = self._parse_portfolio_id(task.payload.get("portfolio_id"))
        window = task.payload.get("window", self.DEFAULT_WINDOW)

        async with self._analytics() as service:
            corr = self._unwrap(
                "correlation matrix",
                await service.calculate_correlation_matrix(portfolio_id, window=window),
            )

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "portfolio_id": portfolio_id,
                "analysis_type": "correlation",
                "symbols": corr.get("symbols", []),
                "correlation_matrix": corr.get("correlation_matrix", []),
                "window": corr.get("window", window),
                "start_date": corr.get("start_date"),
                "end_date": corr.get("end_date"),
            }
        )

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def _assess_risk_level(self, metrics: Dict) -> str:
        """Bucket the portfolio by 95% VaR as a share of exposure.

        A relative threshold rather than an absolute cash amount, so the rating
        means the same thing for a 100k portfolio and a 100m one.
        """
        var_pct = metrics.get("var_95_pct")
        if var_pct is None:
            return "UNKNOWN"
        var_pct = abs(var_pct)
        if var_pct >= self.HIGH_RISK_VAR_PCT:
            return "HIGH"
        if var_pct >= self.MEDIUM_RISK_VAR_PCT:
            return "MEDIUM"
        return "LOW"

    def _default_scenarios(self) -> List[Dict]:
        """Default stress test scenarios, expressed as benchmark moves."""
        return [
            {"name": "Market Crash", "description": "20% market decline", "shock": -0.2},
            {"name": "Rate Hike", "description": "100bp rate increase", "shock": -0.05},
            {"name": "Recession", "description": "Economic recession", "shock": -0.15},
            {"name": "Tech Selloff", "description": "Tech sector -30%", "shock": -0.1},
            {"name": "Black Swan", "description": "Extreme tail event", "shock": -0.35},
        ]
