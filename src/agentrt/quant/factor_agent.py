# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Factor Agent - Factor analysis and multi-factor model construction.

This agent provides comprehensive factor analytics including:
- Factor performance analysis (IC, IR, turnover)
- Factor attribution and decomposition
- Multi-factor model construction
- Factor correlation and clustering analysis
"""

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats

from ..base import BaseAgent, AgentTask, AgentResult, AgentStatus
from ..registry import register_agent
from .mock_guard import require_real_data

logger = logging.getLogger(__name__)


@register_agent("factor", task_types=["performance_analysis", "attribution", "multi_factor_model", "correlation_analysis", "factor_decay"])
class FactorAgent(BaseAgent):
    """
    Agent for factor analysis and multi-factor model construction.

    Supports factor performance analysis, attribution, correlation analysis,
    and multi-factor model building.

    Task Payload:
        type: str - Analysis type
        factor: str - Factor name (for single factor analysis)
        factors: List[str] - Factor names (for multi-factor)
        universe: List[str] - Stock universe
        start_date: str - Analysis start date
        end_date: str - Analysis end date
        holding_period: int - Holding period in days
    """

    SUPPORTED_TASKS = {
        "performance_analysis",
        "attribution",
        "multi_factor_model",
        "correlation_analysis",
        "factor_decay"
    }

    # Built-in factor definitions
    FACTOR_DEFINITIONS = {
        "momentum_20d": {"type": "momentum", "lookback": 20},
        "momentum_60d": {"type": "momentum", "lookback": 60},
        "momentum_120d": {"type": "momentum", "lookback": 120},
        "volatility_20d": {"type": "volatility", "lookback": 20},
        "volatility_60d": {"type": "volatility", "lookback": 60},
        "volume_ratio": {"type": "volume", "lookback": 20},
        "rsi_14d": {"type": "rsi", "lookback": 14},
        "ma_deviation": {"type": "ma_deviation", "fast": 10, "slow": 50},
    }

    def __init__(self, agent_id: Optional[str] = None):
        super().__init__(agent_id)
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._factor_cache: Dict[str, pd.DataFrame] = {}

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute factor analysis task."""
        self.status = AgentStatus.RUNNING

        try:
            analysis_type = task.payload.get("type", "performance_analysis")

            if analysis_type == "attribution":
                result = await self._run_attribution(task)
            elif analysis_type == "multi_factor_model":
                result = await self._run_multi_factor_model(task)
            elif analysis_type == "correlation_analysis":
                result = await self._run_correlation_analysis(task)
            elif analysis_type == "factor_decay":
                result = await self._run_factor_decay(task)
            else:
                result = await self._run_performance_analysis(task)

            self.status = AgentStatus.COMPLETED
            return result

        except Exception as e:
            logger.error(f"Factor analysis failed: {e}", exc_info=True)
            self.status = AgentStatus.FAILED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=str(e)
            )

    async def _fetch_price_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch price data for multiple symbols."""
        data = {}

        for symbol in symbols:
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self._data_cache:
                data[symbol] = self._data_cache[cache_key]
                continue

            require_real_data(
                f"price history for {symbol}",
                "FactorAgent has no market data source wired up; connect it to the "
                "market data service before using its factor output.",
            )
            df = self._generate_mock_price_data(symbol, start_date, end_date)
            self._data_cache[cache_key] = df
            data[symbol] = df

        return data

    def _generate_mock_price_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Generate mock price data."""
        require_real_data("random draws in _generate_mock_price_data",
                          "synthetic data for local development only (ALLOW_MOCK_AGENT_DATA=true)")
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = pd.Timestamp.now()

        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = end - pd.Timedelta(days=500)

        dates = pd.date_range(start=start, end=end, freq='B')
        n = len(dates)

        np.random.seed(hash(symbol) % (2**32))

        initial_price = 50 + np.random.random() * 150
        returns = np.random.normal(0.0005, 0.02, n)
        prices = initial_price * np.exp(np.cumsum(returns))

        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        volume = np.random.randint(1000000, 10000000, n)

        return pd.DataFrame({
            'open': low + (high - low) * np.random.random(n),
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=dates)

    def _calculate_factor(
        self,
        price_data: Dict[str, pd.DataFrame],
        factor_name: str,
        factor_def: Dict = None
    ) -> pd.DataFrame:
        """Calculate factor values for all symbols."""
        if factor_def is None:
            factor_def = self.FACTOR_DEFINITIONS.get(factor_name, {"type": "momentum", "lookback": 20})

        factor_type = factor_def.get("type", "momentum")
        lookback = factor_def.get("lookback", 20)

        factor_values = {}

        for symbol, df in price_data.items():
            if factor_type == "momentum":
                # Price momentum
                factor_values[symbol] = df['close'].pct_change(lookback)
            elif factor_type == "volatility":
                # Rolling volatility
                factor_values[symbol] = df['close'].pct_change().rolling(lookback).std() * np.sqrt(252)
            elif factor_type == "volume":
                # Volume ratio
                factor_values[symbol] = df['volume'] / df['volume'].rolling(lookback).mean()
            elif factor_type == "rsi":
                # RSI
                factor_values[symbol] = self._calculate_rsi(df['close'], lookback)
            elif factor_type == "ma_deviation":
                # MA deviation
                fast = factor_def.get("fast", 10)
                slow = factor_def.get("slow", 50)
                ma_fast = df['close'].rolling(fast).mean()
                ma_slow = df['close'].rolling(slow).mean()
                factor_values[symbol] = (ma_fast - ma_slow) / ma_slow
            else:
                # Default: momentum
                factor_values[symbol] = df['close'].pct_change(lookback)

        return pd.DataFrame(factor_values)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_forward_returns(
        self,
        price_data: Dict[str, pd.DataFrame],
        holding_period: int
    ) -> pd.DataFrame:
        """Calculate forward returns for all symbols."""
        forward_returns = {}

        for symbol, df in price_data.items():
            forward_returns[symbol] = df['close'].pct_change(holding_period).shift(-holding_period)

        return pd.DataFrame(forward_returns)

    def _calculate_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate Information Coefficient (rank correlation) time series."""
        ic_series = []
        dates = factor_values.index.intersection(forward_returns.index)

        for date in dates:
            factor_row = factor_values.loc[date].dropna()
            return_row = forward_returns.loc[date].dropna()

            common = factor_row.index.intersection(return_row.index)
            if len(common) < 5:
                continue

            # Spearman rank correlation
            corr, _ = stats.spearmanr(factor_row[common], return_row[common])
            if not np.isnan(corr):
                ic_series.append({'date': date, 'ic': corr})

        return pd.DataFrame(ic_series).set_index('date')['ic'] if ic_series else pd.Series()

    async def _run_performance_analysis(self, task: AgentTask) -> AgentResult:
        """Run single factor performance analysis."""
        factor_name = task.payload.get("factor", "momentum_20d")
        universe = task.payload.get("universe", ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
        start_date = task.payload.get("start_date")
        end_date = task.payload.get("end_date")
        holding_period = task.payload.get("holding_period", 20)

        # Fetch data
        price_data = await self._fetch_price_data(universe, start_date, end_date)

        # Calculate factor and forward returns
        factor_def = self.FACTOR_DEFINITIONS.get(factor_name)
        factor_values = self._calculate_factor(price_data, factor_name, factor_def)
        forward_returns = self._calculate_forward_returns(price_data, holding_period)

        # Calculate IC series
        ic_series = self._calculate_ic(factor_values, forward_returns)

        if len(ic_series) == 0:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error="Insufficient data for IC calculation"
            )

        # Calculate metrics
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_hit_rate = float((ic_series > 0).mean())

        # T-statistic
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else 0

        # Quintile analysis
        quintile_returns = self._calculate_quintile_returns(factor_values, forward_returns)

        # Turnover
        turnover = self._calculate_turnover(factor_values)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "type": "performance_analysis",
                "factor": factor_name,
                "period": f"{start_date} to {end_date}",
                "holding_period": holding_period,
                "universe_size": len(universe),
                "metrics": {
                    "ic_mean": round(ic_mean, 4),
                    "ic_std": round(ic_std, 4),
                    "ic_ir": round(ic_ir, 4),
                    "ic_hit_rate": round(ic_hit_rate, 4),
                    "t_stat": round(t_stat, 4),
                    "turnover": round(turnover, 4),
                    "quintile_spread": round(quintile_returns.get("Q5", 0) - quintile_returns.get("Q1", 0), 4)
                },
                "quintile_returns": quintile_returns,
                "ic_series": [{"date": str(d), "ic": round(v, 4)} for d, v in ic_series.tail(50).items()],
                "assessment": self._assess_factor_quality(ic_mean, ic_ir, t_stat)
            }
        )

    def _calculate_quintile_returns(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate average returns by factor quintile."""
        quintile_returns = {f"Q{i}": [] for i in range(1, 6)}

        dates = factor_values.index.intersection(forward_returns.index)

        for date in dates:
            factor_row = factor_values.loc[date].dropna()
            return_row = forward_returns.loc[date].dropna()

            common = factor_row.index.intersection(return_row.index)
            if len(common) < 5:
                continue

            # Rank into quintiles
            ranks = factor_row[common].rank(pct=True)
            for symbol in common:
                rank = ranks[symbol]
                ret = return_row[symbol]
                if not np.isnan(ret):
                    quintile = min(5, int(rank * 5) + 1)
                    quintile_returns[f"Q{quintile}"].append(ret)

        return {
            k: round(np.mean(v) * 100, 4) if v else 0
            for k, v in quintile_returns.items()
        }

    def _calculate_turnover(self, factor_values: pd.DataFrame) -> float:
        """Calculate average factor turnover."""
        if len(factor_values) < 2:
            return 0

        turnovers = []
        for i in range(1, len(factor_values)):
            prev = factor_values.iloc[i-1].rank(pct=True).dropna()
            curr = factor_values.iloc[i].rank(pct=True).dropna()

            common = prev.index.intersection(curr.index)
            if len(common) > 0:
                turnover = np.abs(curr[common] - prev[common]).mean()
                turnovers.append(turnover)

        return float(np.mean(turnovers)) if turnovers else 0

    def _assess_factor_quality(self, ic_mean: float, ic_ir: float, t_stat: float) -> Dict[str, Any]:
        """Assess factor quality based on metrics."""
        quality_score = 0
        comments = []

        # IC assessment
        if abs(ic_mean) >= 0.05:
            quality_score += 3
            comments.append("Strong IC mean")
        elif abs(ic_mean) >= 0.03:
            quality_score += 2
            comments.append("Good IC mean")
        elif abs(ic_mean) >= 0.01:
            quality_score += 1
            comments.append("Weak IC mean")
        else:
            comments.append("Very weak IC mean")

        # IR assessment
        if ic_ir >= 0.5:
            quality_score += 3
            comments.append("Excellent IC stability")
        elif ic_ir >= 0.3:
            quality_score += 2
            comments.append("Good IC stability")
        elif ic_ir >= 0.1:
            quality_score += 1
            comments.append("Moderate IC stability")
        else:
            comments.append("Poor IC stability")

        # Statistical significance
        if abs(t_stat) >= 3:
            quality_score += 2
            comments.append("Highly significant")
        elif abs(t_stat) >= 2:
            quality_score += 1
            comments.append("Statistically significant")
        else:
            comments.append("Not statistically significant")

        # Overall rating
        if quality_score >= 7:
            rating = "EXCELLENT"
        elif quality_score >= 5:
            rating = "GOOD"
        elif quality_score >= 3:
            rating = "FAIR"
        else:
            rating = "POOR"

        return {
            "rating": rating,
            "score": quality_score,
            "max_score": 8,
            "comments": comments
        }

    async def _run_attribution(self, task: AgentTask) -> AgentResult:
        """Run factor attribution analysis."""
        portfolio_id = task.payload.get("portfolio_id")
        factors = task.payload.get("factors", ["momentum", "value", "quality", "volatility"])
        start_date = task.payload.get("start_date")
        end_date = task.payload.get("end_date")

        # The attribution below runs on a hard-coded sample portfolio and random
        # contributions: nothing in it describes the caller's portfolio.
        require_real_data(
            f"factor attribution for portfolio {portfolio_id}",
            "Factor attribution is not implemented (sample positions, random contributions); "
            "use the Agent Runtime factor tool on real EOD bars instead.",
        )

        # Sample positions (mock mode only)
        positions = {
            "AAPL": 0.2, "GOOGL": 0.15, "MSFT": 0.15,
            "AMZN": 0.1, "META": 0.1, "NVDA": 0.1,
            "TSLA": 0.1, "JPM": 0.1
        }

        universe = list(positions.keys())
        price_data = await self._fetch_price_data(universe, start_date, end_date)

        # Calculate factor exposures and contributions
        factor_exposures = {}
        factor_contributions = {}

        for factor in factors:
            factor_values = self._calculate_factor(price_data, f"{factor}_20d")

            # Calculate weighted exposure
            if len(factor_values) > 0:
                last_values = factor_values.iloc[-1]
                exposure = sum(positions.get(s, 0) * last_values.get(s, 0)
                             for s in positions.keys() if s in last_values.index)
                factor_exposures[factor] = round(exposure, 4)

                # Mock contribution
                require_real_data(
                    f"return contribution of factor '{factor}'",
                    "Factor attribution is not implemented; regress portfolio returns on "
                    "the factor returns to obtain real contributions.",
                )
                factor_contributions[factor] = round(np.random.uniform(-0.02, 0.05), 4)

        # Calculate total return
        total_return = sum(factor_contributions.values())
        specific_return = round(np.random.uniform(-0.02, 0.05), 4)
        factor_contributions["specific"] = specific_return
        total_return += specific_return

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "type": "attribution",
                "portfolio_id": portfolio_id,
                "period": f"{start_date} to {end_date}",
                "total_return": round(total_return, 4),
                "factor_contributions": factor_contributions,
                "factor_exposures": factor_exposures,
                "contribution_breakdown": {
                    "factor_return": round(total_return - specific_return, 4),
                    "specific_return": specific_return,
                    "total_return": round(total_return, 4)
                }
            }
        )

    async def _run_multi_factor_model(self, task: AgentTask) -> AgentResult:
        """Build and evaluate multi-factor model."""
        factors = task.payload.get("factors", {
            "momentum": {"weight": 0.3, "lookback": 20},
            "value": {"weight": 0.25},
            "quality": {"weight": 0.25},
            "volatility": {"weight": 0.2, "lookback": 60}
        })
        universe = task.payload.get("universe", ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM"])
        start_date = task.payload.get("start_date")
        end_date = task.payload.get("end_date")

        price_data = await self._fetch_price_data(universe, start_date, end_date)

        # Calculate composite scores
        composite_scores = pd.DataFrame(index=universe)

        for factor_name, config in factors.items():
            weight = config.get("weight", 0.25)
            lookback = config.get("lookback", 20)

            factor_def = {"type": factor_name, "lookback": lookback}
            factor_values = self._calculate_factor(price_data, factor_name, factor_def)

            if len(factor_values) > 0:
                # Standardize (z-score)
                last_values = factor_values.iloc[-1]
                z_scores = (last_values - last_values.mean()) / last_values.std()
                composite_scores[factor_name] = z_scores * weight

        # Calculate composite
        composite_scores['composite'] = composite_scores.sum(axis=1)
        composite_scores = composite_scores.sort_values('composite', ascending=False)

        # Rankings
        rankings = [
            {"symbol": symbol, "score": round(row['composite'], 4), "rank": i+1}
            for i, (symbol, row) in enumerate(composite_scores.iterrows())
        ]

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "type": "multi_factor_model",
                "factors": list(factors.keys()),
                "weights": {k: v.get("weight", 0.25) for k, v in factors.items()},
                "universe_size": len(universe),
                "rankings": rankings,
                "top_picks": rankings[:3],
                "bottom_picks": rankings[-3:],
                "score_distribution": {
                    "mean": round(composite_scores['composite'].mean(), 4),
                    "std": round(composite_scores['composite'].std(), 4),
                    "min": round(composite_scores['composite'].min(), 4),
                    "max": round(composite_scores['composite'].max(), 4)
                }
            }
        )

    async def _run_correlation_analysis(self, task: AgentTask) -> AgentResult:
        """Analyze factor correlations."""
        factors = task.payload.get("factors", ["momentum_20d", "volatility_20d", "rsi_14d", "ma_deviation"])
        universe = task.payload.get("universe", ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
        start_date = task.payload.get("start_date")
        end_date = task.payload.get("end_date")

        price_data = await self._fetch_price_data(universe, start_date, end_date)

        # Calculate all factors
        factor_dfs = {}
        for factor_name in factors:
            factor_def = self.FACTOR_DEFINITIONS.get(factor_name)
            factor_values = self._calculate_factor(price_data, factor_name, factor_def)
            # Average across symbols for correlation
            factor_dfs[factor_name] = factor_values.mean(axis=1)

        # Create correlation matrix
        combined = pd.DataFrame(factor_dfs).dropna()
        corr_matrix = combined.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for i, f1 in enumerate(factors):
            for f2 in factors[i+1:]:
                corr = corr_matrix.loc[f1, f2]
                if abs(corr) > 0.5:
                    high_corr_pairs.append({
                        "factor1": f1,
                        "factor2": f2,
                        "correlation": round(corr, 4)
                    })

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "type": "correlation_analysis",
                "factors": factors,
                "correlation_matrix": {
                    f1: {f2: round(corr_matrix.loc[f1, f2], 4) for f2 in factors}
                    for f1 in factors
                },
                "high_correlation_pairs": high_corr_pairs,
                "recommendations": self._get_correlation_recommendations(high_corr_pairs)
            }
        )

    def _get_correlation_recommendations(self, high_corr_pairs: List[Dict]) -> List[str]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []

        if not high_corr_pairs:
            recommendations.append("Factor set shows good diversification")
        else:
            for pair in high_corr_pairs:
                if pair["correlation"] > 0.7:
                    recommendations.append(
                        f"Consider removing either {pair['factor1']} or {pair['factor2']} "
                        f"(correlation: {pair['correlation']})"
                    )
                elif pair["correlation"] > 0.5:
                    recommendations.append(
                        f"Monitor correlation between {pair['factor1']} and {pair['factor2']}"
                    )

        recommendations.append("Consider orthogonalization if using correlated factors")
        return recommendations

    async def _run_factor_decay(self, task: AgentTask) -> AgentResult:
        """Analyze factor decay over different holding periods."""
        factor_name = task.payload.get("factor", "momentum_20d")
        universe = task.payload.get("universe", ["AAPL", "GOOGL", "MSFT", "AMZN", "META"])
        start_date = task.payload.get("start_date")
        end_date = task.payload.get("end_date")
        max_holding_period = task.payload.get("max_holding_period", 60)

        price_data = await self._fetch_price_data(universe, start_date, end_date)
        factor_def = self.FACTOR_DEFINITIONS.get(factor_name)
        factor_values = self._calculate_factor(price_data, factor_name, factor_def)

        # Calculate IC for different holding periods
        holding_periods = [1, 5, 10, 20, 40, 60]
        holding_periods = [p for p in holding_periods if p <= max_holding_period]

        decay_results = []
        for period in holding_periods:
            forward_returns = self._calculate_forward_returns(price_data, period)
            ic_series = self._calculate_ic(factor_values, forward_returns)

            if len(ic_series) > 0:
                decay_results.append({
                    "holding_period": period,
                    "ic_mean": round(float(ic_series.mean()), 4),
                    "ic_std": round(float(ic_series.std()), 4),
                    "ic_ir": round(float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0, 4)
                })

        # Estimate half-life
        if len(decay_results) >= 2:
            ics = [r["ic_mean"] for r in decay_results]
            periods = [r["holding_period"] for r in decay_results]
            half_life = self._estimate_half_life(periods, ics)
        else:
            half_life = None

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.COMPLETED,
            result={
                "type": "factor_decay",
                "factor": factor_name,
                "decay_curve": decay_results,
                "half_life_days": half_life,
                "optimal_holding_period": decay_results[np.argmax([r["ic_ir"] for r in decay_results])]["holding_period"] if decay_results else None,
                "recommendations": self._get_decay_recommendations(decay_results, half_life)
            }
        )

    def _estimate_half_life(self, periods: List[int], ics: List[float]) -> Optional[int]:
        """Estimate factor half-life using exponential decay."""
        if len(periods) < 2 or ics[0] == 0:
            return None

        try:
            # Simple exponential decay fit
            initial_ic = abs(ics[0])
            target = initial_ic / 2

            for i, ic in enumerate(ics):
                if abs(ic) <= target:
                    return periods[i]

            return periods[-1]  # Hasn't decayed to half yet
        except Exception:
            return None

    def _get_decay_recommendations(self, decay_results: List[Dict], half_life: Optional[int]) -> List[str]:
        """Generate recommendations based on decay analysis."""
        recommendations = []

        if not decay_results:
            return ["Insufficient data for decay analysis"]

        if half_life:
            if half_life <= 5:
                recommendations.append(f"Factor decays quickly (half-life: {half_life} days) - use short holding periods")
            elif half_life <= 20:
                recommendations.append(f"Factor has moderate persistence (half-life: {half_life} days)")
            else:
                recommendations.append(f"Factor is persistent (half-life: {half_life} days) - suitable for longer holding")

        # Find optimal period
        best_ir_result = max(decay_results, key=lambda x: x["ic_ir"])
        recommendations.append(f"Optimal holding period: {best_ir_result['holding_period']} days (IR: {best_ir_result['ic_ir']})")

        return recommendations
