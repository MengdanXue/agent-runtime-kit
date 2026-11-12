# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tools that compute on bars returned by MarketDataTool.

They never fetch data themselves: the bars come in as input, so every number
they produce traces back to a MarketDataTool call (``derived_from``). All
computation is delegated to existing code - the strategy registry in
``src.strategies`` and the factor definitions in ``FactorAgent`` - so these
tools add no new trading logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..runtime.contract import ActionLevel
from .base import Provenance, Tool, ToolContext, ToolErrorCode, ToolResult
from .market_data import Bar, bars_to_frame, bars_with_close_only

SIGNAL_LABELS = {2: "STRONG_BUY", 1: "BUY", 0: "HOLD", -1: "SELL", -2: "STRONG_SELL"}
_LOOKBACK_KEYS = ("period", "lookback", "window")


def required_bars(params: Dict[str, Any]) -> int:
    """Bars a strategy needs before its latest signal means anything.

    Strategies fill their warm-up rows with 0 (= HOLD); reading such a row as a
    signal would report "hold" for what is really "not enough history". The
    longest period/lookback/window parameter bounds the warm-up.
    """
    lengths = [int(v) for k, v in params.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)
               and any(key in k for key in _LOOKBACK_KEYS)]
    return (max(lengths) if lengths else 1) + 1


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _derived(source: str, ctx: ToolContext, **detail: Any) -> Provenance:
    return Provenance(source=source, kind="derived", detail=detail,
                      derived_from=list(ctx.input_call_ids))


class BarsInput(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    bars: List[Bar] = Field(..., min_length=2)


# --------------------------------------------------------------------------- signals

class SignalInput(BarsInput):
    strategies: List[str] = Field(
        default_factory=lambda: ["momentum", "rsi_reversion", "bollinger_reversion", "macd", "ma_cross"],
        min_length=1, max_length=10)


class StrategySignal(BaseModel):
    strategy: str
    signal: int
    label: str
    bars_used: int


class SignalOutput(BaseModel):
    symbol: str
    as_of: str
    signals: List[StrategySignal]
    unavailable: Dict[str, str]
    consensus_score: float = Field(..., description="Mean of available signals, -2..2")
    consensus_label: str


class SignalTool(Tool):
    name = "signal"
    description = "Latest signal of registered strategies (strategy port) computed on the given bars."
    Input = SignalInput
    Output = SignalOutput

    async def run(self, params: SignalInput, ctx: ToolContext) -> ToolResult:
        from agentrt.ports.strategies import get_strategy

        bars = [b.model_dump() for b in params.bars]
        frame = bars_to_frame(bars, params.symbol)
        signals, unavailable = [], {}
        for name in params.strategies:
            try:
                strategy = get_strategy(name)
            except ValueError as exc:
                return ToolResult.failure(ToolErrorCode.INVALID_INPUT, str(exc))
            need = required_bars(strategy.params)
            if len(frame) < need:
                unavailable[name] = f"needs {need} bars, got {len(frame)}"
                continue
            try:
                series = strategy.generate_signals(strategy.preprocess_data(frame.copy()))
            except Exception as exc:
                unavailable[name] = f"strategy failed: {type(exc).__name__}: {exc}"
                continue
            last = _finite(series.iloc[-1]) if len(series) else None
            if last is None:
                unavailable[name] = "no signal on the latest bar"
                continue
            value = max(-2, min(2, int(round(last))))
            signals.append({"strategy": name, "signal": value,
                            "label": SIGNAL_LABELS[value], "bars_used": len(frame)})
        if not signals:
            return ToolResult.failure(
                ToolErrorCode.INSUFFICIENT_DATA,
                "No strategy produced a signal: " + "; ".join(f"{k}: {v}" for k, v in unavailable.items()))
        score = sum(s["signal"] for s in signals) / len(signals)
        label = SIGNAL_LABELS[max(-2, min(2, int(round(score))))]
        output = {"symbol": params.symbol, "as_of": bars[-1]["date"], "signals": signals,
                  "unavailable": unavailable, "consensus_score": round(score, 4),
                  "consensus_label": label}
        return ToolResult.success(output, _derived(
            "src.strategies.generate_signals", ctx, strategies=params.strategies,
            bars=len(bars), bars_with_close_only=bars_with_close_only(bars)))


# -------------------------------------------------------------------------- backtest

class BacktestInput(BarsInput):
    strategy: str
    params: Dict[str, Any] = Field(default_factory=dict)
    initial_capital: float = Field(100_000, gt=0)
    commission: float = Field(0.001, ge=0, le=0.05)
    slippage: float = Field(0.0005, ge=0, le=0.05)


class BacktestOutput(BaseModel):
    symbol: str
    strategy: str
    period: List[str]
    bars_used: int
    trade_count: int
    final_equity: Optional[float]
    metrics: Dict[str, Optional[float]]
    unavailable_metrics: List[str]


class BacktestTool(Tool):
    name = "backtest"
    description = "Simulate a registered strategy (BaseStrategy.run) over the given bars."
    action_level = ActionLevel.SIMULATE
    Input = BacktestInput
    Output = BacktestOutput

    async def run(self, params: BacktestInput, ctx: ToolContext) -> ToolResult:
        from agentrt.ports.strategies import get_strategy

        try:
            strategy = get_strategy(params.strategy, params.params or None)
        except ValueError as exc:
            return ToolResult.failure(ToolErrorCode.INVALID_INPUT, str(exc))
        bars = [b.model_dump() for b in params.bars]
        need = required_bars(strategy.params)
        if len(bars) < need:
            return ToolResult.failure(
                ToolErrorCode.INSUFFICIENT_DATA,
                f"{params.strategy} needs {need} bars, got {len(bars)}")
        frame = bars_to_frame(bars, params.symbol)
        result = strategy.run(frame, initial_capital=params.initial_capital,
                              commission=params.commission, slippage=params.slippage)
        metrics, unavailable = {}, []
        for key, value in (result.metrics or {}).items():
            number = _finite(value)
            metrics[key] = number
            if number is None:
                unavailable.append(key)
        equity = _finite(result.equity_curve.iloc[-1]) if len(result.equity_curve) else None
        output = {"symbol": params.symbol, "strategy": params.strategy,
                  "period": [bars[0]["date"], bars[-1]["date"]], "bars_used": len(bars),
                  "trade_count": len(result.trades), "final_equity": equity,
                  "metrics": metrics, "unavailable_metrics": unavailable}
        return ToolResult.success(output, _derived(
            "src.strategies.BaseStrategy.run", ctx, strategy=params.strategy,
            params=strategy.params, commission=params.commission, slippage=params.slippage))


# ---------------------------------------------------------------------------- factors

class FactorInput(BarsInput):
    factors: List[str] = Field(
        default_factory=lambda: ["momentum_20d", "volatility_20d", "rsi_14d", "volume_ratio"],
        min_length=1, max_length=10)


class FactorOutput(BaseModel):
    symbol: str
    as_of: str
    values: Dict[str, float]
    unavailable: Dict[str, str]


class FactorTool(Tool):
    name = "factor"
    description = "Latest factor exposures (FactorAgent definitions) computed on the given bars."
    Input = FactorInput
    Output = FactorOutput

    async def run(self, params: FactorInput, ctx: ToolContext) -> ToolResult:
        from agentrt.quant.factor_agent import FactorAgent

        definitions = FactorAgent.FACTOR_DEFINITIONS
        unknown = [f for f in params.factors if f not in definitions]
        if unknown:
            return ToolResult.failure(
                ToolErrorCode.INVALID_INPUT,
                f"Unknown factors {unknown}; available: {sorted(definitions)}")
        bars = [b.model_dump() for b in params.bars]
        frame = bars_to_frame(bars, params.symbol)
        calculator = FactorAgent(agent_id="factor-tool")
        values, unavailable = {}, {}
        for name in params.factors:
            if name == "volume_ratio" and all(b.get("volume") is None for b in bars):
                unavailable[name] = "no volume in the bars"
                continue
            column = calculator._calculate_factor({params.symbol: frame}, name)[params.symbol]
            latest = _finite(column.iloc[-1])
            if latest is None:
                unavailable[name] = f"not defined on the latest bar ({len(bars)} bars)"
            else:
                values[name] = latest
        if not values:
            return ToolResult.failure(
                ToolErrorCode.INSUFFICIENT_DATA,
                "No factor could be computed: " + "; ".join(f"{k}: {v}" for k, v in unavailable.items()))
        return ToolResult.success(
            {"symbol": params.symbol, "as_of": bars[-1]["date"], "values": values,
             "unavailable": unavailable},
            _derived("agentrt.quant.factor_agent.FactorAgent._calculate_factor", ctx,
                     factors=params.factors, bars=len(bars)))
