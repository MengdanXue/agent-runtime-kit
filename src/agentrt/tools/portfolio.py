# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PortfolioTool and RiskTool.

Both enforce portfolio ownership themselves (the same rule as
``verify_portfolio_access``) rather than trusting the caller to have checked.
Portfolio-scope risk is delegated unchanged to the existing RiskAgent
computation (PortfolioAnalyticsService); symbol-scope risk is plain return
statistics over bars that came from MarketDataTool.
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .analysis import _derived, _finite
from .base import Provenance, Tool, ToolContext, ToolErrorCode, ToolResult
from .market_data import Bar


@asynccontextmanager
async def _default_session() -> AsyncIterator[Any]:
    from agentrt.ports.db import open_session  # configured by the host; raises PortNotConfigured otherwise
    async with open_session() as session:
        yield session


async def check_portfolio_access(session: Any, portfolio_id: int,
                                 ctx: ToolContext) -> Optional[ToolResult]:
    """None when the caller owns the portfolio, otherwise the failure to return."""
    if ctx.user_id is None:
        return ToolResult.failure(ToolErrorCode.UNAUTHENTICATED, "No authenticated user")
    from agentrt.ports.portfolio import PortfolioForbidden, PortfolioNotFound, provider

    try:
        await provider().verify_access(portfolio_id, ctx.user_id, session)
    except PortfolioNotFound:
        return ToolResult.failure(ToolErrorCode.NOT_FOUND, f"Portfolio {portfolio_id} does not exist")
    except PortfolioForbidden:
        return ToolResult.failure(ToolErrorCode.FORBIDDEN, f"User may not access portfolio {portfolio_id}")
    return None


# -------------------------------------------------------------------------- portfolio

class PortfolioInput(BaseModel):
    portfolio_id: int = Field(..., gt=0)


class PositionSummary(BaseModel):
    symbol: str
    quantity: Optional[float]
    direction: Optional[str]
    current_value: Optional[float]
    currency: Optional[str]


class PortfolioOutput(BaseModel):
    portfolio_id: int
    name: Optional[str]
    base_currency: Optional[str]
    position_count: int
    positions: List[PositionSummary]


class PortfolioTool(Tool):
    name = "portfolio"
    description = "Positions of a portfolio the caller owns (portfolio provider)."
    Input = PortfolioInput
    Output = PortfolioOutput

    def __init__(self, session_factory: Callable[[], Any] = _default_session):
        self._session_factory = session_factory

    async def run(self, params: PortfolioInput, ctx: ToolContext) -> ToolResult:
        from agentrt.ports.portfolio import provider

        async with self._session_factory() as session:
            denied = await check_portfolio_access(session, params.portfolio_id, ctx)
            if denied:
                return denied
            store = provider()
            info = await store.portfolio_info(params.portfolio_id, session)
            rows = await store.positions(params.portfolio_id, session)
        positions = [{
            "symbol": r.get("symbol"),
            "quantity": _finite(r.get("quantity")),
            "direction": None if r.get("direction") is None else str(r.get("direction")),
            "current_value": _finite(r.get("current_value")),
            "currency": r.get("currency"),
        } for r in rows]
        output = {"portfolio_id": params.portfolio_id, "name": info.get("portfolio_name"),
                  "base_currency": info.get("base_currency"),
                  "position_count": len(positions), "positions": positions}
        return ToolResult.success(output, Provenance(
            source="portfolio-provider:portfolios,positions", kind="database",
            detail={"service": type(store).__name__, "rows": len(rows)}))


# ------------------------------------------------------------------------------- risk

HIGH_VOLATILITY = 0.60      # annualized
MEDIUM_VOLATILITY = 0.30
SEVERE_DRAWDOWN = -0.35
MIN_RETURNS_FOR_RISK = 20


class RiskInput(BaseModel):
    portfolio_id: Optional[int] = Field(None, gt=0)
    symbol: Optional[str] = None
    bars: Optional[List[Bar]] = None
    window: str = Field("1y", pattern=r"^(1m|3m|6m|1y|3y)$")

    @model_validator(mode="after")
    def _one_scope(self) -> "RiskInput":
        if (self.portfolio_id is None) == (self.bars is None):
            raise ValueError("give exactly one of portfolio_id or bars")
        if self.bars is not None and not self.symbol:
            raise ValueError("symbol is required with bars")
        return self


class RiskOutput(BaseModel):
    scope: str                      # portfolio | symbol
    subject: str
    risk_level: str                 # low | medium | high | unknown
    metrics: Dict[str, Optional[float]]
    unavailable: Dict[str, str]
    policy: Dict[str, float]


class RiskTool(Tool):
    name = "risk"
    description = ("Risk of a portfolio the caller owns (RiskAgent / PortfolioAnalyticsService) "
                   "or of one symbol from its bars.")
    Input = RiskInput
    Output = RiskOutput

    def __init__(self, session_factory: Callable[[], Any] = _default_session,
                 risk_agent_factory: Optional[Callable[[], Any]] = None):
        self._session_factory = session_factory
        self._risk_agent_factory = risk_agent_factory

    async def run(self, params: RiskInput, ctx: ToolContext) -> ToolResult:
        if params.portfolio_id is not None:
            return await self._portfolio_risk(params, ctx)
        return self._symbol_risk(params, ctx)

    async def _portfolio_risk(self, params: RiskInput, ctx: ToolContext) -> ToolResult:
        from agentrt.base import AgentStatus, AgentTask
        from agentrt.quant.risk_agent import RiskAgent

        async with self._session_factory() as session:
            denied = await check_portfolio_access(session, params.portfolio_id, ctx)
        if denied:
            return denied
        agent = (self._risk_agent_factory or (lambda: RiskAgent(agent_id="risk-tool")))()
        outcome = await agent._execute_with_tracking(AgentTask(
            agent_type="risk",
            payload={"type": "risk_analysis", "portfolio_id": params.portfolio_id,
                     "window": params.window}))
        if outcome.status != AgentStatus.COMPLETED:
            message = outcome.error or "risk analysis failed"
            code = (ToolErrorCode.INSUFFICIENT_DATA if "No risk metric could be computed" in message
                    else ToolErrorCode.UNAVAILABLE)
            return ToolResult.failure(code, message)
        body = outcome.result or {}
        metrics = {k: _finite(v) for k, v in (body.get("metrics") or {}).items()}
        return ToolResult.success(
            {"scope": "portfolio", "subject": str(params.portfolio_id),
             # RiskAgent reports HIGH/MEDIUM/LOW/UNKNOWN; the tool contract is lower case
             "risk_level": str(body.get("risk_level") or "unknown").lower(), "metrics": metrics,
             "unavailable": {k: str(v) for k, v in (body.get("unavailable") or {}).items()},
             "policy": {"var_high_pct": RiskAgent.HIGH_RISK_VAR_PCT,
                        "var_medium_pct": RiskAgent.MEDIUM_RISK_VAR_PCT}},
            Provenance(source="postgres:positions+eod via PortfolioAnalyticsService",
                       kind="database", detail={"computed_by": "RiskAgent._full_risk_analysis",
                                                "window": params.window}))

    def _symbol_risk(self, params: RiskInput, ctx: ToolContext) -> ToolResult:
        closes = np.array([b.close for b in params.bars], dtype=float)
        returns = np.diff(closes) / closes[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) < MIN_RETURNS_FOR_RISK:
            return ToolResult.failure(
                ToolErrorCode.INSUFFICIENT_DATA,
                f"{params.symbol}: {len(returns)} daily returns, need {MIN_RETURNS_FOR_RISK}")
        volatility = float(np.std(returns, ddof=1) * math.sqrt(252))
        equity = np.cumprod(1 + returns)
        drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1))
        var_95 = float(-np.quantile(returns, 0.05))
        if volatility >= HIGH_VOLATILITY or drawdown <= SEVERE_DRAWDOWN:
            level = "high"
        elif volatility >= MEDIUM_VOLATILITY:
            level = "medium"
        else:
            level = "low"
        return ToolResult.success(
            {"scope": "symbol", "subject": params.symbol, "risk_level": level,
             "metrics": {"annualized_volatility": round(volatility, 6),
                         "max_drawdown": round(drawdown, 6),
                         "historical_var_95_daily": round(var_95, 6),
                         "returns_used": float(len(returns))},
             "unavailable": {},
             "policy": {"high_volatility": HIGH_VOLATILITY, "medium_volatility": MEDIUM_VOLATILITY,
                        "severe_drawdown": SEVERE_DRAWDOWN}},
            _derived("close-to-close returns", ctx, returns=len(returns)))
