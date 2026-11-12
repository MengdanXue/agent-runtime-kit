# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent tool layer: Agent -> Tool -> Service/Data."""

from .analysis import BacktestTool, FactorTool, SignalTool
from .base import (
    Provenance,
    Tool,
    ToolContext,
    ToolErrorCode,
    ToolRegistry,
    ToolResult,
)
from .execution import ExecutionTool
from .market_data import MarketDataTool
from .portfolio import PortfolioTool, RiskTool


def default_tools() -> ToolRegistry:
    return ToolRegistry([
        MarketDataTool(),
        PortfolioTool(),
        RiskTool(),
        SignalTool(),
        BacktestTool(),
        FactorTool(),
        ExecutionTool(),
    ])


__all__ = [
    "BacktestTool", "ExecutionTool", "FactorTool", "MarketDataTool", "PortfolioTool",
    "Provenance", "RiskTool", "SignalTool", "Tool", "ToolContext", "ToolErrorCode",
    "ToolRegistry", "ToolResult", "default_tools",
]
