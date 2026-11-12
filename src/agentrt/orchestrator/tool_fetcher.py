# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DataSourceAgent fetcher backed by MarketDataTool.

Registered as the ``traderx_eod`` source so DataSourceAgent serves the
product's own EOD bars instead of random prices. A symbol the tool cannot
serve is left out of the result (with the reason kept in ``errors``); it is
never filled in.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..tools.base import ToolContext
from ..tools.market_data import MarketDataTool
from .data_source_agent import (
    DataFreshness,
    DataSourceConfig,
    DataSourceType,
    MarketData,
)

SOURCE_NAME = "traderx_eod"

SOURCE_CONFIG = DataSourceConfig(
    name=SOURCE_NAME,
    type=DataSourceType.MARKET_DATA,
    freshness=DataFreshness.EOD,
    priority=0,
    timeout=30.0,
)


class ToolBackedFetcher:
    def __init__(self, tool: Optional[MarketDataTool] = None, days: int = 180):
        self.tool = tool or MarketDataTool()
        self.days = days
        self.errors: Dict[str, Dict[str, Any]] = {}

    async def fetch(self, symbols: List[str], data_type: str,
                    context: Dict[str, Any]) -> Dict[str, MarketData]:
        ctx = context.get("tool_context") or ToolContext()
        out: Dict[str, MarketData] = {}
        for symbol in symbols:
            result = await self.tool.invoke({"symbol": symbol, "days": self.days}, ctx)
            if not result.ok:
                self.errors[symbol] = result.error_dict()
                continue
            self.errors.pop(symbol, None)
            bars = result.output["bars"]
            last = bars[-1]
            previous = bars[-2]["close"] if len(bars) > 1 else None
            out[symbol] = MarketData(
                symbol=result.output["symbol"],
                timestamp=datetime.fromisoformat(last["date"]),
                source=SOURCE_NAME,
                open=last["open"], high=last["high"], low=last["low"], close=last["close"],
                volume=last["volume"], last=last["close"],
                change_pct=None if not previous else (last["close"] / previous - 1) * 100,
                extra={"as_of": last["date"], "bars": result.output["count"],
                       "provenance": result.provenance.to_dict()},
            )
        return out
