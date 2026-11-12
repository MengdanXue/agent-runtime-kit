# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MarketDataTool - daily bars from the product's own EOD store.

Source of truth is the Java backend's ``GET /market/eod/{symbol}``, which reads
the ``eod_data`` table - the same data the /market/:symbol charts use. Nothing
is fetched from third parties and nothing is filled in: a symbol without rows is
``NOT_FOUND``, an unreachable backend is ``UNAVAILABLE``, a slow one ``TIMEOUT``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import httpx
import pandas as pd
from pydantic import BaseModel, Field

from .base import Provenance, Tool, ToolContext, ToolErrorCode, ToolResult


class Bar(BaseModel):
    date: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None


def bars_to_frame(bars: List[Dict[str, Any]], symbol: str = "") -> pd.DataFrame:
    """OHLCV frame for strategy/factor code. Missing OHLV fall back to close
    only where the strategy code needs a value; the substitution is reported by
    the caller through ``bars_with_close_only``."""
    frame = pd.DataFrame(bars)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    for column in ("open", "high", "low"):
        frame[column] = frame[column].fillna(frame["close"])
    frame["volume"] = frame["volume"].fillna(0.0)
    frame.attrs["symbol"] = symbol
    return frame


def bars_with_close_only(bars: List[Dict[str, Any]]) -> int:
    return sum(1 for b in bars if b.get("high") is None or b.get("low") is None)


class MarketDataInput(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    days: int = Field(180, ge=5, le=3650, description="Calendar days of history")


class MarketDataOutput(BaseModel):
    symbol: str
    bars: List[Bar]
    count: int
    first_date: str
    last_date: str
    last_close: float


def _service_token() -> Optional[str]:
    from agentrt.ports.market import service_token  # AGENTRT_EOD_TOKEN, if any
    return service_token()


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class MarketDataTool(Tool):
    name = "market_data"
    description = "Daily OHLCV bars for one symbol from the configured EOD HTTP service."
    Input = MarketDataInput
    Output = MarketDataOutput
    timeout_seconds = 30.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        request_timeout: float = 10.0,
        service_token: Callable[[], Optional[str]] = _service_token,
        today: Callable[[], date] = date.today,
    ):
        if base_url is None:
            from agentrt.ports.market import base_url as default_base_url  # AGENTRT_EOD_BASE_URL
            base_url = default_base_url()
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = request_timeout
        self._service_token = service_token
        self._today = today

    async def run(self, params: MarketDataInput, ctx: ToolContext) -> ToolResult:
        symbol = params.symbol.strip().upper()
        end = self._today()
        start = end - timedelta(days=params.days)
        path = f"/market/eod/{quote(symbol, safe='')}"
        headers = {}
        if ctx.authorization:
            headers["Authorization"] = ctx.authorization
        else:
            token = self._service_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout,
                                         transport=self._transport) as client:
                response = await client.get(
                    path, params={"startDate": start.isoformat(), "endDate": end.isoformat()},
                    headers=headers)
        except httpx.TimeoutException as exc:
            return ToolResult.failure(ToolErrorCode.TIMEOUT, f"EOD backend timed out: {exc!r}")
        except httpx.TransportError as exc:
            return ToolResult.failure(ToolErrorCode.UNAVAILABLE, f"EOD backend unreachable: {exc!r}")

        if response.status_code == 404:
            return ToolResult.failure(ToolErrorCode.NOT_FOUND, f"No EOD data for {symbol}")
        if response.status_code in (401, 403):
            return ToolResult.failure(
                ToolErrorCode.UNAVAILABLE,
                f"EOD backend rejected the credentials (HTTP {response.status_code})")
        if response.status_code >= 400:
            return ToolResult.failure(
                ToolErrorCode.UNAVAILABLE, f"EOD backend answered HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError:
            return ToolResult.failure(ToolErrorCode.UNAVAILABLE, "EOD backend returned non-JSON")
        if not isinstance(body, dict) or body.get("code") not in (200, None):
            return ToolResult.failure(
                ToolErrorCode.UNAVAILABLE, f"EOD backend reported an error: {str(body)[:200]}")

        rows = body.get("data") or []
        bars: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            close = _num(row.get("close", row.get("closePrice")))
            day = row.get("date") or row.get("tradeDate")
            if close is None or not day:
                continue  # a row without a close is not a bar; it is dropped, not guessed
            bars.append({
                "date": str(day)[:10],
                "open": _num(row.get("open", row.get("openPrice"))),
                "high": _num(row.get("high", row.get("highPrice"))),
                "low": _num(row.get("low", row.get("lowPrice"))),
                "close": close,
                "volume": _num(row.get("volume")),
            })
        if not bars:
            return ToolResult.failure(
                ToolErrorCode.NOT_FOUND,
                f"No EOD rows for {symbol} between {start.isoformat()} and {end.isoformat()}")

        bars.sort(key=lambda b: b["date"])
        output = {
            "symbol": symbol,
            "bars": bars,
            "count": len(bars),
            "first_date": bars[0]["date"],
            "last_date": bars[-1]["date"],
            "last_close": bars[-1]["close"],
        }
        provenance = Provenance(
            source=f"java:{self.base_url}{path}",
            kind="database",
            detail={
                "table": "eod_data",
                "requested_range": [start.isoformat(), end.isoformat()],
                "rows_received": len(rows),
                "rows_used": len(bars),
                "rows_dropped_without_close": len(rows) - len(bars),
            },
        )
        return ToolResult.success(output, provenance)
