# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ExecutionTool - PAPER ONLY.

There is no exchange client here and no live mode to switch on: the tool writes a
paper order ticket filled at the reference price the caller observed through
MarketDataTool. No randomness, no network. Its action level is EXECUTE, so the
runtime refuses to call it unless a human approved this exact order; the tool
checks the approval itself as well, so it cannot be reached around the gate.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from ..runtime.contract import ActionLevel
from .base import Provenance, Tool, ToolContext, ToolErrorCode, ToolResult

PAPER_ONLY = True


class ExecutionInput(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0, le=1_000_000)
    reference_price: float = Field(..., gt=0)
    reference_date: str


class ExecutionOutput(BaseModel):
    mode: Literal["PAPER"]
    order_id: str
    status: Literal["PAPER_FILLED"]
    symbol: str
    side: str
    quantity: float
    fill_price: float
    notional: float
    approved_by: int
    proposal_id: str


def order_fingerprint(params: Dict[str, Any]) -> Dict[str, Any]:
    """The fields an approval is bound to."""
    return {k: params[k] for k in ("symbol", "side", "quantity", "reference_price", "reference_date")}


class ExecutionTool(Tool):
    name = "execution"
    description = "PAPER ONLY order ticket at the observed reference price; requires human approval."
    action_level = ActionLevel.EXECUTE
    Input = ExecutionInput
    Output = ExecutionOutput

    async def run(self, params: ExecutionInput, ctx: ToolContext) -> ToolResult:
        approval = ctx.approval or {}
        requested = order_fingerprint(params.model_dump())
        if not approval.get("approved_by") or approval.get("order") != requested:
            return ToolResult.failure(
                ToolErrorCode.APPROVAL_REQUIRED,
                "EXECUTE requires a human approval bound to this exact order")
        output = {
            "mode": "PAPER",
            "order_id": f"paper-{uuid.uuid4()}",
            "status": "PAPER_FILLED",
            "symbol": params.symbol,
            "side": params.side,
            "quantity": params.quantity,
            "fill_price": params.reference_price,
            "notional": round(params.quantity * params.reference_price, 6),
            "approved_by": approval["approved_by"],
            "proposal_id": approval["proposal_id"],
        }
        return ToolResult.success(output, Provenance(
            source="paper-ledger (no exchange connection)", kind="paper_simulation",
            detail={"fill_rule": "fill at reference close observed by market_data",
                    "reference_date": params.reference_date},
            derived_from=list(ctx.input_call_ids)))
