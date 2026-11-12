# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tool abstraction for the agent runtime.

Agents do not read the database, call internal services or generate data
themselves: they call tools, and tools call services. Every tool

* declares an input schema and an output schema (pydantic models), both enforced
  on every call - a tool whose output drifts from its schema fails the call;
* returns either output *with provenance* or a typed error - there is no third
  "worked, sort of" state and no path that fills in missing data;
* declares an :class:`~agentrt.runtime.contract.ActionLevel`, which the
  runtime's approval gate uses before the call is made.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from ..runtime.contract import ActionLevel, utcnow_iso

logger = logging.getLogger(__name__)


class ToolErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"                    # the requested entity/data does not exist
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"    # data exists but too little to compute
    TIMEOUT = "TIMEOUT"                        # upstream did not answer in time
    UNAVAILABLE = "UNAVAILABLE"                # upstream unreachable or erroring
    INVALID_INPUT = "INVALID_INPUT"            # input failed the schema or a rule
    UNAUTHENTICATED = "UNAUTHENTICATED"        # no caller identity
    FORBIDDEN = "FORBIDDEN"                    # caller may not access the resource
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"    # EXECUTE without a human approval
    INTERNAL = "INTERNAL"                      # bug or schema violation inside the tool


RETRYABLE_ERRORS = frozenset({ToolErrorCode.TIMEOUT, ToolErrorCode.UNAVAILABLE})


@dataclass
class Provenance:
    """Where a tool's output came from."""

    source: str                       # e.g. "java:/api/market/eod/AAPL"
    kind: str                         # database | service | derived | paper_simulation
    detail: Dict[str, Any] = field(default_factory=dict)
    derived_from: List[str] = field(default_factory=list)  # call_ids of inputs
    as_of: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "kind": self.kind, "detail": self.detail,
                "derived_from": self.derived_from, "as_of": self.as_of}


@dataclass
class ToolResult:
    ok: bool
    output: Optional[Dict[str, Any]] = None
    provenance: Optional[Provenance] = None
    error_code: Optional[ToolErrorCode] = None
    error_message: Optional[str] = None

    @property
    def retryable(self) -> bool:
        return self.error_code in RETRYABLE_ERRORS

    @classmethod
    def success(cls, output: Dict[str, Any], provenance: Provenance) -> "ToolResult":
        return cls(ok=True, output=output, provenance=provenance)

    @classmethod
    def failure(cls, code: ToolErrorCode, message: str) -> "ToolResult":
        return cls(ok=False, error_code=code, error_message=message)

    def error_dict(self) -> Optional[Dict[str, Any]]:
        if self.ok:
            return None
        return {"code": self.error_code.value, "message": self.error_message,
                "retryable": self.retryable}


@dataclass
class ToolContext:
    """Who is asking, and on whose behalf a tool may act."""

    user_id: Optional[int] = None
    authorization: Optional[str] = None  # "Bearer ..." forwarded to Java
    task_id: Optional[str] = None
    # Set by the runtime only when a human approved this exact EXECUTE action.
    approval: Optional[Dict[str, Any]] = None
    call_id: Optional[str] = None
    # call_ids whose outputs were passed in as this call's input
    input_call_ids: List[str] = field(default_factory=list)


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    action_level: ClassVar[ActionLevel] = ActionLevel.ANALYSIS
    Input: ClassVar[Type[BaseModel]]
    Output: ClassVar[Type[BaseModel]]
    timeout_seconds: ClassVar[float] = 60.0

    @abstractmethod
    async def run(self, params: BaseModel, ctx: ToolContext) -> ToolResult:
        """Do the work. Return ToolResult.success(...) or ToolResult.failure(...)."""

    async def invoke(self, raw_input: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            params = self.Input.model_validate(raw_input)
        except ValidationError as exc:
            return ToolResult.failure(ToolErrorCode.INVALID_INPUT, _validation_message(exc))

        try:
            result = await asyncio.wait_for(self.run(params, ctx), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return ToolResult.failure(
                ToolErrorCode.TIMEOUT, f"{self.name} did not finish within {self.timeout_seconds}s")
        except Exception as exc:  # a bug inside the tool is a failed call, never a result
            logger.exception("Tool %s raised", self.name)
            return ToolResult.failure(ToolErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")

        if not result.ok:
            return result
        if result.provenance is None:
            return ToolResult.failure(
                ToolErrorCode.INTERNAL, f"{self.name} returned output without provenance")
        try:
            validated = self.Output.model_validate(result.output)
        except ValidationError as exc:
            return ToolResult.failure(
                ToolErrorCode.INTERNAL,
                f"{self.name} output violates its schema: {_validation_message(exc)}")
        result.output = validated.model_dump(mode="json")
        return result

    @classmethod
    def describe(cls) -> Dict[str, Any]:
        return {
            "name": cls.name,
            "description": cls.description,
            "action_level": cls.action_level.value,
            "input_schema": cls.Input.model_json_schema(),
            "output_schema": cls.Output.model_json_schema(),
            "error_codes": [c.value for c in ToolErrorCode],
        }


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


class ToolRegistry:
    def __init__(self, tools: Optional[List[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def describe(self) -> List[Dict[str, Any]]:
        return [tool.describe() for tool in self._tools.values()]
