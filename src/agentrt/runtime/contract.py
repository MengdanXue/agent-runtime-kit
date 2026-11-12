# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agent Runtime Contract.

Every agent invocation - a runtime loop run or a legacy ``/dispatch`` call -
produces one :class:`AgentRun`. It is the audit record that answers:

* what the agent called            -> ``tool_calls``
* what data it used                -> ``data_provenance`` (+ each call's provenance)
* why it reached its result        -> ``trace`` (plan / observation / decision events)
* where it failed                  -> ``status``, ``stop_reason``, the failing call

Serialized shape (``AgentRun.to_dict``)::

    {
      "task_id": "", "agent": "", "input": {},
      "tool_calls": [], "observations": [], "result": {},
      "status": "RUNNING|SUCCESS|FAILED",
      "data_provenance": {}, "trace": [],
      "action_level": "...", "stop_reason": "...", "started_at": "...", "finished_at": "..."
    }
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus(str, Enum):
    RUNNING = "RUNNING"   # started in the background; the contract grows until it finishes
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ActionLevel(str, Enum):
    """How far an action reaches beyond reading data.

    ANALYSIS  - read and compute; runs automatically.
    PROPOSE   - produce a recommendation or plan; no side effects.
    SIMULATE  - run a simulation (backtest, paper preview); no side effects.
    EXECUTE   - act on an account; always requires a human approval.
    """

    ANALYSIS = "ANALYSIS"
    PROPOSE = "PROPOSE"
    SIMULATE = "SIMULATE"
    EXECUTE = "EXECUTE"

    @property
    def rank(self) -> int:
        return _LEVEL_RANK[self]


_LEVEL_RANK = {
    ActionLevel.ANALYSIS: 0,
    ActionLevel.PROPOSE: 1,
    ActionLevel.SIMULATE: 2,
    ActionLevel.EXECUTE: 3,
}


@dataclass
class ToolCall:
    """One tool invocation, whatever its outcome."""

    call_id: str
    tool: str
    action_level: str
    input: Dict[str, Any]
    status: str  # OK | ERROR | BLOCKED
    attempt: int = 1
    error: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
    started_at: str = field(default_factory=utcnow_iso)
    duration_ms: int = 0
    agent: Optional[str] = None  # which workflow agent made the call (None: single-agent loop)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "agent": self.agent,
            "tool": self.tool,
            "action_level": self.action_level,
            "input": self.input,
            "status": self.status,
            "attempt": self.attempt,
            "error": self.error,
            "output_summary": self.output_summary,
            "provenance": self.provenance,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class Observation:
    """What the agent learned from one tool call; the planner reads these."""

    call_id: str
    tool: str
    ok: bool
    summary: Dict[str, Any]
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
            "error_code": self.error_code,
        }


@dataclass
class TraceEvent:
    seq: int
    kind: str  # goal | plan | tool_call | observation | decision | approval | final
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "message": self.message,
                "data": self.data, "at": self.at}


@dataclass
class AgentRun:
    agent: str
    input: Dict[str, Any]
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner_user_id: Optional[int] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)
    status: RunStatus = RunStatus.FAILED
    data_provenance: Dict[str, Any] = field(default_factory=dict)
    trace: List[TraceEvent] = field(default_factory=list)
    action_level: ActionLevel = ActionLevel.ANALYSIS
    stop_reason: Optional[str] = None
    error_code: Optional[str] = None
    started_at: str = field(default_factory=utcnow_iso)
    finished_at: Optional[str] = None
    persisted: Optional[bool] = None  # None: not archived by this code path

    def event(self, kind: str, message: str, **data: Any) -> TraceEvent:
        ev = TraceEvent(seq=len(self.trace) + 1, kind=kind, message=message, data=data)
        self.trace.append(ev)
        return ev

    def raise_level(self, level: ActionLevel) -> None:
        if level.rank > self.action_level.rank:
            self.action_level = level

    def finish(self, status: RunStatus, result: Dict[str, Any], stop_reason: str,
               error_code: Optional[str] = None) -> "AgentRun":
        self.status = status
        self.result = result
        self.stop_reason = stop_reason
        self.error_code = error_code
        self.finished_at = utcnow_iso()
        self.event("final", stop_reason, status=status.value, error_code=error_code)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "input": self.input,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "observations": [o.to_dict() for o in self.observations],
            "result": self.result,
            "status": self.status.value,
            "data_provenance": self.data_provenance,
            "trace": [e.to_dict() for e in self.trace],
            "action_level": self.action_level.value,
            "stop_reason": self.stop_reason,
            "error_code": self.error_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "persisted": self.persisted,
        }
