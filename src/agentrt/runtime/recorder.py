# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run log and the contract wrapper for legacy quant-agent calls.

Legacy agents (``/dispatch``, ``/quant/*``) do not go through the tool layer.
Wrapping them does not change what they do; it records what happened in the
same contract, and says plainly that their tool calls are not observable and
their provenance is only what the agent itself declares.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .contract import AgentRun, RunStatus

RUN_LOG_CAPACITY = 1000


class RunLog:
    """In-process record of recent runs; readable only by the run's owner."""

    def __init__(self, capacity: int = RUN_LOG_CAPACITY):
        self._capacity = capacity
        self._runs: "OrderedDict[str, AgentRun]" = OrderedDict()
        self._lock = threading.Lock()

    def add(self, run: AgentRun) -> None:
        with self._lock:
            self._runs[run.task_id] = run
            self._runs.move_to_end(run.task_id)
            while len(self._runs) > self._capacity:
                self._runs.popitem(last=False)

    def get(self, task_id: str, user_id: Optional[int]) -> Optional[AgentRun]:
        with self._lock:
            run = self._runs.get(task_id)
        if run is None or user_id is None or run.owner_user_id != user_id:
            return None
        return run

    def recent(self, user_id: Optional[int], limit: int = 20) -> List[AgentRun]:
        if user_id is None:
            return []
        with self._lock:
            owned = [r for r in self._runs.values() if r.owner_user_id == user_id]
        return list(reversed(owned))[:limit]


def _jsonable_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in (payload or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, dict)):
            out[key] = value
        else:
            out[key] = f"<{type(value).__name__}>"
    return out


def legacy_run(task: Any, result: Any, agent_class: Optional[str],
               user_id: Optional[int]) -> AgentRun:
    """Contract for a legacy agent task (AgentTask + AgentResult)."""
    status_value = getattr(getattr(result, "status", None), "value", str(getattr(result, "status", "")))
    succeeded = status_value == "completed"
    body = getattr(result, "result", None)
    run = AgentRun(agent=f"legacy:{agent_class or task.agent_type}",
                   input={"agent_type": task.agent_type, "payload": _jsonable_payload(task.payload)},
                   task_id=task.task_id, owner_user_id=user_id)
    run.event("goal", f"legacy dispatch to {task.agent_type}")
    run.event("plan", "legacy agent - fixed internal logic; tool calls are not observable")
    declared = body.get("data_source") if isinstance(body, dict) else None
    run.data_provenance = {
        "declared_by_agent": declared,
        "tool_layer": False,
        "note": "legacy agent does not use the tool layer; provenance is only what it declares",
    }
    if succeeded:
        run.event("decision", "agent completed")
        return run.finish(RunStatus.SUCCESS, body if isinstance(body, dict) else {"value": body},
                          "legacy agent completed")
    error = getattr(result, "error", None) or "legacy agent failed"
    run.event("decision", f"agent failed: {error}")
    return run.finish(RunStatus.FAILED, {"error": error}, error, f"LEGACY_{status_value.upper() or 'FAILED'}")


class SqlRunArchive:
    """agent_runs in PostgreSQL (Flyway V32): the contract survives a restart.

    Persisting is part of the audit trail, so a failure is reported on the run
    (``persisted = False``) instead of being hidden.
    """

    def __init__(self, session_factory=None):
        self._session_factory = session_factory

    def _factory(self):
        if self._session_factory is not None:
            return self._session_factory
        from .memory import _default_session
        return _default_session

    async def save(self, run: AgentRun) -> bool:
        import json
        import logging
        from sqlalchemy import text

        try:
            async with self._factory()() as session:
                await session.execute(text(
                    "INSERT INTO agent_runs (task_id, user_id, agent, status, action_level, "
                    "stop_reason, contract) VALUES (:task_id, :user_id, :agent, :status, "
                    ":action_level, :stop_reason, CAST(:contract AS JSONB)) "
                    "ON CONFLICT (task_id) DO UPDATE SET status = EXCLUDED.status, "
                    "action_level = EXCLUDED.action_level, stop_reason = EXCLUDED.stop_reason, "
                    "contract = EXCLUDED.contract"),
                    {"task_id": run.task_id, "user_id": run.owner_user_id, "agent": run.agent,
                     "status": run.status.value, "action_level": run.action_level.value,
                     "stop_reason": run.stop_reason,
                     "contract": json.dumps(run.to_dict(), ensure_ascii=False, default=str)})
            run.persisted = True
        except Exception as exc:
            logging.getLogger(__name__).error("agent run %s not persisted: %s", run.task_id, exc)
            run.persisted = False
        return run.persisted

    async def load(self, task_id: str, user_id: Optional[int]) -> Optional[Dict[str, Any]]:
        import json
        from sqlalchemy import text

        if user_id is None:
            return None
        async with self._factory()() as session:
            result = await session.execute(text(
                "SELECT contract FROM agent_runs WHERE task_id = :task_id AND user_id = :user_id"),
                {"task_id": task_id, "user_id": user_id})
            row = result.fetchone()
        if row is None:
            return None
        contract = row[0]
        return json.loads(contract) if isinstance(contract, str) else dict(contract)
