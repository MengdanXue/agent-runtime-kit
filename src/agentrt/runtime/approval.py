# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human approval boundary.

    ANALYSIS  runs automatically
    PROPOSE   output only (no tool is PROPOSE; it is what a run produces)
    SIMULATE  runs automatically - simulations have no side effects
    EXECUTE   never runs on the agent's own decision

When a run reaches an EXECUTE action, the gate blocks it and stores the exact
action as a :class:`Proposal`. It can only be carried out by a later request
from the same authenticated user that names the proposal id - the approval is
bound to the stored action, so a caller cannot approve one order and submit
another, and the agent has no way to produce an approval itself.

A proposal moves through::

    PENDING --approve (owner, before expiry)--> APPROVED --paper fill--> EXECUTED
                                                         \\-tool error--> FAILED
    PENDING --expiry--> EXPIRED

Every transition is a compare-and-set on the current status, so a proposal is
approved at most once and executed at most once, even under concurrent
requests. :class:`SqlProposalStore` keeps proposals in ``agent_proposals``
(Flyway V34): a pending proposal, its approval state and its link to the run
that produced it (``task_id``) and to the run that executed it survive a
restart. :class:`ProposalStore` is the in-process equivalent (tests, no DB).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .contract import ActionLevel, utcnow_iso

logger = logging.getLogger(__name__)

PROPOSAL_TTL_SECONDS = int(os.getenv("AGENT_PROPOSAL_TTL_SECONDS", str(15 * 60)))
PROPOSAL_STATUSES = ("PENDING", "APPROVED", "EXECUTED", "FAILED", "EXPIRED")


class ApprovalError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code  # NOT_FOUND | FORBIDDEN | EXPIRED | CONSUMED


class ProposalStoreError(Exception):
    """The proposal could not be stored or updated (e.g. the database is down)."""


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


@dataclass
class Proposal:
    proposal_id: str
    owner_user_id: int
    tool: str
    tool_input: Dict[str, Any]
    rationale: str
    evidence: Dict[str, Any]
    source_task_id: str
    input_call_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)
    expires_at_epoch: float = field(default_factory=lambda: time.time() + PROPOSAL_TTL_SECONDS)
    status: str = "PENDING"
    approved_at: Optional[str] = None
    executed_at: Optional[str] = None
    execution_task_id: Optional[str] = None
    execution: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    persisted: bool = False

    def effective_status(self, now: Optional[float] = None) -> str:
        if self.status == "PENDING" and (now or time.time()) > self.expires_at_epoch:
            return "EXPIRED"
        return self.status

    def to_dict(self) -> Dict[str, Any]:
        return {"proposal_id": self.proposal_id, "tool": self.tool, "mode": "PAPER",
                "action": self.tool_input, "rationale": self.rationale, "evidence": self.evidence,
                "source_task_id": self.source_task_id, "created_at": self.created_at,
                "expires_at": _iso(self.expires_at_epoch),
                "expires_in_seconds": max(0, int(self.expires_at_epoch - time.time())),
                "status": self.effective_status(), "approved_at": self.approved_at,
                "executed_at": self.executed_at, "execution_task_id": self.execution_task_id,
                "execution": self.execution, "error": self.error, "persisted": self.persisted,
                "action_level": ActionLevel.EXECUTE.value}


def _refusal(proposal: Optional[Proposal], proposal_id: str, user_id: Optional[int]) -> ApprovalError:
    """Why ``proposal`` cannot be approved by ``user_id`` now."""
    if proposal is None:
        return ApprovalError("NOT_FOUND", f"No proposal {proposal_id}")
    if user_id is None or proposal.owner_user_id != user_id:
        return ApprovalError("FORBIDDEN", "Only the user who received a proposal can approve it")
    status = proposal.effective_status()
    if status == "EXPIRED":
        return ApprovalError("EXPIRED", f"Proposal {proposal_id} has expired")
    return ApprovalError("CONSUMED", f"Proposal {proposal_id} is already {status}")


class ProposalStore:
    """In-process proposals (tests and single-process use; lost on restart)."""

    def __init__(self, ttl_seconds: int = PROPOSAL_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._items: Dict[str, Proposal] = {}
        self._lock = threading.Lock()

    async def create(self, **kwargs: Any) -> Proposal:
        proposal = Proposal(proposal_id=str(uuid.uuid4()),
                            expires_at_epoch=time.time() + self._ttl, **kwargs)
        with self._lock:
            self._items[proposal.proposal_id] = proposal
        return proposal

    async def approve(self, proposal_id: str, user_id: Optional[int]) -> Proposal:
        """Consume a pending proposal on behalf of the human who owns it."""
        with self._lock:
            proposal = self._items.get(proposal_id)
            if (proposal is None or user_id is None or proposal.owner_user_id != user_id
                    or proposal.effective_status() != "PENDING"):
                if proposal is not None and proposal.effective_status() == "EXPIRED":
                    proposal.status = "EXPIRED"
                raise _refusal(proposal, proposal_id, user_id)
            proposal.status = "APPROVED"
            proposal.approved_at = utcnow_iso()
            return proposal

    async def finish(self, proposal_id: str, execution_task_id: str, ok: bool,
                     execution: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Proposal:
        """APPROVED -> EXECUTED (paper fill recorded) or FAILED."""
        with self._lock:
            proposal = self._items.get(proposal_id)
            if proposal is None or proposal.status != "APPROVED":
                raise ProposalStoreError(f"proposal {proposal_id} is not awaiting execution")
            proposal.status = "EXECUTED" if ok else "FAILED"
            proposal.executed_at = utcnow_iso()
            proposal.execution_task_id = execution_task_id
            proposal.execution = execution
            proposal.error = error
            return proposal

    async def get(self, proposal_id: str, user_id: Optional[int]) -> Optional[Proposal]:
        with self._lock:
            proposal = self._items.get(proposal_id)
        if proposal is None or user_id is None or proposal.owner_user_id != user_id:
            return None
        return proposal

    async def list(self, user_id: Optional[int], status: Optional[str] = None,
                   limit: int = 20) -> List[Proposal]:
        if user_id is None:
            return []
        with self._lock:
            owned = [p for p in self._items.values() if p.owner_user_id == user_id]
        if status:
            owned = [p for p in owned if p.effective_status() == status]
        return sorted(owned, key=lambda p: p.created_at, reverse=True)[:limit]


# ------------------------------------------------------------------------ PostgreSQL

@asynccontextmanager
async def _default_session() -> AsyncIterator[Any]:
    from agentrt.ports.db import open_session  # configured by the host; raises PortNotConfigured otherwise
    async with open_session() as session:
        yield session


_COLUMNS = ("id, task_id, user_id, status, action, payload, created_at, expires_at, approved_at, "
            "executed_at, execution_task_id")
_NOW = "(now() AT TIME ZONE 'utc')"


def _ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)  # stored as UTC


def _row_to_proposal(row: Any) -> Proposal:
    m = row._mapping
    action = m["action"] if not isinstance(m["action"], str) else json.loads(m["action"])
    payload = m["payload"] if not isinstance(m["payload"], str) else json.loads(m["payload"])
    created, expires = _ts(m["created_at"]), _ts(m["expires_at"])
    approved, executed = _ts(m["approved_at"]), _ts(m["executed_at"])
    return Proposal(
        proposal_id=m["id"], owner_user_id=int(m["user_id"]), tool=action["tool"],
        tool_input=dict(action["input"]), rationale=payload.get("rationale", ""),
        evidence=payload.get("evidence") or {}, source_task_id=m["task_id"],
        input_call_ids=list(payload.get("input_call_ids") or []),
        created_at=created.isoformat(), expires_at_epoch=expires.timestamp(), status=m["status"],
        approved_at=approved.isoformat() if approved else None,
        executed_at=executed.isoformat() if executed else None,
        execution_task_id=m["execution_task_id"], execution=payload.get("execution"),
        error=payload.get("error"), persisted=True)


class SqlProposalStore:
    """agent_proposals in PostgreSQL (Flyway V34); same contract as :class:`ProposalStore`."""

    def __init__(self, session_factory: Callable[[], Any] = _default_session,
                 ttl_seconds: int = PROPOSAL_TTL_SECONDS):
        self._session_factory = session_factory
        self._ttl = ttl_seconds

    async def _one(self, sql: str, params: Dict[str, Any]) -> Optional[Proposal]:
        from sqlalchemy import text

        try:
            async with self._session_factory() as session:  # the session commits on exit
                row = (await session.execute(text(sql), params)).fetchone()
        except Exception as exc:
            raise ProposalStoreError(f"proposal store unavailable: {exc}") from exc
        return _row_to_proposal(row) if row is not None else None

    async def create(self, *, owner_user_id: int, tool: str, tool_input: Dict[str, Any],
                     rationale: str, evidence: Dict[str, Any], source_task_id: str,
                     input_call_ids: Optional[List[str]] = None) -> Proposal:
        proposal = await self._one(
            f"INSERT INTO agent_proposals (id, task_id, user_id, status, action, payload, created_at, "
            f"expires_at) VALUES (:id, :task_id, :user_id, 'PENDING', CAST(:action AS JSONB), "
            f"CAST(:payload AS JSONB), {_NOW}, {_NOW} + make_interval(secs => :ttl)) "
            f"RETURNING {_COLUMNS}",
            {"id": str(uuid.uuid4()), "task_id": source_task_id, "user_id": owner_user_id,
             "action": json.dumps({"tool": tool, "input": tool_input}, ensure_ascii=False),
             "payload": json.dumps({"rationale": rationale, "evidence": evidence,
                                    "input_call_ids": list(input_call_ids or [])},
                                   ensure_ascii=False, default=str),
             "ttl": float(self._ttl)})
        if proposal is None:
            raise ProposalStoreError("proposal not stored")
        return proposal

    async def approve(self, proposal_id: str, user_id: Optional[int]) -> Proposal:
        if user_id is not None:
            approved = await self._one(
                f"UPDATE agent_proposals SET status = 'APPROVED', approved_at = {_NOW} "
                f"WHERE id = :id AND user_id = :user_id AND status = 'PENDING' AND expires_at > {_NOW} "
                f"RETURNING {_COLUMNS}", {"id": proposal_id, "user_id": user_id})
            if approved is not None:
                return approved
        current = await self._one(f"SELECT {_COLUMNS} FROM agent_proposals WHERE id = :id",
                                  {"id": proposal_id})
        if current is not None and current.owner_user_id == user_id and current.effective_status() == "EXPIRED":
            await self._one(f"UPDATE agent_proposals SET status = 'EXPIRED' WHERE id = :id "
                            f"AND status = 'PENDING' RETURNING {_COLUMNS}", {"id": proposal_id})
        raise _refusal(current, proposal_id, user_id)

    async def finish(self, proposal_id: str, execution_task_id: str, ok: bool,
                     execution: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Proposal:
        done = await self._one(
            f"UPDATE agent_proposals SET status = :status, executed_at = {_NOW}, "
            f"execution_task_id = :execution_task_id, "
            f"payload = payload || CAST(:extra AS JSONB) "
            f"WHERE id = :id AND status = 'APPROVED' RETURNING {_COLUMNS}",
            {"id": proposal_id, "status": "EXECUTED" if ok else "FAILED",
             "execution_task_id": execution_task_id,
             "extra": json.dumps({"execution": execution, "error": error}, ensure_ascii=False, default=str)})
        if done is None:
            raise ProposalStoreError(f"proposal {proposal_id} is not awaiting execution")
        return done

    async def get(self, proposal_id: str, user_id: Optional[int]) -> Optional[Proposal]:
        if user_id is None:
            return None
        return await self._one(f"SELECT {_COLUMNS} FROM agent_proposals WHERE id = :id AND user_id = :user_id",
                               {"id": proposal_id, "user_id": user_id})

    async def list(self, user_id: Optional[int], status: Optional[str] = None,
                   limit: int = 20) -> List[Proposal]:
        from sqlalchemy import text

        if user_id is None:
            return []
        sql = f"SELECT {_COLUMNS} FROM agent_proposals WHERE user_id = :user_id"
        params: Dict[str, Any] = {"user_id": user_id, "limit": int(limit)}
        if status == "EXPIRED":
            sql += f" AND (status = 'EXPIRED' OR (status = 'PENDING' AND expires_at <= {_NOW}))"
        elif status == "PENDING":
            sql += f" AND status = 'PENDING' AND expires_at > {_NOW}"
        elif status:
            sql += " AND status = :status"
            params["status"] = status
        sql += " ORDER BY created_at DESC, id LIMIT :limit"
        try:
            async with self._session_factory() as session:
                rows = (await session.execute(text(sql), params)).fetchall()
        except Exception as exc:
            raise ProposalStoreError(f"proposal store unavailable: {exc}") from exc
        return [_row_to_proposal(r) for r in rows]


def gate(level: ActionLevel, approval: Optional[Dict[str, Any]]) -> Optional[str]:
    """Reason to block, or None when the action may run now."""
    if level in (ActionLevel.ANALYSIS, ActionLevel.SIMULATE):
        return None
    if level == ActionLevel.EXECUTE and approval and approval.get("approved_by"):
        return None
    return f"{level.value} requires a human approval"
