# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Proposal store: who may approve, at most one approval and one execution,
expiry, and the agent_proposals SQL (compare-and-set on the status)."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pytest

from agentrt.runtime.approval import (
    ApprovalError, ProposalStore, ProposalStoreError, SqlProposalStore,
)

ORDER = {"symbol": "AAPL", "side": "sell", "quantity": 5.0, "reference_price": 239.7,
         "reference_date": "2026-09-19"}


def new(store, owner=2, **over):
    kwargs = dict(owner_user_id=owner, tool="execution", tool_input=dict(ORDER),
                  rationale="reduce AAPL", evidence={"call_ids": ["call-2"]},
                  source_task_id="task-1", input_call_ids=["call-2"])
    kwargs.update(over)
    return store.create(**kwargs)


# ------------------------------------------------------------------------ in-process

@pytest.mark.asyncio
async def test_only_the_owner_approves_and_only_once():
    store = ProposalStore()
    p = await new(store)
    with pytest.raises(ApprovalError) as stranger:
        await store.approve(p.proposal_id, 3)
    assert stranger.value.code == "FORBIDDEN" and p.status == "PENDING"  # untouched
    with pytest.raises(ApprovalError) as anonymous:
        await store.approve(p.proposal_id, None)
    assert anonymous.value.code == "FORBIDDEN"
    approved = await store.approve(p.proposal_id, 2)
    assert approved.status == "APPROVED" and approved.approved_at
    with pytest.raises(ApprovalError) as again:
        await store.approve(p.proposal_id, 2)
    assert again.value.code == "CONSUMED"
    with pytest.raises(ApprovalError) as missing:
        await store.approve("no-such-id", 2)
    assert missing.value.code == "NOT_FOUND"


@pytest.mark.asyncio
async def test_concurrent_approvals_let_exactly_one_through():
    store = ProposalStore()
    p = await new(store)
    outcomes = await asyncio.gather(*(store.approve(p.proposal_id, 2) for _ in range(5)),
                                    return_exceptions=True)
    assert sum(1 for o in outcomes if not isinstance(o, Exception)) == 1
    assert {o.code for o in outcomes if isinstance(o, ApprovalError)} == {"CONSUMED"}


@pytest.mark.asyncio
async def test_an_executed_proposal_cannot_be_approved_or_executed_again():
    store = ProposalStore()
    p = await new(store)
    with pytest.raises(ProposalStoreError):
        await store.finish(p.proposal_id, "exec-1", True, {"mode": "PAPER"})  # not approved yet
    await store.approve(p.proposal_id, 2)
    done = await store.finish(p.proposal_id, "exec-1", True, {"mode": "PAPER", "order_id": "paper-1"})
    assert (done.status, done.execution_task_id, done.execution["order_id"]) == ("EXECUTED", "exec-1", "paper-1")
    with pytest.raises(ApprovalError) as again:
        await store.approve(p.proposal_id, 2)
    assert again.value.code == "CONSUMED" and "EXECUTED" in str(again.value)
    with pytest.raises(ProposalStoreError):
        await store.finish(p.proposal_id, "exec-2", True, {"mode": "PAPER"})


@pytest.mark.asyncio
async def test_a_failed_execution_is_recorded_and_final():
    store = ProposalStore()
    p = await new(store)
    await store.approve(p.proposal_id, 2)
    failed = await store.finish(p.proposal_id, "exec-1", False, error="APPROVAL_REQUIRED")
    assert (failed.status, failed.error) == ("FAILED", "APPROVAL_REQUIRED")
    with pytest.raises(ApprovalError):
        await store.approve(p.proposal_id, 2)


@pytest.mark.asyncio
async def test_an_expired_proposal_cannot_be_approved():
    store = ProposalStore(ttl_seconds=60)
    p = await new(store)
    p.expires_at_epoch = time.time() - 1
    assert p.to_dict()["status"] == "EXPIRED"
    with pytest.raises(ApprovalError) as exc:
        await store.approve(p.proposal_id, 2)
    assert exc.value.code == "EXPIRED" and p.status == "EXPIRED"


@pytest.mark.asyncio
async def test_reads_are_scoped_to_the_owner():
    store = ProposalStore()
    mine, theirs = await new(store), await new(store, owner=3)
    assert await store.get(mine.proposal_id, 2) is mine
    assert await store.get(theirs.proposal_id, 2) is None
    assert [p.proposal_id for p in await store.list(2)] == [mine.proposal_id]
    assert await store.list(2, "EXECUTED") == [] and await store.list(None) == []
    d = mine.to_dict()
    assert (d["mode"], d["action"], d["action_level"], d["source_task_id"]) == (
        "PAPER", ORDER, "EXECUTE", "task-1")


# ------------------------------------------------------------------------ PostgreSQL

class _Row:
    def __init__(self, data):
        self._mapping = data


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class ScriptedSession:
    """Answers each statement with the next scripted row list; records the SQL."""

    def __init__(self, *answers, fail=False):
        self.answers, self.fail, self.statements = list(answers), fail, []

    async def execute(self, statement, params):
        if self.fail:
            raise ConnectionError("could not connect to server")
        self.statements.append((" ".join(str(statement).split()), params))
        return _Result(self.answers.pop(0) if self.answers else [])

    def factory(self):
        @asynccontextmanager
        async def _session():
            yield self
        return _session


def _db_row(**over):
    now = datetime(2026, 9, 23, 12, 0)
    row = {"id": "p-1", "task_id": "task-1", "user_id": 2, "status": "PENDING",
           "action": {"tool": "execution", "input": dict(ORDER)},
           "payload": json.dumps({"rationale": "reduce AAPL", "evidence": {"call_ids": ["call-2"]},
                                  "input_call_ids": ["call-2"]}),
           "created_at": now, "expires_at": now + timedelta(days=3650), "approved_at": None,
           "executed_at": None, "execution_task_id": None}
    row.update(over)
    return [_Row(row)]


@pytest.mark.asyncio
async def test_sql_create_stores_the_exact_action_as_pending_with_an_expiry():
    session = ScriptedSession(_db_row())
    p = await new(SqlProposalStore(session.factory(), ttl_seconds=900))
    sql, params = session.statements[0]
    assert sql.startswith("INSERT INTO agent_proposals") and "'PENDING'" in sql
    assert "(now() AT TIME ZONE 'utc') + make_interval(secs => :ttl)" in sql and params["ttl"] == 900.0
    assert json.loads(params["action"]) == {"tool": "execution", "input": ORDER}
    assert json.loads(params["payload"])["input_call_ids"] == ["call-2"]
    assert (params["task_id"], params["user_id"]) == ("task-1", 2)
    assert (p.persisted, p.status, p.tool_input, p.created_at) == (
        True, "PENDING", ORDER, "2026-09-23T12:00:00+00:00")


@pytest.mark.asyncio
async def test_sql_approval_is_a_compare_and_set_by_owner_status_and_expiry():
    approved = _db_row(status="APPROVED", approved_at=datetime(2026, 9, 23, 12, 5))
    session = ScriptedSession(approved)
    p = await SqlProposalStore(session.factory()).approve("p-1", 2)
    sql, params = session.statements[0]
    assert sql.startswith("UPDATE agent_proposals SET status = 'APPROVED'")
    assert ("WHERE id = :id AND user_id = :user_id AND status = 'PENDING' "
            "AND expires_at > (now() AT TIME ZONE 'utc')") in sql
    assert params == {"id": "p-1", "user_id": 2}
    assert (p.status, p.approved_at) == ("APPROVED", "2026-09-23T12:05:00+00:00")


@pytest.mark.asyncio
@pytest.mark.parametrize("current,user,code", [
    ([], 2, "NOT_FOUND"),
    (_db_row(), 3, "FORBIDDEN"),
    (_db_row(status="APPROVED"), 2, "CONSUMED"),
    (_db_row(status="EXECUTED"), 2, "CONSUMED"),
    (_db_row(expires_at=datetime(2026, 1, 1)), 2, "EXPIRED"),
])
async def test_sql_refusals_say_why(current, user, code):
    session = ScriptedSession([], current)  # the compare-and-set matched no row
    with pytest.raises(ApprovalError) as exc:
        await SqlProposalStore(session.factory()).approve("p-1", user)
    assert exc.value.code == code
    assert session.statements[1][0].startswith("SELECT")
    if code == "EXPIRED":  # the stale PENDING row is closed
        assert "SET status = 'EXPIRED'" in session.statements[2][0]
        assert "AND status = 'PENDING'" in session.statements[2][0]
    else:
        assert len(session.statements) == 2


@pytest.mark.asyncio
async def test_sql_finish_only_moves_an_approved_proposal_and_keeps_the_execution():
    executed = _db_row(status="EXECUTED", approved_at=datetime(2026, 9, 23, 12, 5),
                       executed_at=datetime(2026, 9, 23, 12, 5, 1), execution_task_id="exec-1",
                       payload={"rationale": "r", "evidence": {}, "execution": {"order_id": "paper-1"}})
    session = ScriptedSession(executed)
    done = await SqlProposalStore(session.factory()).finish("p-1", "exec-1", True, {"order_id": "paper-1"})
    sql, params = session.statements[0]
    assert "WHERE id = :id AND status = 'APPROVED'" in sql and "payload = payload || CAST(:extra AS JSONB)" in sql
    assert params["status"] == "EXECUTED" and json.loads(params["extra"])["execution"] == {"order_id": "paper-1"}
    assert (done.status, done.execution_task_id, done.execution) == ("EXECUTED", "exec-1", {"order_id": "paper-1"})
    with pytest.raises(ProposalStoreError):  # no APPROVED row: already executed
        await SqlProposalStore(ScriptedSession([]).factory()).finish("p-1", "exec-2", True, {})


@pytest.mark.asyncio
async def test_sql_reads_are_owner_scoped_and_pending_excludes_expired():
    session = ScriptedSession(_db_row(), _db_row())
    store = SqlProposalStore(session.factory())
    await store.get("p-1", 2)
    await store.list(2, "PENDING")
    get_sql, get_params = session.statements[0]
    assert get_sql.endswith("WHERE id = :id AND user_id = :user_id") and get_params["user_id"] == 2
    list_sql, _ = session.statements[1]
    assert "AND status = 'PENDING' AND expires_at > (now() AT TIME ZONE 'utc')" in list_sql
    assert await store.get("p-1", None) is None and await store.list(None) == []


@pytest.mark.asyncio
async def test_sql_outage_is_an_error_not_a_silent_success():
    store = SqlProposalStore(ScriptedSession(fail=True).factory())
    for call in (new(store), store.approve("p-1", 2), store.finish("p-1", "e", True), store.list(2)):
        with pytest.raises(ProposalStoreError):
            await call
