# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent memory guardrails, recall, and the SQL stores (fake session: unit tests)."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import pytest

from agentrt.runtime.contract import AgentRun, RunStatus
from agentrt.runtime.memory import (
    InMemoryMemoryStore, MemoryItem, MemoryValidationError, SqlMemoryStore, rank_memories,
    validate_content,
)
from agentrt.runtime.recorder import SqlRunArchive

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


# -------------------------------------------------------------------------- guardrails

@pytest.mark.parametrize("key,value", [
    ("risk_tolerance", "reckless"),
    ("max_single_position_pct", 0),
    ("max_single_position_pct", 150),
    ("max_single_position_pct", True),
    ("forbid_auto_trading", False),       # the prohibition cannot be relaxed
    ("allow_margin", True),               # not a whitelisted preference
])
def test_invalid_preferences_are_rejected(key, value):
    with pytest.raises(MemoryValidationError):
        validate_content("preference", {"key": key, "value": value}, None)


@pytest.mark.parametrize("content", [
    {"subject": "AAPL", "recommendation": "BUY", "reason": "x", "last_close": 190.0},
    {"subject": "AAPL", "recommendation": "BUY", "reason": "x", "goal": "g",
     "risk_level": "low", "as_of": "2026-09-01", "action_level": "PROPOSE", "extra": {"bars": []}},
    {"key": "risk_tolerance", "value": "low", "price": 1},
])
def test_memory_never_stores_market_data(content):
    kind = "preference" if "key" in content else "decision"
    with pytest.raises(MemoryValidationError, match="market data"):
        validate_content(kind, content, "task-1")


def test_a_decision_names_its_run_and_has_a_fixed_shape():
    good = {"subject": "AAPL", "recommendation": "HOLD", "reason": "consensus 0.2"}
    with pytest.raises(MemoryValidationError, match="source_task_id"):
        validate_content("decision", good, None)
    with pytest.raises(MemoryValidationError, match="unexpected decision fields"):
        validate_content("decision", {**good, "note": "free text"}, "task-1")
    with pytest.raises(MemoryValidationError, match="string fields"):
        validate_content("decision", {"subject": "AAPL", "recommendation": "HOLD"}, "task-1")
    assert validate_content("decision", good, "task-1") == good


# ------------------------------------------------------------------------------ recall

@pytest.mark.asyncio
async def test_latest_preference_wins_and_users_are_isolated():
    store = InMemoryMemoryStore()
    await store.set_preference(2, "risk_tolerance", "high")
    newest = await store.set_preference(2, "risk_tolerance", "medium")
    await store.set_preference(3, "risk_tolerance", "low")
    context = await store.recall(2)
    assert context.preferences["risk_tolerance"]["value"] == "medium"
    assert context.preferences["risk_tolerance"]["memory_id"] == newest.id
    assert (await store.recall(3)).preference("risk_tolerance") == "low"
    assert (await store.recall(None)).available is False


def _item(id, memory_type, content, days_old=0, importance=3, task="t"):
    created = (NOW - timedelta(days=days_old)).isoformat()
    return MemoryItem(id=id, user_id=2, memory_type=memory_type, content=content,
                      source_task_id=task, importance=importance, created_at=created)


def test_retrieval_ranks_the_goal_subject_first_and_explains_the_score():
    rows = [_item(1, "decision", {"subject": "MSFT", "recommendation": "SELL", "reason": "b"}),
            _item(2, "decision", {"subject": "AAPL", "recommendation": "BUY", "reason": "a"}, days_old=3),
            _item(3, "feedback", {"subject": "AAPL", "verdict": "disagree", "comment": "too risky"},
                  days_old=1, importance=4)]
    relevant, ignored = rank_memories(rows, ["AAPL"], NOW)
    assert [r.item.id for r in relevant] == [3, 2]            # feedback outranks the older analysis
    assert relevant[0].components["subject_match"] == 1.0
    assert 0.35 <= relevant[1].score <= 1
    assert ignored == []  # MSFT is another subject: not relevant, not shown as ignored either


def test_stale_and_superseded_memories_are_set_aside_not_used():
    rows = [_item(1, "decision", {"subject": "AAPL", "recommendation": "SELL", "reason": "old"}, days_old=60),
            _item(2, "decision", {"subject": "AAPL", "recommendation": "HOLD", "reason": "newer"}, days_old=5),
            _item(3, "decision", {"subject": "AAPL", "recommendation": "BUY", "reason": "newest"}, days_old=1)]
    relevant, ignored = rank_memories(rows, ["AAPL"], NOW)
    assert [r.item.id for r in relevant] == [3]
    assert {r.item.id: r.status for r in ignored} == {2: "superseded", 1: "stale"}


@pytest.mark.asyncio
async def test_retrieve_context_renders_constraints_and_history_as_context_lines():
    store = InMemoryMemoryStore()
    await store.set_preference(2, "forbid_auto_trading", True)
    await store.set_preference(2, "risk_tolerance", "medium")
    await store.add(2, "decision", {"subject": "AAPL", "recommendation": "REDUCE_EXPOSURE",
                                    "reason": "over limit", "risk_level": "high"}, "t1")
    await store.add(2, "proposal", {"subject": "AAPL", "side": "sell", "quantity": 5,
                                    "status": "AWAITING_APPROVAL"}, "t2")
    context = await store.retrieve_context("分析 AAPL 风险", 2, ["AAPL"])
    lines = context.lines()
    assert lines[:2] == ["用户风险偏好：medium（memory #2）", "用户禁止自动交易（memory #1）"]
    assert any("AAPL 分析结论：REDUCE_EXPOSURE，风险 high" in line for line in lines)
    assert any("AAPL 交易提案：sell 5 股，状态 AWAITING_APPROVAL" in line for line in lines)
    assert context.to_dict()["note"].startswith("context only")
    assert (await store.retrieve_context("x", None)).available is False


@pytest.mark.parametrize("memory_type,content,message", [
    ("proposal", {"subject": "AAPL", "side": "hold", "status": "x"}, "buy or sell"),
    ("proposal", {"subject": "AAPL", "side": "buy", "status": "x", "quantity": -1}, "positive number"),
    ("proposal", {"subject": "AAPL", "side": "buy", "status": "x", "reference_price": 190}, "market data"),
    ("feedback", {"subject": "AAPL", "verdict": "love it"}, "verdict must be"),
    ("feedback", {"subject": "AAPL", "verdict": "agree", "comment": "x" * 501}, "500 characters"),
])
def test_proposal_and_feedback_memories_are_validated(memory_type, content, message):
    with pytest.raises(MemoryValidationError, match=message):
        validate_content(memory_type, content, "task-1")
    with pytest.raises(MemoryValidationError, match="source_task_id"):
        validate_content("feedback", {"subject": "AAPL", "verdict": "agree"}, None)


@pytest.mark.asyncio
async def test_an_unavailable_store_is_reported_not_fatal():
    class Broken(InMemoryMemoryStore):
        async def _select(self, *a):
            raise ConnectionError("database down")

    context = await Broken().retrieve_context("分析 AAPL", 2, ["AAPL"])
    assert context.available is False and "database down" in context.unavailable_reason
    assert context.preferences == {} and context.items == ()


# ------------------------------------------------------------------------ SQL (fake db)

class _Row:
    def __init__(self, mapping):
        self._mapping = mapping

    def __getitem__(self, i):
        return list(self._mapping.values())[i]


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeSession:
    def __init__(self, rows=None, fail=False):
        self.statements, self.rows, self.fail = [], rows or [], fail

    async def execute(self, statement, params):
        if self.fail:
            raise ConnectionError("relation agent_runs does not exist")
        self.statements.append((" ".join(str(statement).split()), params))
        return _Result(self.rows)

    def factory(self):
        @asynccontextmanager
        async def _session():
            yield self
        return _session


def _db_row(**over):
    row = {"id": 7, "user_id": 2, "memory_type": "preference",
           "content": {"key": "risk_tolerance", "value": "medium"}, "source_task_id": None,
           "importance": 5, "created_at": datetime(2026, 9, 1, 12, 0)}
    row.update(over)
    return _Row(row)


@pytest.mark.asyncio
async def test_sql_store_inserts_validated_json_and_reads_back():
    session = FakeSession(rows=[_db_row()])
    store = SqlMemoryStore(session.factory())
    item = await store.set_preference(2, "risk_tolerance", "medium")
    sql, params = session.statements[0]
    assert sql.startswith("INSERT INTO agent_memory") and "CAST(:content AS JSONB)" in sql
    assert "(now() AT TIME ZONE 'utc')" in sql  # stored as UTC, read back as UTC
    assert json.loads(params["content"]) == {"key": "risk_tolerance", "value": "medium"}
    assert (item.id, item.created_at) == (7, "2026-09-01T12:00:00+00:00")
    with pytest.raises(MemoryValidationError):
        await store.set_preference(2, "risk_tolerance", "yolo")
    assert len(session.statements) == 1  # nothing written for rejected content


@pytest.mark.asyncio
async def test_sql_store_retrieves_candidates_by_subject_or_recency_in_postgres():
    session = FakeSession(rows=[_db_row(content='{"key": "risk_tolerance", "value": "low"}')])
    context = await SqlMemoryStore(session.factory()).retrieve_context("分析 aapl", 2, ["aapl"])
    prefs_sql, prefs_params = session.statements[0]
    assert "WHERE user_id = :user_id AND memory_type = :memory_type" in prefs_sql
    assert prefs_sql.endswith("ORDER BY created_at DESC, id DESC LIMIT :limit")
    cand_sql, cand_params = session.statements[1]
    assert "UPPER(content->>'subject') = ANY(CAST(:subjects AS TEXT[]))" in cand_sql
    assert "created_at >= :since" in cand_sql and "memory_type <> 'preference'" in cand_sql
    assert cand_params["subjects"] == ["AAPL"] and cand_params["user_id"] == 2
    assert context.preference("risk_tolerance") == "low"  # JSON text column decoded


@pytest.mark.asyncio
async def test_run_archive_reports_whether_the_contract_was_persisted():
    run = AgentRun(agent="traderx.workflow", input={}, owner_user_id=2)
    run.finish(RunStatus.SUCCESS, {"recommendation": "HOLD"}, "done")
    session = FakeSession()
    assert await SqlRunArchive(session.factory()).save(run) is True
    sql, params = session.statements[0]
    assert "ON CONFLICT (task_id) DO UPDATE" in sql
    assert json.loads(params["contract"])["task_id"] == run.task_id and run.persisted is True

    assert await SqlRunArchive(FakeSession(fail=True).factory()).save(run) is False
    assert run.persisted is False


@pytest.mark.asyncio
async def test_run_archive_load_is_owner_scoped():
    session = FakeSession(rows=[_Row({"contract": json.dumps({"task_id": "t1"})})])
    archive = SqlRunArchive(session.factory())
    assert await archive.load("t1", 2) == {"task_id": "t1"}
    assert "AND user_id = :user_id" in session.statements[0][0]
    assert session.statements[0][1] == {"task_id": "t1", "user_id": 2}
    assert await archive.load("t1", None) is None
