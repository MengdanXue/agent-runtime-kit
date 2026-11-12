# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent memory - relational, context only - and its retriever.

Memory types, stored in ``agent_memory`` (Flyway V32/V33):

* ``preference`` - a user constraint from a fixed whitelist (risk tolerance,
  maximum single-position weight, the auto-trading prohibition). Latest value
  per key wins.
* ``decision``   - a previous analysis: what a run concluded (subject,
  recommendation, reason), linked to that run (``source_task_id``).
* ``proposal``   - a previous trade proposal and its status.
* ``feedback``   - the user's verdict on a run's conclusion.

Retrieval (``retrieve_context(goal, user_id)``) is PostgreSQL-first: candidate
rows are selected by user, subject (``content->>'subject'``) and recency, then
ranked by a deterministic relevance score::

    score = 0.50 * subject_match + 0.20 * recency + 0.15 * importance/5 + 0.15 * type_weight

Stale memories (older than their type's horizon) and superseded ones (an older
analysis or proposal for the same subject) are set aside and reported, never
used. Memory is context, never data: market-data fields are rejected on write,
and no agent reads a price, a risk figure or a tool result from memory.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MEMORY_TYPES = ("preference", "decision", "proposal", "feedback")
TYPE_LABELS = {"preference": "user_constraint", "decision": "previous_analysis",
               "proposal": "previous_proposal", "feedback": "user_feedback"}
MAX_CONTENT_BYTES = 4096

RISK_TOLERANCES = ("low", "medium", "high")

# Keys a memory may never carry: market data belongs to the tool layer.
MARKET_DATA_KEYS = frozenset({
    "price", "prices", "close", "open", "high", "low", "last_close", "bars", "volume",
    "quote", "quotes", "ohlc", "ohlcv", "market_data", "eod", "reference_price",
})

DECISION_REQUIRED = ("subject", "recommendation", "reason")
DECISION_OPTIONAL = ("action_level", "as_of", "risk_level", "goal", "confidence")
PROPOSAL_REQUIRED = ("subject", "side", "status")
PROPOSAL_OPTIONAL = ("quantity", "proposal_id", "reason", "goal")
FEEDBACK_REQUIRED = ("subject", "verdict")
FEEDBACK_OPTIONAL = ("comment", "about_recommendation")
FEEDBACK_VERDICTS = ("agree", "disagree")

# Retrieval
STALE_AFTER_DAYS = {"decision": 30, "proposal": 30, "feedback": 90}
TYPE_WEIGHT = {"feedback": 1.0, "proposal": 0.8, "decision": 0.6}
RECENCY_HALF_LIFE_DAYS = 14
MIN_RELEVANCE = 0.35
CANDIDATE_WINDOW_DAYS = 90
CANDIDATE_LIMIT = 200


class MemoryValidationError(ValueError):
    """Content rejected by the memory guardrails."""


def _validate_preference_value(key: str, value: Any) -> Any:
    if key == "risk_tolerance":
        if value not in RISK_TOLERANCES:
            raise MemoryValidationError(f"risk_tolerance must be one of {RISK_TOLERANCES}")
        return value
    if key == "max_single_position_pct":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 100:
            raise MemoryValidationError("max_single_position_pct must be a number in (0, 100]")
        return float(value)
    if key == "forbid_auto_trading":
        if value is not True:
            # The platform never trades on an agent's own decision; this cannot be relaxed.
            raise MemoryValidationError("forbid_auto_trading can only be true")
        return True
    raise MemoryValidationError(
        f"unknown preference '{key}'; allowed: risk_tolerance, max_single_position_pct, forbid_auto_trading")


PREFERENCE_KEYS = ("risk_tolerance", "max_single_position_pct", "forbid_auto_trading")


def _market_keys(value: Any, path: str = "") -> List[str]:
    found = []
    if isinstance(value, dict):
        for key, inner in value.items():
            where = f"{path}.{key}" if path else str(key)
            if str(key).lower() in MARKET_DATA_KEYS:
                found.append(where)
            found.extend(_market_keys(inner, where))
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            found.extend(_market_keys(inner, f"{path}[{i}]"))
    return found


def _check_shape(content: Dict[str, Any], kind: str, required: Sequence[str],
                 optional: Sequence[str]) -> None:
    missing = [k for k in required if not isinstance(content.get(k), str) or not content[k]]
    if missing:
        raise MemoryValidationError(f"a {kind} needs string fields {missing}")
    extra = set(content) - set(required) - set(optional)
    if extra:
        raise MemoryValidationError(f"unexpected {kind} fields {sorted(extra)}")


def validate_content(memory_type: str, content: Any,
                     source_task_id: Optional[str]) -> Dict[str, Any]:
    if memory_type not in MEMORY_TYPES:
        raise MemoryValidationError(f"memory_type must be one of {MEMORY_TYPES}")
    if not isinstance(content, dict):
        raise MemoryValidationError("content must be an object")
    try:
        encoded = json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise MemoryValidationError(f"content is not JSON serializable: {exc}") from exc
    if len(encoded.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise MemoryValidationError(f"content exceeds {MAX_CONTENT_BYTES} bytes")
    market = _market_keys(content)
    if market:
        raise MemoryValidationError(
            f"memory may not carry market data ({', '.join(market)}); prices come from tools only")

    if memory_type == "preference":
        if set(content) != {"key", "value"}:
            raise MemoryValidationError("a preference is exactly {key, value}")
        return {"key": content["key"],
                "value": _validate_preference_value(content["key"], content["value"])}
    if not source_task_id:
        raise MemoryValidationError(f"a {memory_type} must name the run it is about (source_task_id)")
    if memory_type == "decision":
        _check_shape(content, "decision", DECISION_REQUIRED, DECISION_OPTIONAL)
    elif memory_type == "proposal":
        _check_shape(content, "proposal", PROPOSAL_REQUIRED, PROPOSAL_OPTIONAL)
        if content["side"] not in ("buy", "sell"):
            raise MemoryValidationError("a proposal side is buy or sell")
        quantity = content.get("quantity")
        if quantity is not None and (isinstance(quantity, bool) or not isinstance(quantity, (int, float))
                                     or quantity <= 0):
            raise MemoryValidationError("a proposal quantity is a positive number")
    else:
        _check_shape(content, "feedback", FEEDBACK_REQUIRED, FEEDBACK_OPTIONAL)
        if content["verdict"] not in FEEDBACK_VERDICTS:
            raise MemoryValidationError(f"feedback verdict must be one of {FEEDBACK_VERDICTS}")
        if len(str(content.get("comment") or "")) > 500:
            raise MemoryValidationError("feedback comment is limited to 500 characters")
    return dict(content)


@dataclass(frozen=True)
class MemoryItem:
    id: int
    user_id: int
    memory_type: str
    content: Dict[str, Any]
    source_task_id: Optional[str]
    importance: int
    created_at: str

    @property
    def subject(self) -> str:
        return str(self.content.get("subject") or "").upper()

    def age_days(self, now: datetime) -> float:
        try:
            created = datetime.fromisoformat(self.created_at)
        except ValueError:
            return float("inf")
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return max(0.0, (now - created).total_seconds() / 86400)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "memory_type": self.memory_type, "kind": TYPE_LABELS[self.memory_type],
                "content": dict(self.content), "source_task_id": self.source_task_id,
                "importance": self.importance, "created_at": self.created_at}


@dataclass(frozen=True)
class RetrievedMemory:
    item: MemoryItem
    score: float
    components: Dict[str, float]
    status: str        # relevant | stale | superseded | other_subject | low_relevance
    line: str

    def to_dict(self) -> Dict[str, Any]:
        return {**self.item.to_dict(), "relevance": self.score, "components": dict(self.components),
                "status": self.status, "line": self.line}


_PREF_LINES = {
    "risk_tolerance": "用户风险偏好：{value}",
    "max_single_position_pct": "单一持仓上限：{value:g}%",
    "forbid_auto_trading": "用户禁止自动交易",
}


def _date(item: MemoryItem) -> str:
    return item.created_at[:10]


def render_line(item: MemoryItem) -> str:
    c = item.content
    if item.memory_type == "decision":
        risk = f"，风险 {c['risk_level']}" if c.get("risk_level") else ""
        return f"{_date(item)} {c['subject']} 分析结论：{c['recommendation']}{risk}（run {item.source_task_id}）"
    if item.memory_type == "proposal":
        qty = f" {c['quantity']:g} 股" if isinstance(c.get("quantity"), (int, float)) else ""
        return f"{_date(item)} {c['subject']} 交易提案：{c['side']}{qty}，状态 {c['status']}"
    if item.memory_type == "feedback":
        verdict = "同意" if c["verdict"] == "agree" else "不同意"
        about = f"（{c['about_recommendation']}）" if c.get("about_recommendation") else ""
        comment = f"：{c['comment']}" if c.get("comment") else ""
        return f"{_date(item)} 用户{verdict} {c['subject']} 的结论{about}{comment}"
    return json.dumps(c, ensure_ascii=False)


@dataclass(frozen=True)
class MemoryContext:
    """What the planner and agents may read. Immutable; context only."""

    preferences: Dict[str, Any] = field(default_factory=dict)
    items: Tuple[RetrievedMemory, ...] = ()        # relevant, ranked
    ignored: Tuple[RetrievedMemory, ...] = ()      # stale / superseded / unrelated, for the trace
    available: bool = True
    unavailable_reason: Optional[str] = None
    query: Dict[str, Any] = field(default_factory=dict)

    @property
    def decisions(self) -> Tuple[MemoryItem, ...]:
        return tuple(r.item for r in self.items if r.item.memory_type == "decision")

    @property
    def feedback(self) -> Tuple[MemoryItem, ...]:
        return tuple(r.item for r in self.items if r.item.memory_type == "feedback")

    @property
    def memory_ids(self) -> List[int]:
        return [p["memory_id"] for p in self.preferences.values()] + [r.item.id for r in self.items]

    def preference(self, key: str) -> Any:
        entry = self.preferences.get(key)
        return entry["value"] if entry else None

    def lines(self) -> List[str]:
        out = [_PREF_LINES[k].format(value=self.preferences[k]["value"])
               + f"（memory #{self.preferences[k]['memory_id']}）"
               for k in PREFERENCE_KEYS if k in self.preferences]
        out += [f"{r.line}（相关度 {r.score:.2f}）" for r in self.items]
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "unavailable_reason": self.unavailable_reason,
                "preferences": {k: dict(v) for k, v in self.preferences.items()},
                "relevant": [r.to_dict() for r in self.items],
                "ignored": [r.to_dict() for r in self.ignored],
                "decisions": [d.to_dict() for d in self.decisions],
                "lines": self.lines(), "query": dict(self.query),
                "memory_ids": self.memory_ids,
                "note": "context only; never used as market data, risk figures or tool results"}


def score_memory(item: MemoryItem, subjects: Iterable[str], now: datetime) -> Tuple[float, Dict[str, float]]:
    subjects = {s.upper() for s in subjects}
    age = item.age_days(now)
    components = {
        "subject_match": 1.0 if item.subject and item.subject in subjects else 0.0,
        "recency": round(math.exp(-age * math.log(2) / RECENCY_HALF_LIFE_DAYS), 3) if age != float("inf") else 0.0,
        "importance": round(item.importance / 5, 3),
        "type_weight": TYPE_WEIGHT.get(item.memory_type, 0.5),
    }
    score = (0.5 * components["subject_match"] + 0.2 * components["recency"]
             + 0.15 * components["importance"] + 0.15 * components["type_weight"])
    return round(score, 3), components


def rank_memories(candidates: Iterable[MemoryItem], subjects: Sequence[str], now: datetime,
                  limit: int = 5) -> Tuple[List[RetrievedMemory], List[RetrievedMemory]]:
    """Relevant memories (ranked) and the ones set aside, with the reason."""
    relevant: List[RetrievedMemory] = []
    ignored: List[RetrievedMemory] = []
    newest_seen: set = set()
    for item in sorted(candidates, key=lambda i: (i.created_at, i.id), reverse=True):
        if item.memory_type == "preference":
            continue
        score, components = score_memory(item, subjects, now)
        if item.age_days(now) > STALE_AFTER_DAYS.get(item.memory_type, 30):
            status = "stale"
        elif item.memory_type in ("decision", "proposal") and (item.memory_type, item.subject) in newest_seen:
            status = "superseded"
        elif subjects and not components["subject_match"]:
            status = "other_subject"
        elif score < MIN_RELEVANCE:
            status = "low_relevance"
        else:
            status = "relevant"
        if item.memory_type in ("decision", "proposal") and status in ("relevant", "other_subject",
                                                                       "low_relevance"):
            newest_seen.add((item.memory_type, item.subject))
        entry = RetrievedMemory(item=item, score=score, components=components, status=status,
                                line=render_line(item))
        (relevant if status == "relevant" else ignored).append(entry)
    relevant.sort(key=lambda r: r.score, reverse=True)
    ignored_shown = [r for r in ignored if r.status in ("stale", "superseded")][:10]
    return relevant[:limit], ignored_shown


class MemoryStore(ABC):
    """Validation and retrieval; subclasses only store and fetch rows."""

    @abstractmethod
    async def _insert(self, user_id: int, memory_type: str, content: Dict[str, Any],
                      source_task_id: Optional[str], importance: int) -> MemoryItem: ...

    @abstractmethod
    async def _select(self, user_id: int, memory_type: Optional[str], limit: int) -> List[MemoryItem]: ...

    async def _candidates(self, user_id: int, subjects: Sequence[str], since: datetime) -> List[MemoryItem]:
        """Non-preference rows about the subjects, or recent ones (SqlMemoryStore does this in SQL)."""
        wanted = set(subjects)
        rows = await self._select(user_id, None, CANDIDATE_LIMIT)
        return [r for r in rows if r.memory_type != "preference"
                and (r.subject in wanted or r.created_at >= since.isoformat())]

    async def add(self, user_id: int, memory_type: str, content: Dict[str, Any],
                  source_task_id: Optional[str] = None, importance: int = 1) -> MemoryItem:
        if not isinstance(user_id, int):
            raise MemoryValidationError("memory needs an authenticated user")
        if not 1 <= int(importance) <= 5:
            raise MemoryValidationError("importance must be 1..5")
        clean = validate_content(memory_type, content, source_task_id)
        return await self._insert(user_id, memory_type, clean, source_task_id, int(importance))

    async def set_preference(self, user_id: int, key: str, value: Any) -> MemoryItem:
        return await self.add(user_id, "preference", {"key": key, "value": value}, importance=5)

    async def list(self, user_id: int, memory_type: Optional[str] = None,
                   limit: int = 50) -> List[MemoryItem]:
        return await self._select(user_id, memory_type, limit)

    async def preferences(self, user_id: int) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for item in await self._select(user_id, "preference", 200):  # newest first
            key = item.content.get("key")
            if key in PREFERENCE_KEYS and key not in latest:
                latest[key] = {"value": item.content.get("value"), "memory_id": item.id,
                               "set_at": item.created_at}
        return latest

    async def retrieve_context(self, goal: str, user_id: Optional[int],
                               subjects: Sequence[str] = (), now: Optional[datetime] = None,
                               limit: int = 5) -> MemoryContext:
        """Relevant memories for a goal: user constraints plus ranked history."""
        if user_id is None:
            return MemoryContext(available=False, unavailable_reason="no authenticated user")
        now = now or datetime.now(timezone.utc)
        since = now - timedelta(days=CANDIDATE_WINDOW_DAYS)
        subjects = [s.upper() for s in subjects if s]
        try:
            prefs = await self.preferences(user_id)
            candidates = await self._candidates(user_id, subjects, since)
        except Exception as exc:  # memory is context: its absence is reported, not fatal
            logger.warning("agent memory unavailable: %s", exc)
            return MemoryContext(available=False, unavailable_reason=f"{type(exc).__name__}: {exc}")
        relevant, ignored = rank_memories(candidates, subjects, now, limit)
        return MemoryContext(preferences=prefs, items=tuple(relevant), ignored=tuple(ignored),
                             query={"goal": goal[:200], "subjects": subjects,
                                    "since": since.isoformat(), "candidates": len(candidates)})

    async def recall(self, user_id: Optional[int], subject: Optional[str] = None,
                     limit: int = 5) -> MemoryContext:
        return await self.retrieve_context("", user_id, [subject] if subject else (), limit=limit)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryMemoryStore(MemoryStore):
    """Process-local store with the same guardrails (tests, and no-DB development)."""

    def __init__(self) -> None:
        self._rows: List[MemoryItem] = []
        self._lock = threading.Lock()

    async def _insert(self, user_id, memory_type, content, source_task_id, importance):
        with self._lock:
            item = MemoryItem(id=len(self._rows) + 1, user_id=user_id, memory_type=memory_type,
                              content=dict(content), source_task_id=source_task_id,
                              importance=importance, created_at=_now())
            self._rows.append(item)
            return item

    async def _select(self, user_id, memory_type, limit):
        with self._lock:
            rows = [r for r in self._rows if r.user_id == user_id
                    and (memory_type is None or r.memory_type == memory_type)]
        return list(reversed(rows))[:limit]


@asynccontextmanager
async def _default_session() -> AsyncIterator[Any]:
    from agentrt.ports.db import open_session  # configured by the host; raises PortNotConfigured otherwise
    async with open_session() as session:
        yield session


def _row_to_item(row: Any) -> MemoryItem:
    m = row._mapping
    content = m["content"]
    if isinstance(content, str):
        content = json.loads(content)
    created = m["created_at"]
    if hasattr(created, "tzinfo") and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)  # stored as UTC
    return MemoryItem(id=int(m["id"]), user_id=int(m["user_id"]), memory_type=m["memory_type"],
                      content=dict(content), source_task_id=m["source_task_id"],
                      importance=int(m["importance"]),
                      created_at=created.isoformat() if hasattr(created, "isoformat") else str(created))


_COLUMNS = "id, user_id, memory_type, content, source_task_id, importance, created_at"


class SqlMemoryStore(MemoryStore):
    """agent_memory in PostgreSQL (Flyway V32/V33)."""

    def __init__(self, session_factory: Callable[[], Any] = _default_session):
        self._session_factory = session_factory

    async def _insert(self, user_id, memory_type, content, source_task_id, importance):
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(text(
                "INSERT INTO agent_memory (user_id, memory_type, content, source_task_id, importance, created_at) "
                "VALUES (:user_id, :memory_type, CAST(:content AS JSONB), :source_task_id, :importance, "
                "(now() AT TIME ZONE 'utc')) "
                f"RETURNING {_COLUMNS}"),
                {"user_id": user_id, "memory_type": memory_type,
                 "content": json.dumps(content, ensure_ascii=False),
                 "source_task_id": source_task_id, "importance": importance})
            row = result.fetchone()  # the session context manager commits
        return _row_to_item(row)

    async def _select(self, user_id, memory_type, limit):
        from sqlalchemy import text

        sql = f"SELECT {_COLUMNS} FROM agent_memory WHERE user_id = :user_id"
        params: Dict[str, Any] = {"user_id": user_id, "limit": int(limit)}
        if memory_type:
            sql += " AND memory_type = :memory_type"
            params["memory_type"] = memory_type
        sql += " ORDER BY created_at DESC, id DESC LIMIT :limit"
        async with self._session_factory() as session:
            result = await session.execute(text(sql), params)
            rows = result.fetchall()
        return [_row_to_item(r) for r in rows]

    async def _candidates(self, user_id, subjects, since):
        from sqlalchemy import text

        async with self._session_factory() as session:
            result = await session.execute(text(
                f"SELECT {_COLUMNS} FROM agent_memory "
                "WHERE user_id = :user_id AND memory_type <> 'preference' "
                "AND (UPPER(content->>'subject') = ANY(CAST(:subjects AS TEXT[])) OR created_at >= :since) "
                "ORDER BY created_at DESC, id DESC LIMIT :limit"),
                {"user_id": user_id, "subjects": list(subjects),
                 "since": since.astimezone(timezone.utc).replace(tzinfo=None), "limit": CANDIDATE_LIMIT})
            rows = result.fetchall()
        return [_row_to_item(r) for r in rows]
