# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared set-up for the multi-agent workflow tests.

The workflow, planner, agents, memory guardrails, approval store and tools are
the real code. Replaced are only the outermost backends (as in the v1
evaluation suite): the Java EOD endpoint, the database session behind the
portfolio tools, RiskAgent's analytics service, and - where a test says so -
the language model, by a scripted client that returns fixed JSON.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

import httpx
import pytest

from agentrt.quant.risk_agent import RiskAgent
from agentrt.runtime.approval import ProposalStore
from agentrt.runtime.llm import LLMError
from agentrt.runtime.loop import AgentLoop
from agentrt.runtime.memory import InMemoryMemoryStore
from agentrt.runtime.planning import PlanningService
from agentrt.runtime.recorder import RunLog
from agentrt.runtime.workflow import DecisionWorkflow
from agentrt.tools import (
    BacktestTool, ExecutionTool, FactorTool, MarketDataTool, PortfolioTool, RiskTool,
    SignalTool, ToolContext, ToolRegistry,
)

from .test_agent_evaluation import FakeDB, FakeJava, OTHER, OWNER, _bars, _position  # noqa: F401

TODAY = date(2026, 9, 1)


def calm_uptrend(n: int = 120) -> List[Dict[str, Any]]:
    """Ends 2026-08-28 for n=120: fresh relative to TODAY."""
    import math
    return _bars([0.004 + 0.002 * math.sin(i) for i in range(n)])


def wild(n: int = 120) -> List[Dict[str, Any]]:
    return _bars([0.08 if i % 2 else -0.075 for i in range(n)])


def sell_setup(n: int = 120) -> List[Dict[str, Any]]:
    """Downtrend with a two-day drop at the end: momentum and MACD both say SELL (consensus -0.6)."""
    import math
    return _bars([-0.004 + 0.001 * math.sin(i) for i in range(n - 2)] + [-0.01, -0.01])


class WindowedJava(FakeJava):
    """FakeJava that honours startDate/endDate like the real EOD endpoint."""

    def handler(self, request: httpx.Request) -> httpx.Response:
        response = super().handler(request)
        if response.status_code != 200:
            return response
        start, end = request.url.params.get("startDate"), request.url.params.get("endDate")
        body = response.json()
        body["data"] = [r for r in body["data"] if (not start or r["date"] >= start)
                        and (not end or r["date"] <= end)]
        return httpx.Response(200, json=body)


def position(symbol: str, quantity: float) -> Dict[str, Any]:
    row = _position(symbol)
    row["quantity"] = quantity
    return row


class ScriptedLLM:
    """A language model that answers with fixed JSON (or raises), in order.

    Positional answers are plans; ``review=[...]`` and ``decision=[...]`` script
    the critic and the decision agent. A call with no scripted answer raises
    LLMError, which callers treat like any model failure (rule fallback).
    """

    model = "scripted-test-model"
    provider = "mock"          # never a real model: verification reports TEST_ONLY
    endpoint = "scripted://test"

    def __init__(self, *plans: Any, review: Optional[List[Any]] = None,
                 decision: Optional[List[Any]] = None):
        self.answers = {"plan": list(plans), "review": list(review or []),
                        "decision": list(decision or [])}
        self.prompts: List[Dict[str, Any]] = []
        self.calls: List[str] = []
        self.last_call: Dict[str, Any] = {}

    def available(self):
        return True, None

    async def generate(self, system: str, prompt: str, schema_name: str, schema: Dict[str, Any]):
        self.calls.append(schema_name)
        self.last_call = {"model": self.model, "schema": schema_name}  # like the SDK client: per call
        payload = json.loads(prompt)
        if schema_name == "plan":
            self.prompts.append(payload)
        queue = self.answers.get(schema_name, [])
        if not queue:
            raise LLMError(f"no scripted {schema_name} answer")
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return json.loads(json.dumps(answer))


class MemoryArchive:
    """SqlRunArchive stand-in that keeps contracts in a dict."""

    def __init__(self):
        self.saved: Dict[str, Any] = {}

    async def save(self, run) -> bool:
        run.persisted = True
        self.saved[run.task_id] = (run.owner_user_id, run.to_dict())
        return True

    async def load(self, task_id: str, user_id: Optional[int]):
        entry = self.saved.get(task_id)
        return entry[1] if entry and user_id is not None and entry[0] == user_id else None


def owner_ctx() -> ToolContext:
    return ToolContext(user_id=OWNER, authorization="Bearer test-token")


class Harness:
    def __init__(self, workflow: DecisionWorkflow, loop: AgentLoop, java: FakeJava,
                 memory: InMemoryMemoryStore):
        self.workflow, self.loop, self.java, self.memory = workflow, loop, java, memory

    async def run(self, goal: str, portfolio_id: Optional[int] = None, symbol: Optional[str] = None,
                  ctx: Optional[ToolContext] = None):
        return await self.workflow.run(goal, ctx or owner_ctx(), symbol=symbol, portfolio_id=portfolio_id)


@pytest.fixture
def make_workflow(stub_analytics):
    def _make(java: FakeJava, db: Optional[FakeDB] = None,
              risk_overrides: Optional[Dict[str, Any]] = None, llm: Any = None,
              memory: Optional[InMemoryMemoryStore] = None, today: date = TODAY) -> Harness:
        db = db or FakeDB({}, {})

        def risk_agent():
            agent = RiskAgent(agent_id="eval-risk")
            stub_analytics(agent, risk_overrides)
            return agent

        tools = ToolRegistry([
            MarketDataTool(base_url="http://java.test/api", transport=httpx.MockTransport(java.handler),
                           service_token=lambda: None, today=lambda: today),
            PortfolioTool(session_factory=db.session_factory()),
            RiskTool(session_factory=db.session_factory(), risk_agent_factory=risk_agent),
            SignalTool(), BacktestTool(), FactorTool(), ExecutionTool(),
        ])
        memory = memory or InMemoryMemoryStore()
        proposals, run_log = ProposalStore(), RunLog()
        loop = AgentLoop(tools, proposals=proposals, run_log=run_log)
        workflow = DecisionWorkflow(tools, PlanningService(tools, llm), memory, proposals=proposals,
                                    run_log=run_log, today=lambda: today)
        return Harness(workflow, loop, java, memory)
    return _make


async def standard_preferences(memory: InMemoryMemoryStore, user_id: int = OWNER) -> None:
    """The preferences named in the product brief: medium risk, 20% cap, no auto trading."""
    await memory.set_preference(user_id, "risk_tolerance", "medium")
    await memory.set_preference(user_id, "max_single_position_pct", 20)
    await memory.set_preference(user_id, "forbid_auto_trading", True)


def agents_in_order(run_dict: Dict[str, Any]) -> List[str]:
    return [e["data"]["agent"] for e in run_dict["trace"] if e["kind"] == "agent_report"]


def calls_by_agent(run_dict: Dict[str, Any]) -> List[tuple]:
    return [(c["agent"], c["tool"]) for c in run_dict["tool_calls"]]
