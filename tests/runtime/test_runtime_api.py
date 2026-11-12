# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/api/v1/agents/runtime/* and the contract attached to legacy dispatch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from agentrt import AgentStatus
from agentrt.runtime.contract import AgentRun, RunStatus
from agentrt.runtime.loop import AgentLoop
from agentrt.runtime.recorder import RunLog
from agentrt.tools import ToolRegistry
from agentrt.tools.base import Provenance, Tool, ToolErrorCode, ToolResult
from pydantic import BaseModel

from .workflow_fixtures import MemoryArchive


def _request(user_id, authorization="Bearer t"):
    request = MagicMock()
    request.state = MagicMock()
    request.state.user_id = user_id
    request.headers = {"Authorization": authorization}
    return request


class _In(BaseModel):
    symbol: str
    days: int = 180


class _Out(BaseModel):
    symbol: str
    bars: list
    count: int
    first_date: str
    last_date: str
    last_close: float


class _NotFoundMarketData(Tool):
    name = "market_data"
    description = "always NOT_FOUND"
    Input = _In
    Output = _Out

    async def run(self, params, ctx):
        return ToolResult.failure(ToolErrorCode.NOT_FOUND, f"No EOD data for {params.symbol}")


@pytest.mark.parametrize("code,status", [
    (None, 200), ("NOT_FOUND", 404), ("FORBIDDEN", 403), ("UNAUTHENTICATED", 401),
    ("INSUFFICIENT_DATA", 422), ("NEEDS_USER_INPUT", 422), ("TIMEOUT", 504),
    ("UNAVAILABLE", 503), ("APPROVAL_CONSUMED", 409), ("INTERNAL", 500),
])
def test_failed_runs_are_not_http_success(code, status):
    from agentrt.api.routes import run_http_status

    run = AgentRun(agent="a", input={})
    run.finish(RunStatus.SUCCESS if code is None else RunStatus.FAILED, {}, "x", code)
    assert run_http_status(run) == status


@pytest.mark.asyncio
async def test_runtime_run_requires_a_user():
    from agentrt.api.routes import RuntimeRunRequest, runtime_run

    body = RuntimeRunRequest(goal={"intent": "analyze_symbol", "symbol": "AAPL"})
    with pytest.raises(HTTPException) as exc:
        await runtime_run(body=body, http_request=_request(None),
                          loop=AgentLoop(ToolRegistry([_NotFoundMarketData()])),
                          archive=MemoryArchive())
    assert exc.value.status_code == 401


def test_runtime_request_takes_exactly_one_of_goal_or_approval():
    from agentrt.api.routes import RuntimeRunRequest

    with pytest.raises(ValueError):
        RuntimeRunRequest()
    with pytest.raises(ValueError):
        RuntimeRunRequest(goal={"intent": "analyze_symbol", "symbol": "A"}, approve_proposal_id="p")


@pytest.mark.asyncio
async def test_runtime_run_returns_the_contract_with_the_failure_status(monkeypatch):
    from agentrt.api import routes as agent_routes

    log = RunLog()
    monkeypatch.setattr(agent_routes, "_run_log", log)
    loop = AgentLoop(ToolRegistry([_NotFoundMarketData()]), run_log=log)
    archive = MemoryArchive()
    response = await agent_routes.runtime_run(
        body=agent_routes.RuntimeRunRequest(goal={"intent": "analyze_symbol", "symbol": "ZZZ"}),
        http_request=_request(2), loop=loop, archive=archive)
    body = json.loads(response.body)
    assert response.status_code == 404
    assert set(body) >= {"task_id", "agent", "input", "tool_calls", "observations", "result",
                         "status", "data_provenance", "trace"}
    assert body["status"] == "FAILED" and body["error_code"] == "NOT_FOUND"

    # the record is readable by its owner only
    detail = agent_routes.runtime_run_detail
    assert (await detail(body["task_id"], _request(2), archive=archive))["task_id"] == body["task_id"]
    with pytest.raises(HTTPException) as exc:
        await detail(body["task_id"], _request(3), archive=archive)
    assert exc.value.status_code == 404
    assert [r["task_id"] for r in await agent_routes.runtime_runs(_request(2))] == [body["task_id"]]
    assert await agent_routes.runtime_runs(_request(3)) == []

    # the contract was archived, so it survives the in-process log (e.g. a restart)
    assert body["persisted"] is True
    monkeypatch.setattr(agent_routes, "_run_log", RunLog())
    assert (await detail(body["task_id"], _request(2), archive=archive))["error_code"] == "NOT_FOUND"
    with pytest.raises(HTTPException) as exc:
        await detail(body["task_id"], _request(3), archive=archive)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_legacy_dispatch_carries_a_contract_record(monkeypatch):
    from agentrt.api import routes as agent_routes
    from agentrt.api.routes import TaskRequest

    log = RunLog()
    monkeypatch.setattr(agent_routes, "_run_log", log)

    class _Orchestrator:
        async def dispatch(self, task):
            return MagicMock(task_id=task.task_id, status=AgentStatus.FAILED, result=None,
                             error="Refusing to return fabricated trading signals", duration_ms=0)

    db = MagicMock()
    response = await agent_routes.dispatch_task(
        request=TaskRequest(agent_type="signal", payload={"symbol": "AAPL"}),
        http_request=_request(2), orchestrator=_Orchestrator(), db=db)
    assert response.status == "failed"  # existing fields unchanged
    run = response.run
    assert run["agent"] == "legacy:SignalAgent"
    assert run["status"] == "FAILED" and run["tool_calls"] == []
    assert run["data_provenance"]["tool_layer"] is False
    assert log.get(response.task_id, 2) is not None


@pytest.mark.asyncio
async def test_tools_endpoint_describes_schemas():
    from agentrt.api.routes import runtime_tools
    from agentrt.tools import default_tools

    tools = {t["name"]: t for t in await runtime_tools(loop=AgentLoop(default_tools()))}
    assert set(tools) == {"market_data", "portfolio", "risk", "signal", "backtest", "factor", "execution"}
    assert tools["execution"]["action_level"] == "EXECUTE"
    assert tools["backtest"]["action_level"] == "SIMULATE"
    assert all("properties" in t["input_schema"] and "properties" in t["output_schema"]
               for t in tools.values())


def test_provenance_is_mandatory_for_tool_output():
    import asyncio

    class _NoProvenance(_NotFoundMarketData):
        async def run(self, params, ctx):
            return ToolResult(ok=True, output={"symbol": "A", "bars": [], "count": 0,
                                               "first_date": "", "last_date": "", "last_close": 1.0})

    result = asyncio.run(_NoProvenance().invoke({"symbol": "A"}, MagicMock()))
    assert not result.ok and result.error_code == ToolErrorCode.INTERNAL


def test_output_schema_is_enforced():
    import asyncio

    class _BadOutput(_NotFoundMarketData):
        async def run(self, params, ctx):
            return ToolResult.success({"symbol": "A"}, Provenance(source="x", kind="database"))

    result = asyncio.run(_BadOutput().invoke({"symbol": "A"}, MagicMock()))
    assert not result.ok and "violates its schema" in result.error_message
