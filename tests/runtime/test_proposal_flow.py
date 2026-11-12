# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Risk detection -> Decision -> Proposal -> Owner approval -> PAPER execution.

The workflow, agents, proposal store, approval path and ExecutionTool are the
real code (see workflow_fixtures for the replaced backends).
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi import HTTPException

from agentrt.runtime.approval import ProposalStore, ProposalStoreError
from agentrt.tools import ToolContext

from .test_runtime_api import _request
from .workflow_fixtures import (
    OTHER, OWNER, FakeDB, FakeJava, calm_uptrend, make_workflow, owner_ctx, position,  # noqa: F401
    standard_preferences,
)

pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

LOW_PORTFOLIO_RISK = {"var_95": {"var_amount": 1000.0, "var_pct": 1.0}}
SALE_GOAL = "我的AAPL仓位超过上限，减持 5 股 AAPL"


async def over_limit(make_workflow, **kw):
    """AAPL is 25% of what portfolio 7 holds (10 AAPL, 30 MSFT at the same close); the limit is 20%."""
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK, **kw)
    await standard_preferences(h.memory)
    return h


@pytest.mark.asyncio
async def test_a_sale_that_reduces_an_over_limit_position_becomes_a_proposal(make_workflow):
    h = await over_limit(make_workflow)
    d = (await h.run(SALE_GOAL, portfolio_id=7)).to_dict()
    r = d["result"]
    # risk detection: AAPL above the 20% limit -> veto on new exposure
    assert r["risk"]["status"] == "VETO"
    assert any("AAPL is 25.0% of the portfolio" in v for v in r["risk"]["vetoes"])
    sale = next(c for c in r["risk"]["checks"] if c["check"] == "sell_quantity")
    assert sale["status"] == "reduces" and "25.0% -> 14.29%" in sale["detail"]
    # decision: reduce; the requested sale is what the critic's position-limit issue asks for
    assert r["recommendation"] == "REDUCE_EXPOSURE" and d["action_level"] == "PROPOSE"
    trade = r["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and trade["mode"] == "PAPER"
    assert trade["reduces_exposure"] is True and trade["addresses"] == ["POSITION_LIMIT"]
    assert trade["trade_check"]["within_limit_after"] is True
    order = trade["proposal"]["action"]
    close = calm_uptrend()[-1]["close"]
    assert order == {"symbol": "AAPL", "side": "sell", "quantity": 5.0, "reference_price": close,
                     "reference_date": calm_uptrend()[-1]["date"]}
    assert r["approval_required"] is True
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]  # nothing executed by the run
    # the critic says plainly what the sale does not fix
    codes = {i["code"]: i for i in r["critic"]["issues"]}
    assert codes["OTHER_POSITIONS_OVER_LIMIT"]["severity"] == "LOW"
    assert "MSFT 75.0% -> 85.7143%" in codes["OTHER_POSITIONS_OVER_LIMIT"]["message"]
    assert any(n["code"] == "SALE_REDUCES_POSITION" for n in r["critic"]["notes"])
    approval = [e for e in d["trace"] if e["kind"] == "approval"]
    assert approval and approval[-1]["data"]["proposal_id"] == trade["proposal"]["proposal_id"]


@pytest.mark.asyncio
async def test_owner_approval_executes_once_on_paper_and_links_both_runs(make_workflow):
    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]

    stranger = await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))
    assert stranger.error_code == "APPROVAL_FORBIDDEN"
    assert (await h.loop.proposals.get(pid, OWNER)).status == "PENDING"  # untouched

    done = await h.loop.execute_approved(pid, owner_ctx())
    assert done.status.value == "SUCCESS"
    fill = done.result["execution"]
    assert (fill["mode"], fill["status"], fill["side"], fill["quantity"]) == ("PAPER", "PAPER_FILLED", "sell", 5.0)
    assert fill["fill_price"] == calm_uptrend()[-1]["close"] and fill["proposal_id"] == pid
    stored = await h.loop.proposals.get(pid, OWNER)
    assert stored.status == "EXECUTED" and stored.execution_task_id == done.task_id
    assert stored.source_task_id == run.task_id and stored.execution["order_id"] == fill["order_id"]
    assert done.data_provenance["source_kinds"] == ["paper_simulation"]
    trace = [e["message"] for e in done.to_dict()["trace"]]
    assert any(f"proposal {pid} is now EXECUTED" == m for m in trace)

    again = await h.loop.execute_approved(pid, owner_ctx())
    assert again.error_code == "APPROVAL_CONSUMED" and "EXECUTED" in again.stop_reason
    assert [c.tool for c in again.tool_calls] == []  # the execution tool was not reached


@pytest.mark.asyncio
async def test_selling_more_than_is_held_is_a_short_sale_and_is_vetoed(make_workflow):
    h = await over_limit(make_workflow)
    r = (await h.run("我的AAPL仓位超过上限，减持 50 股 AAPL", portfolio_id=7)).result
    assert r["trade"]["status"] == "NOT_PROPOSED" and "Risk Agent veto" in r["trade"]["reason"]
    assert any("short sale" in v for v in r["risk"]["vetoes"])
    assert not h.loop.proposals._items


@pytest.mark.asyncio
async def test_a_buy_under_the_same_veto_is_not_proposed(make_workflow):
    h = await over_limit(make_workflow)
    r = (await h.run("我的AAPL仓位，买入 5 股 AAPL", portfolio_id=7)).result
    assert r["trade"]["status"] == "NOT_PROPOSED" and "Risk Agent veto" in r["trade"]["reason"]
    assert not h.loop.proposals._items


@pytest.mark.asyncio
async def test_a_sale_is_held_back_while_another_blocking_issue_remains(make_workflow):
    h = await over_limit(make_workflow, today=date(2026, 10, 30))  # last bar 2026-08-28: stale
    r = (await h.run(SALE_GOAL, portfolio_id=7)).result
    assert r["trade"]["status"] == "NOT_PROPOSED"
    assert r["trade"]["reason"] == "critic requested a revision: STALE_DATA"
    assert not h.loop.proposals._items


@pytest.mark.asyncio
async def test_without_a_portfolio_a_sale_still_needs_a_sell_decision(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    r = (await h.run("卖出 5 股 AAPL")).result
    assert r["recommendation"] == "HOLD"
    assert r["trade"] == {"status": "NOT_PROPOSED", "reason": "decision is HOLD, not SELL"}


@pytest.mark.asyncio
async def test_no_proposal_when_it_cannot_be_stored(make_workflow):
    class DownStore(ProposalStore):
        async def create(self, **kwargs):
            raise ProposalStoreError("proposal store unavailable: connection refused")

    h = await over_limit(make_workflow)
    h.workflow.proposals = DownStore()
    r = (await h.run(SALE_GOAL, portfolio_id=7)).result
    assert r["trade"]["status"] == "NOT_PROPOSED" and "could not be stored" in r["trade"]["reason"]
    assert r["approval_required"] is False


@pytest.mark.asyncio
async def test_an_approval_that_cannot_be_recorded_executes_nothing(make_workflow):
    class DownApproval(ProposalStore):
        async def approve(self, proposal_id, user_id):
            raise ProposalStoreError("proposal store unavailable: connection refused")

    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    h.loop.proposals = DownApproval()
    done = await h.loop.execute_approved(run.result["trade"]["proposal"]["proposal_id"], owner_ctx())
    assert done.error_code == "UNAVAILABLE" and done.tool_calls == []


# ----------------------------------------------------------------------------- API

@pytest.mark.asyncio
async def test_proposal_endpoints_are_owner_scoped(make_workflow):
    from agentrt.api import routes as agent_routes

    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]
    store = h.loop.proposals

    pending = await agent_routes.runtime_proposals(_request(OWNER), status="PENDING", proposals=store)
    assert [p["proposal_id"] for p in pending] == [pid] and pending[0]["source_task_id"] == run.task_id
    assert await agent_routes.runtime_proposals(_request(OTHER), proposals=store) == []
    one = await agent_routes.runtime_proposal(pid, _request(OWNER), proposals=store)
    assert one["status"] == "PENDING" and one["action"]["side"] == "sell"
    with pytest.raises(HTTPException) as exc:
        await agent_routes.runtime_proposal(pid, _request(OTHER), proposals=store)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        await agent_routes.runtime_proposals(_request(None), proposals=store)
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        await agent_routes.runtime_proposals(_request(OWNER), status="DONE", proposals=store)
    assert exc.value.status_code == 422

    body = agent_routes.RuntimeRunRequest(approve_proposal_id=pid)
    response = await agent_routes.runtime_run(body, _request(OWNER), loop=h.loop,
                                              archive=h.workflow.archive or _Archive(), memory=h.memory)
    assert response.status_code == 200
    duplicate = await agent_routes.runtime_run(body, _request(OWNER), loop=h.loop,
                                               archive=_Archive(), memory=h.memory)
    assert duplicate.status_code == 409
    assert json.loads(duplicate.body)["error_code"] == "APPROVAL_CONSUMED"
    executed = await agent_routes.runtime_proposals(_request(OWNER), status="EXECUTED", proposals=store)
    assert [p["proposal_id"] for p in executed] == [pid]
    remembered = (await h.memory.list(OWNER, "proposal"))[0]
    assert remembered.content["status"] == "APPROVED_PAPER_EXECUTED"


class _Archive:
    async def save(self, run):
        run.persisted = True
        return True


@pytest.mark.asyncio
async def test_proposal_endpoints_report_a_store_outage():
    from agentrt.api import routes as agent_routes

    class Down(ProposalStore):
        async def list(self, *a, **k):
            raise ProposalStoreError("proposal store unavailable")

        async def get(self, *a, **k):
            raise ProposalStoreError("proposal store unavailable")

    for call in (agent_routes.runtime_proposals(_request(OWNER), proposals=Down()),
                 agent_routes.runtime_proposal("p-1", _request(OWNER), proposals=Down())):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 503
