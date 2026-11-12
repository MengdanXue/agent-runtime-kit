# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-agent workflow mechanics and its API (see workflow_fixtures for what is real)."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timezone

import pytest
from fastapi import HTTPException

from agentrt.runtime.contract import AgentRun
from agentrt.runtime.memory import InMemoryMemoryStore, MemoryItem
from agentrt.runtime.workflow import ToolRunner, make_report
from agentrt.tools import ToolContext

from .test_runtime_api import _request
from .workflow_fixtures import (
    OTHER, OWNER, FakeDB, FakeJava, MemoryArchive, ScriptedLLM, agents_in_order, calls_by_agent,
    calm_uptrend, make_workflow, owner_ctx, position, sell_setup, standard_preferences, wild,
)

pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

LOW_PORTFOLIO_RISK = {"var_95": {"var_amount": 1000.0, "var_pct": 1.0}}


# ------------------------------------------------------------------------- the path

@pytest.mark.asyncio
async def test_position_review_runs_every_agent_in_order_on_real_tools(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK)
    await standard_preferences(h.memory)
    d = (await h.run("分析我的AAPL仓位风险", portfolio_id=7)).to_dict()

    kinds = [e["kind"] for e in d["trace"]]
    assert kinds[:2] == ["goal", "memory"] and kinds[-1] == "final"
    assert kinds.index("memory") < kinds.index("plan") < kinds.index("tool_call")
    # the critic still reviews after a veto, but nothing is re-analysed
    assert agents_in_order(d) == ["research", "risk", "critic", "decision"]
    assert "replan" not in kinds
    assert calls_by_agent(d) == [("research", "portfolio"), ("research", "market_data"),
                                 ("research", "signal"), ("risk", "risk"), ("risk", "risk"),
                                 ("risk", "market_data")]
    # exposure is quantity x the latest close the tools returned, never a stored price
    close = calm_uptrend()[-1]["close"]
    exposure = d["result"]["risk"]["exposure"]
    assert exposure["values"] == {"AAPL": round(10 * close, 2), "MSFT": round(30 * close, 2)}
    assert exposure["weights_pct"] == {"AAPL": 25.0, "MSFT": 75.0}
    # the 20% limit comes from memory and is cited by id; the held position is to be reduced
    assert d["result"]["recommendation"] == "REDUCE_EXPOSURE"
    assert any("above the user's 20.0% limit (memory #2)" in v for v in d["result"]["risk"]["vetoes"])
    assert d["status"] == "SUCCESS" and d["action_level"] == "PROPOSE"
    assert d["data_provenance"]["synthetic_data_used"] is False


@pytest.mark.asyncio
async def test_the_decision_sees_the_more_severe_of_symbol_and_portfolio_risk(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend()})  # symbol risk low
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10)]})
    d = (await make_workflow(java, db).run("分析我的AAPL仓位风险", portfolio_id=7)).to_dict()  # VaR 5%: high
    assert d["result"]["risk"]["level"] == "high"
    assert d["result"]["portfolio_action"] == "REDUCE_EXPOSURE"  # quant DecisionAgent rule for high risk


@pytest.mark.asyncio
async def test_the_decision_is_remembered_with_the_run_that_made_it(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    run = await h.run("分析 AAPL")
    remembered = await h.memory.list(OWNER, "decision")
    assert run.result["memory_written"] == remembered[0].id
    assert remembered[0].source_task_id == run.task_id
    assert remembered[0].content["subject"] == "AAPL"
    assert remembered[0].content["recommendation"] == run.result["recommendation"]
    assert "last_close" not in remembered[0].content

    # the next run on the same subject reads it back as context
    second = (await h.run("再分析一次 AAPL")).to_dict()
    memory_event = next(e for e in second["trace"] if e["kind"] == "memory")
    assert memory_event["data"]["memory"]["decisions"][0]["source_task_id"] == run.task_id


# ---------------------------------------------------------------- agent isolation

def test_agent_reports_cannot_be_changed_by_another_agent():
    report = make_report("research", "OK", "view HOLD", {"view": "HOLD", "market": {}}, ["call-1"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.status = "VETO"
    findings = report.findings
    findings["view"] = "BUY"
    assert report.findings["view"] == "HOLD"


@pytest.mark.asyncio
async def test_tool_runner_enforces_each_agents_allowance(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    run = AgentRun(agent="t", input={}, owner_user_id=OWNER)
    runner = ToolRunner(run, h.workflow.tools, owner_ctx())
    _, res = await runner.call("research", "risk", {"symbol": "AAPL"}, "not my job", ("market_data",))
    assert res.error_code.value == "FORBIDDEN"
    _, res = await runner.call("decision", "execution",
                               {"symbol": "AAPL", "side": "buy", "quantity": 1,
                                "reference_price": 1, "reference_date": "2026-01-01"},
                               "try to trade", ("execution",))
    assert res.error_code.value == "APPROVAL_REQUIRED"
    _, res = await runner.call("research", "signal", {"symbol": "AAPL", "bars": "$bars"}, "no bars yet",
                               ("signal",))
    assert res.error_code.value == "INVALID_INPUT"
    assert [c.status for c in run.tool_calls] == ["BLOCKED"] * 3
    assert [c.agent for c in run.tool_calls] == ["research", "decision", "research"]
    assert h.java.requests == []


@pytest.mark.asyncio
async def test_the_trace_keeps_every_planner_attempt(make_workflow):
    from .test_planning import llm_plan

    bad = llm_plan()
    bad["steps"].append({"tool": "execution", "purpose": "buy now", "required_inputs": []})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=ScriptedLLM(bad, llm_plan()))
    d = (await h.run("分析 AAPL")).to_dict()
    attempts = [e["data"] for e in d["trace"]
                if e["kind"] == "plan" and e["data"].get("agent") == "planner" and "outcome" in e["data"]]
    assert [(a["planner"], a["outcome"]) for a in attempts] == [("llm", "rejected"), ("llm", "accepted")]
    assert attempts[0]["planner_output"] == bad and attempts[0]["validation"]["valid"] is False
    assert d["trace"][[e["data"] for e in d["trace"]].index(attempts[0])]["message"].startswith(
        "llm planner (scripted-test-model): rejected")
    planner = d["result"]["planner"]
    assert (planner["planner"], planner["model"], planner["fallback"]) == ("llm", "scripted-test-model", False)
    assert [{k: v for k, v in a.items() if k != "agent"} for a in attempts] == planner["attempts"]


# ------------------------------------------------------------------ approval path

@pytest.mark.asyncio
async def test_a_supported_trade_becomes_a_proposal_that_only_the_owner_can_approve(make_workflow):
    series = sell_setup(80)  # ends 2026-07-19; drawdown -28%: risk low, consensus SELL
    h = make_workflow(FakeJava({"AAPL": series}), today=date(2026, 7, 21))
    run = await h.run("分析 AAPL，考虑卖出 5 股")
    d = run.to_dict()
    assert d["result"]["recommendation"] == "SELL"
    trade = d["result"]["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and trade["mode"] == "PAPER"
    assert d["result"]["approval_required"] is True and d["action_level"] == "PROPOSE"
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]  # nothing executed
    order = trade["proposal"]["action"]
    assert order["reference_price"] == series[-1]["close"]  # bound to the observed close
    assert "approval" in [e["kind"] for e in d["trace"]]

    pid = trade["proposal"]["proposal_id"]
    denied = await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))
    assert denied.error_code == "APPROVAL_FORBIDDEN"
    done = await h.loop.execute_approved(pid, owner_ctx())
    assert done.status.value == "SUCCESS"
    assert done.result["execution"]["mode"] == "PAPER"
    assert done.result["execution"]["quantity"] == 5
    again = await h.loop.execute_approved(pid, owner_ctx())
    assert again.error_code == "APPROVAL_CONSUMED"


class _ProviderLLM(ScriptedLLM):
    def __init__(self, provider, model, *plans, **kw):
        super().__init__(*plans, **kw)
        self.provider, self.model = provider, model


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model", [("anthropic", "claude-opus-5"), ("deepseek", "deepseek-chat")])
async def test_no_provider_can_plan_or_decide_its_way_past_risk_and_approval(make_workflow, provider, model):
    from .test_planning import llm_plan

    execute_now = llm_plan()
    execute_now["steps"].append({"tool": "execution", "purpose": "sell 5 now", "required_inputs": []})
    llm = _ProviderLLM(provider, model, execute_now, execute_now,
                       decision=[{**GROUNDED, "decision": "SELL", "action_level": "EXECUTE"}])
    h = make_workflow(FakeJava({"AAPL": sell_setup(80)}), llm=llm, today=date(2026, 7, 21))
    d = (await h.run("分析 AAPL，考虑卖出 5 股")).to_dict()

    planner = d["result"]["planner"]
    assert [a["outcome"] for a in planner["attempts"]] == ["rejected", "rejected", "accepted"]
    assert all(a["provider"] == provider for a in planner["attempts"] if a["planner"] == "llm")
    assert planner["planner"] == "rule"  # the model's plans never ran
    assert agents_in_order(d)[-3:] == ["risk", "critic", "decision"]
    assert "action_level 'EXECUTE' is not allowed" in d["result"]["decision"]["fallback_reason"]
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    trade = d["result"]["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and d["action_level"] == "PROPOSE"
    pid = trade["proposal"]["proposal_id"]
    assert (await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))).error_code == "APPROVAL_FORBIDDEN"
    done = await h.loop.execute_approved(pid, owner_ctx())
    assert done.result["execution"]["mode"] == "PAPER"  # only after the owner approved


@pytest.mark.asyncio
async def test_a_plan_with_unknown_strategies_never_reaches_a_tool_and_risk_still_gates(make_workflow):
    from .test_planning import with_named_input

    placeholder = with_named_input("strategies", "registered")
    llm = _ProviderLLM("deepseek", "deepseek-chat", placeholder, placeholder)
    h = make_workflow(FakeJava({"AAPL": sell_setup(80)}), llm=llm, today=date(2026, 7, 21))
    d = (await h.run("分析 AAPL，考虑卖出 5 股")).to_dict()

    planner = d["result"]["planner"]
    assert [a["outcome"] for a in planner["attempts"]] == ["rejected", "rejected", "accepted"]
    assert planner["planner"] == "rule"
    assert all(c["status"] == "OK" for c in d["tool_calls"])  # no tool ever saw "registered"
    assert agents_in_order(d)[-3:] == ["risk", "critic", "decision"]
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    trade = d["result"]["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and d["action_level"] == "PROPOSE"
    pid = trade["proposal"]["proposal_id"]
    assert (await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))).error_code == "APPROVAL_FORBIDDEN"
    assert (await h.loop.execute_approved(pid, owner_ctx())).result["execution"]["mode"] == "PAPER"


# ------------------------------------------------ insufficient history for a backtest

SHORT = 30  # like the demo store: 30 daily bars, while ma_cross needs 31
SHORT_TODAY = date(2026, 6, 1)  # _bars start 2026-05-01, so 30 bars end 2026-05-30: fresh


def _plan_with_backtest(*, portfolio_id=None, strategy="ma_cross"):
    from .test_planning import inp

    steps = [{"tool": "market_data", "purpose": "history",
              "required_inputs": [inp("symbol", "AAPL"), inp("days", SHORT)]},
             {"tool": "signal", "purpose": "signals",
              "required_inputs": [inp("symbol", "AAPL"), inp("bars", from_step=1)]},
             {"tool": "backtest", "purpose": "a representative strategy",
              "required_inputs": [inp("symbol", "AAPL"), inp("bars", from_step=1),
                                  inp("strategy", strategy)]}]
    if portfolio_id is not None:
        steps.insert(0, {"tool": "portfolio", "purpose": "holdings",
                         "required_inputs": [inp("portfolio_id", portfolio_id)]})
        for step in steps[1:]:
            for i in step["required_inputs"]:
                if i["name"] == "bars":
                    i["from_step"] = 2
    return {"goal": "g", "reasoning_summary": "holdings, history, signals and one backtest",
            "steps": steps, "constraints": [], "clarification_question": ""}


async def _short_history_over_limit(make_workflow, llm, failures=None):
    java = FakeJava({"AAPL": calm_uptrend(SHORT), "MSFT": calm_uptrend(SHORT)}, failures=failures)
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK, llm=llm, today=SHORT_TODAY)
    await standard_preferences(h.memory)
    return h


def accepted_intent(run_dict):
    """The goal type of the plan that ran, as the trace recorded it."""
    plans = [e["data"]["plan"] for e in run_dict["trace"]
             if e["kind"] == "plan" and "plan" in (e.get("data") or {})]
    return plans[-1]["subject"]["intent"]


@pytest.mark.asyncio
async def test_a_backtest_without_enough_history_is_an_observation_and_the_review_continues(make_workflow):
    # The model adds a backtest to a sale the user asked for; the store cannot simulate it.
    llm = ScriptedLLM(_plan_with_backtest(portfolio_id=7))
    h = await _short_history_over_limit(make_workflow, llm)
    run = await h.run("我的AAPL仓位超过上限，减持 5 股 AAPL", portfolio_id=7)
    d = run.to_dict()

    assert d["result"]["planner"]["planner"] == "llm"  # the model's plan ran
    backtest = [c for c in d["tool_calls"] if c["tool"] == "backtest"]
    assert [(c["status"], c["error"]["code"]) for c in backtest] == [("ERROR", "INSUFFICIENT_DATA")]
    # the goal type is the user's, not the model's
    assert accepted_intent(d) == "position_review"
    assert ("goal type position_review kept from the goal text; the added steps are optional research"
            in d["result"]["planner"]["notes"])
    research = d["result"]["research"]
    assert research["backtests"] == []  # nothing was made up for the missing simulation
    assert "backtest/ma_cross skipped: ma_cross needs 31 bars, got 30" in research["skipped"]
    # risk, critic and decision still ran, and the sale is proposed, not executed
    assert agents_in_order(d) == ["research", "risk", "critic", "decision"]
    assert d["result"]["risk"]["status"] == "VETO"
    assert d["result"]["recommendation"] == "REDUCE_EXPOSURE"
    assert d["status"] == "SUCCESS" and d["action_level"] == "PROPOSE"
    trade = d["result"]["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and trade["mode"] == "PAPER"
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    assert d["data_provenance"]["synthetic_data_used"] is False
    # approval and paper execution are unchanged: owner only, once, on paper
    pid = trade["proposal"]["proposal_id"]
    assert (await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))).error_code == "APPROVAL_FORBIDDEN"
    done = await h.loop.execute_approved(pid, owner_ctx())
    assert done.result["execution"]["mode"] == "PAPER" and done.result["execution"]["quantity"] == 5
    assert (await h.loop.execute_approved(pid, owner_ctx())).error_code == "APPROVAL_CONSUMED"


@pytest.mark.asyncio
async def test_a_strategy_goal_still_fails_when_no_strategy_can_be_simulated(make_workflow):
    # When the user asked for a strategy, a missing simulation is the answer failing, not a detail.
    llm = ScriptedLLM(_plan_with_backtest())
    h = make_workflow(FakeJava({"AAPL": calm_uptrend(SHORT)}), llm=llm, today=SHORT_TODAY)
    d = (await h.run("给 AAPL 推荐一个策略")).to_dict()

    assert d["result"]["planner"]["planner"] == "llm"
    assert d["status"] == "FAILED" and d["error_code"] == "INSUFFICIENT_DATA"
    assert d["stop_reason"] == "no candidate strategy could be simulated"
    assert "risk" not in agents_in_order(d)
    assert "trade" not in (d["result"] or {})


@pytest.mark.asyncio
async def test_an_unknown_symbol_in_an_optional_step_fails_the_run(make_workflow):
    from .test_planning import inp, llm_plan

    raw = llm_plan()
    raw["steps"].append({"tool": "market_data", "purpose": "compare",
                         "required_inputs": [inp("symbol", "ZZZZ"), inp("days", 180)]})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=ScriptedLLM(raw))
    d = (await h.run("分析 AAPL，对比 ZZZZ")).to_dict()

    assert d["result"]["planner"]["planner"] == "llm"
    unknown = [c for c in d["tool_calls"] if c["tool"] == "market_data" and c["status"] != "OK"]
    assert [c["error"]["code"] for c in unknown] == ["NOT_FOUND"]
    assert d["status"] == "FAILED" and d["error_code"] == "NOT_FOUND"
    assert "risk" not in agents_in_order(d)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["FORBIDDEN", "INTERNAL", "INVALID_INPUT", "NOT_FOUND"])
async def test_a_real_error_in_an_optional_backtest_fails_the_run(make_workflow, monkeypatch, code):
    from agentrt.tools import BacktestTool
    from agentrt.tools.base import ToolErrorCode, ToolResult

    async def failing(self, params, ctx):
        return ToolResult.failure(ToolErrorCode(code), f"backtest {code}")

    monkeypatch.setattr(BacktestTool, "run", failing)
    llm = ScriptedLLM(_plan_with_backtest(portfolio_id=7, strategy="momentum"))
    h = await _short_history_over_limit(make_workflow, llm)
    d = (await h.run("我的AAPL仓位超过上限，减持 5 股 AAPL", portfolio_id=7)).to_dict()

    backtest = [c for c in d["tool_calls"] if c["tool"] == "backtest"]
    assert [c["error"]["code"] for c in backtest] == [code]  # not retried, not skipped
    assert d["status"] == "FAILED" and d["error_code"] == code
    assert d["stop_reason"] == f"step backtest failed: backtest {code}"
    assert "risk" not in agents_in_order(d)
    assert "trade" not in (d["result"] or {})


@pytest.mark.asyncio
async def test_an_optional_backtest_timing_out_after_its_retries_is_skipped(make_workflow, monkeypatch):
    from agentrt.tools import BacktestTool
    from agentrt.tools.base import ToolErrorCode, ToolResult

    async def slow(self, params, ctx):
        return ToolResult.failure(ToolErrorCode.TIMEOUT, "backtest timed out")

    monkeypatch.setattr(BacktestTool, "run", slow)
    llm = ScriptedLLM(_plan_with_backtest(portfolio_id=7, strategy="momentum"))
    h = await _short_history_over_limit(make_workflow, llm)
    d = (await h.run("我的AAPL仓位超过上限，减持 5 股 AAPL", portfolio_id=7)).to_dict()

    backtest = [c for c in d["tool_calls"] if c["tool"] == "backtest"]
    assert [c["attempt"] for c in backtest] == [1, 2, 3]  # the existing retry policy
    assert "backtest/momentum skipped: backtest timed out" in d["result"]["research"]["skipped"]
    assert d["status"] == "SUCCESS" and d["result"]["trade"]["status"] == "AWAITING_APPROVAL"


@pytest.mark.asyncio
async def test_a_symbol_analysis_stays_one_when_the_planner_adds_the_portfolio(make_workflow):
    from .test_planning import inp, llm_plan

    raw = llm_plan()  # market_data + signal for AAPL, plus the portfolio the user selected
    raw["steps"].insert(0, {"tool": "portfolio", "purpose": "context",
                            "required_inputs": [inp("portfolio_id", 7)]})
    for step in raw["steps"][1:]:
        for i in step["required_inputs"]:
            if i["name"] == "bars":
                i["from_step"] = 2
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()}), db,
                      risk_overrides=LOW_PORTFOLIO_RISK, llm=ScriptedLLM(raw))
    d = (await h.run("分析 AAPL", portfolio_id=7)).to_dict()

    assert d["result"]["planner"]["planner"] == "llm"
    assert accepted_intent(d) == "analyze_symbol"  # not a position review
    assert ("research", "portfolio") in calls_by_agent(d)  # the added step still ran
    assert d["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_a_portfolio_risk_goal_stays_one_when_the_planner_adds_a_symbol(make_workflow):
    from .test_planning import inp

    raw = {"goal": "g", "reasoning_summary": "holdings, then a closer look at AAPL",
           "steps": [{"tool": "portfolio", "purpose": "holdings",
                      "required_inputs": [inp("portfolio_id", 7)]},
                     {"tool": "market_data", "purpose": "AAPL history",
                      "required_inputs": [inp("symbol", "AAPL"), inp("days", 180)]},
                     {"tool": "signal", "purpose": "AAPL signals",
                      "required_inputs": [inp("symbol", "AAPL"), inp("bars", from_step=2)]}],
           "constraints": [], "clarification_question": ""}
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()}), db,
                      risk_overrides=LOW_PORTFOLIO_RISK, llm=ScriptedLLM(raw))
    d = (await h.run("看看组合风险", portfolio_id=7)).to_dict()

    assert d["result"]["planner"]["planner"] == "llm"
    assert accepted_intent(d) == "portfolio_risk"  # not a position review of AAPL
    assert {("research", "market_data"), ("research", "signal")} <= set(calls_by_agent(d))
    assert d["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_a_real_tool_error_on_a_required_step_still_fails_the_run(make_workflow):
    # The market-data backend is down for every attempt: no bars, so nothing may be analysed.
    llm = ScriptedLLM(_plan_with_backtest(portfolio_id=7))
    h = await _short_history_over_limit(make_workflow, llm, failures=["down"] * 3)
    d = (await h.run("我的AAPL仓位超过上限，减持 5 股 AAPL", portfolio_id=7)).to_dict()

    market = [c for c in d["tool_calls"] if c["tool"] == "market_data"]
    assert market and all(c["status"] == "ERROR" for c in market)
    assert d["status"] == "FAILED" and d["error_code"] == "UNAVAILABLE"
    assert "backtest" not in [c["tool"] for c in d["tool_calls"]]
    assert "risk" not in agents_in_order(d)
    assert "trade" not in (d["result"] or {})


# ------------------------------------------------------------------------ failures

@pytest.mark.asyncio
async def test_unauthenticated_and_foreign_portfolio_runs_fail_without_data(make_workflow):
    db = FakeDB({7: OTHER}, {7: [position("AAPL", 10)]})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), db)
    anonymous = await h.run("分析 AAPL", ctx=ToolContext(user_id=None))
    assert anonymous.error_code == "UNAUTHENTICATED" and anonymous.tool_calls == []
    foreign = await h.run("分析我的AAPL仓位", portfolio_id=7)
    assert foreign.error_code == "FORBIDDEN"
    assert set(foreign.result) == {"error", "agents", "planner"}
    assert await h.memory.list(OWNER, "decision") == []  # nothing remembered from a failed run


@pytest.mark.asyncio
async def test_memory_outage_is_reported_and_the_run_continues(make_workflow):
    class Down(InMemoryMemoryStore):
        async def _select(self, *a):
            raise ConnectionError("agent_memory unreachable")

        async def _insert(self, *a):
            raise ConnectionError("agent_memory unreachable")

    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), memory=Down())
    d = (await h.run("分析 AAPL")).to_dict()
    assert d["status"] == "SUCCESS"
    assert d["result"]["memory_used"]["available"] is False
    assert d["result"]["memory_written"] is None
    assert any(e["kind"] == "memory_write" and "not remembered" in e["message"] for e in d["trace"])


@pytest.mark.asyncio
async def test_a_price_smuggled_into_memory_is_never_used(make_workflow):
    memory = InMemoryMemoryStore()
    # a row that bypassed validation (e.g. written by hand into the table)
    memory._rows.append(MemoryItem(id=99, user_id=OWNER, memory_type="decision",
                                   content={"subject": "AAPL", "recommendation": "BUY",
                                            "reason": "old", "last_close": 1.0},
                                   source_task_id="old", importance=5,
                                   created_at=datetime.now(timezone.utc).isoformat()))
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), memory=memory)
    d = (await h.run("分析 AAPL")).to_dict()
    assert d["result"]["research"]["market"]["AAPL"]["last_close"] == calm_uptrend()[-1]["close"]
    retrieved = [e for e in d["trace"] if e["kind"] == "memory"][0]["data"]["memory"]
    assert retrieved["relevant"][0]["id"] == 99  # read as context ...
    assert all(s["kind"] != "memory" for s in d["data_provenance"]["sources"])  # ... never as data


# ---------------------------------------------------------------------------- API

@pytest.mark.asyncio
async def test_workflow_endpoint_requires_a_user_and_returns_the_contract(make_workflow):
    from agentrt.api import routes as agent_routes

    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    h.workflow.archive = MemoryArchive()
    body = agent_routes.WorkflowRequest(goal_text="分析 AAPL")
    with pytest.raises(HTTPException) as exc:
        await agent_routes.runtime_workflow(body, _request(None), workflow=h.workflow)
    assert exc.value.status_code == 401
    response = await agent_routes.runtime_workflow(body, _request(OWNER), workflow=h.workflow)
    d = json.loads(response.body)
    assert response.status_code == 200 and d["agent"] == "traderx.workflow"
    assert d["persisted"] is True and d["task_id"] in h.workflow.archive.saved

    unclear = await agent_routes.runtime_workflow(agent_routes.WorkflowRequest(goal_text="帮我看看"),
                                                  _request(OWNER), workflow=h.workflow)
    assert unclear.status_code == 422
    assert json.loads(unclear.body)["result"]["question"]


@pytest.mark.asyncio
async def test_memory_endpoints_validate_and_scope_to_the_caller():
    from agentrt.api import routes as agent_routes

    store = InMemoryMemoryStore()
    Pref = agent_routes.PreferenceRequest
    saved = await agent_routes.runtime_set_preference(
        Pref(key="max_single_position_pct", value=20), _request(OWNER), memory=store)
    assert saved["content"] == {"key": "max_single_position_pct", "value": 20.0}
    with pytest.raises(HTTPException) as exc:
        await agent_routes.runtime_set_preference(
            Pref(key="forbid_auto_trading", value=False), _request(OWNER), memory=store)
    assert exc.value.status_code == 422
    mine = await agent_routes.runtime_memory(_request(OWNER), memory=store)
    assert mine["preferences"]["max_single_position_pct"]["value"] == 20.0
    theirs = await agent_routes.runtime_memory(_request(OTHER), memory=store)
    assert theirs["preferences"] == {}
    with pytest.raises(HTTPException) as exc:
        await agent_routes.runtime_memory(_request(None), memory=store)
    assert exc.value.status_code == 401


# --------------------------------------------------------------- critic (structured)

@pytest.mark.asyncio
async def test_critic_review_is_structured_and_never_edits_results(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK)
    await standard_preferences(h.memory)
    d = (await h.run("分析我的AAPL仓位风险", portfolio_id=7)).to_dict()
    critic = next(r for r in d["result"]["agents"] if r["agent"] == "critic")["findings"]
    assert (critic["valid"], critic["verdict"]) == (False, "request_revision")
    assert set(critic) >= {"valid", "verdict", "issues", "notes", "reviewer"}
    assert not ({"recommendation", "decision", "trade"} & set(critic))  # it reviews, it does not decide
    issue = next(i for i in critic["issues"] if i["code"] == "POSITION_LIMIT")
    assert (issue["type"], issue["severity"]) == ("constraint_violation", "HIGH")
    assert issue["suggestion"] == "reduce exposure to within the limit"


@pytest.mark.asyncio
async def test_llm_critic_can_add_issues_but_not_remove_the_rule_findings(make_workflow):
    llm = ScriptedLLM(review=[{"issues": [
        {"type": "logic_gap", "severity": "LOW", "message": "the momentum view rests on one strategy",
         "suggestion": "check another window", "remedy_days": 0, "remedy_factors": ""}]}])
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK, llm=llm)
    await standard_preferences(h.memory)
    d = (await h.run("分析我的AAPL仓位风险", portfolio_id=7)).to_dict()
    issues = d["result"]["critic"]["issues"]
    assert [(i["code"], i["source"]) for i in issues] == [("POSITION_LIMIT", "rule"), ("LLM_REVIEW", "llm")]
    assert d["result"]["critic"]["reviewer"] == "rule+llm"


@pytest.mark.asyncio
async def test_llm_critic_citing_unknown_numbers_is_not_used(make_workflow):
    llm = ScriptedLLM(review=[{"issues": [
        {"type": "risk_conflict", "severity": "HIGH", "message": "AAPL will fall to 150 next week",
         "suggestion": "sell", "remedy_days": 0, "remedy_factors": ""}]}])
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm)
    d = (await h.run("分析 AAPL")).to_dict()
    critic = d["result"]["critic"]
    assert critic["reviewer"] == "rule" and critic["valid"] is True
    assert "numbers not in the reports" in critic["fallback_reason"]


# ------------------------------------------------------------------ decision (LLM)

GROUNDED = {"decision": "HOLD", "confidence": "high", "action_level": "PROPOSE",
            "rationale": ["the consensus score is 0.4 across 5 strategies, short of a directional call"]}


@pytest.mark.asyncio
async def test_a_grounded_llm_decision_is_accepted(make_workflow):
    llm = ScriptedLLM(decision=[GROUNDED])
    d = (await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")).to_dict()
    decision = d["result"]["decision"]
    assert (decision["decided_by"], decision["model"], decision["decision"]) == ("llm", "scripted-test-model", "HOLD")
    assert decision["confidence"] == "high" and decision["fallback_reason"] is None
    assert d["result"]["rationale"] == GROUNDED["rationale"]
    assert llm.calls == ["plan", "review", "decision"]  # plan and review fell back (nothing scripted)
    assert d["result"]["planner"]["fallback_reason"].startswith("model error")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,reason", [
    ({**GROUNDED, "rationale": ["fair value is 312.5, well above today"]}, "numbers not in the reports"),
    ({**GROUNDED, "decision": "BUY_MORE"}, "is not allowed here"),
    ({**GROUNDED, "action_level": "EXECUTE"}, "action_level 'EXECUTE' is not allowed"),
    ({**GROUNDED, "rationale": []}, "rationale needs 1-6 points"),
])
async def test_an_llm_decision_outside_the_rules_falls_back(make_workflow, answer, reason):
    llm = ScriptedLLM(decision=[answer])
    d = (await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")).to_dict()
    decision = d["result"]["decision"]
    assert decision["decided_by"] == "rule" and reason in decision["fallback_reason"]
    assert decision["decision"] == "HOLD"  # the rule decision on the same reports


@pytest.mark.asyncio
async def test_an_llm_cannot_buy_through_a_risk_veto(make_workflow):
    llm = ScriptedLLM(decision=[{**GROUNDED, "decision": "BUY", "confidence": "high"}])
    h = make_workflow(FakeJava({"TSLA": wild()}), llm=llm)
    d = (await h.run("分析 TSLA，买入 10 股")).to_dict()
    decision = d["result"]["decision"]
    assert decision["decided_by"] == "rule" and "'BUY' is not allowed here" in decision["fallback_reason"]
    assert d["result"]["recommendation"] == "NO_NEW_EXPOSURE"
    assert d["result"]["trade"]["status"] == "NOT_PROPOSED" and not h.loop.proposals._items


@pytest.mark.asyncio
async def test_llm_confidence_is_capped_by_an_unresolved_critic(make_workflow):
    short = calm_uptrend(22)
    llm = ScriptedLLM(decision=[{**GROUNDED, "rationale": ["only 3 strategies had a signal"]}])
    h = make_workflow(FakeJava({"AAPL": short}), llm=llm, today=date(2026, 5, 24))
    d = (await h.run("分析 AAPL")).to_dict()
    assert d["result"]["decision"]["decided_by"] == "llm"
    assert d["result"]["confidence"] == "low"
    assert any(n.startswith("confidence capped at low") for n in d["result"]["notes"])


# ----------------------------------------------------------- async, feedback, memory

@pytest.mark.asyncio
async def test_a_started_run_is_visible_while_running_and_finishes(make_workflow):
    import asyncio
    from agentrt.api.routes import run_http_status

    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    run = h.workflow.start("分析 AAPL", owner_ctx())
    assert run.status.value == "RUNNING" and run_http_status(run) == 202
    assert h.workflow.run_log.get(run.task_id, OWNER) is run
    assert [e.kind for e in run.trace] == ["goal"]
    for _ in range(200):
        if run.status.value != "RUNNING":
            break
        await asyncio.sleep(0.01)
    assert run.status.value == "SUCCESS" and run.to_dict()["trace"][-1]["kind"] == "final"


@pytest.mark.asyncio
async def test_feedback_is_remembered_and_retrieved_for_the_next_goal(make_workflow, monkeypatch):
    from agentrt.api import routes as agent_routes

    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    monkeypatch.setattr(agent_routes, "_run_log", h.workflow.run_log)
    archive = MemoryArchive()
    first = await h.run("分析 AAPL")
    body = agent_routes.FeedbackRequest(task_id=first.task_id, verdict="disagree", comment="仓位太重")
    saved = await agent_routes.runtime_feedback(body, _request(OWNER), memory=h.memory, archive=archive)
    assert saved["content"] == {"subject": "AAPL", "verdict": "disagree", "comment": "仓位太重",
                                "about_recommendation": first.result["recommendation"]}
    assert saved["source_task_id"] == first.task_id
    with pytest.raises(HTTPException) as exc:  # someone else's run cannot be judged
        await agent_routes.runtime_feedback(body, _request(OTHER), memory=h.memory, archive=archive)
    assert exc.value.status_code == 404

    second = (await h.run("再分析 AAPL")).to_dict()
    assert any("用户不同意 AAPL 的结论" in line for line in second["result"]["memory_used"]["lines"])
    notes = second["result"]["critic"]["notes"]
    assert any(n["code"] == "USER_DISAGREED_BEFORE" for n in notes)


@pytest.mark.asyncio
async def test_proposals_and_their_approval_are_remembered(make_workflow):
    from agentrt.api.routes import _remember_approval

    h = make_workflow(FakeJava({"AAPL": sell_setup(80)}), today=date(2026, 7, 21))
    run = await h.run("分析 AAPL，考虑卖出 5 股")
    proposals = await h.memory.list(OWNER, "proposal")
    assert proposals[0].content["status"] == "AWAITING_APPROVAL"
    assert "reference_price" not in proposals[0].content  # never a price in memory
    approved = await h.loop.execute_approved(run.result["trade"]["proposal"]["proposal_id"], owner_ctx())
    await _remember_approval(approved, OWNER, h.memory)
    latest = (await h.memory.list(OWNER, "proposal"))[0]
    assert latest.content["status"] == "APPROVED_PAPER_EXECUTED" and latest.source_task_id == run.task_id


@pytest.mark.asyncio
async def test_data_sources_are_summarised_by_kind(make_workflow):
    d = (await make_workflow(FakeJava({"AAPL": calm_uptrend()})).run("分析 AAPL")).to_dict()
    summary = {g["label"]: g["sources"] for g in d["result"]["data_summary"]}
    assert list(summary) == ["PostgreSQL", "Derived analytics"]
    assert summary["PostgreSQL"] == ["eod_data（AAPL，经 Java /market/eod）"]


@pytest.mark.asyncio
async def test_memory_notes_compare_views_with_views_only(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    await h.memory.add(OWNER, "decision", {"subject": "AAPL", "recommendation": "REDUCE_EXPOSURE",
                                           "reason": "over limit"}, "t-old")
    notes = (await h.run("分析 AAPL")).result["critic"]["notes"]
    assert not any(n["code"] == "DIFFERS_FROM_MEMORY" for n in notes)  # a decision is not a view
    await h.memory.add(OWNER, "decision", {"subject": "AAPL", "recommendation": "SELL", "reason": "x"}, "t-2")
    notes = (await h.run("分析 AAPL")).result["critic"]["notes"]
    assert [n["code"] for n in notes] == ["DIFFERS_FROM_MEMORY"]
