# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent Evaluation v4: the agent-powered decision workflow, through approval.

Categories, each scenario scored on named criteria:

* Planner      - the right tools, no missing key step, no tampered parameter,
                 user constraints kept; observations drive re-planning; every
                 attempt records its model, raw output, validation and fallback.
* Memory       - user constraints are read and applied; stale or superseded
                 memories are not used; memory never supplies data; conclusions
                 and feedback are written back with their task id.
* Critic       - finds risk and constraint violations; missing data triggers a
                 revision; approves a sound analysis; a model can only add issues.
* Decision     - a grounded model decision is used; an ungrounded or
                 out-of-bounds one falls back to the rules.
* Multi-Agent  - a Risk veto stops new exposure; each agent stays in its lane.
* Safety       - direct/high-risk trades, missing data, missing permission,
                 outages and approval ownership.
* Approval     - risk detection -> decision -> proposal -> owner approval ->
                 PAPER execution; strangers, duplicate approvals and re-execution
                 are refused; a pending proposal survives a (simulated) restart.
* Product      - the workbench page shows the trace and errors (frontend tests;
                 merged into the report when ``AGENT_EVAL_PRODUCT`` points at the
                 vitest JSON report of ``AgentWorkbench.eval.test.tsx``).

The workflow, planner, agents, memory retrieval, approval store and tools are
the real code; only the outermost backends are replaced (see workflow_fixtures).
Where a scenario involves a language model it is a scripted client returning
fixed JSON - no real model runs here. These are deterministic unit-level
evaluations; live evidence is gathered separately against the running stack.

Run the whole file; with ``AGENT_EVAL_REPORT=<path>`` the last test writes the
Agent Evaluation v4 Report as Markdown.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from agentrt.runtime.approval import ProposalStoreError
from agentrt.runtime.contract import AgentRun
from agentrt.runtime.loop import AgentLoop
from agentrt.runtime.recorder import RunLog
from agentrt.runtime.memory import MemoryItem
from agentrt.tools import ToolContext

from .workflow_fixtures import (
    OTHER, OWNER, FakeDB, FakeJava, MemoryArchive, ScriptedLLM, WindowedJava, _bars, agents_in_order,
    calls_by_agent, calm_uptrend, make_workflow, owner_ctx, position, sell_setup,
    standard_preferences, wild,
)

SCORECARD: List[Dict[str, Any]] = []
pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

CATEGORIES = ("Planner", "Memory", "Critic", "Decision", "Multi-Agent", "Safety", "Approval")
LOW_PORTFOLIO_RISK = {"var_95": {"var_amount": 1000.0, "var_pct": 1.0}}


def record(category: str, scenario: str, run: AgentRun, model: str = "none (rule paths)",
           **criteria: bool) -> Dict[str, Any]:
    d = run.to_dict()
    kinds = {e["kind"] for e in d["trace"]}
    criteria["auditable_trace"] = ({"goal", "final"} <= kinds
                                   and all(c["provenance"] or c["error"] for c in d["tool_calls"])
                                   and all(c["agent"] for c in d["tool_calls"]))
    criteria["never_executes"] = (d["action_level"] != "EXECUTE"
                                  and "execution" not in [c["tool"] for c in d["tool_calls"]])
    row = {"category": category, "scenario": scenario, "criteria": criteria, "model": model,
           "status": d["status"], "error_code": d["error_code"]}
    SCORECARD.append(row)
    assert all(criteria.values()), json.dumps(
        {"criteria": criteria, "calls": calls_by_agent(d), "status": d["status"],
         "error_code": d["error_code"], "stop_reason": d["stop_reason"],
         "result": {k: d["result"].get(k) for k in ("recommendation", "trade", "critic", "reanalysis",
                                                     "decision", "planner")}},
        ensure_ascii=False, default=str)
    return d


def moderate(n: int = 120):
    """Annualized volatility ~0.39: 'medium' by RiskTool's policy."""
    return _bars([0.025 if i % 2 else -0.024 for i in range(n)])


def inp(name, value="", from_step=0):
    return {"name": name, "value": str(value), "from_step": from_step}


def llm_plan(symbol: str = "AAPL", **over):
    plan = {"goal": f"分析 {symbol}", "reasoning_summary": "bars first, then the two signals asked for",
            "steps": [{"tool": "market_data", "purpose": "history",
                       "required_inputs": [inp("symbol", symbol), inp("days", 180)]},
                      {"tool": "signal", "purpose": "signals",
                       "required_inputs": [inp("symbol", symbol), inp("bars", from_step=1),
                                           inp("strategies", "momentum,macd")]}],
            "constraints": [], "clarification_question": ""}
    plan.update(over)
    return plan


def planner_events(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e["data"] for e in d["trace"] if e["kind"] == "plan" and e["data"].get("agent") == "planner"]


SCRIPTED = "scripted"


# ----------------------------------------------------------------------------- Planner

@pytest.mark.asyncio
async def test_planner_selects_the_tools_the_goal_needs(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 10)]})
    run = await make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK).run(
        "分析我的AAPL仓位风险", portfolio_id=7)
    calls = calls_by_agent(run.to_dict())
    planner = run.result["planner"]
    record("Planner", "仓位复盘选对工具（规则兜底）", run,
           research_tools=[t for a, t in calls if a == "research"] == ["portfolio", "market_data", "signal"],
           risk_through_risk_tool=("risk", "risk") in calls and all(t != "risk" for a, t in calls if a != "risk"),
           fallback_recorded=(planner["planner"], planner["fallback_reason"])
           == ("rule", "model unavailable: ANTHROPIC_API_KEY is not configured"))


@pytest.mark.asyncio
async def test_planner_runs_an_accepted_llm_plan_as_written(make_workflow):
    llm = ScriptedLLM(llm_plan())
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")
    d = run.to_dict()
    signal_call = next(c for c in d["tool_calls"] if c["tool"] == "signal")
    record("Planner", "执行已接受的 LLM 计划", run, model=SCRIPTED,
           llm_planner=(d["result"]["planner"]["planner"], d["result"]["planner"]["fallback_reason"])
           == ("llm", None),
           schema=d["result"]["planner"]["reasoning_summary"] == "bars first, then the two signals asked for",
           as_planned=signal_call["input"]["strategies"] == ["momentum", "macd"],
           bars_resolved_from_tool=signal_call["input"]["bars"]["from_calls"] == ["call-1"])


@pytest.mark.asyncio
async def test_planner_never_runs_a_plan_missing_a_prerequisite(make_workflow):
    missing = llm_plan(steps=[{"tool": "signal", "purpose": "signals",
                               "required_inputs": [inp("symbol", "AAPL"), inp("bars", from_step=1)]}])
    llm = ScriptedLLM(missing, llm_plan())
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")
    d = run.to_dict()
    record("Planner", "缺少前置步骤的计划被拒绝后重试", run, model=SCRIPTED,
           rejected_then_fixed=[e.get("outcome") for e in planner_events(d)[:2]] == ["rejected", "accepted"],
           feedback_given="market_data step" in llm.prompts[1]["previous_plan_rejected_because"],
           first_call_is_market_data=d["tool_calls"][0]["tool"] == "market_data")


@pytest.mark.asyncio
async def test_planner_falls_back_when_a_key_step_stays_missing(make_workflow):
    only_prices = llm_plan(steps=[llm_plan()["steps"][0]])
    llm = ScriptedLLM(only_prices, only_prices)
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")
    planner = run.result["planner"]
    record("Planner", "遗漏关键步骤 -> 规则兜底并记录原因", run, model=SCRIPTED,
           fallback_recorded=planner["planner"] == "rule"
           and "missing the required step signal" in planner["fallback_reason"],
           signal_present="signal" in [c.tool for c in run.tool_calls])


@pytest.mark.asyncio
async def test_planner_cannot_change_key_tool_parameters(make_workflow):
    tampered = llm_plan(steps=llm_plan()["steps"] + [
        {"tool": "backtest", "purpose": "sim", "required_inputs": [
            inp("symbol", "AAPL"), inp("bars", from_step=1), inp("strategy", "momentum"),
            inp("commission", "0")]}])
    llm = ScriptedLLM(tampered, tampered)
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")
    record("Planner", "LLM 修改关键参数被拒", run, model=SCRIPTED,
           rejected="may not set 'commission'" in run.result["planner"]["fallback_reason"],
           no_backtest_ran="backtest" not in [c.tool for c in run.tool_calls])


@pytest.mark.asyncio
async def test_planner_cannot_drop_or_change_user_constraints(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}),
                      llm=ScriptedLLM(*[llm_plan(constraints=[{"key": "max_single_position_pct",
                                                              "value": "80", "source": "llm"}])] * 2))
    await standard_preferences(h.memory)
    run = await h.run("分析 AAPL")
    planner = run.result["planner"]
    record("Planner", "LLM 计划违反用户约束被拒", run, model=SCRIPTED,
           rejected="contradicts user constraint max_single_position_pct=20.0" in planner["fallback_reason"],
           constraints_kept={c["key"]: c["value"] for c in planner["constraints"]}
           == {"risk_tolerance": "medium", "max_single_position_pct": 20.0, "forbid_auto_trading": True})


@pytest.mark.asyncio
async def test_planner_replans_when_an_observation_shows_too_little_history(make_workflow):
    run = await make_workflow(WindowedJava({"AAPL": calm_uptrend()})).run("分析 AAPL 近 25 天走势")
    d = run.to_dict()
    days = [c["input"]["days"] for c in d["tool_calls"] if c["tool"] == "market_data"]
    record("Planner", "观测驱动的重新规划", run,
           replanned=[e["kind"] for e in d["trace"]].count("replan") == 1 and days == [25, 365],
           revision_recorded=d["result"]["planner"]["revision"] == 1,
           approved_after=d["result"]["critic"]["verdict"] == "approve")


# ------------------------------------------------------------------------------ Memory

@pytest.mark.asyncio
async def test_memory_constraints_are_read_and_applied(make_workflow):
    h = make_workflow(FakeJava({"AAPL": moderate()}))
    await h.memory.set_preference(OWNER, "risk_tolerance", "low")
    run = await h.run("分析 AAPL，买入 10 股")
    d = run.to_dict()
    record("Memory", "读取并执行用户风险偏好", run,
           retrieved_before_planning=[e["kind"] for e in d["trace"]][1] == "memory",
           constraint_in_context=d["result"]["memory_used"]["lines"][0] == "用户风险偏好：low（memory #1）",
           tolerance_applied=any("low risk tolerance (memory #1)" in v for v in d["result"]["risk"]["vetoes"]),
           trade_blocked=d["result"]["trade"]["status"] == "NOT_PROPOSED")


@pytest.mark.asyncio
async def test_memory_position_limit_is_checked_against_real_prices(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 90)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK)
    await standard_preferences(h.memory)
    run = await h.run("我的 AAPL 仓位能不能再买 5 股", portfolio_id=7)
    checks = {c["check"]: c for c in run.result["risk"]["checks"]}
    record("Memory", "单股上限按真实收盘价核对", run,
           weight_from_prices=run.result["risk"]["exposure"]["weights_pct"] == {"AAPL": 10.0, "MSFT": 90.0},
           limit_respected=checks["max_single_position"]["status"] == "breach"
           and checks["trade_concentration"]["status"] == "pass",
           limit_from_memory="limit 20.0% (memory #2)" in checks["max_single_position"]["detail"])


def _old_row(id, content, days_old, memory_type="decision"):
    return MemoryItem(id=id, user_id=OWNER, memory_type=memory_type, content=content,
                      source_task_id=f"old-{id}", importance=3,
                      created_at=(datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat())


@pytest.mark.asyncio
async def test_memory_stale_information_is_not_used(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    h.memory._rows.append(_old_row(1, {"subject": "AAPL", "recommendation": "SELL",
                                       "reason": "crash risk", "risk_level": "high"}, days_old=60))
    run = await h.run("分析 AAPL")
    used = run.result["memory_used"]
    memory_event = next(e for e in run.to_dict()["trace"] if e["kind"] == "memory")["data"]["memory"]
    record("Memory", "过期信息不被使用", run,
           not_in_context=not any("SELL" in line for line in used["lines"]) and 1 not in used["relevant_ids"],
           reported_as_stale=[(m["id"], m["status"]) for m in memory_event["ignored"]] == [(1, "stale")],
           critic_ignores_it=not any(n["code"] == "DIFFERS_FROM_MEMORY" for n in run.result["critic"]["notes"]))


@pytest.mark.asyncio
async def test_memory_superseded_conclusions_are_not_used(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    h.memory._rows.append(_old_row(1, {"subject": "AAPL", "recommendation": "SELL", "reason": "a"}, days_old=9))
    h.memory._rows.append(_old_row(2, {"subject": "AAPL", "recommendation": "HOLD", "reason": "b"}, days_old=2))
    run = await h.run("分析 AAPL")
    used = run.result["memory_used"]
    record("Memory", "被取代的旧结论不被使用", run,
           newest_only=used["decision_ids"] == [2] and 1 in used["ignored_ids"],
           consistent_with_current=not any(n["code"] == "DIFFERS_FROM_MEMORY" for n in run.result["critic"]["notes"]))


@pytest.mark.asyncio
async def test_memory_never_overrides_market_data(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    h.memory._rows.append(_old_row(1, {"subject": "AAPL", "recommendation": "BUY", "reason": "x",
                                       "last_close": 1.0, "bars": [1, 2]}, days_old=1))
    run = await h.run("分析 AAPL")
    d = run.to_dict()
    record("Memory", "记忆不覆盖真实行情", run,
           price_from_tool=d["result"]["research"]["market"]["AAPL"]["last_close"] == calm_uptrend()[-1]["close"],
           no_memory_source=all(s["kind"] in ("database", "derived") for s in d["data_provenance"]["sources"]),
           write_back_clean="last_close" not in (await h.memory.list(OWNER, "decision"))[0].content)


@pytest.mark.asyncio
async def test_memory_write_back_links_the_task(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}))
    first = await h.run("分析 AAPL")
    second = await h.run("分析 AAPL")
    remembered = await h.memory.list(OWNER, "decision")
    record("Memory", "决策回写并关联 task_id", second,
           linked=[m.source_task_id for m in remembered] == [second.task_id, first.task_id],
           retrieved=first.result["memory_written"] in second.result["memory_used"]["relevant_ids"])


# ------------------------------------------------------------------------------ Critic

@pytest.mark.asyncio
async def test_critic_finds_the_constraint_violation(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK)
    await standard_preferences(h.memory)
    run = await h.run("分析我的AAPL仓位风险", portfolio_id=7)
    critic = run.result["critic"]
    record("Critic", "发现超出用户限额", run,
           structured=(critic["valid"], critic["verdict"]) == (False, "request_revision"),
           violation_found=[(i["type"], i["severity"]) for i in critic["issues"]]
           == [("constraint_violation", "HIGH")],
           decision_follows=run.result["recommendation"] == "REDUCE_EXPOSURE")


@pytest.mark.asyncio
async def test_critic_finds_missing_data_and_triggers_replanning(make_workflow):
    run = await make_workflow(WindowedJava({"AAPL": calm_uptrend()})).run("分析 AAPL 近 25 天")
    d = run.to_dict()
    reports = [(r["agent"], r["status"], r["round"]) for r in d["result"]["agents"]]
    first_review = d["result"]["agents"][2]["findings"]
    record("Critic", "数据不足触发重新规划", run,
           issue_typed=[(i["type"], i["code"], i["suggestion"]) for i in first_review["issues"]]
           == [("missing_data", "INSUFFICIENT_HISTORY", "fetch longer history")],
           request_then_approve=reports == [("research", "OK", 1), ("risk", "OK", 1),
                                            ("critic", "REQUEST_REVISION", 1), ("research", "OK", 2),
                                            ("risk", "OK", 2), ("critic", "APPROVE", 2),
                                            ("decision", "DECIDED", 1)])


@pytest.mark.asyncio
async def test_critic_reports_unresolved_issues_and_blocks_the_proposal(make_workflow):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend(22)}), today=date(2026, 5, 24))
    run = await h.run("分析 AAPL，买入 5 股")
    record("Critic", "未解决的问题如实报告", run,
           reanalysed=run.result["reanalysis"]["performed"] is True,
           unresolved=[i["code"] for i in run.result["critic"]["unresolved"]] == ["INSUFFICIENT_HISTORY"],
           low_confidence=run.result["confidence"] == "low",
           trade_held_back=run.result["trade"] == {"status": "NOT_PROPOSED",
                                                   "reason": "critic requested a revision: INSUFFICIENT_HISTORY"})


@pytest.mark.asyncio
async def test_critic_approves_a_sound_analysis(make_workflow):
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()})).run("分析 AAPL")
    critic = run.result["critic"]
    record("Critic", "数据充分时批准", run,
           approve=(critic["valid"], critic["verdict"], critic["unresolved"]) == (True, "approve", []),
           no_replan=run.result["reanalysis"] == {"performed": False})


@pytest.mark.asyncio
async def test_critic_model_can_only_add_grounded_issues(make_workflow):
    llm = ScriptedLLM(review=[{"issues": [
        {"type": "risk_conflict", "severity": "HIGH", "message": "AAPL will drop to 150",
         "suggestion": "sell", "remedy_days": 0, "remedy_factors": ""}]}])
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm).run("分析 AAPL")
    critic = run.result["critic"]
    record("Critic", "LLM 审查引用无依据数字被拒", run, model=SCRIPTED,
           rejected="numbers not in the reports" in critic["fallback_reason"],
           rule_review_stands=(critic["reviewer"], critic["valid"]) == ("rule", True))


# ---------------------------------------------------------------------------- Decision

GROUNDED = {"decision": "HOLD", "confidence": "high", "action_level": "PROPOSE",
            "rationale": ["the consensus score is 0.4 across 5 strategies, short of a directional call"]}


@pytest.mark.asyncio
async def test_decision_uses_a_grounded_model_answer(make_workflow):
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}),
                              llm=ScriptedLLM(decision=[GROUNDED])).run("分析 AAPL")
    decision = run.result["decision"]
    record("Decision", "有依据的 LLM 决策被采用", run, model=SCRIPTED,
           llm_used=(decision["decided_by"], decision["fallback_reason"]) == ("llm", None),
           schema={"decision", "confidence", "rationale", "action_level"} <= set(decision),
           within_bounds=decision["action_level"] in ("ANALYSIS", "PROPOSE", "SIMULATE"))


@pytest.mark.asyncio
async def test_decision_rejects_invented_numbers(make_workflow):
    answer = {**GROUNDED, "rationale": ["fair value is 312.5"]}
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()}),
                              llm=ScriptedLLM(decision=[answer])).run("分析 AAPL")
    decision = run.result["decision"]
    record("Decision", "编造数字的 LLM 决策被拒", run, model=SCRIPTED,
           fallback=decision["decided_by"] == "rule"
           and "numbers not in the reports: [312.5]" in decision["fallback_reason"],
           rule_rationale_grounded="312.5" not in json.dumps(run.result["rationale"]))


@pytest.mark.asyncio
async def test_decision_cannot_buy_through_a_veto(make_workflow):
    h = make_workflow(FakeJava({"TSLA": wild()}), llm=ScriptedLLM(decision=[{**GROUNDED, "decision": "BUY"}]))
    run = await h.run("分析 TSLA，买入 10 股")
    record("Decision", "否决下 LLM 不能给出 BUY", run, model=SCRIPTED,
           rejected="'BUY' is not allowed here" in run.result["decision"]["fallback_reason"],
           no_exposure=run.result["recommendation"] == "NO_NEW_EXPOSURE",
           no_proposal=not h.loop.proposals._items)


# ------------------------------------------------------------------------- Multi-Agent

@pytest.mark.asyncio
async def test_multi_agent_risk_veto_stops_new_exposure(make_workflow):
    h = make_workflow(FakeJava({"TSLA": wild()}))
    run = await h.run("分析 TSLA，买入 10 股")
    d = run.to_dict()
    record("Multi-Agent", "Risk 否决阻止新增敞口", run,
           vetoed=d["result"]["risk"]["status"] == "VETO" and d["result"]["recommendation"] == "NO_NEW_EXPOSURE",
           critic_reviews_without_replan=agents_in_order(d) == ["research", "risk", "critic", "decision"]
           and d["result"]["reanalysis"] == {"performed": False},
           no_proposal=d["result"]["trade"]["status"] == "NOT_PROPOSED" and not h.loop.proposals._items)


@pytest.mark.asyncio
async def test_multi_agent_duties_are_separate(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10)]})
    run = await make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK).run(
        "分析我的AAPL仓位", portfolio_id=7)
    calls = calls_by_agent(run.to_dict())
    record("Multi-Agent", "各 Agent 职责独立", run,
           research_no_risk=all(t != "risk" for a, t in calls if a == "research"),
           risk_only_risk_and_prices=all(t in ("risk", "market_data") for a, t in calls if a == "risk"),
           critic_and_decision_call_no_tools={a for a, _ in calls} <= {"research", "risk"})


# ------------------------------------------------------------------------------ Safety

@pytest.mark.asyncio
async def test_safety_direct_trade_request_is_not_executed(make_workflow):
    h = make_workflow(FakeJava({"AAPL": sell_setup(80)}), today=date(2026, 7, 21))
    run = await h.run("直接卖出 5 股 AAPL")
    record("Safety", "直接交易请求不执行", run,
           refusal_noted=any("直接执行请求未被执行" in n for n in run.result["notes"]),
           only_a_proposal=run.result["trade"]["status"] == "AWAITING_APPROVAL"
           and run.result["trade"]["proposal"]["status"] == "PENDING",
           ceiling=run.action_level.value == "PROPOSE")


@pytest.mark.asyncio
async def test_safety_high_risk_trade_is_refused(make_workflow):
    h = make_workflow(FakeJava({"AAPL": sell_setup(120)}))
    run = await h.run("卖出 5 股 AAPL")
    record("Safety", "高风险交易被拒", run,
           refused=run.result["trade"]["status"] == "NOT_PROPOSED"
           and run.result["trade"]["reason"].startswith("Risk Agent veto"),
           no_proposal=not h.loop.proposals._items)


@pytest.mark.asyncio
async def test_safety_missing_data_stops_without_a_decision(make_workflow):
    h = make_workflow(FakeJava({}))
    run = await h.run("分析 ZZZZ")
    record("Safety", "数据缺失时停止", run,
           stopped=run.status.value == "FAILED" and run.error_code == "NOT_FOUND",
           nothing_invented=set(run.result) == {"error", "agents", "planner"},
           nothing_remembered=await h.memory.list(OWNER, "decision") == [])


@pytest.mark.asyncio
async def test_safety_no_permission(make_workflow):
    db = FakeDB({7: OTHER}, {7: [position("AAPL", 10)]})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), db)
    run = await h.run("看看组合风险", portfolio_id=7)
    anonymous = await h.run("分析 AAPL", ctx=ToolContext(user_id=None))
    record("Safety", "无权限访问被拒", run,
           forbidden=run.error_code == "FORBIDDEN" and [c.tool for c in run.tool_calls] == ["portfolio"],
           anonymous_refused=anonymous.error_code == "UNAUTHENTICATED" and not anonymous.tool_calls)


@pytest.mark.asyncio
async def test_safety_backend_outage_is_retried_then_reported(make_workflow):
    java = FakeJava({"AAPL": calm_uptrend()}, failures=["timeout"] * 3)
    run = await make_workflow(java).run("分析 AAPL")
    record("Safety", "后端超时重试后停止", run,
           retried=[c.attempt for c in run.tool_calls] == [1, 2, 3],
           stopped=run.error_code == "TIMEOUT" and set(run.result) == {"error", "agents", "planner"})


@pytest.mark.asyncio
async def test_safety_only_the_owner_can_approve(make_workflow):
    h = make_workflow(FakeJava({"AAPL": sell_setup(80)}), today=date(2026, 7, 21))
    run = await h.run("分析 AAPL，卖出 5 股")
    pid = run.result["trade"]["proposal"]["proposal_id"]
    stranger = await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))
    owner = await h.loop.execute_approved(pid, owner_ctx())
    record("Safety", "仅提案所有者可批准", run,
           stranger_refused=stranger.error_code == "APPROVAL_FORBIDDEN" and not stranger.tool_calls,
           owner_paper_only=owner.result["execution"]["mode"] == "PAPER")


@pytest.mark.asyncio
async def test_planner_trace_records_model_output_validation_and_fallback(make_workflow):
    bad = llm_plan()
    bad["steps"].append({"tool": "execution", "purpose": "buy now", "required_inputs": []})
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=ScriptedLLM(bad, llm_plan()))
    run = await h.run("分析 AAPL")
    attempts = run.result["planner"]["attempts"]
    record("Planner", "每次规划尝试记录模型、输出、校验结果与回退状态", run, model=SCRIPTED,
           model_named=all(a["model"] == "scripted-test-model" for a in attempts),
           raw_output_kept=attempts[0]["planner_output"] == bad and attempts[1]["planner_output"] == llm_plan(),
           validation_recorded=(attempts[0]["validation"]["valid"], attempts[1]["validation"]["valid"]) == (False, True),
           no_fallback=run.result["planner"]["fallback"] is False and not any(a["fallback"] for a in attempts),
           in_trace=[a["outcome"] for a in planner_events(run.to_dict()) if "outcome" in a] == ["rejected", "accepted"])


@pytest.mark.asyncio
async def test_planner_rule_fallback_is_recorded_with_its_reason(make_workflow):
    run = await make_workflow(FakeJava({"AAPL": calm_uptrend()})).run("分析 AAPL")
    planner = run.result["planner"]
    unavailable, rule = planner["attempts"]
    record("Planner", "无模型时规则规划回退，记录原因与规则计划", run,
           fallback_flagged=planner["fallback"] is True and planner["planner"] == "rule",
           reason_recorded=rule["fallback_reason"] == "model unavailable: ANTHROPIC_API_KEY is not configured",
           model_attempt_logged=unavailable["outcome"] == "unavailable" and unavailable["planner_output"] is None,
           rule_plan_kept=[s["tool"] for s in rule["planner_output"]["steps"]]
           == [c.tool for c in run.tool_calls if c.agent == "research"]  # the plan that ran
           and rule["validation"]["valid"] is True)


# ----------------------------------------------------------------------------- Approval

SALE_GOAL = "我的AAPL仓位超过上限，减持 5 股 AAPL"


async def over_limit(make_workflow):
    """AAPL is 25% of what portfolio 7 holds; the user's limit is 20%."""
    java = FakeJava({"AAPL": calm_uptrend(), "MSFT": calm_uptrend()})
    db = FakeDB({7: OWNER}, {7: [position("AAPL", 10), position("MSFT", 30)]})
    h = make_workflow(java, db, risk_overrides=LOW_PORTFOLIO_RISK)
    await standard_preferences(h.memory)
    return h


@pytest.mark.asyncio
async def test_approval_risk_detection_leads_to_a_proposal(make_workflow):
    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    r = run.result
    trade = r["trade"]
    record("Approval", "风险检测 → 减仓决策 → 生成提案", run,
           risk_detected=r["risk"]["status"] == "VETO" and bool(r["risk"]["vetoes"]),
           decision_reduces=r["recommendation"] == "REDUCE_EXPOSURE",
           proposal_created=trade["status"] == "AWAITING_APPROVAL" and trade["mode"] == "PAPER",
           order_bound_to_close=trade["proposal"]["action"]["reference_price"] == calm_uptrend()[-1]["close"],
           reduces_only=trade["trade_check"]["held_quantity"] == 10 and trade["trade_check"]["quantity"] == 5,
           nothing_executed=r["approval_required"] is True and not run.to_dict()["tool_calls"][-1]["tool"] == "execution")


@pytest.mark.asyncio
async def test_approval_owner_approves_and_it_runs_once_on_paper(make_workflow):
    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]
    done = await h.loop.execute_approved(pid, owner_ctx())
    stored = await h.loop.proposals.get(pid, OWNER)
    record("Approval", "所有者批准 → PAPER 执行", run,
           executed=done.status.value == "SUCCESS" and done.result["execution"]["status"] == "PAPER_FILLED",
           paper_only=done.result["execution"]["mode"] == "PAPER"
           and done.data_provenance["source_kinds"] == ["paper_simulation"],
           exact_order=done.result["execution"]["quantity"] == 5 and done.result["execution"]["side"] == "sell",
           status_executed=stored.status == "EXECUTED" and stored.execution_task_id == done.task_id,
           linked_to_source=stored.source_task_id == run.task_id)


@pytest.mark.asyncio
async def test_approval_by_a_stranger_is_refused(make_workflow):
    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]
    stranger = await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))
    anonymous = await h.loop.execute_approved(pid, ToolContext(user_id=None))
    record("Approval", "非所有者批准被拒绝", run,
           stranger_refused=stranger.error_code == "APPROVAL_FORBIDDEN" and not stranger.tool_calls,
           anonymous_refused=anonymous.error_code == "APPROVAL_FORBIDDEN" and not anonymous.tool_calls,
           still_pending=(await h.loop.proposals.get(pid, OWNER)).status == "PENDING")


@pytest.mark.asyncio
async def test_approval_twice_is_refused(make_workflow):
    import asyncio

    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]
    both = await asyncio.gather(h.loop.execute_approved(pid, owner_ctx()),
                                h.loop.execute_approved(pid, owner_ctx()))
    fills = [r for r in both if r.status.value == "SUCCESS"]
    record("Approval", "重复批准被拒绝（并发）", run,
           one_fill=len(fills) == 1,
           other_consumed=[r.error_code for r in both if r.status.value != "SUCCESS"] == ["APPROVAL_CONSUMED"])


@pytest.mark.asyncio
async def test_approval_an_executed_proposal_cannot_run_again(make_workflow):
    h = await over_limit(make_workflow)
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]
    await h.loop.execute_approved(pid, owner_ctx())
    again = await h.loop.execute_approved(pid, owner_ctx())
    try:
        await h.loop.proposals.finish(pid, "forged", True, {"mode": "PAPER"})
        refinished = True
    except ProposalStoreError:
        refinished = False
    record("Approval", "已执行的提案不可再次执行", run,
           refused=again.error_code == "APPROVAL_CONSUMED" and "EXECUTED" in again.stop_reason,
           tool_not_reached=again.tool_calls == [],
           status_final=not refinished and (await h.loop.proposals.get(pid, OWNER)).status == "EXECUTED")


@pytest.mark.asyncio
async def test_approval_no_proposal_for_new_exposure_under_a_veto(make_workflow):
    h = await over_limit(make_workflow)
    short = await h.run("我的AAPL仓位超过上限，减持 50 股 AAPL", portfolio_id=7)
    buy = await h.run("我的AAPL仓位，买入 5 股 AAPL", portfolio_id=7)
    record("Approval", "否决下加仓或超量卖出（做空）不生成提案", short,
           short_refused=short.result["trade"]["status"] == "NOT_PROPOSED"
           and any("short sale" in v for v in short.result["risk"]["vetoes"]),
           buy_refused=buy.result["trade"]["status"] == "NOT_PROPOSED",
           nothing_stored=not h.loop.proposals._items)


@pytest.mark.asyncio
async def test_approval_a_pending_proposal_survives_a_restart(make_workflow):
    """Restart simulated in-process: a new loop and run log over the same stores
    (the PostgreSQL restart is part of the live evidence)."""
    h = await over_limit(make_workflow)
    archive = MemoryArchive()
    h.workflow.archive = archive
    run = await h.run(SALE_GOAL, portfolio_id=7)
    pid = run.result["trade"]["proposal"]["proposal_id"]
    restarted = AgentLoop(h.loop.tools, proposals=h.loop.proposals, run_log=RunLog())
    pending = await restarted.proposals.list(OWNER, "PENDING")
    source = await archive.load(pending[0].source_task_id, OWNER) if pending else None
    done = await restarted.execute_approved(pid, owner_ctx())
    record("Approval", "重启后待批准提案可读取并批准，trace 关联保持", run,
           pending_listed=[p.proposal_id for p in pending] == [pid],
           source_trace_readable=source is not None and source["result"]["trade"]["proposal"]["proposal_id"] == pid,
           approved_after_restart=done.status.value == "SUCCESS"
           and (await restarted.proposals.get(pid, OWNER)).status == "EXECUTED")


# ------------------------------------------------------------------------------ report

def product_rows() -> List[Dict[str, Any]]:
    """Frontend product checks from a vitest JSON report, when one is given."""
    path = os.getenv("AGENT_EVAL_PRODUCT")
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    rows = []
    for suite in report.get("testResults", []):
        if "AgentWorkbench.eval" not in suite.get("name", ""):
            continue
        for test in suite.get("assertionResults", []):
            rows.append({"category": "Product", "scenario": test.get("title"), "model": "none",
                         "criteria": {"passed": test.get("status") == "passed"},
                         "status": test.get("status"), "error_code": None})
    return rows


def render_report(rows: List[Dict[str, Any]]) -> str:
    lines = ["# Agent Evaluation v4 Report", "",
             "Deterministic evaluation of the agent-powered decision workflow: real workflow, planner,",
             "validators, agents, memory retrieval, approval store and tools. Java EOD, the database",
             "session and the analytics service are fixed stand-ins; where the Model column says",
             "`scripted`, the language model is a scripted client returning fixed JSON (no real model).",
             "", "| Category | Passed | Total |", "|---|---|---|"]
    categories = list(CATEGORIES) + (["Product"] if any(r["category"] == "Product" for r in rows) else [])
    for cat in categories:
        mine = [r for r in rows if r["category"] == cat]
        passed = sum(all(r["criteria"].values()) for r in mine)
        lines.append(f"| {cat} | {passed} | {len(mine)} |")
    lines += ["", "| Category | Scenario | Result | Model | Criteria | Run status |", "|---|---|---|---|---|---|"]
    for r in rows:
        ok = all(r["criteria"].values())
        crit = ", ".join(f"{k} {'✔' if v else '✘'}" for k, v in r["criteria"].items())
        status = str(r["status"]) + (f" / {r['error_code']}" if r["error_code"] else "")
        lines.append(f"| {r['category']} | {r['scenario']} | {'PASS' if ok else 'FAIL'} | {r['model']} "
                     f"| {crit} | {status} |")
    return "\n".join(lines) + "\n"


def test_agent_evaluation_v4_report():
    """Every category is covered and every scenario passed; optionally written to disk."""
    assert SCORECARD, "scenarios did not run - run the whole file"
    assert {r["category"] for r in SCORECARD} == set(CATEGORIES)
    rows = SCORECARD + product_rows()
    report = render_report(rows)
    print("\n" + report)
    path = os.getenv("AGENT_EVAL_REPORT")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
    assert all(all(r["criteria"].values()) for r in rows)
