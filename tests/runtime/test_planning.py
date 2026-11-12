# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Planner v3: one plan schema for LLM and rule plans, one validator for both.

The language model is a scripted client (unit tests); the SDK client itself is
exercised against a mocked HTTP transport - no real model is called here.
"""

from __future__ import annotations

import json

import httpx2
import pytest

from agentrt.runtime.llm import AnthropicStructuredLLM, LLMError, default_llm
from agentrt.runtime.memory import InMemoryMemoryStore, MemoryContext
from agentrt.runtime.planning import (
    Clarification, Plan, PlanValidationError, PlanningService, goal_facts, rule_plan, validate_plan,
)
from agentrt.tools import default_tools

from .workflow_fixtures import ScriptedLLM

pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

TOOLS = default_tools()


def inp(name, value="", from_step=0):
    return {"name": name, "value": str(value), "from_step": from_step}


def llm_plan(symbol="AAPL", **over):
    plan = {"goal": f"分析 {symbol}", "reasoning_summary": "read the bars, then compute signals",
            "steps": [{"tool": "market_data", "purpose": "history",
                       "required_inputs": [inp("symbol", symbol), inp("days", 180)]},
                      {"tool": "signal", "purpose": "signals",
                       "required_inputs": [inp("symbol", symbol), inp("bars", from_step=1)]}],
            "constraints": [], "clarification_question": ""}
    plan.update(over)
    return plan


def tools_of(plan: Plan):
    return [s.tool for s in plan.steps]


async def memory_with(**prefs):
    store = InMemoryMemoryStore()
    for key, value in prefs.items():
        await store.set_preference(2, key, value)
    return await store.retrieve_context("", 2)


def validate(raw, text="分析 AAPL", memory=None, portfolio=None):
    return validate_plan(raw, TOOLS, "llm", goal_facts(text, None, portfolio), memory)


# ------------------------------------------------------------------------ plan schema

@pytest.mark.asyncio
@pytest.mark.parametrize("goal,portfolio,intent,tools", [
    ("分析我的AAPL仓位风险", 2, "position_review", ["portfolio", "market_data", "signal"]),
    ("分析 AAPL", None, "analyze_symbol", ["market_data", "signal", "factor"]),
    ("给 MSFT 推荐一个策略", None, "strategy_suggestion",
     ["market_data", "backtest", "backtest", "backtest", "backtest"]),
    ("看看组合风险", 2, "portfolio_risk", ["portfolio"]),
])
async def test_rule_plan_uses_the_shared_schema_and_records_the_fallback(goal, portfolio, intent, tools):
    memory = await memory_with(max_single_position_pct=20)
    plan, attempts = await PlanningService(TOOLS).plan(goal, memory, portfolio_hint=portfolio)
    d = plan.to_dict()
    assert set(d) >= {"goal", "reasoning_summary", "steps", "constraints"}
    assert all(set(s) == {"tool", "purpose", "required_inputs"} for s in d["steps"])
    assert (plan.intent, tools_of(plan)) == (intent, tools)
    assert (d["planner"], d["fallback_reason"]) == ("rule", "model unavailable: ANTHROPIC_API_KEY is not configured")
    assert d["constraints"] == [{"key": "max_single_position_pct", "value": 20.0, "source": "memory #1"}]
    assert attempts[0]["outcome"] == "unavailable"
    bars_inputs = [i for s in d["steps"] for i in s["required_inputs"] if i["name"] == "bars"]
    assert all(i.get("from_step") for i in bars_inputs)  # bars always referenced, never supplied


@pytest.mark.asyncio
async def test_rule_planner_asks_when_the_goal_names_nothing():
    plan, _ = await PlanningService(TOOLS).plan("帮我看看", MemoryContext())
    assert isinstance(plan, Clarification)
    plan, _ = await PlanningService(TOOLS).plan("分析我的 AAPL 仓位", MemoryContext())
    assert isinstance(plan, Clarification) and "portfolio" in plan.reason


def test_goal_facts_come_from_the_text_not_the_model():
    facts = goal_facts("直接卖出 5 股 TSLA，看近 60 天", None, 3)
    assert facts.symbols == ("TSLA",) and facts.portfolio_id == 3
    assert facts.trade == {"side": "sell", "quantity": 5.0}
    assert facts.direct_execution_requested is True and facts.days == 60


# ------------------------------------------------------------------------- validation

def _mutate_steps(fn):
    raw = llm_plan()
    fn(raw)
    return raw


@pytest.mark.parametrize("text,portfolio,intent", [
    ("分析我的AAPL仓位风险", 2, "position_review"),
    ("分析 AAPL", None, "analyze_symbol"),
    ("给 AAPL 推荐一个策略", None, "strategy_suggestion"),
])
def test_a_backtest_the_model_adds_does_not_change_the_goal_type(text, portfolio, intent):
    raw = llm_plan()
    raw["steps"].append({"tool": "backtest", "purpose": "sim", "required_inputs": [
        inp("symbol", "AAPL"), inp("bars", from_step=1), inp("strategy", "momentum")]})
    if portfolio is not None:
        raw["steps"].insert(0, {"tool": "portfolio", "purpose": "p",
                                "required_inputs": [inp("portfolio_id", portfolio)]})
        for s in raw["steps"][1:]:
            for i in s["required_inputs"]:
                if i["name"] == "bars":
                    i["from_step"] = 2
    plan = validate(raw, text=text, portfolio=portfolio)
    assert plan.intent == intent
    assert "backtest" in tools_of(plan)  # the step itself is kept


def test_a_backtest_in_another_goal_does_not_relax_the_required_steps():
    raw = llm_plan()
    raw["steps"][1] = {"tool": "backtest", "purpose": "sim", "required_inputs": [
        inp("symbol", "AAPL"), inp("bars", from_step=1), inp("strategy", "momentum")]}
    with pytest.raises(PlanValidationError, match="missing the required step signal"):
        validate(raw, text="分析 AAPL")


def _plan(*tools, symbol="AAPL", portfolio_id=2):
    """A plan from the given optional tools; bars always come from its market_data step."""
    steps = []
    if "portfolio" in tools:
        steps.append({"tool": "portfolio", "purpose": "p", "required_inputs": [inp("portfolio_id", portfolio_id)]})
    if "market_data" in tools:
        steps.append({"tool": "market_data", "purpose": "m",
                       "required_inputs": [inp("symbol", symbol), inp("days", 180)]})
        bars = len(steps)
        for tool, extra in (("signal", []), ("factor", []), ("backtest", [inp("strategy", "momentum")])):
            if tool in tools:
                steps.append({"tool": tool, "purpose": tool, "required_inputs": [
                    inp("symbol", symbol), inp("bars", from_step=bars)] + extra})
    return {"goal": "g", "reasoning_summary": "r", "steps": steps, "constraints": [],
            "clarification_question": ""}


@pytest.mark.parametrize("text,portfolio,tools,intent", [
    ("分析 AAPL", 2, ("portfolio", "market_data", "signal"), "analyze_symbol"),
    ("看看组合风险", 2, ("portfolio", "market_data", "signal"), "portfolio_risk"),
    ("分析我的AAPL仓位风险", 2, ("portfolio", "market_data", "signal", "backtest"), "position_review"),
])
def test_the_planner_may_add_steps_but_keeps_the_users_goal_type(text, portfolio, tools, intent):
    plan = validate(_plan(*tools), text=text, portfolio=portfolio)
    assert plan.intent == intent
    assert tools_of(plan) == list(tools)  # every added step is kept and runs
    assert f"goal type {intent} kept from the goal text; the added steps are optional research" in plan.notes


def test_no_combination_of_added_steps_changes_the_goal_type():
    import itertools

    goals = [("分析我的AAPL仓位风险", 2), ("分析 AAPL", None), ("分析 AAPL", 2),
             ("给 AAPL 推荐一个策略", None), ("看看组合风险", 2)]
    optional = ("portfolio", "market_data", "signal", "factor", "backtest")
    accepted = 0
    for text, portfolio in goals:
        facts = goal_facts(text, None, portfolio)
        for n in range(1, len(optional) + 1):
            for tools in itertools.combinations(optional, n):
                try:
                    plan = validate_plan(_plan(*tools), TOOLS, "llm", facts, None)
                except PlanValidationError:
                    continue  # the validator's own rules; not relaxed here
                accepted += 1
                assert plan.intent == facts.expected_intent, (text, portfolio, tools, plan.intent)
    assert accepted >= 16  # the check really covered plans of every goal type


def test_keeping_the_goal_type_does_not_relax_its_required_steps():
    # a position review still needs the portfolio and the signal, whatever else the model adds
    for tools in (("market_data", "signal", "backtest"), ("portfolio", "market_data", "backtest")):
        with pytest.raises(PlanValidationError, match="missing the required step"):
            validate(_plan(*tools), text="分析我的AAPL仓位风险", portfolio=2)


@pytest.mark.parametrize("mutate,message", [
    (lambda p: p["steps"].append({"tool": "execution", "purpose": "buy", "required_inputs": []}),
     "execution cannot be planned"),
    (lambda p: p["steps"][1]["required_inputs"].__setitem__(1, inp("bars", "[190.1, 191.2]")),
     "may not supply market data"),
    (lambda p: p["steps"][1]["required_inputs"].append(inp("close", "190")), "may not set 'close'"),
    (lambda p: p["steps"].append({"tool": "backtest", "purpose": "sim", "required_inputs": [
        inp("symbol", "AAPL"), inp("bars", from_step=1), inp("strategy", "momentum"),
        inp("commission", "0")]}), "may not set 'commission'"),
    (lambda p: p["steps"].append({"tool": "signal", "purpose": "x", "required_inputs": [
        inp("symbol", "AAPL"), inp("bars", from_step=2)]}), "bars must come from an earlier market_data step"),
    (lambda p: p["steps"].pop(0), "bars must come from an earlier market_data step"),
    (lambda p: p["steps"].pop(1), "missing the required step signal"),
    (lambda p: p["steps"].append({"tool": "shell", "purpose": "x", "required_inputs": []}),
     "unknown or forbidden tool"),
    (lambda p: p["steps"][0].update(purpose=""), "needs a purpose"),
    (lambda p: p["steps"][0]["required_inputs"].__setitem__(1, inp("days", 99999)), "within 5..3650"),
    (lambda p: p.update(reasoning_summary=""), "needs a reasoning_summary"),
])
def test_validator_rejects_unsafe_or_incomplete_plans(mutate, message):
    with pytest.raises(PlanValidationError, match=message):
        validate(_mutate_steps(mutate))


def test_validator_keeps_goal_facts():
    other_symbol = llm_plan(symbol="TSLA")
    with pytest.raises(PlanValidationError, match="may not switch to TSLA"):
        validate(other_symbol, text="分析 AAPL")
    other_portfolio = llm_plan()
    other_portfolio["steps"].insert(0, {"tool": "portfolio", "purpose": "p",
                                        "required_inputs": [inp("portfolio_id", 9)]})
    for s in other_portfolio["steps"][2:]:
        for i in s["required_inputs"]:
            if i["name"] == "bars":
                i["from_step"] = 2
    with pytest.raises(PlanValidationError, match="portfolio the user selected \\(2\\)"):
        validate(other_portfolio, text="分析我的 AAPL 仓位", portfolio=2)


def test_validator_requires_what_the_goal_text_asks_for():
    # the user asked about their position; a plan that skips the portfolio is incomplete
    with pytest.raises(PlanValidationError, match="missing the required step portfolio"):
        validate(llm_plan(), text="分析我的AAPL仓位风险", portfolio=2)


@pytest.mark.asyncio
async def test_validator_enforces_user_constraints():
    memory = await memory_with(max_single_position_pct=20, forbid_auto_trading=True)
    contradicts = llm_plan(constraints=[{"key": "max_single_position_pct", "value": "50", "source": "x"}])
    with pytest.raises(PlanValidationError, match="contradicts user constraint max_single_position_pct=20.0"):
        validate(contradicts, memory=memory)
    invented = llm_plan(constraints=[{"key": "risk_tolerance", "value": "high", "source": "x"}])
    with pytest.raises(PlanValidationError, match="memory holds no such constraint"):
        validate(invented, memory=memory)
    omitted = validate(llm_plan(), memory=memory)
    assert {c["key"] for c in omitted.constraints} == {"max_single_position_pct", "forbid_auto_trading"}
    assert any("restored from memory" in n for n in omitted.notes)


def test_validator_drops_risk_steps_because_the_risk_agent_owns_risk():
    raw = llm_plan()
    raw["steps"].append({"tool": "risk", "purpose": "risk", "required_inputs": [inp("symbol", "AAPL")]})
    plan = validate(raw)
    assert "risk" not in tools_of(plan) and any("Risk Agent" in n for n in plan.notes)


def with_named_input(name, value, tool="signal"):
    raw = llm_plan()
    if tool == "signal":
        raw["steps"][1]["required_inputs"].append(inp(name, value))
    else:
        raw["steps"].append({"tool": tool, "purpose": tool, "required_inputs": [
            inp("symbol", "AAPL"), inp("bars", from_step=1), inp(name, value)]})
    return raw


# placeholders the DeepSeek planner produced in a real run instead of registry names
@pytest.mark.parametrize("name,value,tool", [
    ("strategies", "registered", "signal"),
    ("strategies", "all_registered", "signal"),
    ("strategies", "momentum,all", "signal"),
    ("factors", "all", "factor"),
    ("factors", "all_registered", "factor"),
    ("strategy", "registered", "backtest"),
])
def test_validator_rejects_names_outside_the_registries(name, value, tool):
    with pytest.raises(PlanValidationError, match=f"unknown {name} .*use names from allowed_values"):
        validate(with_named_input(name, value, tool))


def test_validator_accepts_registry_names():
    plan = validate(with_named_input("strategies", "macd,turtle"))
    assert [s.arguments for s in plan.steps if s.tool == "signal"][0]["strategies"] == ["macd", "turtle"]
    plan = validate(with_named_input("factors", "rsi_14d", "factor"))
    assert [s.arguments for s in plan.steps if s.tool == "factor"][0]["factors"] == ["rsi_14d"]


# ------------------------------------------------------------------------ LLM planner

@pytest.mark.asyncio
async def test_llm_plan_is_accepted_and_sees_retrieved_memory():
    memory = await memory_with(max_single_position_pct=20)
    llm = ScriptedLLM(llm_plan())
    plan, attempts = await PlanningService(TOOLS, llm).plan("分析 AAPL", memory)
    assert (plan.planner, plan.model, plan.fallback_reason) == ("llm", "scripted-test-model", None)
    assert plan.reasoning_summary == "read the bars, then compute signals"
    assert attempts[0]["outcome"] == "accepted"
    prompt = llm.prompts[0]
    assert prompt["relevant_memories"] == ["单一持仓上限：20%（memory #1）"]
    assert prompt["user_constraints"] == [{"key": "max_single_position_pct", "value": "20.0",
                                           "source": "memory #1"}]
    assert {t["name"] for t in prompt["tools"]} == {"market_data", "portfolio", "signal", "factor", "backtest"}


@pytest.mark.asyncio
async def test_the_planner_sees_the_names_each_tool_accepts_and_they_run():
    from agentrt.quant.factor_agent import FactorAgent
    from agentrt.ports.strategies import strategy_names

    llm = ScriptedLLM(with_named_input("strategies", "macd,rsi_reversion"))
    plan, _ = await PlanningService(TOOLS, llm).plan("分析 AAPL", MemoryContext())
    catalog = {t["name"]: t.get("allowed_values") for t in llm.prompts[0]["tools"]}
    assert catalog["signal"] == {"strategies": strategy_names()}
    assert catalog["backtest"] == {"strategy": strategy_names()}
    assert catalog["factor"] == {"factors": sorted(FactorAgent.FACTOR_DEFINITIONS)}
    assert catalog["market_data"] is None and catalog["portfolio"] is None
    # a name taken from the catalog passes the validator and the tool itself
    signal = [s for s in plan.steps if s.tool == "signal"][0]
    assert (plan.planner, signal.arguments["strategies"]) == ("llm", ["macd", "rsi_reversion"])
    TOOLS.get("signal").Input.model_validate({**signal.arguments, "bars": [{"date": "2026-01-01", "close": 1.0}] * 2})


@pytest.mark.asyncio
async def test_an_unknown_strategy_gets_the_allowed_names_back_and_a_second_try():
    llm = ScriptedLLM(with_named_input("strategies", "registered"), with_named_input("strategies", "macd"))
    plan, attempts = await PlanningService(TOOLS, llm).plan("分析 AAPL", MemoryContext())
    assert [a["outcome"] for a in attempts] == ["rejected", "accepted"] and plan.planner == "llm"
    feedback = llm.prompts[1]["previous_plan_rejected_because"]
    assert "unknown strategies ['registered']" in feedback and "macd" in feedback


@pytest.mark.asyncio
async def test_unknown_names_twice_fall_back_to_a_rule_plan_with_registry_names():
    bad = with_named_input("strategies", "all_registered")
    plan, attempts = await PlanningService(TOOLS, ScriptedLLM(bad, bad)).plan("分析 AAPL", MemoryContext())
    assert [a["outcome"] for a in attempts] == ["rejected", "rejected", "accepted"]
    assert plan.planner == "rule" and "unknown strategies ['all_registered']" in plan.fallback_reason
    from agentrt.quant.factor_agent import FactorAgent

    signal = [s for s in plan.steps if s.tool == "signal"][0]
    assert "strategies" not in signal.arguments  # the tool's default set, all registered
    factors = [n for s in plan.steps if s.tool == "factor" for n in s.arguments["factors"]]
    assert factors and set(factors) <= set(FactorAgent.FACTOR_DEFINITIONS)


@pytest.mark.asyncio
async def test_llm_may_infer_a_symbol_the_text_does_not_spell_out():
    llm = ScriptedLLM(llm_plan(goal="分析苹果公司"))
    plan, _ = await PlanningService(TOOLS, llm).plan("分析苹果公司的风险", MemoryContext())
    assert plan.planner == "llm" and plan.symbol == "AAPL"
    assert any("inferred by the planner" in n for n in plan.notes)


@pytest.mark.asyncio
async def test_rejected_llm_plan_gets_feedback_and_a_second_try():
    bad = llm_plan()
    bad["steps"].append({"tool": "execution", "purpose": "buy now", "required_inputs": []})
    llm = ScriptedLLM(bad, llm_plan())
    plan, attempts = await PlanningService(TOOLS, llm).plan("分析 AAPL", MemoryContext())
    assert plan.planner == "llm" and [a["outcome"] for a in attempts] == ["rejected", "accepted"]
    assert "execution cannot be planned" in llm.prompts[1]["previous_plan_rejected_because"]


@pytest.mark.asyncio
@pytest.mark.parametrize("answers,outcomes,reason", [
    ((LLMError("HTTP 529 overloaded"),), ["model_error"], "model error: HTTP 529 overloaded"),
    ((llm_plan(steps=[]), llm_plan(steps=[])), ["rejected", "rejected"],
     "LLM plan rejected by the validator: plan needs at least one step"),
])
async def test_llm_failure_falls_back_to_rules_and_records_why(answers, outcomes, reason):
    plan, attempts = await PlanningService(TOOLS, ScriptedLLM(*answers)).plan("分析 AAPL", MemoryContext())
    assert (plan.planner, plan.fallback_reason) == ("rule", reason)
    assert [a["outcome"] for a in attempts] == outcomes + ["accepted"]
    assert attempts[-1]["fallback_reason"] == reason


@pytest.mark.asyncio
async def test_every_planner_attempt_records_model_output_validation_and_fallback():
    bad = llm_plan()
    bad["steps"].append({"tool": "execution", "purpose": "buy now", "required_inputs": []})
    plan, (rejected, accepted) = await PlanningService(TOOLS, ScriptedLLM(bad, llm_plan())).plan(
        "分析 AAPL", MemoryContext())
    assert (rejected["model"], rejected["outcome"], rejected["planner_output"]) == (
        "scripted-test-model", "rejected", bad)
    assert rejected["validation"]["valid"] is False
    assert "execution cannot be planned" in rejected["validation"]["errors"][0]
    assert (accepted["outcome"], accepted["planner_output"], accepted["fallback"]) == ("accepted", llm_plan(), False)
    assert accepted["validation"] == {"valid": True, "errors": [], "adjustments": plan.notes}
    assert accepted["call"]["model"] == "scripted-test-model"


@pytest.mark.asyncio
async def test_a_rule_fallback_records_the_rule_plan_it_ran_and_why():
    plan, (unavailable, rule) = await PlanningService(TOOLS).plan("分析 AAPL", MemoryContext())
    reason = "model unavailable: ANTHROPIC_API_KEY is not configured"
    assert (unavailable["outcome"], unavailable["fallback"], unavailable["detail"]) == ("unavailable", True, reason)
    assert (rule["planner"], rule["fallback"], rule["fallback_reason"]) == ("rule", True, reason)
    assert [s["tool"] for s in rule["planner_output"]["steps"]] == [s.tool for s in plan.steps]
    assert rule["validation"]["valid"] is True


@pytest.mark.asyncio
async def test_a_model_error_attempt_has_no_output_to_validate():
    _, (error, rule) = await PlanningService(TOOLS, ScriptedLLM(LLMError("HTTP 529 overloaded"))).plan(
        "分析 AAPL", MemoryContext())
    assert (error["outcome"], error["planner_output"], error["validation"], error["detail"]) == (
        "model_error", None, None, "HTTP 529 overloaded")
    assert error["call"]["model"] == "scripted-test-model"
    assert rule["fallback_reason"] == "model error: HTTP 529 overloaded"


@pytest.mark.asyncio
async def test_llm_can_ask_the_user():
    llm = ScriptedLLM(llm_plan(clarification_question="哪只股票？"))
    plan, _ = await PlanningService(TOOLS, llm).plan("分析一下", MemoryContext())
    assert plan == Clarification("哪只股票？", "planner needs more information")


# --------------------------------------------------------------------------- revision

@pytest.mark.asyncio
async def test_rule_revision_widens_the_window_and_adds_factor_evidence():
    service = PlanningService(TOOLS)
    plan, _ = await service.plan("分析我的 AAPL 仓位，近 25 天", MemoryContext(), portfolio_hint=2)
    revised, log = await service.revise(plan, "too few bars", {"days": 365, "factors": ["momentum_20d"]}, [])
    assert log["outcome"] == "accepted" and revised.revision == 1
    assert (revised.intent, revised.symbol, revised.portfolio_id) == (plan.intent, "AAPL", 2)
    md = [s for s in revised.steps if s.tool == "market_data"][0]
    assert md.arguments["days"] == 365 and "factor" in tools_of(revised)
    none, log = await service.revise(plan, "stale data", {}, [])
    assert none is None and log["outcome"] == "no_remedy"


@pytest.mark.asyncio
async def test_llm_revision_is_validated_against_the_original_goal():
    llm = ScriptedLLM(llm_plan(), llm_plan(symbol="TSLA"))  # the revision tries to change the subject
    service = PlanningService(TOOLS, llm)
    plan, _ = await service.plan("分析 AAPL", MemoryContext())
    revised, log = await service.revise(plan, "issue", {"days": 365}, [])
    assert log["planner"] == "rule" and "may not switch to TSLA" in log["fallback_reason"]
    assert (revised.symbol, revised.planner) == ("AAPL", "rule")


def test_rule_plan_output_passes_its_own_validator():
    facts = goal_facts("分析我的AAPL仓位风险", None, 2)
    raw = rule_plan(facts)
    assert json.loads(json.dumps(raw)) == raw          # plain JSON, same shape as the LLM's
    assert isinstance(validate_plan(raw, TOOLS, "rule", facts), Plan)


# ---------------------------------------------------------------------- the SDK client

def _client(handler, **kw):
    transport = httpx2.MockTransport(handler)
    return AnthropicStructuredLLM(api_key="test-key", http_client=httpx2.AsyncClient(transport=transport),
                                  max_retries=0, **kw)


def _message(text=None, stop_reason="end_turn", **extra):
    content = [{"type": "text", "text": text}] if text is not None else []
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}, **extra}


@pytest.mark.asyncio
async def test_sdk_client_requests_structured_output_with_default_fallbacks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://not-the-model-api.invalid")
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_message('{"steps": []}'))

    client = _client(handler)
    assert await client.generate("sys", "prompt", "plan", {"type": "object"}) == {"steps": []}
    assert seen["url"].startswith("https://api.anthropic.com/v1/messages")
    assert seen["headers"]["x-api-key"] == "test-key"
    assert "server-side-fallback-2026-07-01" in seen["headers"]["anthropic-beta"]
    body = seen["body"]
    assert body["model"] == "claude-opus-5" and body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "medium",
                                     "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert "tool_choice" not in body
    assert client.last_call["usage"] == {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize("response,message", [
    (httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}), "HTTP 500"),
    (httpx2.Response(200, json=_message(None, stop_reason="refusal")), "declined"),
    (httpx2.Response(200, json=_message('{"steps": [', stop_reason="max_tokens")), "truncated"),
    (httpx2.Response(200, json=_message("BUY AAPL now")), "invalid JSON"),
    (httpx2.Response(200, json=_message("[1, 2]")), "non-object"),
])
async def test_sdk_client_turns_bad_answers_into_llm_errors(response, message):
    with pytest.raises(LLMError, match=message):
        await _client(lambda request: response).generate("s", "p", "plan", {})


@pytest.mark.asyncio
async def test_sdk_client_records_each_call_even_when_it_fails():
    ok = httpx2.Response(200, json=_message('{"steps": []}'), headers={"request-id": "req_ok"})
    denied = httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error",
                                                                  "message": "invalid x-api-key"}},
                             headers={"request-id": "req_denied"})
    answers = [ok, denied]
    client = _client(lambda request: answers.pop(0))
    await client.generate("s", "p", "plan", {})
    assert client.last_call["request_id"] == "req_ok"
    with pytest.raises(LLMError, match="HTTP 401"):
        await client.generate("s", "p", "plan", {})
    assert client.last_call == {"model": "claude-opus-5", "provider": "anthropic",
                                "endpoint": "https://api.anthropic.com", "request_id": "req_denied",
                                "stop_reason": None, "usage": None}  # not the previous call's


@pytest.mark.parametrize("base_url,provider", [
    ("https://api.anthropic.com", "anthropic"),
    ("https://api.deepseek.com/anthropic", "deepseek"),
    ("http://localhost:9000", "unknown:localhost"),
    ("https://api.anthropic.com.evil.example", "unknown:api.anthropic.com.evil.example"),
    ("https://api.anthropic.com@evil.example", "unknown:evil.example"),
    ("https://evil.example/api.deepseek.com", "unknown:evil.example"),
])
def test_the_client_names_its_provider_from_the_endpoint_host(base_url, provider, monkeypatch):
    monkeypatch.delenv("AGENT_LLM_PROVIDER", raising=False)
    client = AnthropicStructuredLLM(api_key="k", base_url=base_url)
    assert client.provider == provider
    ok, reason = client.available()
    assert ok is (provider in ("anthropic", "deepseek"))
    if not ok:
        assert "unknown provider" in reason


def _clear_llm_env(monkeypatch):
    for name in ("AGENT_LLM_PROVIDER", "AGENT_LLM_API_KEY", "AGENT_LLM_BASE_URL", "AGENT_LLM_MODEL",
                 "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_deepseek_is_configured_by_environment_with_its_official_defaults(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "test-key")
    client = AnthropicStructuredLLM()
    assert (client.provider, client.endpoint, client.model) == (
        "deepseek", "https://api.deepseek.com/anthropic", "deepseek-chat")
    assert client.available() == (True, None)
    assert default_llm() is not None


def test_an_anthropic_key_is_never_sent_to_another_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-only-key")
    client = AnthropicStructuredLLM()
    assert client.api_key is None
    assert client.available() == (False, "model unavailable: AGENT_LLM_API_KEY is not configured")


@pytest.mark.parametrize("env,reason", [
    ({"AGENT_LLM_PROVIDER": "foo"}, "model unavailable: unknown provider 'foo'"),
    ({"AGENT_LLM_PROVIDER": "deepseek", "AGENT_LLM_BASE_URL": "https://api.anthropic.com"},
     "model unavailable: provider 'deepseek' does not match the endpoint 'api.anthropic.com'"),
    ({"AGENT_LLM_PROVIDER": "anthropic", "AGENT_LLM_BASE_URL": "https://api.deepseek.com/anthropic"},
     "model unavailable: provider 'anthropic' does not match the endpoint 'api.deepseek.com'"),
    ({"AGENT_LLM_BASE_URL": "https://llm.example.invalid"}, "model unavailable: unknown provider 'llm.example.invalid'"),
])
def test_an_unknown_or_mismatched_provider_is_never_used(monkeypatch, env, reason):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("AGENT_LLM_API_KEY", "test-key")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    client = default_llm()
    assert client is not None and client.available() == (False, reason)  # configured, so the reason is reported
    assert client.provider.startswith(("unknown:", "mismatch:"))


@pytest.mark.asyncio
async def test_deepseek_requests_carry_the_schema_and_no_anthropic_only_parameters(monkeypatch):
    _clear_llm_env(monkeypatch)
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        answer = _message('```json\n{"steps": []}\n```')
        answer["model"] = "deepseek-chat"
        return httpx2.Response(200, json=answer)

    client = _client(handler, provider="deepseek")
    assert await client.generate("sys", "prompt", "plan", {"type": "object"}) == {"steps": []}
    assert seen["url"].startswith("https://api.deepseek.com/anthropic/v1/messages")
    body = seen["body"]
    assert body["model"] == "deepseek-chat" and body["max_tokens"] == 8000
    assert "output_config" not in body and "fallbacks" not in body
    assert "anthropic-beta" not in seen["headers"]
    assert '"type": "object"' in body["system"]  # the schema is in the instructions
    assert client.last_call["provider"] == "deepseek" and client.last_call["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_client_without_a_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicStructuredLLM()
    assert client.available() == (False, "model unavailable: ANTHROPIC_API_KEY is not configured")
    assert default_llm() is None
    with pytest.raises(LLMError):
        await client.generate("s", "p", "plan", {})
