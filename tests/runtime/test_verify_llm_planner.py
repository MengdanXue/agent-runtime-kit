# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/testing/verify_llm_planner.py: provider-aware verification.

VERIFIED only for a known provider on its official endpoint with a model of
that provider's family; mock models are TEST_ONLY, a rule fallback is PARTIAL,
unknown or mismatched providers are FAILED.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agentrt.runtime import llm as llm_module

from .test_planning import llm_plan
from .workflow_fixtures import FakeJava, ScriptedLLM, calm_uptrend, make_workflow  # noqa: F401

pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

_PATH = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "verify_llm_planner.py"
_spec = importlib.util.spec_from_file_location("verify_llm_planner", _PATH)
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)

ENDPOINTS = {"anthropic": "https://api.anthropic.com", "deepseek": "https://api.deepseek.com/anthropic"}


class ProviderLike(ScriptedLLM):
    """Scripted answers reported the way the SDK client reports a real provider's call."""

    def __init__(self, *plans, provider="anthropic", served="claude-opus-5", endpoint=None):
        super().__init__(*plans)
        self.provider, self.model, self._served = provider, served, served
        self.endpoint = endpoint or ENDPOINTS.get(provider, "https://llm.example.invalid")

    async def generate(self, system, prompt, schema_name, schema):
        answer = await super().generate(system, prompt, schema_name, schema)
        self.last_call = {"model": self._served, "request_id": "req_test", "stop_reason": "end_turn"}
        return answer


def status_for(llm, available=True, reason=None):
    return {"llm_available": available, "llm_model": getattr(llm, "model", None),
            "llm_provider": getattr(llm, "provider", None), "llm_endpoint": getattr(llm, "endpoint", None),
            "reason": reason}


async def verify_with(make_workflow, llm, status=None, goal="分析 AAPL"):
    h = make_workflow(FakeJava({"AAPL": calm_uptrend()}), llm=llm)
    run = (await h.run(goal)).to_dict()
    return verify.assess(status or status_for(llm), run)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,served", [("anthropic", "claude-opus-5"), ("deepseek", "deepseek-chat")])
async def test_a_known_provider_on_its_official_endpoint_is_verified(make_workflow, provider, served):
    code, report = await verify_with(make_workflow, ProviderLike(llm_plan(), provider=provider, served=served))
    assert code == verify.VERIFIED and report["verification_status"] == "VERIFIED", report["problems"]
    assert (report["provider"], report["model"], report["endpoint"], report["planner_mode"]) == (
        provider, served, ENDPOINTS[provider], "llm")
    accepted = report["attempts"][-1]
    assert accepted["planner_output"] == llm_plan() and accepted["validation"]["valid"] is True
    assert accepted["request_id"] == "req_test" and accepted["endpoint"] == ENDPOINTS[provider]
    kinds = [e["kind"] for e in report["trace"]]
    assert kinds[0] == "goal" and kinds[-1] == "final" and "tool_call" in kinds


@pytest.mark.asyncio
async def test_a_mock_model_is_test_only_never_verified(make_workflow):
    code, report = await verify_with(make_workflow, ScriptedLLM(llm_plan()))
    assert code == verify.TEST_ONLY and report["verification_status"] == "TEST_ONLY"
    assert report["provider"] == "mock"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,served,endpoint,problem", [
    ("anthropic", "claude-opus-5", "https://api.deepseek.com/anthropic", "not anthropic's official API"),
    ("deepseek", "deepseek-chat", "https://api.anthropic.com", "not deepseek's official API"),
    ("anthropic", "deepseek-chat", None, "not a claude model"),
    ("deepseek", "claude-opus-5", None, "not a deepseek model"),
])
async def test_a_compatible_endpoint_or_foreign_model_never_passes_for_a_provider(
        make_workflow, provider, served, endpoint, problem):
    llm = ProviderLike(llm_plan(), provider=provider, served=served, endpoint=endpoint)
    code, report = await verify_with(make_workflow, llm)
    assert code == verify.FAILED and report["verification_status"] == "FAILED"
    assert any(problem in p for p in report["problems"]), report["problems"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["unknown:llm.example.invalid", "unknown:foo",
                                      "mismatch:deepseek@api.anthropic.com"])
async def test_an_unknown_or_mismatched_provider_fails(make_workflow, provider):
    status = {"llm_available": False, "llm_model": None, "llm_provider": provider, "llm_endpoint": None,
              "reason": "model unavailable: unknown provider"}
    code, report = await verify_with(make_workflow, None, status)
    assert code == verify.FAILED and report["verification_status"] == "FAILED"
    assert any("provider" in p for p in report["problems"])


@pytest.mark.asyncio
async def test_no_usable_model_is_partial_with_the_fallback_reason(make_workflow):
    status = {"llm_available": False, "llm_model": None, "llm_provider": None, "llm_endpoint": None,
              "reason": "model unavailable: ANTHROPIC_API_KEY is not configured"}
    code, report = await verify_with(make_workflow, None, status)
    assert code == verify.PARTIAL and report["verification_status"] == "PARTIAL"
    assert report["planner_mode"] == "rule"
    assert report["fallback_reason"] == "model unavailable: ANTHROPIC_API_KEY is not configured"
    assert any(c["status"] == "OK" for c in report["tool_calls"])


@pytest.mark.asyncio
async def test_a_model_that_fails_at_run_time_is_partial(make_workflow):
    from agentrt.runtime.llm import LLMError

    llm = ProviderLike(LLMError("model endpoint unreachable: connection refused"), provider="deepseek",
                       served="deepseek-chat")
    code, report = await verify_with(make_workflow, llm)
    assert code == verify.PARTIAL and report["planner_mode"] == "rule"
    assert report["fallback_reason"] == "model error: model endpoint unreachable: connection refused"
    assert report["provider"] == "deepseek"  # the configured provider is still reported


@pytest.mark.asyncio
async def test_a_plan_the_validator_rejects_is_partial_not_verified(make_workflow):
    bad = llm_plan(steps=[])
    code, report = await verify_with(make_workflow, ProviderLike(bad, bad, provider="deepseek",
                                                                 served="deepseek-chat"))
    assert code == verify.PARTIAL and report["planner_mode"] == "rule"
    assert report["fallback_reason"].startswith("LLM plan rejected by the validator")
    assert [a["outcome"] for a in report["attempts"]] == ["rejected", "rejected", "accepted"]


STRATEGY_GOAL = "给 AAPL 推荐一个策略"


def backtest_plan(strategy):
    plan = llm_plan()
    plan["steps"].append({"tool": "backtest", "purpose": "simulate", "required_inputs": [
        {"name": "symbol", "value": "AAPL", "from_step": 0}, {"name": "bars", "value": "", "from_step": 1},
        {"name": "strategy", "value": strategy, "from_step": 0}]})
    return plan


@pytest.mark.asyncio
async def test_a_strategy_needing_more_history_than_stored_is_expected_insufficient_data(make_workflow):
    # 120 stored bars; tsmom needs 253. The tool says so - nothing is filled in. The user asked
    # for a strategy, so a strategy that cannot be simulated ends the run.
    llm = ProviderLike(backtest_plan("tsmom"), provider="deepseek", served="deepseek-v4-flash")
    code, report = await verify_with(make_workflow, llm, goal=STRATEGY_GOAL)
    assert code == verify.EXPECTED_INSUFFICIENT_DATA, report["problems"]
    assert report["verification_status"] == "EXPECTED_INSUFFICIENT_DATA"
    assert (report["provider"], report["planner_mode"], report["run_status"]) == ("deepseek", "llm", "FAILED")
    backtest = [c for c in report["tool_calls"] if c["tool"] == "backtest"]
    assert [c["error_code"] for c in backtest] == ["INSUFFICIENT_DATA"]
    assert report["problems"] == [f"backtest: INSUFFICIENT_DATA - {backtest[0]['error']}"]
    assert "needs 253 bars, got 120" in backtest[0]["error"]


@pytest.mark.asyncio
async def test_insufficient_data_never_hides_a_provider_problem(make_workflow):
    llm = ProviderLike(backtest_plan("tsmom"), provider="deepseek", served="gpt-4o")
    code, report = await verify_with(make_workflow, llm, goal=STRATEGY_GOAL)
    assert code == verify.FAILED
    assert any("not a deepseek model" in p for p in report["problems"])


@pytest.mark.asyncio
async def test_an_optional_backtest_without_history_is_reported_and_the_analysis_still_verifies(make_workflow):
    # The model adds a backtest to a symbol analysis: the tool reports the missing history, the
    # run completes on the other steps, and the report still shows the failed call.
    llm = ProviderLike(backtest_plan("tsmom"), provider="deepseek", served="deepseek-v4-flash")
    code, report = await verify_with(make_workflow, llm)
    assert code == verify.VERIFIED, report["problems"]
    assert (report["provider"], report["planner_mode"], report["run_status"]) == ("deepseek", "llm", "SUCCESS")
    backtest = [c for c in report["tool_calls"] if c["tool"] == "backtest"]
    assert [c["error_code"] for c in backtest] == ["INSUFFICIENT_DATA"]
    assert "needs 253 bars, got 120" in backtest[0]["error"]


def test_a_run_failing_for_another_tool_error_stays_failed():
    status = {"llm_available": True, "llm_provider": "deepseek", "llm_model": "deepseek-chat",
              "llm_endpoint": ENDPOINTS["deepseek"], "reason": None}
    attempt = {"planner": "llm", "outcome": "accepted", "provider": "deepseek", "endpoint": ENDPOINTS["deepseek"],
               "served_model": "deepseek-v4-flash", "planner_output": llm_plan(), "validation": {"valid": True}}
    run = {"task_id": "t", "status": "FAILED", "trace": [],
           "result": {"planner": {"planner": "llm", "fallback": False, "attempts": [attempt]}},
           "tool_calls": [{"call_id": "call-1", "tool": "market_data", "status": "OK"},
                          {"call_id": "call-2", "tool": "signal", "status": "ERROR",
                           "error": {"code": "INVALID_INPUT", "message": "Unknown strategy: registered"}},
                          {"call_id": "call-3", "tool": "backtest", "status": "ERROR",
                           "error": {"code": "INSUFFICIENT_DATA", "message": "tsmom needs 253 bars, got 30"}}]}
    code, report = verify.assess(status, run)
    assert code == verify.FAILED
    assert report["problems"] == ["the run ended FAILED; failed tool calls: signal INVALID_INPUT, "
                                  "backtest INSUFFICIENT_DATA"]


def test_the_verifier_knows_the_same_official_endpoints_as_the_client():
    for name, spec in llm_module.PROVIDERS.items():
        assert verify.OFFICIAL[name] == {"hosts": spec["hosts"], "family": spec["family"]}
    assert set(verify.OFFICIAL) == set(llm_module.PROVIDERS)
