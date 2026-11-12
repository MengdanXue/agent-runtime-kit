# Copyright 2026 Mengdan Xue
# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks for the S2c sandbox data and the sale -> approval path.

No real model is called here. ``ScriptedLLM`` answers are labelled provider "mock";
they test what the validators do with a given model answer, never what a real model
would answer. Run:

    python -m pytest -q validation/real_llm/test_s2c.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1] / "src"), str(HERE.parents[1]), str(HERE)]

import run_real_llm_e2e as e2e  # noqa: E402
import s2c_data as D  # noqa: E402
from agentrt.quant.risk_agent import RiskAgent  # noqa: E402
from agentrt.runtime.review import ALLOWED_FACTORS, CRITIC_SYSTEM  # noqa: E402
from agentrt.tools import ToolContext  # noqa: E402
from tests.runtime.workflow_fixtures import OTHER, OWNER, TODAY, ScriptedLLM, owner_ctx  # noqa: E402

# Tests that run the decision workflow need named strategies (the research step's signal
# tool); without a provider they are skipped with that reason (see conftest.py). Everything
# else - schema, the required-review block at critic level, the option-2 gate, the policy,
# the data - runs without quantrisk-core.
needs_strategies = pytest.mark.needs_strategies


def issue(message, severity="HIGH", type_="risk_conflict", days=0, factors="", category="other"):
    # a schema-complete model issue; "other" is not on the research-quality whitelist, so a
    # MEDIUM/HIGH one blocks. category=None builds an answer that misses the required field.
    out = {"type": type_, "severity": severity, "message": message, "suggestion": "check it",
           "remedy_days": days, "remedy_factors": factors}
    if category is not None:
        out["category"] = category
    return out


EMPTY_REVIEW = {"issues": []}   # a usable model review that finds nothing


# ------------------------------------------------------------------------ data consistency

@pytest.fixture(scope="module")
def data():
    series = D.make_series(TODAY)
    return series, D.position_rows(series)


def test_every_symbol_has_its_own_series(data):
    series, _ = data
    closes = {s: tuple(b["close"] for b in bars) for s, bars in series.items()}
    assert len(set(closes.values())) == len(closes)          # no two symbols share a path
    a, m = np.array(closes["AAPL"]), np.array(closes["MSFT"])
    assert not np.allclose(a, m)


def test_bars_are_weekdays_ending_the_last_weekday_before_today(data):
    series, _ = data
    for bars in series.values():
        days = [date.fromisoformat(b["date"]) for b in bars]
        assert all(d.weekday() < 5 for d in days) and days == sorted(set(days))
        assert days[-1] == date(2026, 8, 31) and len(days) == D.N_BARS
        assert all(b["low"] <= min(b["open"], b["close"]) <= max(b["open"], b["close"]) <= b["high"]
                   for b in bars)


def test_rerunning_gives_the_same_data():
    assert D.make_series(TODAY) == D.make_series(TODAY)


def test_positions_and_portfolio_row_agree_with_the_series(data):
    series, rows = data
    for r in rows:
        last = series[r["symbol"]][-1]["close"]
        assert r["current_price"] == last
        assert r["current_value"] == round(r["quantity"] * last, 2)
        assert r["unrealized_pnl"] == round(r["quantity"] * (last - r["average_cost"]), 2)
    row = D.portfolio_row(7, rows)
    assert row["total_value"] == row["equity"] == round(sum(r["current_value"] for r in rows) + D.CASH, 2)
    assert row["cash_balance"] == D.CASH


def test_weights_put_only_aapl_over_the_cap_and_the_sale_fixes_it(data):
    _, rows = data
    values = {r["symbol"]: r["current_value"] for r in rows}
    total = sum(values.values())                            # exposure weights exclude cash
    weights = {s: v / total * 100 for s, v in values.items()}
    assert weights["AAPL"] > 20 and all(w < 20 for s, w in weights.items() if s != "AAPL")
    sold = 5 * rows[0]["current_price"]
    after = {s: (v - (sold if s == "AAPL" else 0)) / (total - sold) * 100 for s, v in values.items()}
    assert all(w <= 20 for w in after.values())


def test_portfolio_risk_payloads_match_an_independent_recomputation(data):
    series, rows = data
    an = D.ConsistentAnalytics(series, rows)
    value = sum(r["current_value"] for r in rows) + D.CASH
    # independent path: day-by-day weighted sum (no matrix code)
    closes = {r["symbol"]: [b["close"] for b in series[r["symbol"]]] for r in rows}
    n = D.WINDOW_RETURNS
    port = []
    for t in range(D.N_BARS - n, D.N_BARS):
        port.append(sum(r["current_value"] / value * (closes[r["symbol"]][t] / closes[r["symbol"]][t - 1] - 1)
                        for r in rows))
    port = np.array(port)
    var95 = (asyncio.run(an.calculate_var(7, 0.05)))
    var99 = (asyncio.run(an.calculate_var(7, 0.01)))
    cvar = asyncio.run(an.calculate_cvar(7))
    vol = asyncio.run(an.calculate_portfolio_volatility(7))
    beta = asyncio.run(an.calculate_beta(7))
    assert var95["sample_size"] == n and var95["equity"] == round(value, 2)
    assert var95["var_amount"] == pytest.approx(-np.percentile(port, 5) * value, abs=0.01)
    assert var95["var_pct"] == pytest.approx(var95["var_amount"] / value * 100, abs=1e-3)
    assert var99["var_amount"] >= var95["var_amount"]
    assert cvar["cvar_amount"] >= var95["var_amount"]
    assert vol["portfolio_volatility"] == pytest.approx(port.std(ddof=1), abs=1e-6)
    spy = np.array([b["close"] for b in series["SPY"]])
    bench = spy[-n:] / spy[-n - 1:-1] - 1
    assert beta["beta"] == pytest.approx(np.polyfit(bench, port, 1)[0], abs=1e-3)


def test_portfolio_risk_level_follows_the_unchanged_var_thresholds(data):
    series, rows = data
    var_pct = asyncio.run(D.ConsistentAnalytics(series, rows).calculate_var(7, 0.05))["var_pct"]
    assert (RiskAgent.HIGH_RISK_VAR_PCT, RiskAgent.MEDIUM_RISK_VAR_PCT) == (5.0, 2.0)
    assert RiskAgent()._assess_risk_level({"var_95_pct": var_pct}) == (
        "HIGH" if var_pct >= 5 else "MEDIUM" if var_pct >= 2 else "LOW")


# ------------------------------------------------------------------- rules stay as they were

def test_factor_whitelist_is_unchanged_and_named_in_the_critic_prompt():
    assert ALLOWED_FACTORS == ("momentum_20d", "volatility_20d", "rsi_14d", "momentum_60d", "volatility_60d")
    assert all(f in CRITIC_SYSTEM for f in ALLOWED_FACTORS)


# ----------------------------------------------------------------------------- workflow

async def sale(llm=None, stale=False):
    h, _, _ = await e2e.s2c_harness(llm, stale=stale)
    run = await h.run(e2e.SALE_GOAL, portfolio_id=7)
    return h, run, run.to_dict()


@pytest.mark.asyncio
@needs_strategies
async def test_legit_sale_on_consistent_data_reaches_approval_and_executes_once_on_paper():
    h, run, d = await sale()
    r = d["result"]
    assert d["status"] == "SUCCESS"
    trade = r["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and trade["mode"] == "PAPER"
    assert trade["addresses"] == ["POSITION_LIMIT"] and trade["trade_check"]["within_limit_after"] is True
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    assert {i["code"] for i in r["critic"]["unresolved"]} <= {"POSITION_LIMIT"}
    pid = trade["proposal"]["proposal_id"]

    stranger = await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))
    assert stranger.error_code == "APPROVAL_FORBIDDEN"
    assert (await h.loop.proposals.get(pid, OWNER)).status == "PENDING"
    assert any(e["kind"] == "approval" for e in stranger.to_dict()["trace"])
    done = await h.loop.execute_approved(pid, owner_ctx())
    fill = done.result["execution"]
    assert (fill["mode"], fill["status"], fill["quantity"]) == ("PAPER", "PAPER_FILLED", 5.0)
    assert fill["fill_price"] == D.make_series(TODAY)["AAPL"][-1]["close"]
    again = await h.loop.execute_approved(pid, owner_ctx())
    assert again.error_code == "APPROVAL_CONSUMED" and again.tool_calls == []
    assert (await h.loop.proposals.get(pid, OWNER)).status == "EXECUTED"


@pytest.mark.asyncio
@needs_strategies
async def test_sale_on_stale_data_stays_blocked():
    _, _, d = await sale(stale=True)
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "STALE_DATA" in trade["reason"]
    assert "proposal" not in trade or not trade.get("proposal")
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    assert any(e["kind"] == "approval" and "no trade proposal" in e["message"] for e in d["trace"])


@pytest.mark.asyncio
@needs_strategies
async def test_a_review_that_cites_a_number_not_in_the_reports_is_rejected():
    llm = ScriptedLLM(review=[{"issues": [issue("AAPL returned 87.3 percent over the window")]}])
    _, _, d = await sale(llm)
    c = d["result"]["critic"]
    assert c["reviewer"] == "rule" and "cites numbers not in the reports" in c["fallback_reason"]
    # was AWAITING_APPROVAL ("the rule review alone does not block"): that let a rejected model
    # review count as "no risk". Since the fallback fix a required review that is unusable blocks.
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "LLM_REVIEW_INVALID" in trade["reason"]
    assert c["llm_review"]["status"] == "invalid" and c["llm_review"]["error"]["field"] == "message"


@pytest.mark.asyncio
@needs_strategies
async def test_a_review_that_names_a_strategy_as_factor_is_rejected():
    llm = ScriptedLLM(review=[{"issues": [issue("momentum evidence is thin", factors="tsmom")]}])
    _, _, d = await sale(llm)
    c = d["result"]["critic"]
    assert c["reviewer"] == "rule" and "remedy_factors must be among" in c["fallback_reason"]


@pytest.mark.asyncio
@needs_strategies
async def test_a_grounded_high_llm_issue_still_blocks_the_sale():
    llm = ScriptedLLM(review=[{"issues": [issue("the sale leaves the financial sector exposure unexamined")]}])
    _, _, d = await sale(llm)
    c = d["result"]["critic"]
    assert c["reviewer"] == "rule+llm"
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "LLM_REVIEW" in trade["reason"]
    assert "execution" not in [x["tool"] for x in d["tool_calls"]]


# ------------------------------------------------------- root-cause round (S2C_FINAL_ROOT_CAUSE)
# Cases 1-10 of the brief. Tests named *_currently_blocks pin today's behaviour for the
# B-class (research-quality) issues; they are expected to change only if the rule does.

import json  # noqa: E402

from agentrt.runtime.planner import Goal  # noqa: E402

REAL_PLAN = json.loads((HERE / "fixtures" / "s2c_real_plan.json").read_text(encoding="utf-8"))["plan"]


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def decisions(d):
    return [x for x in _walk(d) if "decided_by" in x]


@pytest.mark.asyncio
@needs_strategies
async def test_c3_selling_more_than_is_held_is_refused():
    h, _, _ = await e2e.s2c_harness(None)
    d = (await h.run("我的AAPL仓位超过上限，减持 50 股 AAPL", portfolio_id=7)).to_dict()
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "Risk Agent veto" in trade["reason"]
    assert any(c["check"] == "sell_quantity" and c["status"] == "veto" for c in d["result"]["risk"]["checks"])
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]


@pytest.mark.asyncio
@needs_strategies
async def test_c5_decision_without_action_level_falls_back_to_the_rule_decision():
    llm = ScriptedLLM(review=[{"issues": []}],
                      decision=[{"decision": "REDUCE_EXPOSURE", "confidence": "low", "rationale": ["sell 5"]}])
    _, _, d = await sale(llm)
    ds = decisions(d)
    assert ds and ds[0]["decided_by"] == "rule" and "action_level None is not allowed" in ds[0]["fallback_reason"]
    assert d["result"]["recommendation"] == "REDUCE_EXPOSURE"
    assert d["result"]["trade"]["status"] == "AWAITING_APPROVAL"   # the fallback keeps the safe path


@pytest.mark.asyncio
@needs_strategies
async def test_c6_enough_data_but_a_losing_backtest_does_not_block_under_the_rules():
    # the real model's plan (120 calendar days, ma_cross backtest), rule review and decision
    _, _, d = await sale(ScriptedLLM(REAL_PLAN, review=[EMPTY_REVIEW] * 2))
    bt = d["result"]["research"]["backtests"]
    assert bt and bt[0]["strategy"] == "ma_cross" and bt[0]["sharpe_ratio"] < 0
    assert d["result"]["research"]["market"]["AAPL"]["count"] == 86   # reported as is, not padded
    assert d["result"]["critic"]["reviewer"] == "rule+llm" and not d["result"]["critic"]["issues"][1:]
    assert d["result"]["trade"]["status"] == "AWAITING_APPROVAL"


SHORT_SAMPLE = issue("factor and signal estimates rest on a short sample", severity="MEDIUM",
                     type_="missing_data", days=365, category="sample_length")
# the same finding outside the research-quality whitelist: option 2 cannot downgrade it
SHORT_SAMPLE_OTHER = {**SHORT_SAMPLE, "category": "other"}


@pytest.mark.asyncio
@needs_strategies
async def test_c7_llm_research_quality_issue_is_reanalysed_and_still_blocks_if_unresolved():
    # Option 1 (approved 2026-11-12): the concentration veto no longer skips re-analysis
    # for a reducing sale; the round-2 critic repeats the issue, so the sale stays blocked
    llm = ScriptedLLM(REAL_PLAN, review=[{"issues": [SHORT_SAMPLE]}, {"issues": [SHORT_SAMPLE_OTHER]}])
    _, _, d = await sale(llm)
    r = d["result"]
    assert r["critic"]["reviewer"] == "rule+llm"
    assert r["reanalysis"]["performed"] is True and r["reanalysis"]["remedy"] == {"days": 365}
    assert any(e["message"].startswith("re-analysis allowed: the only veto is the concentration limit")
               for e in d["trace"])
    assert [(a["agent"], a["round"]) for a in r["agents"]].count(("research", 2)) == 1   # one round only
    risk2 = [a for a in r["agents"] if (a["agent"], a["round"]) == ("risk", 2)][0]
    assert risk2["status"] == "VETO" and "AAPL" in risk2["findings"]["concentration_breaches"]  # veto kept
    trade = r["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "LLM_REVIEW" in trade["reason"]


@pytest.mark.asyncio
@needs_strategies
async def test_c7b_research_quality_issue_resolved_by_reanalysis_reaches_human_approval():
    llm = ScriptedLLM(REAL_PLAN, review=[{"issues": [SHORT_SAMPLE]}, EMPTY_REVIEW])  # round 2: resolved
    _, _, d = await sale(llm)
    r = d["result"]
    assert r["reanalysis"]["performed"] is True
    assert r["research"]["market"]["AAPL"]["count"] > 86                  # real longer history, not padded
    assert r["research"]["market"]["AAPL"]["days_requested"] == 365
    assert r["risk"]["vetoes"]                                            # the veto still stands
    assert {i["code"] for i in r["critic"]["unresolved"]} <= {"POSITION_LIMIT"}
    assert r["trade"]["status"] == "AWAITING_APPROVAL" and r["trade"]["mode"] == "PAPER"
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]


def _veto_report(vetoes, breaches, reduces):
    from agentrt.runtime.reports import make_report
    return make_report("risk", "VETO", "; ".join(vetoes),
                       {"vetoes": vetoes, "concentration_breaches": breaches,
                        "trade_check": {"reduces_exposure": reduces}}, [])


@pytest.mark.parametrize("side,vetoes,breaches,reduces,allowed", [
    ("sell", ["AAPL over limit"], {"AAPL": 40.0}, True, True),
    ("buy", ["AAPL over limit"], {"AAPL": 40.0}, None, False),                    # new exposure
    ("sell", ["AAPL over limit", "short sale"], {"AAPL": 40.0}, False, False),   # sells more than held
    ("sell", ["AAPL over limit", "AAPL risk level is high"], {"AAPL": 40.0}, True, False),  # second veto
    ("sell", ["portfolio 7 risk level is high"], {"MSFT": 40.0}, True, False),   # veto not on this symbol
])
def test_option1_reanalysis_only_for_a_reducing_sale_whose_only_veto_is_its_concentration(
        side, vetoes, breaches, reduces, allowed):
    from types import SimpleNamespace
    from agentrt.runtime.workflow import _reducing_sale_of_breach
    plan = SimpleNamespace(symbol="AAPL", trade={"side": side, "quantity": 5})
    assert _reducing_sale_of_breach(plan, _veto_report(vetoes, breaches, reduces)) is allowed


class RecordingLLM(ScriptedLLM):
    async def generate(self, system, prompt, schema_name, schema):
        self.seen = getattr(self, "seen", []) + [(schema_name, prompt)]
        return await super().generate(system, prompt, schema_name, schema)


def llm_prompt(llm, schema_name):
    return next(p for s, p in llm.seen if s == schema_name)


@pytest.mark.asyncio
@needs_strategies
async def test_report_fields_d1_to_d5():
    llm = RecordingLLM(REAL_PLAN, review=[{"issues": []}])
    _, _, d = await sale(llm)
    r = d["result"]
    assert r["research"]["market"]["AAPL"]["days_requested_unit"] == "calendar"            # D1
    assert "risk vetoed new exposure: no directional call" in r["rationale"]                 # D2
    assert not any("risk was not assessed" in x for x in r["rationale"])
    review_prompt = json.loads(llm_prompt(llm, "review"))
    assert "best_strategy" not in review_prompt["research"]                                  # D3
    pr = review_prompt["risk"]["portfolio_risk"]
    assert pr["policy"] == {"var_high_pct": RiskAgent.HIGH_RISK_VAR_PCT,
                            "var_medium_pct": RiskAgent.MEDIUM_RISK_VAR_PCT}                # D4
    assert pr["max_drawdown_sign"] == "positive_magnitude"                                  # D5
    assert review_prompt["risk"]["symbol_risk"]["max_drawdown_sign"] == "negative"
    assert "Do not repeat issues_already_found" in CRITIC_SYSTEM


@pytest.mark.asyncio
@needs_strategies
async def test_c8_llm_hard_risk_issue_blocks():
    llm = ScriptedLLM(review=[{"issues": [
        issue("the sale is priced off a close that the reports do not show for today", type_="risk_conflict")]}])
    _, _, d = await sale(llm)
    assert d["result"]["critic"]["reviewer"] == "rule+llm"
    assert d["result"]["trade"]["status"] == "NOT_PROPOSED"


@pytest.mark.asyncio
@needs_strategies
async def test_c9_execution_without_approval_is_blocked_and_only_proposed():
    h, series, _ = await e2e.s2c_harness(None)
    goal = Goal(intent="analyze_symbol", symbol="AAPL", strategies=["momentum"],
                trade={"side": "sell", "quantity": 5})
    d = (await h.loop.run(goal, owner_ctx())).to_dict()
    blocked = [c for c in d["tool_calls"] if c["tool"] == "execution"]
    assert blocked and all(c["status"] == "BLOCKED" and c["error"]["code"] == "APPROVAL_REQUIRED"
                           for c in blocked)
    pid = d["result"]["execution"]["proposal"]["proposal_id"]
    assert (await h.loop.proposals.get(pid, OWNER)).status == "PENDING"   # nothing filled


@pytest.mark.asyncio
@needs_strategies
async def test_c10_every_rejection_and_fallback_is_in_the_audit_trace():
    llm = ScriptedLLM(review=[{"issues": [issue("AAPL returned 87.3 percent")]}],
                      decision=[{"decision": "REDUCE_EXPOSURE", "confidence": "low", "rationale": ["x"]}])
    h, run, d = await sale(llm)
    reports = [e for e in d["trace"] if e["kind"] == "agent_report"]
    text = json.dumps(reports, ensure_ascii=False)
    assert "cites numbers not in the reports" in text            # review rejected -> rule review
    assert "action_level None is not allowed" in text            # decision rejected -> rule decision
    assert any(e["kind"] == "approval" for e in d["trace"])      # proposal recorded
    _, _, stale = await sale(stale=True)
    assert any(e["kind"] == "approval" and "no trade proposal" in e["message"] for e in stale["trace"])


# ------------------------------------------------------------- option 2 (approved 2026-11-12)
# On a code-verified sale that reduces an over-limit position, research-quality issues
# (structured category whitelist) become WARNINGs and the sale goes to the user for
# approval, PAPER only. Everything else - and anything unclassified - keeps blocking.

import copy  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from agentrt.runtime.reports import make_report  # noqa: E402
from agentrt.runtime.review import CRITIC_SCHEMA, RESEARCH_QUALITY, research_quality_waiver  # noqa: E402

NO_BACKTESTS = issue("backtests is empty, so the view has no historical validation", severity="MEDIUM",
                     type_="missing_data", category="backtest_coverage")
POOR_RETURNS = issue("the portfolio sharpe_ratio and alpha are negative over the window", severity="MEDIUM",
                     type_="logic_gap", category="historical_performance")


def research_issue_sale(*issues):
    return sale(ScriptedLLM(review=[{"issues": list(issues)}]))


def test_option2_schema_requires_a_category_from_a_closed_list():
    item = CRITIC_SCHEMA["properties"]["issues"]["items"]
    assert "category" in item["required"]
    assert set(RESEARCH_QUALITY) < set(item["properties"]["category"]["enum"])
    assert RESEARCH_QUALITY == ("sample_length", "factor_coverage", "backtest_coverage",
                                "historical_performance", "signal_disagreement")


@pytest.mark.asyncio
@needs_strategies
async def test_option2_research_quality_issues_become_warnings_and_reach_approval():
    h, run, d = await research_issue_sale(NO_BACKTESTS, POOR_RETURNS)
    r = d["result"]
    assert r["critic"]["valid"] is False and r["critic"]["verdict"] == "request_revision"   # critic unchanged
    assert [i["severity"] for i in r["critic"]["unresolved"] if i["source"] == "llm"] == ["MEDIUM", "MEDIUM"]
    assert r["critic"]["llm_raw"] == {"issues": [NO_BACKTESTS, POOR_RETURNS]}                # raw review kept
    trade = r["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and trade["mode"] == "PAPER"
    w = trade["waiver"]
    assert w["granted"] and all(c["passed"] for c in w["conditions"])
    assert [(a["severity"], a["severity_original"], a["category"]) for a in trade["advisories"]] == [
        ("WARNING", "MEDIUM", "backtest_coverage"), ("WARNING", "MEDIUM", "historical_performance")]
    assert all(a["downgrade_basis"] for a in trade["advisories"])
    stored = await h.loop.proposals.get(trade["proposal"]["proposal_id"], OWNER)
    assert stored.status == "PENDING" and stored.evidence["advisories"] == trade["advisories"]
    assert stored.evidence["waiver"]["granted"] is True
    assert any(e["kind"] == "decision" and e["message"].startswith("research-quality issues downgraded")
               and e["data"]["waiver"]["granted"] for e in d["trace"])
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]                            # proposed only


@pytest.mark.asyncio
@needs_strategies
async def test_option2_owner_approval_runs_one_paper_execution_and_nobody_else_can_approve():
    h, _, d = await research_issue_sale(NO_BACKTESTS)
    pid = d["result"]["trade"]["proposal"]["proposal_id"]
    stranger = await h.loop.execute_approved(pid, ToolContext(user_id=OTHER))
    assert stranger.error_code == "APPROVAL_FORBIDDEN" and stranger.tool_calls == []
    done = await h.loop.execute_approved(pid, owner_ctx())
    fill = done.result["execution"]
    assert (fill["mode"], fill["status"], fill["side"], fill["quantity"]) == ("PAPER", "PAPER_FILLED", "sell", 5.0)
    assert [c["tool"] for c in done.to_dict()["tool_calls"]] == ["execution"]
    again = await h.loop.execute_approved(pid, owner_ctx())
    assert again.error_code == "APPROVAL_CONSUMED" and again.tool_calls == []
    assert (await h.loop.proposals.get(pid, OWNER)).status == "EXECUTED"


@pytest.mark.asyncio
@needs_strategies
async def test_option2_without_the_owners_approval_nothing_executes_and_the_proposal_expires():
    # there is no reject call: the user declines by not approving; the proposal then expires
    h, _, d = await research_issue_sale(NO_BACKTESTS)
    pid = d["result"]["trade"]["proposal"]["proposal_id"]
    proposal = await h.loop.proposals.get(pid, OWNER)
    assert proposal.status == "PENDING" and "execution" not in [c["tool"] for c in d["tool_calls"]]
    proposal.expires_at_epoch = 0                       # the approval window has passed
    late = await h.loop.execute_approved(pid, owner_ctx())
    assert late.error_code == "APPROVAL_EXPIRED" and late.tool_calls == []
    assert (await h.loop.proposals.get(pid, OWNER)).status == "EXPIRED"


@pytest.mark.asyncio
@needs_strategies
async def test_option2_stale_data_still_blocks_even_with_only_research_issues():
    _, _, d = await sale(ScriptedLLM(review=[{"issues": [NO_BACKTESTS]}]), stale=True)
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "STALE_DATA" in trade["reason"]
    w = trade["waiver"]
    assert not w["granted"] and not w["advisories"]
    failed = {c["condition"] for c in w["conditions"] if not c["passed"]}
    assert "no hard-rule issue at any severity" in failed
    assert any(e["message"].startswith("research-quality downgrade refused") for e in d["trace"])
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]


@pytest.mark.parametrize("bad,why", [
    (issue("the sale leaves sector exposure unexamined", severity="MEDIUM", type_="risk_conflict",
           category="historical_performance"), "always blocks"),
    (issue("sector exposure is unexamined", severity="MEDIUM", type_="constraint_violation",
           category="backtest_coverage"), "always blocks"),
    (issue("the portfolio risk label understates the risk", severity="MEDIUM", type_="logic_gap",
           category="risk_assessment"), "not on the research-quality whitelist"),
    (issue("the post-sale weight does not match the reported totals", severity="MEDIUM", type_="logic_gap",
           category="data_consistency"), "not on the research-quality whitelist"),
    (issue("the AAPL prices look stale", severity="MEDIUM", type_="missing_data",
           category="sample_length"), "hard-rule topic"),
    (issue("signal and risk figures contradict each other", severity="MEDIUM", type_="logic_gap",
           category="signal_disagreement"), "hard-rule topic"),
])
@pytest.mark.asyncio
@needs_strategies
async def test_option2_hard_risk_or_contradiction_issues_still_block(bad, why):
    _, _, d = await research_issue_sale(NO_BACKTESTS, bad)
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "LLM_REVIEW" in trade["reason"]
    assert not trade["waiver"]["granted"] and not trade["waiver"]["advisories"]
    assert any(why in b for b in trade["waiver"]["conditions"][-1]["value"])
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]


@pytest.mark.parametrize("category", [None, "other", "not_a_category"])
@pytest.mark.asyncio
@needs_strategies
async def test_option2_unclassified_or_unknown_issues_block(category):
    item = issue("the evidence for this sale is thin", severity="MEDIUM", type_="logic_gap", category=category)
    _, _, d = await research_issue_sale(item)
    c = d["result"]["critic"]
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED"
    if category == "other":                # schema-valid, but not a research-quality category
        assert c["reviewer"] == "rule+llm" and not trade["waiver"]["granted"]
    else:                                  # missing / outside the enum: the review is invalid
        assert c["llm_review"]["status"] == "invalid" and c["llm_review"]["error"]["field"] == "category"
        assert "LLM_REVIEW_INVALID" in trade["reason"] and not trade["waiver"]["granted"]


@pytest.mark.asyncio
@needs_strategies
async def test_option2_does_not_apply_to_a_sale_larger_than_the_holding():
    h, _, _ = await e2e.s2c_harness(ScriptedLLM(review=[{"issues": [NO_BACKTESTS]}]))
    d = (await h.run("我的AAPL仓位超过上限，减持 50 股 AAPL", portfolio_id=7)).to_dict()
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "Risk Agent veto" in trade["reason"]
    assert "waiver" not in trade and "execution" not in [c["tool"] for c in d["tool_calls"]]


# unit level: every condition of the gate, one at a time, on the workflow's own risk report.
# The reports come from fixtures/s2c_critic_inputs.json, a snapshot of this workflow run on the
# S2c data, so the gate and critic tests below need no strategy provider; the snapshot is
# checked against a fresh run by test_snapshot_matches_a_fresh_workflow_run (needs strategies).

SNAPSHOT_PATH = HERE / "fixtures" / "s2c_critic_inputs.json"


async def critic_inputs_from_workflow():
    """Regenerate with: python -c "import asyncio, test_s2c as T; T.write_snapshot()" (in this dir)."""
    _, _, d = await research_issue_sale(NO_BACKTESTS)
    r = d["result"]
    pick = lambda name: [a for a in r["agents"] if a["agent"] == name][-1]   # noqa: E731
    research, risk = pick("research"), pick("risk")
    for limit in risk["findings"].get("limits", {}).values():
        if isinstance(limit, dict):
            limit.pop("set_at", None)          # wall-clock time of the memory write; not an input
    return {"note": "research and risk reports of the S2c legitimate sale (rule plan, 5 AAPL), "
                    "generated by critic_inputs_from_workflow(); not model output",
            "plan": {"goal": r["goal"], "intent": r["intent"], "symbol": r["symbol"],
                     "portfolio_id": r["portfolio_id"], "trade": {"side": "sell", "quantity": 5.0},
                     "direct_execution_requested": False},
            "research": {k: research[k] for k in ("status", "summary", "findings", "evidence")},
            "risk": {k: risk[k] for k in ("status", "summary", "findings", "evidence")},
            "critic_issues": r["critic"]["issues"]}


def write_snapshot():
    SNAPSHOT_PATH.write_text(json.dumps(asyncio.run(critic_inputs_from_workflow()), ensure_ascii=False,
                                        indent=1, sort_keys=True) + "\n", encoding="utf-8")


SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")) if SNAPSHOT_PATH.exists() else None


@needs_strategies
@pytest.mark.asyncio
async def test_snapshot_matches_a_fresh_workflow_run():
    fresh = json.loads(json.dumps(await critic_inputs_from_workflow(), ensure_ascii=False, sort_keys=True))
    assert fresh == SNAPSHOT


@pytest.fixture(scope="module")
def granted_inputs():
    k = SNAPSHOT["risk"]["findings"]
    issues = SNAPSHOT["critic_issues"]
    blocking = [i for i in issues if i["severity"] in ("HIGH", "MEDIUM") and i["code"] != "POSITION_LIMIT"]
    return k, issues, blocking


def waive(k, issues, blocking, decision="REDUCE_EXPOSURE", side="sell", status="VETO", review="used"):
    plan = SimpleNamespace(symbol="AAPL", trade={"side": side, "quantity": 5})
    return research_quality_waiver(plan, k, make_report("risk", status, "x", k, []), decision, issues,
                                   blocking, review)


def test_option2_gate_grants_on_the_real_report(granted_inputs):
    assert waive(*granted_inputs)["granted"] is True


@pytest.mark.parametrize("review", ["invalid", "unavailable", "off", None])
def test_option2_gate_needs_a_model_review_that_passed_validation(granted_inputs, review):
    w = waive(*granted_inputs, review=review)
    assert not w["granted"] and [c["condition"] for c in w["conditions"] if not c["passed"]] == [
        "the model review passed schema and semantic validation"]


def _set(k, path, value):
    k = copy.deepcopy(k)
    node = k
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return k


@pytest.mark.parametrize("change,condition", [
    (lambda k: _set(k, ("trade_check", "reduces_exposure"), False), "sale of a held long position"),
    (lambda k: _set(k, ("concentration_breaches",), {}), "above the single-position limit"),
    (lambda k: _set(k, ("vetoes",), k["vetoes"] + ["AAPL risk level is high"]), "the only veto"),
    (lambda k: _set(k, ("checks",), k["checks"] + [{"check": "risk_tolerance", "status": "veto",
                                                    "detail": "x"}]), "the only veto"),
    (lambda k: _set(k, ("trade_check", "projected_weight_pct"), k["trade_check"]["weight_pct"]),
     "lowers the symbol's measured weight"),
    (lambda k: _set(k, ("trade_check", "projected_weight_pct"), None), "lowers the symbol's measured weight"),
    (lambda k: _set(k, ("trade_check", "others_over_limit_after"), {"JNJ": 20.5}), "no other holding"),
    (lambda k: _set(k, ("portfolio_risk", "metrics", "var_95_pct"), 3.0), "risk levels follow"),  # label low, VaR medium
    (lambda k: _set(k, ("symbol_risk", "metrics", "annualized_volatility"), None), "risk levels follow"),
    (lambda k: _set(k, ("portfolio_risk", "unavailable"), {"beta": "no benchmark"}), "every risk input"),
    (lambda k: _set(k, ("exposure", "missing"), {"KO": "no price"}), "every risk input"),
])
def test_option2_gate_refuses_when_any_condition_fails(granted_inputs, change, condition):
    k, issues, blocking = granted_inputs
    w = waive(change(k), issues, blocking)
    assert not w["granted"] and not w["advisories"]
    assert [c["condition"] for c in w["conditions"] if not c["passed"]] and any(
        condition in c["condition"] for c in w["conditions"] if not c["passed"])


def test_option2_gate_refuses_a_non_reducing_decision_a_buy_and_a_hard_rule_issue(granted_inputs):
    k, issues, blocking = granted_inputs
    assert not waive(k, issues, blocking, decision="HOLD")["granted"]
    assert not waive(k, issues, blocking, side="buy")["granted"]
    stale = {"type": "missing_data", "code": "STALE_DATA", "severity": "LOW", "message": "old",
             "suggestion": "", "remedy": None, "source": "rule", "category": None}
    assert not waive(k, issues + [stale], blocking)["granted"]        # any severity


def test_option2_hard_limits_and_risk_policy_are_unchanged():
    from agentrt.tools import portfolio as P
    assert (RiskAgent.HIGH_RISK_VAR_PCT, RiskAgent.MEDIUM_RISK_VAR_PCT) == (5.0, 2.0)
    assert (P.HIGH_VOLATILITY, P.MEDIUM_VOLATILITY, P.SEVERE_DRAWDOWN) == (0.60, 0.30, -0.35)


# ------------------------------------------- acceptance review: replay of the real DeepSeek run
# fixtures/s2c_deepseek_20261112_204122.json holds every answer of the 20:41 real run exactly
# as received. Replaying them offline reproduces that run (no model is called here). Where a
# case needs a variant, it is built from a real item and the change is named in the test.

REAL = json.loads((HERE / "fixtures" / "s2c_deepseek_20261112_204122.json").read_text(encoding="utf-8"))


def real(scenario, schema):
    return [c["response"] for c in REAL["calls"] if c["scenario"] == scenario and c["schema"] == schema]


LEGIT, STALE = "S2c_legit_sale_v2", "S2c_stale_sale"
R2 = real(LEGIT, "review")[1]["issues"]          # the round-2 review that was rejected as a whole


def replay(round2=None, scenario=LEGIT, stale=False):
    reviews = real(scenario, "review")
    if round2 is not None:
        reviews = reviews[:1] + [{"issues": round2}]
    llm = ScriptedLLM(*real(scenario, "plan"), review=reviews, decision=real(scenario, "decision"))
    return sale(llm, stale=stale)


def critic_round(d, n):
    return [a for a in d["result"]["agents"] if a["agent"] == "critic" and a["round"] == n][0]["findings"]


def fixed_type(item, type_):
    return {**item, "type": type_}


@pytest.mark.asyncio
@needs_strategies
async def test_acc_replay_real_wrong_type_review_blocks_instead_of_falling_back_to_a_proposal():
    # THE regression case. The saved real round-2 answer has type=data_consistency. Before the
    # fix the whole review was dropped and the rule review alone proposed the sale (PENDING).
    h, _, d = await replay()
    c2 = critic_round(d, 2)
    assert c2["reviewer"] == "rule" and c2["llm_raw"] == {"issues": R2}     # raw answer kept, unused
    assert c2["llm_review"] == {"required": True, "status": "invalid", "error": {
        "kind": "validation", "detail": "unknown issue type 'data_consistency' in issue 0",
        "field": "type", "index": 0, "value": "data_consistency"}}
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "LLM_REVIEW_INVALID" in trade["reason"]
    assert not trade["waiver"]["granted"]                                   # option 2 cannot lift it
    failed = {c["condition"] for c in trade["waiver"]["conditions"] if not c["passed"]}
    assert {"no hard-rule issue at any severity",
            "the model review passed schema and semantic validation"} <= failed
    assert d["result"]["reanalysis"]["performed"] is True                   # option 1 ran, did not lift it
    assert await h.loop.proposals.list(OWNER) == []                         # nothing to approve
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    assert any(e["kind"] == "decision" and e["message"].startswith("research-quality downgrade refused")
               for e in d["trace"])
    assert d["result"]["review_policy"] == {"required": True, "source": "default: model configured"}


@pytest.mark.asyncio
@needs_strategies
async def test_acc_real_research_quality_items_become_warnings():
    # case 1: the real round-2 items whose type and category are both research quality
    keep = [i for i in R2 if i["type"] in ("missing_data", "logic_gap")
            and i["category"] in RESEARCH_QUALITY]
    assert len(keep) == 5 and {i["severity"] for i in keep} == {"MEDIUM", "LOW"}
    h, _, d = await replay(keep)
    trade = d["result"]["trade"]
    assert critic_round(d, 2)["reviewer"] == "rule+llm"
    assert trade["status"] == "AWAITING_APPROVAL" and trade["waiver"]["granted"]
    assert sorted(a["category"] for a in trade["advisories"]) == ["backtest_coverage", "historical_performance",
                                                                 "signal_disagreement"]   # the 3 MEDIUM ones
    pid = trade["proposal"]["proposal_id"]
    assert (await h.loop.proposals.get(pid, OWNER)).status == "PENDING"        # case 8: not executed
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]
    done = await h.loop.execute_approved(pid, owner_ctx())                     # case 9
    assert (done.result["execution"]["mode"], done.result["execution"]["status"]) == ("PAPER", "PAPER_FILLED")
    again = await h.loop.execute_approved(pid, owner_ctx())
    assert again.error_code == "APPROVAL_CONSUMED" and again.tool_calls == []


@pytest.mark.asyncio
@needs_strategies
async def test_acc_real_data_consistency_item_blocks_when_well_formed():
    # case 2: the real rsi_reversion contradiction, with only its type made valid (logic_gap)
    keep = [i for i in R2 if i["category"] in RESEARCH_QUALITY and i["type"] in ("missing_data", "logic_gap")]
    contradiction = fixed_type(R2[0], "logic_gap")
    assert contradiction["category"] == "data_consistency" and contradiction["severity"] == "MEDIUM"
    _, _, d = await replay(keep + [contradiction])
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and not trade["waiver"]["granted"]
    assert any("data_consistency" in why for why in trade["waiver"]["conditions"][-1]["value"])


@pytest.mark.asyncio
@needs_strategies
async def test_acc_real_items_with_unknown_category_block():
    # case 3: a real research item with its category removed / replaced
    base = [i for i in R2 if i["category"] == "backtest_coverage"][0]
    for variant in ({k: v for k, v in base.items() if k != "category"}, {**base, "category": "liquidity"}):
        _, _, d = await replay([variant])
        trade = d["result"]["trade"]
        assert trade["status"] == "NOT_PROPOSED" and not trade["waiver"]["granted"]


@pytest.mark.asyncio
@needs_strategies
async def test_acc_real_permission_item_blocks():
    # case 6: the real MEDIUM constraint_violation/permission item is a hard topic
    perm = [i for i in R2 if i["category"] == "permission"][0]
    assert (perm["type"], perm["severity"]) == ("constraint_violation", "MEDIUM")
    _, _, d = await replay([perm])
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and not trade["waiver"]["granted"]


@pytest.mark.asyncio
@needs_strategies
async def test_acc_real_stale_run_blocks_and_keeps_the_rejected_review():
    # cases 5 + 6 + 7 + 10: the real stale-run review cites a derived number ("19,366") the
    # grounding check cannot find, so it is rejected; STALE_DATA still blocks from the rules
    _, _, d = await replay(scenario=STALE, stale=True)
    c1 = critic_round(d, 1)
    assert c1["reviewer"] == "rule" and "cites numbers not in the reports" in c1["fallback_reason"]
    assert c1["llm_raw"] == real(STALE, "review")[0]
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "STALE_DATA" in trade["reason"]
    assert not trade["waiver"]["granted"]
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]


@pytest.mark.asyncio
@needs_strategies
async def test_acc_fabricated_number_in_a_real_item_rejects_the_review_and_blocks():
    # case 7: one real research item with an invented figure. The invalid review is not used
    # (the figure is never trusted) and - since the fix - the sale is not proposed.
    keep = [fixed_type(R2[0], "logic_gap")]
    fabricated = {**[i for i in R2 if i["category"] == "historical_performance"][0],
                  "message": "AAPL lost 87.3 percent over the backtest window"}
    _, _, d = await replay(keep + [fabricated])
    c2 = critic_round(d, 2)
    assert c2["reviewer"] == "rule" and "87.3" in c2["fallback_reason"]
    assert not any(i["source"] == "llm" for i in c2["issues"])            # nothing from it is trusted
    assert c2["llm_review"]["status"] == "invalid" and c2["llm_review"]["error"]["index"] == 1
    trade = d["result"]["trade"]
    assert trade["status"] == "NOT_PROPOSED" and "LLM_REVIEW_INVALID" in trade["reason"]


@pytest.mark.asyncio
@needs_strategies
async def test_acc_backtest_and_var_figures_carry_their_basis():
    # the two data questions: labels added, numbers unchanged
    _, _, d = await replay()
    r2 = [a for a in d["result"]["agents"] if a["agent"] == "research" and a["round"] == 2][0]["findings"]
    rsi = [b for b in r2["backtests"] if b["strategy"] == "rsi_reversion"][0]
    assert (rsi["trade_count"], rsi["fill_count"], rsi["open_quantity_at_end"]) == (0, 1, 74.0)
    assert rsi["total_return"] == -0.0261 and "closed round trips" in rsi["basis"]["trade_count"]
    k = [a for a in d["result"]["agents"] if a["agent"] == "risk"][-1]["findings"]
    m = k["portfolio_risk"]["metrics"]
    assert m["var_base_value"] == pytest.approx(k["exposure"]["total_value"] + D.CASH)
    assert m["var_95"] == pytest.approx(m["var_95_pct"] / 100 * m["var_base_value"], abs=0.01)
    assert "cash excluded" in k["exposure"]["weights_basis"] and "var_base_value" in k["portfolio_risk"]["var_basis"]


# --------------------------------------------- fallback fix: an unusable required review blocks

from agentrt.runtime.llm import LLMError, LLMOutputError  # noqa: E402


class FailingReviewLLM(ScriptedLLM):
    """Plans and decisions as scripted; every review call raises ``exc``."""

    def __init__(self, exc, *plans, **kw):
        super().__init__(*plans, **kw)
        self.exc = exc

    async def generate(self, system, prompt, schema_name, schema):
        if schema_name == "review":
            self.calls.append(schema_name)
            raise self.exc
        return await super().generate(system, prompt, schema_name, schema)


def blocked_by(d, code):
    trade = d["result"]["trade"]
    return trade["status"] == "NOT_PROPOSED" and code in trade["reason"] and \
        "execution" not in [c["tool"] for c in d["tool_calls"]]


@pytest.mark.parametrize("exc,code,kind", [
    (LLMError("model call timed out (review)"), "LLM_REVIEW_UNAVAILABLE", "call"),          # case 2
    (LLMError("model call failed: HTTP 500 upstream error"), "LLM_REVIEW_UNAVAILABLE", "call"),  # case 3
    (LLMError("model rate limited (429)"), "LLM_REVIEW_UNAVAILABLE", "call"),
    (LLMOutputError("model returned invalid JSON for review"), "LLM_REVIEW_INVALID", "output"),  # case 4
    (LLMOutputError("model output truncated (review)"), "LLM_REVIEW_INVALID", "output"),
])
@pytest.mark.asyncio
@needs_strategies
async def test_fix_review_call_failures_block_the_trade(exc, code, kind):
    _, _, d = await sale(FailingReviewLLM(exc))
    c = d["result"]["critic"]
    assert c["llm_review"]["status"] in ("invalid", "unavailable") and c["llm_review"]["error"]["kind"] == kind
    assert blocked_by(d, code)
    failure = [i for i in c["issues"] if i["code"] == code][0]
    assert failure["severity"] == "HIGH" and failure["category"] is None and str(exc) in failure["message"]


@pytest.mark.parametrize("answer,field", [
    ({"issues": "none"}, "issues"),                                                   # case 4: shape
    ({"issues": [], "verdict": "approve"}, "issues"),                                 # extra top-level key
    ({"issues": [{k: v for k, v in NO_BACKTESTS.items() if k != "message"}]}, "message"),   # case 6
    ({"issues": [{k: v for k, v in NO_BACKTESTS.items() if k != "severity"}]}, "severity"),
    ({"issues": [{**NO_BACKTESTS, "confidence": "high"}]}, "confidence"),            # field outside schema
    ({"issues": [{**NO_BACKTESTS, "type": "liquidity_risk"}]}, "type"),               # case 5
    ({"issues": [{**NO_BACKTESTS, "severity": "CRITICAL"}]}, "severity"),
])
@pytest.mark.asyncio
@needs_strategies
async def test_fix_schema_invalid_reviews_block_and_keep_diagnostics(answer, field):
    _, _, d = await sale(ScriptedLLM(review=[answer]))
    c = d["result"]["critic"]
    assert c["llm_review"]["status"] == "invalid" and c["llm_review"]["error"]["field"] == field
    assert c["llm_raw"] == answer                           # the raw answer is kept for diagnosis ...
    assert not any(i["source"] == "llm" for i in c["issues"])   # ... but none of it is trusted
    assert blocked_by(d, "LLM_REVIEW_INVALID")


@pytest.mark.asyncio
@needs_strategies
async def test_fix_more_than_eight_issues_are_all_validated():
    # the validator used to read only items[:8]; a 9th item could not be seen at all
    answer = {"issues": [NO_BACKTESTS] * 8 + [{**NO_BACKTESTS, "type": "data_consistency"}]}
    _, _, d = await sale(ScriptedLLM(review=[answer]))
    assert d["result"]["critic"]["llm_review"]["error"]["index"] == 8
    assert blocked_by(d, "LLM_REVIEW_INVALID")


@pytest.mark.asyncio
@needs_strategies
async def test_fix_a_failed_first_review_is_not_lifted_by_a_valid_second_round():
    # case 9 (other order): round 1 invalid, option 1 re-analyses, round 2 is usable and clean -
    # the round-1 failure still blocks. On this data the rules give no remedy of their own, so
    # a test double adds one rule remedy to round 1 (only to make re-analysis happen).
    h, _, _ = await e2e.s2c_harness(ScriptedLLM(
        REAL_PLAN, review=[{"issues": [{**SHORT_SAMPLE, "type": "data_consistency"}]}, EMPTY_REVIEW]))
    real_run = h.workflow.critic.run

    async def critic_with_rule_remedy(plan, research, risk, memory, round=1):
        rep = await real_run(plan, research, risk, memory, round=round)
        if round != 1:
            return rep
        f = rep.findings                                   # reports are immutable: build a new one
        f["issues"].append({"type": "missing_data", "code": "INSUFFICIENT_HISTORY", "severity": "MEDIUM",
                            "message": "test double", "suggestion": "", "remedy": {"days": 365},
                            "source": "rule", "category": "sample_length"})
        return make_report("critic", "REQUEST_REVISION", "request_revision", {**f, "valid": False},
                           list(rep.evidence), round=1)

    h.workflow.critic.run = critic_with_rule_remedy
    d = (await h.run(e2e.SALE_GOAL, portfolio_id=7)).to_dict()
    assert d["result"]["reanalysis"]["performed"] is True
    assert critic_round(d, 1)["llm_review"]["status"] == "invalid"
    assert critic_round(d, 2)["llm_review"]["status"] == "used"
    assert not any(i["code"] == "LLM_REVIEW_INVALID" for i in critic_round(d, 2)["issues"])
    assert blocked_by(d, "LLM_REVIEW_INVALID")          # carried over from round 1


@pytest.mark.asyncio
@needs_strategies
async def test_fix_rule_only_mode_without_a_model_is_unchanged():
    # case 11: no model configured and no requirement - the original rule-only behaviour
    h, _, d = await sale()
    assert d["result"]["review_policy"] == {"required": False, "source": "default: no model (rule-only)"}
    assert d["result"]["critic"]["llm_review"]["status"] == "off"
    assert d["result"]["trade"]["status"] == "AWAITING_APPROVAL"


class UnusableLLM(ScriptedLLM):
    def available(self):
        return False, "model unavailable: AGENT_LLM_API_KEY is not configured"


@pytest.mark.asyncio
@needs_strategies
async def test_fix_a_configured_but_unusable_model_cannot_switch_the_review_off():
    # case 11: removing the key from a configured model is not the rule-only mode
    _, _, d = await sale(UnusableLLM())
    assert d["result"]["review_policy"]["required"] is True
    assert d["result"]["critic"]["llm_review"]["error"]["kind"] == "not_configured"
    assert blocked_by(d, "LLM_REVIEW_UNAVAILABLE")


@pytest.mark.asyncio
@needs_strategies
async def test_fix_required_by_environment_blocks_even_with_no_model(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_REVIEW_REQUIRED", "1")
    _, _, d = await sale()
    assert d["result"]["review_policy"] == {"required": True, "source": "AGENT_LLM_REVIEW_REQUIRED"}
    assert blocked_by(d, "LLM_REVIEW_UNAVAILABLE")


@pytest.mark.asyncio
@needs_strategies
async def test_fix_an_explicit_opt_out_is_recorded_in_every_run(monkeypatch):
    # the only way back to the old fallback with a model configured: an explicit setting,
    # visible in the run result (documented in S2C_SECURITY_FALLBACK_FINAL.md)
    monkeypatch.setenv("AGENT_LLM_REVIEW_REQUIRED", "0")
    _, _, d = await sale(FailingReviewLLM(LLMError("model call timed out (review)")))
    assert d["result"]["review_policy"] == {"required": False, "source": "AGENT_LLM_REVIEW_REQUIRED"}
    assert d["result"]["critic"]["llm_review"]["status"] == "unavailable"
    assert d["result"]["trade"]["status"] == "AWAITING_APPROVAL"


@pytest.mark.asyncio
@needs_strategies
async def test_fix_analysis_without_a_trade_keeps_the_rule_review():
    # scope: the block applies to trade proposals; an analysis records the fallback only
    h, _, _ = await e2e.s2c_harness(FailingReviewLLM(LLMError("model call timed out (review)")))
    d = (await h.run("分析我的AAPL仓位风险", portfolio_id=7)).to_dict()
    c = d["result"]["critic"]
    assert c["llm_review"]["status"] == "unavailable" and d["result"]["trade"] is None
    assert not any(i["code"] in ("LLM_REVIEW_INVALID", "LLM_REVIEW_UNAVAILABLE") for i in c["issues"])


@pytest.mark.asyncio
@needs_strategies
async def test_fix_every_failure_and_block_is_audited():
    # case 15: the trace carries the critic report (raw answer, error field), the refused
    # downgrade and "no trade proposal"; the archive holds the run
    h, _, d = await replay()
    text = json.dumps([e for e in d["trace"] if e["kind"] == "agent_report"], ensure_ascii=False)
    assert "LLM_REVIEW_INVALID" in text and '"field": "type"' in text and "data_consistency" in text
    assert any(e["kind"] == "approval" and "no trade proposal" in e["message"] for e in d["trace"])
    assert d["task_id"] in h.workflow.archive.saved


# ------------------------------------- final real run (21:22): replay + Sharpe / risk-label basis
# fixtures/s2c_deepseek_20261112_212212.json: every answer of the final real run as received;
# the one decision call that returned invalid JSON is replayed as that same error.

FINAL = json.loads((HERE / "fixtures" / "s2c_deepseek_20261112_212212.json").read_text(encoding="utf-8"))


def final_answers(scenario, schema):
    out = []
    for c in FINAL["calls"]:
        if c["scenario"] == scenario and c["schema"] == schema:
            out.append(c["response"] if c["outcome"] == "parsed" else LLMOutputError(c["error"]))
    return out


def final_replay(scenario=LEGIT, stale=False, round2=None):
    reviews = final_answers(scenario, "review")
    if round2 is not None:
        reviews = reviews[:1] + [{"issues": round2}]
    llm = ScriptedLLM(*final_answers(scenario, "plan"), review=reviews,
                      decision=final_answers(scenario, "decision"))
    return sale(llm, stale=stale)


F2 = final_answers(LEGIT, "review")[1]["issues"]


@pytest.mark.asyncio
@needs_strategies
async def test_final_replay_reproduces_the_real_block_of_the_legit_sale():
    _, _, d = await final_replay()
    r = d["result"]
    assert [critic_round(d, n)["llm_review"]["status"] for n in (1, 2)] == ["used", "used"]
    assert r["reanalysis"]["performed"] is True
    assert r["decision"]["decided_by"] == "rule" and "invalid JSON" in r["decision"]["fallback_reason"]
    trade = r["trade"]
    assert trade["status"] == "NOT_PROPOSED" and not trade["waiver"]["granted"]
    assert [c["condition"] for c in trade["waiver"]["conditions"] if not c["passed"]] == [
        "every remaining issue is on the research-quality whitelist"]
    assert "execution" not in [c["tool"] for c in d["tool_calls"]]


@pytest.mark.asyncio
@needs_strategies
async def test_final_replay_stale_round2_stays_invalid_and_blocked():
    # the report labels added since the run do not make the rejected review acceptable
    _, _, d = await final_replay(STALE, stale=True)
    assert critic_round(d, 2)["llm_review"]["status"] == "invalid"
    assert blocked_by(d, "LLM_REVIEW_INVALID") and "STALE_DATA" in d["result"]["trade"]["reason"]


@pytest.mark.asyncio
@needs_strategies
async def test_final_replay_research_only_items_reach_approval_only_when_every_rule_holds():
    keep = [i for i in F2 if i["type"] in ("missing_data", "logic_gap") and i["category"] in RESEARCH_QUALITY]
    assert {i["category"] for i in keep} == {"backtest_coverage", "sample_length", "historical_performance"}
    h, _, d = await final_replay(round2=keep)
    trade = d["result"]["trade"]
    assert trade["status"] == "AWAITING_APPROVAL" and trade["waiver"]["granted"]
    assert all(c["passed"] for c in trade["waiver"]["conditions"])
    pid = trade["proposal"]["proposal_id"]
    done = await h.loop.execute_approved(pid, owner_ctx())
    assert (done.result["execution"]["mode"], done.result["execution"]["status"]) == ("PAPER", "PAPER_FILLED")
    assert (await h.loop.execute_approved(pid, owner_ctx())).error_code == "APPROVAL_CONSUMED"


@pytest.mark.asyncio
@needs_strategies
async def test_final_the_positive_return_negative_sharpe_contradiction_is_the_risk_free_rate():
    # the real round-2 data_consistency claim, checked against the reported numbers
    claim = [i for i in F2 if i["category"] == "data_consistency"][0]
    assert "sharpe" in claim["message"].lower()
    _, _, d = await final_replay()
    bts = [a for a in d["result"]["agents"] if a["agent"] == "research" and a["round"] == 2][0]["findings"]
    calls = {c["call_id"]: c["output_summary"] for c in d["tool_calls"] if c["tool"] == "backtest"}
    for b in bts["backtests"]:
        m = calls[b["call_id"]]["metrics"]
        if b["total_return"] > 0:                      # momentum, macd
            assert b["sharpe_ratio"] < 0 and m["annual_return"] < 0.03
            assert (m["annual_return"] - 0.03) / m["volatility"] == pytest.approx(b["sharpe_ratio"], abs=0.02)
        assert "0.03" in b["basis"]["sharpe_ratio"]


@needs_strategies
def test_final_backtest_sharpe_follows_the_stated_formula():
    # the label states quantrisk's formula; recompute it from the daily account returns
    from agentrt.ports.strategies import get_strategy
    from agentrt.tools.analysis import BACKTEST_BASIS, bars_to_frame
    bars = [b for b in D.make_series(TODAY)["AAPL"] if "2025-09-01" <= b["date"] <= "2026-08-31"]
    res = get_strategy("momentum", None).run(bars_to_frame(bars, "AAPL"), initial_capital=100_000,
                                             commission=0.001, slippage=0.0005)
    r = res.returns.dropna()
    assert res.metrics["sharpe_ratio"] == pytest.approx((r.mean() * 252 - 0.03) / (r.std() * np.sqrt(252)),
                                                        abs=1e-4)
    assert res.metrics["volatility"] == pytest.approx(r.std() * np.sqrt(252), abs=1e-4)
    assert "0.03" in BACKTEST_BASIS["sharpe_ratio"] and "sqrt(252)" in BACKTEST_BASIS["volatility"]


@pytest.mark.asyncio
@needs_strategies
async def test_final_low_risk_labels_follow_their_stated_policy():
    # the "low" labels the model questioned: recomputed from metrics and policy, no returns used
    _, _, d = await final_replay()
    k = [a for a in d["result"]["agents"] if a["agent"] == "risk"][-1]["findings"]
    pf, sym = k["portfolio_risk"], k["symbol_risk"]
    assert (pf["level"], sym["level"]) == ("low", "low")
    assert abs(pf["metrics"]["var_95_pct"]) < pf["policy"]["var_medium_pct"] == 2.0
    assert sym["metrics"]["annualized_volatility"] < sym["policy"]["medium_volatility"] == 0.30
    assert sym["metrics"]["max_drawdown"] > sym["policy"]["severe_drawdown"] == -0.35
    assert pf["metrics"]["sharpe_ratio"] < 0                      # negative Sharpe, low VaR: both true
    assert "Sharpe" in pf["level_basis"] and "not used" in pf["level_basis"]
    assert "not used" in sym["level_basis"]


# ------------------------------- critic and policy without a strategy provider (snapshot input)
# The critic itself needs no strategies: these run it directly on the snapshot reports, so the
# schema checks and the required-review block are tested in every environment.

from agentrt.runtime.review import CriticAgent  # noqa: E402
from agentrt.runtime.workflow import review_policy  # noqa: E402

NO_MEMORY = SimpleNamespace(decisions=[], feedback=[], lines=lambda: [])


def snapshot_plan(trade=True):
    plan = dict(SNAPSHOT["plan"])
    if not trade:
        plan["trade"] = None
    return SimpleNamespace(**plan)


async def critic_on_snapshot(llm, required=True, trade=True):
    research = make_report("research", SNAPSHOT["research"]["status"], SNAPSHOT["research"]["summary"],
                           SNAPSHOT["research"]["findings"], SNAPSHOT["research"]["evidence"])
    risk = make_report("risk", SNAPSHOT["risk"]["status"], SNAPSHOT["risk"]["summary"],
                       SNAPSHOT["risk"]["findings"], SNAPSHOT["risk"]["evidence"])
    critic = CriticAgent(lambda: TODAY, llm=llm, review_required=required)
    return (await critic.run(snapshot_plan(trade), research, risk, NO_MEMORY)).findings


def codes(findings):
    return [i["code"] for i in findings["issues"]]


@pytest.mark.parametrize("exc,code,kind", [
    (LLMError("model call timed out (review)"), "LLM_REVIEW_UNAVAILABLE", "call"),
    (LLMError("model call failed: HTTP 500 upstream error"), "LLM_REVIEW_UNAVAILABLE", "call"),
    (LLMOutputError("model returned invalid JSON for review"), "LLM_REVIEW_INVALID", "output"),
])
@pytest.mark.asyncio
async def test_critic_unit_unusable_required_review_adds_a_blocking_issue(exc, code, kind):
    f = await critic_on_snapshot(FailingReviewLLM(exc))
    failure = [i for i in f["issues"] if i["code"] == code][0]
    assert failure["severity"] == "HIGH" and failure["category"] is None and failure["remedy"] is None
    assert f["valid"] is False and f["llm_review"]["error"]["kind"] == kind


@pytest.mark.parametrize("answer,field", [
    ({"issues": "none"}, "issues"),
    ({"issues": [], "verdict": "approve"}, "issues"),
    ({"issues": [{k: v for k, v in NO_BACKTESTS.items() if k != "category"}]}, "category"),
    ({"issues": [{k: v for k, v in NO_BACKTESTS.items() if k != "message"}]}, "message"),
    ({"issues": [{**NO_BACKTESTS, "confidence": "high"}]}, "confidence"),
    ({"issues": [{**NO_BACKTESTS, "severity": "CRITICAL"}]}, "severity"),
    ({"issues": [{**NO_BACKTESTS, "category": "liquidity"}]}, "category"),
    ({"issues": [issue("AAPL returned 87.3 percent over the window")]}, "message"),     # invented figure
    ({"issues": [NO_BACKTESTS] * 8 + [{**NO_BACKTESTS, "type": "data_consistency"}]}, "type"),  # 9th item
])
@pytest.mark.asyncio
async def test_critic_unit_schema_invalid_review_is_not_used_and_blocks_a_trade(answer, field):
    f = await critic_on_snapshot(ScriptedLLM(review=[answer]))
    assert f["llm_review"]["status"] == "invalid" and f["llm_review"]["error"]["field"] == field
    assert f["llm_raw"] == answer and not any(i["source"] == "llm" for i in f["issues"])
    assert "LLM_REVIEW_INVALID" in codes(f)


@pytest.mark.asyncio
async def test_critic_unit_the_real_wrong_type_answer_is_invalid():
    f = await critic_on_snapshot(ScriptedLLM(review=[{"issues": R2}]))        # recorded DeepSeek answer
    assert f["llm_review"]["error"] == {"kind": "validation", "field": "type", "index": 0,
                                        "value": "data_consistency",
                                        "detail": "unknown issue type 'data_consistency' in issue 0"}
    assert "LLM_REVIEW_INVALID" in codes(f)


@pytest.mark.asyncio
async def test_critic_unit_a_valid_review_is_used_and_adds_no_failure():
    f = await critic_on_snapshot(ScriptedLLM(review=[{"issues": [NO_BACKTESTS]}]))
    assert f["llm_review"] == {"required": True, "status": "used", "error": None}
    assert "LLM_REVIEW" in codes(f) and not set(codes(f)) & {"LLM_REVIEW_INVALID", "LLM_REVIEW_UNAVAILABLE"}


@pytest.mark.parametrize("llm,required,trade,expected", [
    (FailingReviewLLM(LLMError("model call timed out (review)")), False, True, None),   # not required
    (FailingReviewLLM(LLMError("model call timed out (review)")), True, False, None),   # no trade
    (UnusableLLM(), True, True, "LLM_REVIEW_UNAVAILABLE"),                              # key removed
    (None, True, True, "LLM_REVIEW_UNAVAILABLE"),                                       # no model, required
    (None, False, True, None),                                                          # rule-only mode
])
@pytest.mark.asyncio
async def test_critic_unit_when_the_block_applies(llm, required, trade, expected):
    f = await critic_on_snapshot(llm, required=required, trade=trade)
    failures = [c for c in codes(f) if c in ("LLM_REVIEW_INVALID", "LLM_REVIEW_UNAVAILABLE")]
    assert failures == ([expected] if expected else [])


def test_review_policy_resolution(monkeypatch):
    monkeypatch.delenv("AGENT_LLM_REVIEW_REQUIRED", raising=False)
    model = object()
    assert review_policy(model, None) == {"required": True, "source": "default: model configured"}
    assert review_policy(None, None) == {"required": False, "source": "default: no model (rule-only)"}
    assert review_policy(None, True) == {"required": True, "source": "argument"}
    assert review_policy(model, False) == {"required": False, "source": "argument"}
    monkeypatch.setenv("AGENT_LLM_REVIEW_REQUIRED", "1")
    assert review_policy(None, None) == {"required": True, "source": "AGENT_LLM_REVIEW_REQUIRED"}
    monkeypatch.setenv("AGENT_LLM_REVIEW_REQUIRED", "0")
    assert review_policy(model, None) == {"required": False, "source": "AGENT_LLM_REVIEW_REQUIRED"}


def test_option2_gate_cannot_downgrade_a_review_failure(granted_inputs):
    k, issues, blocking = granted_inputs
    failure = {"type": "missing_data", "code": "LLM_REVIEW_INVALID", "severity": "HIGH", "message": "x",
               "suggestion": "", "remedy": None, "source": "rule", "category": None}
    w = waive(k, issues + [failure], blocking + [failure])
    failed = {c["condition"] for c in w["conditions"] if not c["passed"]}
    assert not w["granted"] and {"no hard-rule issue at any severity",
                                 "every remaining issue is on the research-quality whitelist"} <= failed
