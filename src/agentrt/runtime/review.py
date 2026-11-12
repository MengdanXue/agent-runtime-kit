# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Critic Agent and Portfolio Decision Agent.

Critic - a structured review of the research and risk reports::

    {"valid": false, "verdict": "request_revision",
     "issues": [{"type": "missing_data", "severity": "HIGH",
                 "message": "...", "suggestion": "fetch longer history", "remedy": {...}}]}

Issue types: missing_data | risk_conflict | logic_gap | constraint_violation.
The critic never changes a result; it can only approve or request a revision.
Rule checks always run; when a model is configured it may add issues (never
remove the rule's), limited to the same types and remedies, and every number
it cites must appear in the reports.

Decision - one structured decision from the Research, Risk and Critic reports
plus memory context::

    {"decision": "...", "confidence": "high|medium|low",
     "rationale": ["..."], "action_level": "ANALYSIS|PROPOSE|SIMULATE"}

With a model, the LLM proposes the decision; the validator accepts it only if
the decision is allowed for this goal and these reports (a Risk veto rules out
new exposure, unassessed risk rules out a direction, a strategy must be one
that traded), the level is at most PROPOSE/SIMULATE, the confidence respects
the critic, and every number in the rationale is found in the evidence.
Otherwise - or without a model - the rule decision is used and the fallback
reason recorded. Execution is never decided here: a requested trade becomes a
PAPER proposal only when the decision matches it (a sale of a held long
position also when the decision is to reduce exposure - see ``_trade``), and it
runs only after the user approves it.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from ..tools.base import ToolContext
from .approval import ProposalStore, ProposalStoreError
from .contract import ActionLevel
from .llm import LLMError, StructuredLLM, llm_status
from .memory import MemoryContext
from .planning import DEFAULT_FACTORS, Plan
from .reports import AgentReport, make_report

ISSUE_TYPES = ("missing_data", "risk_conflict", "logic_gap", "constraint_violation")
SEVERITIES = ("HIGH", "MEDIUM", "LOW")
BLOCKING = ("HIGH", "MEDIUM")
STALE_AFTER_DAYS = 10
MAX_UNAVAILABLE_SHARE = 0.4  # this share of strategies without a signal is too little evidence
ALLOWED_FACTORS = ("momentum_20d", "volatility_20d", "rsi_14d", "momentum_60d", "volatility_60d")
CONFIDENCE_ORDER = ("low", "medium", "high")
REDUCING_DECISIONS = ("SELL", "REDUCE_EXPOSURE", "REDUCE_CONCENTRATION")
ADDRESSED_BY_REDUCING = ("POSITION_LIMIT",)  # issues a sale of the breaching position resolves


class ReviewValidationError(ValueError):
    """Model output the review/decision validator refuses."""


# --------------------------------------------------------------------------- grounding

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> List[float]:
    return [float(n) for n in _NUMBER.findall(text or "")]


def evidence_numbers(*sources: Any) -> Set[float]:
    """Every number in the evidence, with the roundings and percent forms a
    writer may legitimately use when quoting it."""
    found: Set[float] = set()
    for source in sources:
        text = source if isinstance(source, str) else json.dumps(source, ensure_ascii=False, default=str)
        for value in _numbers(text):
            for v in (value, value * 100):
                for digits in (0, 1, 2, 3, 4):
                    found.add(round(v, digits))
    return found


def ungrounded(texts: Iterable[str], known: Set[float]) -> List[float]:
    """Numbers in ``texts`` that do not appear in the evidence (0-12 are allowed:
    counts and ordinals such as "3 of 5 strategies" or "2 issues")."""
    bad = []
    for text in texts:
        for value in _numbers(text):
            if value <= 12 and value == int(value):
                continue
            if not any(round(value, d) in known for d in (0, 1, 2, 3, 4)):
                bad.append(value)
    return bad


# ------------------------------------------------------------------------------ critic

CRITIC_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["issues"],
    "properties": {"issues": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["type", "severity", "message", "suggestion", "remedy_days", "remedy_factors"],
        "properties": {"type": {"type": "string", "enum": list(ISSUE_TYPES)},
                       "severity": {"type": "string", "enum": list(SEVERITIES)},
                       "message": {"type": "string"}, "suggestion": {"type": "string"},
                       "remedy_days": {"type": "integer"},
                       "remedy_factors": {"type": "string"}}}}},
}

CRITIC_SYSTEM = """You are the critic of TraderX, a quantitative decision product.
Review the research and risk reports for four kinds of problem: missing_data,
risk_conflict, logic_gap, constraint_violation (against the user's constraints).
You cannot change any result. Report only problems the reports show; quote only
numbers that appear in them. A remedy may only ask for a longer history
(remedy_days, 30-3650) or more factor evidence (remedy_factors, comma-separated);
use 0 and "" when no remedy applies. Return an empty list when the analysis is sound."""


def _issue(issues: List[Dict[str, Any]], type_: str, code: str, severity: str, message: str,
           suggestion: str, remedy: Optional[Dict[str, Any]] = None, source: str = "rule") -> None:
    issues.append({"type": type_, "code": code, "severity": severity, "message": message,
                   "suggestion": suggestion, "remedy": remedy, "source": source})


class CriticAgent:
    """Reviews research and risk; approves or requests a revision - never edits results."""

    name = "critic"

    def __init__(self, today: Callable[[], date] = lambda: datetime.now(timezone.utc).date(),
                 llm: Optional[StructuredLLM] = None):
        self._today = today
        self.llm = llm

    def _rule_review(self, plan: Plan, r: Dict[str, Any], k: Dict[str, Any], risk: AgentReport,
                     memory: MemoryContext) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        issues: List[Dict[str, Any]] = []
        notes: List[Dict[str, Any]] = []
        market = r["market"].get(plan.symbol) if plan.symbol else None
        sig = r.get("signal")
        view = r.get("view")
        level = k.get("level") or "unknown"

        # missing data
        if sig is not None:
            requested = sig["requested"] or 1
            if len(sig["unavailable"]) >= MAX_UNAVAILABLE_SHARE * requested:
                days = (market or {}).get("days_requested") or 0
                _issue(issues, "missing_data", "INSUFFICIENT_HISTORY", "MEDIUM",
                       f"{len(sig['unavailable'])} of {requested} strategies had no signal "
                       f"({sorted(sig['unavailable'])}); the view rests on too few strategies",
                       "fetch longer history",
                       {"days": 365} if days < 365 else None)
        if market:
            try:
                age = (self._today() - date.fromisoformat(market["last_date"][:10])).days
            except ValueError:
                age = None
            if age is not None and age > STALE_AFTER_DAYS:
                _issue(issues, "missing_data", "STALE_DATA", "HIGH",
                       f"latest {plan.symbol} bar is {market['last_date']} ({age} days old)",
                       "load fresh EOD data before deciding")
        if risk.status == "UNASSESSED":
            _issue(issues, "missing_data", "NO_RISK_ASSESSMENT", "HIGH",
                   "risk could not be measured", "restore the data RiskTool needs; no trade may be proposed")

        # logic gaps
        if sig is not None:
            sides = {1 if s["signal"] > 0 else -1 for s in sig["signals"] if s["signal"] != 0}
            factors = (r.get("factors") or {}).get("values") or {}
            if len(sides) == 2 and not factors:
                _issue(issues, "logic_gap", "SIGNAL_CONFLICT", "MEDIUM",
                       "strategies disagree on direction and there is no factor evidence to arbitrate",
                       "add factor evidence", {"factors": list(DEFAULT_FACTORS)})
            elif len(sides) == 2:
                notes.append({"code": "SIGNAL_CONFLICT", "message": f"strategies disagree; factors: {factors}"})
            mom = factors.get("momentum_20d")
            if mom is not None and ((view == "BUY" and mom < 0) or (view == "SELL" and mom > 0)):
                _issue(issues, "logic_gap", "FACTOR_CONTRADICTS_VIEW", "LOW",
                       f"view {view} but momentum_20d is {mom}", "treat the view with caution")

        # risk conflicts
        if view == "BUY" and level == "high":
            _issue(issues, "risk_conflict", "BUY_AGAINST_HIGH_RISK", "HIGH",
                   "the signal view is BUY while measured risk is high", "do not add exposure")
        elif view == "BUY" and level == "medium":
            _issue(issues, "risk_conflict", "BUY_AGAINST_MEDIUM_RISK", "LOW",
                   "the signal view is BUY while measured risk is medium", "size conservatively")
        sym_level = (k.get("symbol_risk") or {}).get("level")
        pf_level = (k.get("portfolio_risk") or {}).get("level")
        if view == "BUY" and sym_level == "low" and pf_level == "high":
            _issue(issues, "risk_conflict", "SYMBOL_VS_PORTFOLIO", "MEDIUM",
                   "the symbol looks calm but the portfolio's risk is high", "review portfolio risk first")

        # constraint violations
        for check in k.get("checks") or []:
            if check["check"] == "max_single_position" and check["status"] == "breach":
                breaches = k.get("concentration_breaches") or {}
                _issue(issues, "constraint_violation", "POSITION_LIMIT", "HIGH",
                       f"positions above the user's single-position limit: {breaches} ({check['detail']})",
                       "reduce exposure to within the limit")
            if check["check"] == "trade_concentration" and check["status"] == "veto":
                _issue(issues, "constraint_violation", "TRADE_LIMIT", "HIGH", check["detail"],
                       "do not propose this trade size")
            if check["check"] == "risk_tolerance" and check["status"] == "veto":
                _issue(issues, "constraint_violation", "RISK_TOLERANCE", "HIGH", check["detail"],
                       "stay within the user's risk tolerance")
        check = k.get("trade_check") or {}
        if check.get("reduces_exposure"):
            if check.get("projected_weight_pct") is not None:
                within = check.get("within_limit_after")
                notes.append({"code": "SALE_REDUCES_POSITION",
                              "message": f"the requested sale reduces {check['symbol']} from {check['weight_pct']}% "
                                         f"to {check['projected_weight_pct']}% of the portfolio"
                                         + ("" if within is None else
                                            " (within the limit)" if within else " (still above the limit)")})
            else:
                notes.append({"code": "SALE_REDUCES_POSITION",
                              "message": f"the requested sale reduces the held {check['symbol']} position"})
            others = check.get("others_over_limit_after") or {}
            if others:
                before = (k.get("exposure") or {}).get("weights_pct") or {}
                listed = ", ".join(f"{s} {before.get(s)}% -> {w}%" for s, w in sorted(others.items()))
                _issue(issues, "constraint_violation", "OTHER_POSITIONS_OVER_LIMIT", "LOW",
                       f"after the sale other holdings stay above the {check['limit_pct']}% limit "
                       f"(share of what is held: {listed})",
                       "review those positions separately; this sale does not change their size")
        if plan.direct_execution_requested:
            _issue(issues, "constraint_violation", "AUTO_TRADING", "LOW",
                   "the goal asks for direct execution", "execution only after the user approves a PAPER proposal")

        # memory context (notes only: memory never overrides data)
        subject = (plan.symbol or "").upper()
        for past in memory.decisions:
            previous = past.content.get("recommendation")
            # compare like with like: an earlier directional view against today's view
            if past.subject == subject and view and previous in ("BUY", "SELL", "HOLD") and previous != view:
                notes.append({"code": "DIFFERS_FROM_MEMORY", "memory_id": past.id,
                              "message": f"previous run {past.source_task_id} concluded "
                                         f"{past.content['recommendation']}; current data says {view}"})
                break
        for fb in memory.feedback:
            if fb.subject == subject and fb.content.get("verdict") == "disagree":
                notes.append({"code": "USER_DISAGREED_BEFORE", "memory_id": fb.id,
                              "message": f"the user disagreed with {fb.content.get('about_recommendation') or 'a previous conclusion'}"
                                         + (f": {fb.content['comment']}" if fb.content.get("comment") else "")})
                break
        if plan.intent == "strategy_suggestion" and not r.get("best_strategy"):
            notes.append({"code": "NO_TRADING_CANDIDATE", "message": "no candidate strategy traded in this window"})
        return issues, notes

    async def _llm_review(self, plan: Plan, r: Dict[str, Any], k: Dict[str, Any],
                          memory: MemoryContext, rule_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evidence = {"goal": plan.goal, "research": _brief_research(r), "risk": _brief_risk(k),
                    "user_constraints": memory.lines()}
        prompt = json.dumps({**evidence, "issues_already_found": rule_issues}, ensure_ascii=False, default=str)
        raw = await self.llm.generate(CRITIC_SYSTEM, prompt, "review", CRITIC_SCHEMA)
        items = raw.get("issues")
        if not isinstance(items, list):
            raise ReviewValidationError("review.issues must be a list")
        known = evidence_numbers(evidence)
        out = []
        for item in items[:8]:
            if item.get("type") not in ISSUE_TYPES or item.get("severity") not in SEVERITIES:
                raise ReviewValidationError(f"unknown issue type/severity {item.get('type')}/{item.get('severity')}")
            message = str(item.get("message") or "").strip()[:300]
            suggestion = str(item.get("suggestion") or "").strip()[:200]
            if not message:
                raise ReviewValidationError("every issue needs a message")
            if ungrounded([message, suggestion], known):
                raise ReviewValidationError(f"issue cites numbers not in the reports: {message}")
            remedy: Dict[str, Any] = {}
            days = item.get("remedy_days") or 0
            if days:
                if not isinstance(days, int) or not 30 <= days <= 3650:
                    raise ReviewValidationError("remedy_days must be 30..3650")
                remedy["days"] = days
            factors = [f.strip() for f in str(item.get("remedy_factors") or "").split(",") if f.strip()]
            if factors:
                if any(f not in ALLOWED_FACTORS for f in factors):
                    raise ReviewValidationError(f"remedy_factors must be among {ALLOWED_FACTORS}")
                remedy["factors"] = factors
            out.append({"type": item["type"], "code": "LLM_REVIEW", "severity": item["severity"],
                        "message": message, "suggestion": suggestion, "remedy": remedy or None,
                        "source": "llm"})
        return out

    async def run(self, plan: Plan, research: AgentReport, risk: AgentReport, memory: MemoryContext,
                  round: int = 1) -> AgentReport:
        r, k = research.findings, risk.findings
        issues, notes = self._rule_review(plan, r, k, risk, memory)
        reviewer, fallback_reason, model = "rule", None, None
        ok, reason = llm_status(self.llm)
        if ok:
            try:
                extra = await self._llm_review(plan, r, k, memory, issues)
                seen = {(i["type"], i["message"]) for i in issues}
                issues += [i for i in extra if (i["type"], i["message"]) not in seen]
                reviewer, model = "rule+llm", self.llm.model
            except (LLMError, ReviewValidationError) as exc:
                fallback_reason = f"LLM review not used: {exc}"
        else:
            fallback_reason = reason
        valid = not any(i["severity"] in BLOCKING for i in issues)
        verdict = "approve" if valid else "request_revision"
        blocking = [i["code"] for i in issues if i["severity"] in BLOCKING]
        summary = "approve" if valid else f"request_revision: {', '.join(blocking)}"
        return make_report(self.name, "APPROVE" if valid else "REQUEST_REVISION", summary,
                           {"valid": valid, "verdict": verdict, "issues": issues, "notes": notes,
                            "reviewer": reviewer, "model": model, "fallback_reason": fallback_reason,
                            "round": round},
                           list(research.evidence) + list(risk.evidence), round=round)


def _brief_research(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in ("intent", "symbol", "market", "signal", "factors", "backtests",
                                   "portfolio", "view", "view_rationale", "skipped", "best_strategy")}


def _brief_risk(k: Dict[str, Any]) -> Dict[str, Any]:
    return {key: k.get(key) for key in ("level", "symbol_risk", "portfolio_risk", "exposure", "checks",
                                         "vetoes", "concentration_breaches")}


# ---------------------------------------------------------------------------- decision

DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["decision", "confidence", "rationale", "action_level"],
    "properties": {"decision": {"type": "string"},
                   "confidence": {"type": "string", "enum": list(CONFIDENCE_ORDER)},
                   "rationale": {"type": "array", "items": {"type": "string"}},
                   "action_level": {"type": "string", "enum": ["ANALYSIS", "PROPOSE", "SIMULATE"]}},
}

DECISION_SYSTEM = """You are the portfolio decision agent of TraderX.
Decide from the research, risk and critic reports and the user's memory context.
Choose the decision from allowed_decisions only; you cannot execute anything, and
a decision is advice or a proposal a human must approve. Respect a Risk veto and
the critic's unresolved issues. rationale: 2-5 short points; quote only numbers
that appear in the reports. Use action_level SIMULATE for strategy suggestions,
PROPOSE when the user should act, ANALYSIS when nothing needs to be done."""


class PortfolioDecisionAgent:
    """Aggregates the reports into one decision; never above PROPOSE / SIMULATE."""

    name = "decision"

    def __init__(self, decision_agent_factory: Optional[Callable[[], Any]] = None,
                 llm: Optional[StructuredLLM] = None):
        self._factory = decision_agent_factory
        self.llm = llm

    async def _quant_decision(self, context: Dict[str, Any], decision_type: str) -> Dict[str, Any]:
        """Portfolio action from the existing quant DecisionAgent (rule path; debate off)."""
        from ..base import AgentStatus, AgentTask
        from ..quant.decision_agent import DecisionAgent

        agent = (self._factory or (lambda: DecisionAgent(agent_id="workflow-decision")))()
        outcome = await agent._execute_with_tracking(AgentTask(
            agent_type="decision", payload={"type": decision_type, "context": context}))
        if outcome.status != AgentStatus.COMPLETED:
            return {"decision": None, "error": outcome.error}
        body = outcome.result or {}
        return {"decision": body.get("decision"), "reasoning": body.get("reasoning"),
                "debate_used": body.get("debate_used", False)}

    @staticmethod
    def _allowed(plan: Plan, r: Dict[str, Any], k: Dict[str, Any], risk: AgentReport) -> List[str]:
        veto = risk.status == "VETO"
        if plan.intent == "strategy_suggestion":
            traded = [b["strategy"] for b in r.get("backtests") or []
                      if b["trade_count"] > 0 and b["sharpe_ratio"] is not None]
            return ["NO_SUGGESTION"] + [f"CONSIDER_STRATEGY:{s}" for s in traded]
        if plan.intent == "portfolio_risk":
            allowed = ["REDUCE_EXPOSURE", "REDUCE_CONCENTRATION", "HEDGE", "REVIEW"]
            if not veto and not k.get("concentration_breaches"):
                allowed.append("HOLD")
            return allowed
        if veto:
            return ["NO_NEW_EXPOSURE", "REDUCE_EXPOSURE", "HOLD"]
        if risk.status != "OK":
            return ["HOLD", "NO_NEW_EXPOSURE"]
        return ["BUY", "SELL", "HOLD", "NO_NEW_EXPOSURE", "REDUCE_EXPOSURE"]

    @staticmethod
    def _confidence_cap(risk: AgentReport, critic: Optional[AgentReport]) -> Tuple[str, Optional[str]]:
        if risk.status == "UNASSESSED":
            return "low", "risk was not assessed"
        if critic is not None and not critic.findings["valid"]:
            return "low", "the critic's review is not resolved"
        return "high", None

    async def _rule_decision(self, plan: Plan, r: Dict[str, Any], k: Dict[str, Any],
                             risk: AgentReport, critic: Optional[AgentReport],
                             quant: Dict[str, Any], cap: str) -> Dict[str, Any]:
        rationale: List[str] = list(r.get("view_rationale") or [])
        view = r.get("view")
        veto = risk.status == "VETO"
        level = ActionLevel.PROPOSE
        if plan.intent == "portfolio_risk":
            decision = quant["decision"] or "REVIEW"
            breaches = k.get("concentration_breaches") or {}
            if breaches:
                rationale.append(f"positions above the single-position limit: {breaches}")
                decision = "REDUCE_CONCENTRATION"
            if decision == "HOLD" and not veto:
                level = ActionLevel.ANALYSIS
        elif plan.intent == "strategy_suggestion":
            best = r.get("best_strategy")
            decision = f"CONSIDER_STRATEGY:{best}" if best else "NO_SUGGESTION"
            rationale.append(f"{best} has the highest Sharpe among candidates that traded (simulation only)"
                             if best else "no candidate strategy traded in this window")
            level = ActionLevel.SIMULATE
        else:
            decision = view or "HOLD"
            if risk.status != "OK" and decision in ("BUY", "SELL"):
                decision = "HOLD"
                rationale.append("risk was not assessed: no directional call")
        if veto:
            if plan.intent != "portfolio_risk":
                held = plan.symbol in ((k.get("exposure") or {}).get("weights_pct") or {})
                decision = "REDUCE_EXPOSURE" if held else "NO_NEW_EXPOSURE"
            rationale.append("Risk Agent veto: " + risk.summary)
        if critic is not None and not critic.findings["valid"]:
            codes = [i["code"] for i in critic.findings["issues"] if i["severity"] in BLOCKING]
            rationale.append("critic requested a revision: " + ", ".join(codes))
        return {"decision": decision, "confidence": "low" if cap == "low" else "medium",
                "rationale": rationale, "action_level": level.value}

    async def _llm_decision(self, plan: Plan, r: Dict[str, Any], k: Dict[str, Any],
                            critic: Optional[AgentReport], memory: MemoryContext,
                            allowed: List[str], cap: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        evidence = {"goal": plan.goal, "intent": plan.intent, "research": _brief_research(r),
                    "risk": _brief_risk(k),
                    "critic": critic.findings if critic is not None else None,
                    "memory_context": memory.lines()}
        prompt = json.dumps({**evidence, "allowed_decisions": allowed,
                             "confidence_cap": cap, "rule_based_decision": rule},
                            ensure_ascii=False, default=str)
        raw = await self.llm.generate(DECISION_SYSTEM, prompt, "decision", DECISION_SCHEMA)
        decision = str(raw.get("decision") or "").strip()
        if decision not in allowed:
            raise ReviewValidationError(f"decision {decision!r} is not allowed here (allowed: {allowed})")
        level = raw.get("action_level")
        if level not in ("ANALYSIS", "PROPOSE", "SIMULATE"):
            raise ReviewValidationError(f"action_level {level!r} is not allowed")
        if plan.intent == "strategy_suggestion" and level != "SIMULATE":
            raise ReviewValidationError("a strategy suggestion is a SIMULATE-level result")
        if decision not in ("HOLD", "NO_SUGGESTION") and level == "ANALYSIS":
            raise ReviewValidationError(f"{decision} is a proposal, not ANALYSIS")
        rationale = [str(x).strip()[:300] for x in raw.get("rationale") or [] if str(x).strip()]
        if not 1 <= len(rationale) <= 6:
            raise ReviewValidationError("rationale needs 1-6 points")
        bad = ungrounded(rationale, evidence_numbers(evidence))
        if bad:
            raise ReviewValidationError(f"rationale cites numbers not in the reports: {bad}")
        confidence = raw.get("confidence")
        if confidence not in CONFIDENCE_ORDER:
            raise ReviewValidationError("confidence must be high, medium or low")
        return {"decision": decision, "confidence": confidence, "rationale": rationale,
                "action_level": level}

    async def run(self, plan: Plan, research: AgentReport, risk: AgentReport,
                  critic: Optional[AgentReport], memory: MemoryContext, proposals: ProposalStore,
                  ctx: ToolContext, task_id: str) -> Tuple[AgentReport, Dict[str, Any], ActionLevel]:
        r, k = research.findings, risk.findings
        view = r.get("view")
        risk_level = k.get("level") or "unknown"
        trend = {"BUY": "bullish", "SELL": "bearish"}.get(view or "", "neutral")
        quant = await self._quant_decision(
            {"risk_level": risk_level, "market_trend": trend, "symbol": plan.symbol or "portfolio"},
            "portfolio_decision" if plan.intent == "portfolio_risk" else "trade_decision")
        allowed = self._allowed(plan, r, k, risk)
        cap, cap_reason = self._confidence_cap(risk, critic)
        rule = await self._rule_decision(plan, r, k, risk, critic, quant, cap)

        decided_by, fallback_reason, model, notes = "rule", None, None, []
        ok, reason = llm_status(self.llm)
        decision = rule
        if ok:
            try:
                decision = await self._llm_decision(plan, r, k, critic, memory, allowed, cap, rule)
                decided_by, model = "llm", self.llm.model
            except (LLMError, ReviewValidationError) as exc:
                fallback_reason = f"LLM decision not used: {exc}"
        else:
            fallback_reason = reason
        if CONFIDENCE_ORDER.index(decision["confidence"]) > CONFIDENCE_ORDER.index(cap):
            notes.append(f"confidence capped at {cap}: {cap_reason}")
            decision = {**decision, "confidence": cap}

        prefs = {key: v["value"] for key, v in memory.preferences.items()}
        constraints = [f"{key}={value} (memory #{memory.preferences[key]['memory_id']})"
                       for key, value in prefs.items()]
        if plan.direct_execution_requested:
            notes.append("直接执行请求未被执行：Agent 不会自主下单，任何执行都必须由用户批准提案后以 PAPER 模式进行。")
        if prefs.get("forbid_auto_trading"):
            notes.append("用户偏好 forbid_auto_trading=true：仅输出建议/提案。")

        unresolved = [] if critic is None or critic.findings["valid"] else \
            [i for i in critic.findings["issues"] if i["severity"] in BLOCKING]
        trade = await self._trade(plan, r, k, risk, decision["decision"], unresolved, proposals, ctx,
                                  task_id, decision["rationale"])
        approval_required = bool(trade and trade.get("status") == "AWAITING_APPROVAL")
        level = ActionLevel(decision["action_level"])
        result = {
            "goal": plan.goal, "intent": plan.intent, "symbol": plan.symbol,
            "portfolio_id": plan.portfolio_id,
            "decision": {**decision, "decided_by": decided_by, "model": model,
                         "fallback_reason": fallback_reason, "allowed": allowed},
            "recommendation": decision["decision"],
            "confidence": decision["confidence"],
            "rationale": decision["rationale"],
            "portfolio_action": quant.get("decision"),
            "action_level": level.value,
            "approval_required": approval_required,
            "trade": trade,
            "suggestion": next((b for b in r.get("backtests") or []
                                if decision["decision"] == f"CONSIDER_STRATEGY:{b['strategy']}"), None),
            "risk": {"status": risk.status, "level": risk_level, "vetoes": k["vetoes"],
                     "checks": k["checks"], "exposure": k.get("exposure")},
            "critic": _critic_summary(critic),
            "research": {"market": r["market"], "signal": r.get("signal"), "factors": r.get("factors"),
                         "backtests": r.get("backtests"), "skipped": r.get("skipped")},
            "constraints_applied": constraints,
            "notes": notes,
            "quant_decision": quant,
            "advice_only": True,
        }
        report = make_report(self.name, "DECIDED",
                             f"{decision['decision']} at {level.value} ({decided_by}, confidence {decision['confidence']})",
                             {"decision": result["decision"], "approval_required": approval_required,
                              "trade_status": (trade or {}).get("status")},
                             list(research.evidence) + list(risk.evidence))
        return report, result, level

    @staticmethod
    async def _trade(plan: Plan, r: Dict[str, Any], k: Dict[str, Any], risk: AgentReport,
                     decision: str, unresolved: List[Dict[str, Any]], proposals: ProposalStore,
                     ctx: ToolContext, task_id: str, rationale: List[str]) -> Optional[Dict[str, Any]]:
        """A PAPER proposal for the user's requested trade - only when every agent supports it.

        A Risk veto rules out new exposure. A sale that only reduces a held long
        position adds none, so it can still be proposed when the decision is to
        sell or reduce, and the critic's position-limit issue for that symbol is
        exactly what the sale addresses; any other unresolved issue still blocks.
        """
        if not plan.trade:
            return None
        side = plan.trade["side"]
        wanted = side.upper()
        check = k.get("trade_check") or {}
        reducing = side == "sell" and bool(check.get("reduces_exposure"))
        if risk.status == "VETO" and not reducing:
            return {"status": "NOT_PROPOSED", "reason": "Risk Agent veto: " + risk.summary}
        if risk.status not in ("OK", "VETO"):
            return {"status": "NOT_PROPOSED", "reason": "risk was not assessed"}
        addressed: List[str] = []
        if reducing and plan.symbol in (k.get("concentration_breaches") or {}):
            addressed = [i["code"] for i in unresolved if i["code"] in ADDRESSED_BY_REDUCING]
        blocking = [i for i in unresolved if i["code"] not in addressed]
        if blocking:
            return {"status": "NOT_PROPOSED",
                    "reason": "critic requested a revision: " + ", ".join(i["code"] for i in blocking)}
        if reducing:
            if decision not in REDUCING_DECISIONS:
                return {"status": "NOT_PROPOSED",
                        "reason": f"decision is {decision}; a sale is proposed only when the decision is "
                                  f"one of {list(REDUCING_DECISIONS)}"}
        elif decision != wanted:
            return {"status": "NOT_PROPOSED", "reason": f"decision is {decision}, not {wanted}"}
        market = r["market"][plan.symbol]
        order = {"symbol": plan.symbol, "side": side, "quantity": plan.trade["quantity"],
                 "reference_price": market["last_close"], "reference_date": market["last_date"]}
        evidence = [market["call_id"], (r.get("signal") or {}).get("call_id"),
                    (k.get("symbol_risk") or {}).get("call_id"),
                    (k.get("portfolio_risk") or {}).get("call_id")]
        evidence = [e for e in dict.fromkeys(evidence) if e]
        try:
            proposal = await proposals.create(
                owner_user_id=ctx.user_id, tool="execution", tool_input=order,
                rationale="; ".join(rationale),
                evidence={"call_ids": evidence, "decision": decision, "risk_status": risk.status,
                          "trade_check": check or None, "addresses": addressed},
                source_task_id=task_id, input_call_ids=evidence)
        except ProposalStoreError as exc:
            return {"status": "NOT_PROPOSED", "reason": f"the proposal could not be stored: {exc}"}
        out = {"status": "AWAITING_APPROVAL", "mode": "PAPER", "proposal": proposal.to_dict(),
               "reduces_exposure": reducing, "addresses": addressed}
        if check:
            out["trade_check"] = check
        return out


def _critic_summary(critic: Optional[AgentReport]) -> Dict[str, Any]:
    if critic is None:
        return {"valid": None, "verdict": None, "issues": [], "notes": [], "unresolved": [],
                "reviewer": None}
    f = critic.findings
    return {"valid": f["valid"], "verdict": f["verdict"], "issues": f["issues"], "notes": f["notes"],
            "unresolved": [] if f["valid"] else [i for i in f["issues"] if i["severity"] in BLOCKING],
            "reviewer": f["reviewer"], "fallback_reason": f["fallback_reason"], "round": f["round"]}
