# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-agent decision workflow.

    User goal -> Memory retrieval -> Planner (LLM, rule fallback) -> Research Agent
              -> Risk Agent -> Critic Agent -> (one re-analysis on request_revision)
              -> Portfolio Decision Agent -> (PAPER proposal, human approval)

Every agent has one duty and returns a frozen :class:`AgentReport`; no agent can
change another agent's report or the plan. Agents reach data only through
:class:`ToolRunner`, which records each call in the run contract (tagged with the
agent), resolves ``"$bars"`` to bars a market_data call actually returned, and
refuses any tool outside the agent's allowance - in particular EXECUTE tools:
the workflow's ceiling is PROPOSE / SIMULATE. A requested trade becomes a
proposal in the shared proposal store (``agent_proposals``); it is carried out
only when its owner approves it through ``/runtime/run`` (PAPER ONLY, once).

* Research Agent  - runs the plan's research steps (market data, signals,
  factors, backtests, portfolio) and states the signal view.
* Risk Agent      - RiskTool for the symbol and the portfolio, exposure weights
  from real closes, the user's limits from memory; may VETO new exposure, and
  checks that a requested sale only reduces a held long position. After a
  veto the Critic still reviews; the only trade that can still be proposed -
  and the only one that may still be re-analysed, when the concentration
  limit is its only veto - is such a reducing sale.
* Critic Agent    - a structured review (see :mod:`.review`); request_revision
  with a remedy sends the plan back to the planner for one revision. When a
  model review is required (``review_policy``) and unusable, a requested
  trade is not proposed.
* Portfolio Decision Agent - one structured decision (LLM with a validator, or
  rules), written back to memory with the run's task id.

The run's status is RUNNING while it executes (``start`` returns immediately
and the contract grows as agents work), then SUCCESS or FAILED.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import replace
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..tools.base import RETRYABLE_ERRORS, ToolContext, ToolErrorCode, ToolRegistry, ToolResult
from .approval import ProposalStore, gate
from .contract import ActionLevel, AgentRun, Observation, RunStatus, ToolCall
from .llm import StructuredLLM
from .loop import _observation_text, provenance_for, summarize_input, summarize_output
from .memory import MemoryContext, MemoryStore, MemoryValidationError
from .planner import MIN_BARS_FOR_ANALYSIS
from .planning import BARS_REF, Clarification, Plan, PlanningService, RESEARCH_TOOLS, goal_facts
from .recorder import RunLog
from .reports import AgentReport, make_report
from .review import REVIEW_FAILURE_CODES, CriticAgent, PortfolioDecisionAgent

logger = logging.getLogger(__name__)

WORKFLOW_AGENT = "traderx.workflow"
MAX_ATTEMPTS = 3
BUY_THRESHOLD = 0.5  # consensus score thresholds, same as the single-agent loop

RISK_ORDER = ("low", "medium", "high")
LONG_DIRECTIONS = ("1", "LONG", "BUY")  # positions.direction values of a long holding
RESEARCH_ALLOWED = RESEARCH_TOOLS
RISK_ALLOWED = ("risk", "market_data")
DATA_LABELS = {"database": "PostgreSQL", "derived": "Derived analytics",
               "paper_simulation": "Paper simulation"}

_BACKGROUND: Set["asyncio.Task[Any]"] = set()


# -------------------------------------------------------------------------- tool runner

class ToolRunner:
    """The only path from a workflow agent to a tool."""

    def __init__(self, run: AgentRun, tools: ToolRegistry, ctx: ToolContext):
        self.run = run
        self.tools = tools
        self.ctx = ctx
        self._bars: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}  # symbol -> (call_id, bars)

    def bars_call(self, symbol: str) -> Optional[str]:
        ref = self._bars.get(symbol)
        return ref[0] if ref else None

    def latest_close(self, symbol: str) -> Optional[float]:
        ref = self._bars.get(symbol)
        return float(ref[1][-1]["close"]) if ref and ref[1] else None

    async def call(self, agent: str, tool_name: str, args: Dict[str, Any], reason: str,
                   allowed: Tuple[str, ...]) -> Tuple[str, ToolResult]:
        tool = self.tools.get(tool_name)
        payload = dict(args)
        input_call_ids: List[str] = []
        refusal: Optional[Tuple[ToolErrorCode, str]] = None
        if tool_name not in allowed or tool is None:
            refusal = (ToolErrorCode.FORBIDDEN, f"{agent} agent may not call {tool_name}")
        elif gate(tool.action_level, None):
            refusal = (ToolErrorCode.APPROVAL_REQUIRED,
                       f"{tool.action_level.value} is never run by a workflow agent")
        elif payload.get("bars") == BARS_REF:
            ref = self._bars.get(str(payload.get("symbol")))
            if ref is None:
                refusal = (ToolErrorCode.INVALID_INPUT,
                           f"no observed bars for {payload.get('symbol')}; market_data must succeed first")
            else:
                input_call_ids = [ref[0]]
                payload["bars"] = ref[1]
        if refusal:
            call_id = f"call-{len(self.run.tool_calls) + 1}"
            level = tool.action_level.value if tool else "UNKNOWN"
            self.run.tool_calls.append(ToolCall(
                call_id=call_id, tool=tool_name, action_level=level,
                input=summarize_input(payload, input_call_ids), status="BLOCKED",
                error={"code": refusal[0].value, "message": refusal[1], "retryable": False},
                agent=agent))
            self.run.event("tool_call", f"{tool_name} blocked: {refusal[1]}", agent=agent,
                           call_id=call_id, tool=tool_name)
            return call_id, ToolResult.failure(*refusal)

        result: ToolResult = ToolResult.failure(ToolErrorCode.INTERNAL, "not called")
        call_id = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            call_id = f"call-{len(self.run.tool_calls) + 1}"
            self.run.event("plan", reason if attempt == 1 else
                           f"retry {tool_name} after {result.error_code.value} (attempt {attempt})",
                           agent=agent, tool=tool_name, call_id=call_id, input_from=input_call_ids)
            started = time.monotonic()
            result = await tool.invoke(payload, replace(self.ctx, call_id=call_id,
                                                        input_call_ids=list(input_call_ids)))
            duration = int((time.monotonic() - started) * 1000)
            summary = summarize_output(tool.name, result.output)
            self.run.tool_calls.append(ToolCall(
                call_id=call_id, tool=tool.name, action_level=tool.action_level.value,
                input=summarize_input(payload, input_call_ids),
                status="OK" if result.ok else "ERROR", attempt=attempt, error=result.error_dict(),
                output_summary=summary,
                provenance=result.provenance.to_dict() if result.provenance else None,
                duration_ms=duration, agent=agent))
            self.run.observations.append(Observation(
                call_id=call_id, tool=tool.name, ok=result.ok,
                summary=(summary if result.ok else result.error_dict()) or {},
                error_code=None if result.ok else result.error_code.value))
            self.run.event("tool_call", f"{tool.name} -> {'OK' if result.ok else result.error_code.value}",
                           agent=agent, call_id=call_id, attempt=attempt, duration_ms=duration)
            self.run.event("observation", _observation_text(tool.name, result), agent=agent,
                           call_id=call_id)
            if result.ok or result.error_code not in RETRYABLE_ERRORS:
                break
        if result.ok:
            self.run.raise_level(tool.action_level)
            if tool.name == "market_data":
                self._bars[result.output["symbol"]] = (call_id, result.output["bars"])
        return call_id, result


# ------------------------------------------------------------------------------ agents

def signal_view(consensus_score: float) -> str:
    if consensus_score >= BUY_THRESHOLD:
        return "BUY"
    if consensus_score <= -BUY_THRESHOLD:
        return "SELL"
    return "HOLD"


class ResearchAgent:
    """Runs the plan's research steps and states what the data says."""

    name = "research"

    async def run(self, plan: Plan, runner: ToolRunner, round: int = 1) -> AgentReport:
        findings: Dict[str, Any] = {"intent": plan.intent, "symbol": plan.symbol,
                                    "portfolio_id": plan.portfolio_id, "plan_revision": plan.revision,
                                    "market": {}, "signal": None, "factors": None, "backtests": [],
                                    "portfolio": None, "view": None, "view_rationale": [],
                                    "skipped": []}
        evidence: List[str] = []
        for step in plan.steps:
            args = step.arguments
            call_id, result = await runner.call(self.name, step.tool, args, step.reason, RESEARCH_ALLOWED)
            symbol = args.get("symbol")
            if not result.ok:
                required = (step.tool == "portfolio"
                            or (step.tool == "market_data" and symbol == plan.symbol)
                            or (step.tool == "signal" and plan.intent in ("analyze_symbol", "position_review")))
                # Only too little history is an observation to carry on with; a transient error
                # that outlived the retries skips an optional step; any other error is real.
                recoverable = not required and (result.error_code == ToolErrorCode.INSUFFICIENT_DATA
                                                or result.error_code in RETRYABLE_ERRORS)
                if not recoverable:
                    return make_report(self.name, "FAILED",
                                       f"{'required step' if required else 'step'} {step.tool} failed: "
                                       f"{result.error_message}",
                                       findings, evidence + [call_id], result.error_code.value, round)
                findings["skipped"].append(f"{step.tool}{'/' + args['strategy'] if 'strategy' in args else ''}"
                                           f" skipped: {result.error_message}")
                continue
            out = result.output
            evidence.append(call_id)
            if step.tool == "market_data":
                findings["market"][out["symbol"]] = {
                    "call_id": call_id, "count": out["count"], "first_date": out["first_date"],
                    "last_date": out["last_date"], "last_close": out["last_close"],
                    "days_requested": args.get("days"), "days_requested_unit": "calendar"}
                if out["symbol"] == plan.symbol and out["count"] < MIN_BARS_FOR_ANALYSIS:
                    return make_report(self.name, "FAILED",
                                       f"only {out['count']} bars for {plan.symbol}, need {MIN_BARS_FOR_ANALYSIS}",
                                       findings, evidence, ToolErrorCode.INSUFFICIENT_DATA.value, round)
            elif step.tool == "portfolio":
                findings["portfolio"] = {"call_id": call_id, "position_count": out["position_count"],
                                         "positions": [{"symbol": p["symbol"], "quantity": p["quantity"],
                                                        "direction": p["direction"]}
                                                       for p in out["positions"]]}
                if plan.intent == "portfolio_risk" and out["position_count"] == 0:
                    return make_report(self.name, "FAILED",
                                       f"portfolio {plan.portfolio_id} has no positions; there is no risk to measure",
                                       findings, evidence, ToolErrorCode.INSUFFICIENT_DATA.value, round)
            elif step.tool == "signal":
                findings["signal"] = {"call_id": call_id, "consensus_score": out["consensus_score"],
                                      "consensus_label": out["consensus_label"],
                                      "signals": [{"strategy": s["strategy"], "signal": s["signal"],
                                                   "label": s["label"]} for s in out["signals"]],
                                      "unavailable": out["unavailable"],
                                      "requested": len(out["signals"]) + len(out["unavailable"])}
            elif step.tool == "factor":
                findings["factors"] = {"call_id": call_id, "values": out["values"],
                                       "unavailable": out["unavailable"]}
            elif step.tool == "backtest":
                m = out["metrics"]
                findings["backtests"].append({"call_id": call_id, "strategy": out["strategy"],
                                              "trade_count": out["trade_count"],
                                              "fill_count": out.get("fill_count"),
                                              "open_quantity_at_end": out.get("open_quantity_at_end"),
                                              "basis": out.get("basis"),
                                              "sharpe_ratio": m.get("sharpe_ratio"),
                                              "total_return": m.get("total_return"),
                                              "max_drawdown": m.get("max_drawdown")})

        if plan.intent == "strategy_suggestion" and not findings["backtests"]:
            return make_report(self.name, "FAILED", "no candidate strategy could be simulated",
                               findings, evidence, ToolErrorCode.INSUFFICIENT_DATA.value, round)
        sig = findings["signal"]
        if sig is not None:
            findings["view"] = signal_view(sig["consensus_score"])
            findings["view_rationale"].append(
                f"consensus of {len(sig['signals'])} strategies = {sig['consensus_score']} ({sig['consensus_label']})")
            if sig["unavailable"]:
                findings["view_rationale"].append(f"strategies without a signal: {sorted(sig['unavailable'])}")
        if findings["backtests"]:
            ranked = sorted((b for b in findings["backtests"]
                             if b["trade_count"] > 0 and b["sharpe_ratio"] is not None),
                            key=lambda b: b["sharpe_ratio"], reverse=True)
            findings["best_strategy"] = ranked[0]["strategy"] if ranked else None
        summary = (f"view {findings['view']}" if findings["view"] else
                   f"{len(findings['backtests'])} backtests" if findings["backtests"] else
                   f"{(findings['portfolio'] or {}).get('position_count', 0)} positions read")
        return make_report(self.name, "OK", summary, findings, evidence, round=round)


class RiskAgent:
    """Risk through RiskTool, exposure from real closes, and the user's limits."""

    name = "risk"

    async def run(self, plan: Plan, research: AgentReport, runner: ToolRunner,
                  memory: MemoryContext, round: int = 1) -> AgentReport:
        r = research.findings
        evidence: List[str] = []
        checks: List[Dict[str, Any]] = []
        vetoes: List[str] = []
        findings: Dict[str, Any] = {"symbol_risk": None, "portfolio_risk": None, "exposure": None,
                                    "limits": {}, "checks": checks, "vetoes": vetoes, "trade_check": None}
        tolerance = memory.preference("risk_tolerance")
        max_pct = memory.preference("max_single_position_pct")
        for key in ("risk_tolerance", "max_single_position_pct", "forbid_auto_trading"):
            if key in memory.preferences:
                findings["limits"][key] = dict(memory.preferences[key])
        if not memory.available:
            findings["limits"]["unavailable"] = memory.unavailable_reason

        symbol_level = None
        if plan.symbol and plan.symbol in r["market"]:
            call_id, res = await runner.call(self.name, "risk", {"symbol": plan.symbol, "bars": BARS_REF},
                                             f"measure {plan.symbol} risk on the observed bars", RISK_ALLOWED)
            if res.ok:
                evidence.append(call_id)
                symbol_level = res.output["risk_level"]
                findings["symbol_risk"] = {"call_id": call_id, "level": symbol_level,
                                           "metrics": res.output["metrics"], "policy": res.output["policy"],
                                           "max_drawdown_sign": "negative",
                                           "level_basis": "annualized_volatility and max_drawdown against "
                                                          "policy only; returns and Sharpe are not used"}
                checks.append({"check": "symbol_risk", "status": "veto" if symbol_level == "high" else "pass",
                               "detail": f"{plan.symbol} risk level {symbol_level}"})
                if symbol_level == "high":
                    vetoes.append(f"{plan.symbol} risk level is high: {res.output['metrics']}")
                if tolerance == "low" and symbol_level == "medium":
                    vetoes.append(f"{plan.symbol} risk level medium exceeds the user's low risk tolerance "
                                  f"(memory #{memory.preferences['risk_tolerance']['memory_id']})")
                    checks.append({"check": "risk_tolerance", "status": "veto",
                                   "detail": "medium risk vs low tolerance"})
                elif tolerance:
                    checks.append({"check": "risk_tolerance", "status": "pass",
                                   "detail": f"{symbol_level} risk vs {tolerance} tolerance"})
            else:
                findings["symbol_risk"] = {"call_id": call_id, "level": "unknown",
                                           "error": res.error_dict()}
                checks.append({"check": "symbol_risk", "status": "unavailable", "detail": res.error_message})

        portfolio_level = None
        positions = (r.get("portfolio") or {}).get("positions") or []
        if plan.portfolio_id is not None and positions:
            call_id, res = await runner.call(self.name, "risk", {"portfolio_id": plan.portfolio_id},
                                             f"portfolio {plan.portfolio_id} risk", RISK_ALLOWED)
            if res.ok:
                evidence.append(call_id)
                portfolio_level = res.output["risk_level"]
                findings["portfolio_risk"] = {"call_id": call_id, "level": portfolio_level,
                                              "metrics": res.output["metrics"],
                                              "unavailable": res.output["unavailable"],
                                              "policy": res.output.get("policy"),
                                              "max_drawdown_sign": "positive_magnitude",
                                              "level_basis": "abs(var_95_pct) against policy only "
                                                             "(var_medium_pct, var_high_pct); returns, "
                                                             "Sharpe and alpha are not used",
                                              "var_basis": "*_pct = amount / var_base_value, the analytics "
                                                           "service's portfolio equity (may include cash); "
                                                           "not the exposure total_value"}
                checks.append({"check": "portfolio_risk",
                               "status": "veto" if portfolio_level == "high" else "pass",
                               "detail": f"portfolio risk level {portfolio_level}"})
                if portfolio_level == "high":
                    vetoes.append(f"portfolio {plan.portfolio_id} risk level is high")
            else:
                findings["portfolio_risk"] = {"call_id": call_id, "level": "unknown",
                                              "error": res.error_dict()}
                checks.append({"check": "portfolio_risk", "status": "unavailable",
                               "detail": res.error_message})
            findings["exposure"] = await self._exposure(plan, positions, runner, evidence)
            self._concentration(plan, findings, max_pct, memory, r, checks, vetoes)
            self._sell_check(plan, positions, findings, max_pct, r, checks, vetoes)

        if plan.intent == "portfolio_risk":
            assessed = portfolio_level not in (None, "unknown")
        else:
            assessed = symbol_level not in (None, "unknown")
        if vetoes:
            status, summary = "VETO", "; ".join(vetoes)
        elif not assessed:
            status, summary = "UNASSESSED", "risk could not be measured; no trade can be proposed"
        else:
            status = "OK"
            summary = f"risk {symbol_level or portfolio_level}; no veto"
        findings["blocks_new_exposure"] = bool(vetoes) or not assessed
        # the overall level is the more severe of what was measured
        measured = [lvl for lvl in (symbol_level, portfolio_level) if lvl in RISK_ORDER]
        findings["level"] = max(measured, key=RISK_ORDER.index) if measured else "unknown"
        return make_report(self.name, status, summary, findings, evidence, round=round)

    async def _exposure(self, plan: Plan, positions: List[Dict[str, Any]], runner: ToolRunner,
                        evidence: List[str]) -> Dict[str, Any]:
        """Position weights from quantity x latest observed close (never from memory)."""
        values: Dict[str, float] = {}
        missing: Dict[str, str] = {}
        derived_from: List[str] = []
        for pos in positions:
            symbol, qty = pos["symbol"], pos.get("quantity")
            if not symbol or qty is None:
                missing[str(symbol)] = "no quantity"
                continue
            call_id = runner.bars_call(symbol)
            if call_id is None:
                call_id, res = await runner.call(self.name, "market_data", {"symbol": symbol, "days": 180},
                                                 f"latest close of held {symbol} for exposure weights",
                                                 RISK_ALLOWED)
                if not res.ok:
                    missing[symbol] = res.error_message or "market data unavailable"
                    continue
            price = runner.latest_close(symbol)
            derived_from.append(call_id)
            evidence.append(call_id)
            values[symbol] = values.get(symbol, 0.0) + abs(float(qty)) * float(price)
        total = sum(values.values())
        weights = {s: round(v / total * 100, 4) for s, v in values.items()} if total > 0 else {}
        return {"weights_pct": weights, "total_value": round(total, 2),
                "weights_basis": "share of the summed value of priced holdings; cash excluded",
                "values": {s: round(v, 2) for s, v in values.items()}, "missing": missing,
                "derived_from": derived_from,
                "method": "abs(quantity) x latest EOD close from market_data"}

    @staticmethod
    def _concentration(plan: Plan, findings: Dict[str, Any], max_pct: Optional[float],
                       memory: MemoryContext, research: Dict[str, Any],
                       checks: List[Dict[str, Any]], vetoes: List[str]) -> None:
        exposure = findings["exposure"] or {}
        weights = exposure.get("weights_pct") or {}
        if max_pct is None:
            checks.append({"check": "max_single_position", "status": "not_configured",
                           "detail": "no max_single_position_pct preference in memory"})
            return
        if exposure.get("missing"):
            checks.append({"check": "max_single_position", "status": "unavailable",
                           "detail": f"no price for {sorted(exposure['missing'])}"})
            return
        source = f"memory #{memory.preferences['max_single_position_pct']['memory_id']}"
        over = {s: w for s, w in weights.items() if w > max_pct}
        findings["concentration_breaches"] = over
        checks.append({"check": "max_single_position", "status": "breach" if over else "pass",
                       "detail": f"limit {max_pct}% ({source}); weights {weights}"})
        if plan.symbol and plan.symbol in over:
            vetoes.append(f"{plan.symbol} is {over[plan.symbol]}% of the portfolio, above the "
                          f"user's {max_pct}% limit ({source}); no new exposure")
        trade = plan.trade
        if trade and trade["side"] == "buy" and plan.symbol in research["market"] and plan.symbol not in over:
            price = research["market"][plan.symbol]["last_close"]
            held = exposure["values"].get(plan.symbol, 0.0)
            added = trade["quantity"] * price
            projected = (held + added) / (exposure["total_value"] + added) * 100
            ok = projected <= max_pct
            checks.append({"check": "trade_concentration", "status": "pass" if ok else "veto",
                           "detail": f"after buying {trade['quantity']:g} {plan.symbol}: {projected:.2f}% "
                                     f"(limit {max_pct}%)"})
            if not ok:
                vetoes.append(f"buying {trade['quantity']:g} {plan.symbol} would make it {projected:.2f}% "
                              f"of the portfolio, above the {max_pct}% limit ({source})")


    @staticmethod
    def _sell_check(plan: Plan, positions: List[Dict[str, Any]], findings: Dict[str, Any],
                    max_pct: Optional[float], research: Dict[str, Any],
                    checks: List[Dict[str, Any]], vetoes: List[str]) -> None:
        """Does the user's requested sale reduce a held long position? Selling more
        than is held would be a short sale - new exposure - and is vetoed."""
        trade = plan.trade
        if not trade or trade["side"] != "sell" or not plan.symbol:
            return
        held = sum(float(p["quantity"]) for p in positions
                   if p["symbol"] == plan.symbol and p.get("quantity") and float(p["quantity"]) > 0
                   and str(p.get("direction")).upper() in LONG_DIRECTIONS)
        quantity = float(trade["quantity"])
        check: Dict[str, Any] = {"side": "sell", "symbol": plan.symbol, "quantity": quantity,
                                 "held_quantity": held, "reduces_exposure": 0 < quantity <= held}
        if not check["reduces_exposure"]:
            findings["trade_check"] = check
            checks.append({"check": "sell_quantity", "status": "veto",
                           "detail": f"selling {quantity:g} {plan.symbol} but {held:g} are held long"})
            vetoes.append(f"selling {quantity:g} {plan.symbol} exceeds the {held:g} held; "
                          f"the excess would be a short sale (new exposure)")
            return
        exposure = findings.get("exposure") or {}
        market = research["market"].get(plan.symbol)
        if market and plan.symbol in (exposure.get("values") or {}) and not exposure.get("missing"):
            sold = quantity * float(market["last_close"])
            remaining = exposure["values"][plan.symbol] - sold
            left = exposure["total_value"] - sold
            projected = remaining / left * 100 if left > 0 else 0.0
            # the other holdings keep their value, so their share of what is held rises
            others = {s: round(v / left * 100, 4) for s, v in exposure["values"].items()
                      if s != plan.symbol and left > 0}
            check.update({"weight_pct": exposure["weights_pct"][plan.symbol],
                          "projected_weight_pct": round(projected, 4), "limit_pct": max_pct,
                          "within_limit_after": None if max_pct is None else projected <= max_pct,
                          "other_weights_after_pct": others,
                          "others_over_limit_after": {} if max_pct is None else
                          {s: w for s, w in others.items() if w > max_pct}})
            detail = (f"after selling {quantity:g} of {held:g} {plan.symbol}: "
                      f"{exposure['weights_pct'][plan.symbol]}% -> {projected:.2f}%"
                      + (f" (limit {max_pct}%)" if max_pct is not None else ""))
        else:
            detail = f"selling {quantity:g} of {held:g} {plan.symbol} held (weight not measurable)"
        findings["trade_check"] = check
        checks.append({"check": "sell_quantity", "status": "reduces", "detail": detail})


# ---------------------------------------------------------------------------- workflow

def _reducing_sale_of_breach(plan: Plan, risk: AgentReport) -> bool:
    """The only veto is the concentration limit on the symbol, and the request is a
    sale that reduces that held long position. Re-analysis may then run; nothing it
    finds can lift the veto or any blocking rule."""
    k = risk.findings
    trade_check = k.get("trade_check") or {}
    return (bool(plan.trade) and plan.trade["side"] == "sell"
            and plan.symbol in (k.get("concentration_breaches") or {})
            and len(k.get("vetoes") or []) == 1
            and trade_check.get("reduces_exposure") is True)


def review_policy(llm: Optional[StructuredLLM], required: Optional[bool]) -> Dict[str, Any]:
    """Must a trade wait for a usable model review? An explicit argument wins, then
    AGENT_LLM_REVIEW_REQUIRED; otherwise yes whenever a model client is configured -
    also one that is configured but unusable (missing key, wrong provider), so
    breaking the model configuration cannot switch the review off. Only a workflow
    with no model at all (and no requirement) keeps the rule-only mode."""
    if required is not None:
        return {"required": bool(required), "source": "argument"}
    env = os.getenv("AGENT_LLM_REVIEW_REQUIRED", "").strip().lower()
    if env:
        return {"required": env in ("1", "true", "yes"), "source": "AGENT_LLM_REVIEW_REQUIRED"}
    return {"required": llm is not None,
            "source": "default: model configured" if llm is not None else "default: no model (rule-only)"}


def data_summary(provenance: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sources grouped for people: PostgreSQL / Derived analytics / Paper simulation."""
    groups: Dict[str, List[str]] = {}
    for s in provenance.get("sources", []):
        label = DATA_LABELS.get(s.get("kind"), s.get("kind") or "unknown")
        detail = s.get("detail") or {}
        if s.get("tool") == "market_data":
            name = f"{detail.get('table', 'eod_data')}（{s['source'].rsplit('/', 1)[-1]}，经 Java /market/eod）"
        else:
            name = s.get("source")
        groups.setdefault(label, [])
        if name not in groups[label]:
            groups[label].append(name)
    order = list(DATA_LABELS.values())
    return [{"label": k, "sources": v} for k, v in sorted(groups.items(),
                                                          key=lambda kv: order.index(kv[0]) if kv[0] in order else 9)]


def _planner_summary(plan: Plan, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which planner produced the plan, with every attempt (model, raw output,
    validator verdict) and whether - and why - the rule planner took over."""
    return {"planner": plan.planner, "model": plan.model, "fallback": plan.planner == "rule",
            "fallback_reason": plan.fallback_reason, "reasoning_summary": plan.reasoning_summary,
            "constraints": plan.constraints, "notes": plan.notes, "revision": plan.revision,
            "attempts": attempts}


class DecisionWorkflow:
    def __init__(self, tools: ToolRegistry, planning: PlanningService, memory: MemoryStore,
                 proposals: Optional[ProposalStore] = None, run_log: Optional[RunLog] = None,
                 archive: Any = None, today: Optional[Callable[[], date]] = None,
                 decision_agent_factory: Optional[Callable[[], Any]] = None,
                 llm: Optional[StructuredLLM] = None, llm_review_required: Optional[bool] = None):
        self.tools = tools
        self.planning = planning
        self.memory = memory
        self.proposals = proposals or ProposalStore()
        self.run_log = run_log or RunLog()
        self.archive = archive
        llm = llm if llm is not None else planning.llm
        self.review_policy = review_policy(llm, llm_review_required)
        required = self.review_policy["required"]
        self.research = ResearchAgent()
        self.risk = RiskAgent()
        self.critic = (CriticAgent(today, llm=llm, review_required=required) if today
                       else CriticAgent(llm=llm, review_required=required))
        self.decision = PortfolioDecisionAgent(decision_agent_factory, llm=llm)

    @staticmethod
    def _report(run: AgentRun, report: AgentReport) -> None:
        run.event("agent_report", f"{report.agent}: {report.status} - {report.summary}",
                  agent=report.agent, report=report.to_dict())

    def _new_run(self, goal_text: str, ctx: ToolContext, symbol: Optional[str],
                 portfolio_id: Optional[int]) -> AgentRun:
        run = AgentRun(agent=WORKFLOW_AGENT, owner_user_id=ctx.user_id, status=RunStatus.RUNNING,
                       input={"goal_text": goal_text, "symbol": symbol, "portfolio_id": portfolio_id})
        run.event("goal", f"goal received: {goal_text}", agent="user")
        self.run_log.add(run)
        return run

    async def run(self, goal_text: str, ctx: ToolContext, symbol: Optional[str] = None,
                  portfolio_id: Optional[int] = None) -> AgentRun:
        run = self._new_run(goal_text, ctx, symbol, portfolio_id)
        await self._execute(run, goal_text, ctx, symbol, portfolio_id)
        return run

    def start(self, goal_text: str, ctx: ToolContext, symbol: Optional[str] = None,
              portfolio_id: Optional[int] = None) -> AgentRun:
        """Start a run in the background; its contract (status RUNNING) grows as agents work."""
        run = self._new_run(goal_text, ctx, symbol, portfolio_id)
        task = asyncio.get_running_loop().create_task(
            self._execute(run, goal_text, ctx, symbol, portfolio_id))
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)
        return run

    async def _execute(self, run: AgentRun, goal_text: str, ctx: ToolContext, symbol: Optional[str],
                       portfolio_id: Optional[int]) -> None:
        try:
            await self._run(run, goal_text, replace(ctx, task_id=run.task_id, approval=None),
                            symbol, portfolio_id)
        except Exception as exc:  # the contract must end in a final state, whatever happened
            logger.exception("workflow run %s failed", run.task_id)
            run.finish(RunStatus.FAILED, {"error": f"internal error: {type(exc).__name__}: {exc}"},
                       "internal error", ToolErrorCode.INTERNAL.value)
        finally:
            if self.archive is not None:
                await self.archive.save(run)
            self.run_log.add(run)

    async def _retrieve(self, run: AgentRun, goal_text: str, ctx: ToolContext,
                        subjects: List[str]) -> MemoryContext:
        memory = await self.memory.retrieve_context(goal_text, ctx.user_id, subjects)
        if memory.available:
            message = (f"retrieved {len(memory.preferences)} user constraint(s) and "
                       f"{len(memory.items)} relevant memorie(s)"
                       + (f"; {len(memory.ignored)} stale/superseded ignored" if memory.ignored else ""))
        else:
            message = f"memory unavailable: {memory.unavailable_reason}"
        run.event("memory", message, agent="memory", memory=memory.to_dict())
        return memory

    async def _run(self, run: AgentRun, goal_text: str, ctx: ToolContext, symbol: Optional[str],
                   portfolio_id: Optional[int]) -> AgentRun:
        if ctx.user_id is None:
            return run.finish(RunStatus.FAILED, {"error": "authentication required"},
                              "no authenticated user", ToolErrorCode.UNAUTHENTICATED.value)

        facts = goal_facts(goal_text, symbol, portfolio_id)
        subjects = list(facts.symbols) + ([f"portfolio:{portfolio_id}"] if portfolio_id else [])
        memory = await self._retrieve(run, goal_text, ctx, subjects)

        plan, attempts = await self.planning.plan(goal_text, memory, symbol, portfolio_id)
        for a in attempts:  # model, raw planner output, validator verdict, fallback status
            who = f"{a['planner']} planner" + (f" ({a['model']})" if a.get("model") else "")
            run.event("plan", f"{who}: {a['outcome']}" + (f" - {a['detail']}" if a.get("detail") else ""),
                      agent="planner", **a)
        if isinstance(plan, Clarification):
            run.event("decision", f"ask user: {plan.question}", agent="planner")
            return run.finish(RunStatus.FAILED, {"question": plan.question}, plan.reason,
                              "NEEDS_USER_INPUT")
        if plan.symbol and plan.symbol not in subjects:
            memory = await self._retrieve(run, goal_text, ctx, subjects + [plan.symbol])
        run.event("plan", f"plan accepted ({plan.planner}): {len(plan.steps)} research step(s)",
                  agent="planner", plan=plan.to_dict())

        runner = ToolRunner(run, self.tools, ctx)
        reports: List[AgentReport] = []
        research = await self.research.run(plan, runner)
        self._report(run, research)
        reports.append(research)
        if research.status == "FAILED":
            return self._fail(run, reports, research, plan, attempts)
        risk = await self.risk.run(plan, research, runner, memory)
        self._report(run, risk)
        reports.append(risk)

        critic = await self.critic.run(plan, research, risk, memory)
        self._report(run, critic)
        reports.append(critic)
        reanalysis: Dict[str, Any] = {"performed": False}
        remedies = [i for i in critic.findings["issues"]
                    if i["severity"] in ("HIGH", "MEDIUM") and i.get("remedy")]
        if risk.status == "VETO" and not _reducing_sale_of_breach(plan, risk):
            if not critic.findings["valid"]:
                run.event("decision", "no re-analysis: the Risk Agent vetoed new exposure", agent="critic")
        elif not critic.findings["valid"] and remedies:
            if risk.status == "VETO":
                run.event("decision", "re-analysis allowed: the only veto is the concentration limit "
                          "this sale reduces; the veto and every blocking rule still apply", agent="critic")
            remedy: Dict[str, Any] = {}
            for i in remedies:
                if i["remedy"].get("days"):
                    remedy["days"] = max(remedy.get("days", 0), i["remedy"]["days"])
                if i["remedy"].get("factors"):
                    remedy["factors"] = list(dict.fromkeys(remedy.get("factors", []) + i["remedy"]["factors"]))
            issue = "; ".join(f"{i['code']}: {i['message']}" for i in remedies)
            revised, log = await self.planning.revise(
                plan, issue, remedy, [o.to_dict() for o in run.observations], memory)
            run.event("replan", f"critic requested a revision -> planner: {log['outcome']}",
                      agent="planner", issue=issue, remedy=remedy,
                      plan=revised.to_dict() if revised else None,
                      **{k: v for k, v in log.items() if k != "outcome"})
            reanalysis = {"performed": revised is not None, "issue": issue, "remedy": remedy,
                          "planner": log}
            if revised is not None:
                research2 = await self.research.run(revised, runner, round=2)
                self._report(run, research2)
                reports.append(research2)
                if research2.status == "FAILED":
                    reanalysis["outcome"] = f"re-analysis failed: {research2.summary}; round-1 findings kept"
                else:
                    plan, research = revised, research2
                    risk = await self.risk.run(plan, research, runner, memory, round=2)
                    self._report(run, risk)
                    reports.append(risk)
                    critic = await self.critic.run(plan, research, risk, memory, round=2)
                    self._report(run, critic)
                    reports.append(critic)
                    reanalysis["outcome"] = "re-analysed with the revised plan"

        failures = [i for rep in reports if rep.agent == "critic"
                    for i in rep.findings["issues"] if i["code"] in REVIEW_FAILURE_CODES]
        decision, result, level = await self.decision.run(
            plan, research, risk, critic, memory, self.proposals, ctx, run.task_id,
            review_failures=failures)
        self._report(run, decision)
        reports.append(decision)
        if level.rank >= ActionLevel.EXECUTE.rank:  # structural guarantee, never expected
            raise AssertionError("the workflow may not reach EXECUTE")
        run.raise_level(level)
        result["action_level"] = run.action_level.value
        result["reanalysis"] = reanalysis
        result["review_policy"] = self.review_policy
        result["planner"] = _planner_summary(plan, attempts)
        result["memory_used"] = {"preferences": {k: v["value"] for k, v in memory.preferences.items()},
                                 "relevant_ids": [m.item.id for m in memory.items],
                                 "decision_ids": [d.id for d in memory.decisions],
                                 "ignored_ids": [m.item.id for m in memory.ignored],
                                 "lines": memory.lines(), "available": memory.available}
        result["agents"] = [rep.to_dict() for rep in reports]
        evidence = list(dict.fromkeys(list(research.evidence) + list(risk.evidence)))
        run.data_provenance = provenance_for(run, evidence)
        result["data_sources"] = sorted({s["source"] for s in run.data_provenance["sources"]})
        result["data_summary"] = data_summary(run.data_provenance)
        waiver = (result["trade"] or {}).get("waiver")
        if waiver is not None:  # option 2: what was downgraded, on which basis, or why not
            run.event("decision", "research-quality issues downgraded to WARNING; the user must approve"
                      if waiver["granted"] else "research-quality downgrade refused: "
                      + "; ".join(c["condition"] for c in waiver["conditions"] if not c["passed"]),
                      agent="decision", waiver=waiver)
        if result["trade"] and result["trade"].get("status") == "AWAITING_APPROVAL":
            run.event("approval", "trade proposal stored; waiting for the user's approval (PAPER)",
                      agent="approval", proposal_id=result["trade"]["proposal"]["proposal_id"])
        else:
            run.event("approval", "no trade proposal: nothing awaits approval, nothing was executed",
                      agent="approval")
        run.event("decision", f"{result['recommendation']} ({run.action_level.value})", agent="decision",
                  evidence=evidence)
        result["memory_written"] = await self._remember(run, plan, result, ctx)
        return run.finish(RunStatus.SUCCESS, result, "decision ready"
                          + (" - awaiting human approval (nothing was executed)"
                             if result["approval_required"] else ""))

    def _fail(self, run: AgentRun, reports: List[AgentReport], failed: AgentReport,
              plan: Plan, attempts: List[Dict[str, Any]]) -> AgentRun:
        run.data_provenance = provenance_for(run, list(failed.evidence))
        return run.finish(RunStatus.FAILED,
                          {"error": failed.summary, "agents": [r.to_dict() for r in reports],
                           "planner": _planner_summary(plan, attempts)},
                          failed.summary, failed.error_code or "INTERNAL")

    async def _remember(self, run: AgentRun, plan: Plan, result: Dict[str, Any],
                        ctx: ToolContext) -> Optional[int]:
        """The conclusion (and any proposal) go back to memory, linked to this run."""
        subject = plan.symbol or f"portfolio:{plan.portfolio_id}"
        content = {"subject": subject, "recommendation": result["recommendation"],
                   "reason": "; ".join(result["rationale"])[:500] or "no rationale",
                   "action_level": run.action_level.value, "goal": plan.goal[:200],
                   "risk_level": str(result["risk"]["level"]), "confidence": result["confidence"]}
        market = result["research"]["market"].get(plan.symbol) if plan.symbol else None
        if market:
            content["as_of"] = market["last_date"]
        written: Optional[int] = None
        try:
            item = await self.memory.add(ctx.user_id, "decision", content,
                                         source_task_id=run.task_id, importance=3)
            written = item.id
            run.event("memory_write", f"decision remembered as memory #{item.id}", agent="memory",
                      memory_id=item.id, source_task_id=run.task_id)
            trade = result.get("trade") or {}
            if trade.get("status") == "AWAITING_APPROVAL":
                order = trade["proposal"]["action"]
                proposal = await self.memory.add(ctx.user_id, "proposal", {
                    "subject": subject, "side": order["side"], "quantity": order["quantity"],
                    "status": "AWAITING_APPROVAL", "proposal_id": trade["proposal"]["proposal_id"],
                    "reason": content["reason"][:300]}, source_task_id=run.task_id, importance=4)
                run.event("memory_write", f"proposal remembered as memory #{proposal.id}",
                          agent="memory", memory_id=proposal.id)
        except (MemoryValidationError, Exception) as exc:  # memory is context; report, don't fail the run
            run.event("memory_write", f"not remembered: {exc}", agent="memory")
        return written
