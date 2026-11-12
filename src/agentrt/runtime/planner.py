# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observation-driven planner.

``next_action(state)`` is called after every observation and decides the single
next step from what has been observed so far - there is no fixed step list. The
same goal can take different paths:

* a retryable tool error (TIMEOUT / UNAVAILABLE) is retried, then the run stops;
* NOT_FOUND / FORBIDDEN / UNAUTHENTICATED stop the run immediately;
* INVALID_INPUT asks the user to correct the goal;
* INSUFFICIENT_DATA stops the run when the tool was required and is skipped
  (and reported) when it was optional;
* a "high" risk observation ends directional analysis early and suppresses any
  trade proposal; an empty portfolio ends the run before risk is computed.

The policy is deterministic and rule-based. No language model is involved:
without model credentials that is the only honest option, and it keeps every
decision reproducible from the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..tools.base import RETRYABLE_ERRORS, ToolErrorCode, ToolResult
from .contract import ActionLevel

MAX_ATTEMPTS = 3
MIN_BARS_FOR_ANALYSIS = 21
DEFAULT_CANDIDATES = ["momentum", "rsi_reversion", "bollinger_reversion", "macd"]
INTENTS = ("analyze_symbol", "portfolio_risk", "strategy_suggestion")


# ------------------------------------------------------------------------ goal & state

@dataclass
class Goal:
    intent: str
    symbol: Optional[str] = None
    portfolio_id: Optional[int] = None
    days: int = 180
    strategies: Optional[List[str]] = None
    factors: Optional[List[str]] = None
    trade: Optional[Dict[str, Any]] = None  # {"side": "buy"|"sell", "quantity": float}

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Step:
    tool: str
    key: str               # distinguishes calls of the same tool (e.g. backtest per strategy)
    call_id: str
    input: Dict[str, Any]
    result: ToolResult
    input_call_ids: List[str] = field(default_factory=list)


@dataclass
class LoopState:
    goal: Goal
    steps: List[Step] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def last(self, tool: str, key: str = "") -> Optional[Step]:
        for step in reversed(self.steps):
            if step.tool == tool and step.key == key:
                return step
        return None

    def note(self, message: str) -> None:
        # next_action re-reads the same failed step on every turn
        if message not in self.notes:
            self.notes.append(message)

    def attempts(self, tool: str, key: str = "") -> int:
        return sum(1 for s in self.steps if s.tool == tool and s.key == key)


# ---------------------------------------------------------------------------- actions

@dataclass
class CallTool:
    tool: str
    input: Dict[str, Any]
    reason: str
    key: str = ""
    input_call_ids: List[str] = field(default_factory=list)
    # result to report if this call is blocked by the approval gate
    draft_result: Optional[Dict[str, Any]] = None


@dataclass
class Finish:
    result: Dict[str, Any]
    reason: str
    level: ActionLevel = ActionLevel.ANALYSIS
    evidence: List[str] = field(default_factory=list)  # call_ids the result rests on


@dataclass
class Stop:
    error_code: str
    reason: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class AskUser:
    question: str
    reason: str


Action = Any  # CallTool | Finish | Stop | AskUser


# ---------------------------------------------------------------------------- planner

class RuleBasedPlanner:
    def next_action(self, state: LoopState) -> Action:
        goal = state.goal
        if goal.intent not in INTENTS:
            return AskUser(f"Unsupported goal '{goal.intent}'. Choose one of {list(INTENTS)}.",
                           "the goal cannot be planned")
        if goal.intent == "portfolio_risk":
            if goal.portfolio_id is None:
                return AskUser("Which portfolio_id should be analysed?", "goal is missing portfolio_id")
            return self._portfolio_risk(state)
        if not goal.symbol:
            return AskUser("Which symbol should be analysed?", "goal is missing symbol")
        if goal.intent == "analyze_symbol":
            return self._analyze_symbol(state)
        return self._strategy_suggestion(state)

    # ---- error policy -------------------------------------------------------------

    def _on_error(self, state: LoopState, step: Step, required: bool) -> Optional[Action]:
        """Action for a failed step, or None to continue without it (optional tools)."""
        code = step.result.error_code
        message = step.result.error_message
        if code in RETRYABLE_ERRORS:
            attempts = state.attempts(step.tool, step.key)
            if attempts < MAX_ATTEMPTS:
                return CallTool(step.tool, step.input, key=step.key,
                                reason=f"retry {step.tool} after {code.value} (attempt {attempts + 1} of {MAX_ATTEMPTS})",
                                input_call_ids=list(step.input_call_ids))
            return Stop(code.value, f"{step.tool} still failing after {attempts} attempts: {message}",
                        [step.call_id])
        if code == ToolErrorCode.INVALID_INPUT:
            return AskUser(f"{step.tool} rejected the input: {message}. Please correct the goal.",
                           f"{step.tool} input invalid")
        if code == ToolErrorCode.INSUFFICIENT_DATA and not required:
            state.note(f"{step.tool}{'/' + step.key if step.key else ''} skipped: {message}")
            return None
        return Stop(code.value, f"{step.tool} failed: {message}", [step.call_id])

    def _market_data(self, state: LoopState) -> Tuple[Optional[Action], Optional[Step]]:
        goal = state.goal
        step = state.last("market_data")
        if step is None:
            return CallTool("market_data", {"symbol": goal.symbol, "days": goal.days},
                            reason=f"need price history for {goal.symbol}"), None
        if not step.result.ok:
            return self._on_error(state, step, required=True), None
        if step.result.output["count"] < MIN_BARS_FOR_ANALYSIS:
            return Stop(ToolErrorCode.INSUFFICIENT_DATA.value,
                        f"only {step.result.output['count']} bars for {goal.symbol}, "
                        f"need {MIN_BARS_FOR_ANALYSIS}", [step.call_id]), None
        return None, step

    def _bars_input(self, md: Step, **extra: Any) -> Dict[str, Any]:
        return {"symbol": md.result.output["symbol"], "bars": md.result.output["bars"], **extra}

    def _symbol_risk(self, state: LoopState, md: Step) -> Tuple[Optional[Action], Optional[Step]]:
        step = state.last("risk")
        if step is None:
            return CallTool("risk", {"symbol": md.result.output["symbol"],
                                     "bars": md.result.output["bars"]},
                            reason="measure the symbol's risk from the observed bars",
                            input_call_ids=[md.call_id]), None
        if not step.result.ok:
            action = self._on_error(state, step, required=False)
            return action, None
        return None, step

    # ---- analyze_symbol -----------------------------------------------------------

    def _analyze_symbol(self, state: LoopState) -> Action:
        goal = state.goal
        action, md = self._market_data(state)
        if action:
            return action
        action, risk = self._symbol_risk(state, md)
        if action:
            return action
        evidence = [md.call_id] + ([risk.call_id] if risk else [])
        risk_level = risk.result.output["risk_level"] if risk else "unknown"

        if risk_level == "high":
            # Observation changes the path: no directional call, no trade proposal.
            return Finish(self._symbol_result(md, risk, None, None, state,
                                              recommendation="NO_NEW_EXPOSURE",
                                              why=["risk level is high: "
                                                   f"{risk.result.output['metrics']}"],
                                              trade_status=self._trade_refused(goal, "risk level is high")),
                          reason="risk too high - directional analysis and trading suppressed",
                          level=ActionLevel.PROPOSE, evidence=evidence)

        signal = state.last("signal")
        if signal is None:
            return CallTool("signal", self._bars_input(md, **({"strategies": goal.strategies}
                                                              if goal.strategies else {})),
                            reason="risk is acceptable; compute strategy signals",
                            input_call_ids=[md.call_id])
        if not signal.result.ok:
            return self._on_error(state, signal, required=True)
        evidence.append(signal.call_id)

        factor = state.last("factor")
        if goal.factors is not None and factor is None:
            return CallTool("factor", self._bars_input(md, factors=goal.factors),
                            reason="goal asks for factor exposures", input_call_ids=[md.call_id])
        if factor is not None and not factor.result.ok:
            action = self._on_error(state, factor, required=False)
            if action:
                return action
            factor = None
        if factor:
            evidence.append(factor.call_id)

        recommendation, why = self._decide(signal, risk_level)
        result = self._symbol_result(md, risk, signal, factor, state, recommendation, why, None)

        if goal.trade:
            side = goal.trade.get("side")
            wanted = "BUY" if side == "buy" else "SELL"
            if recommendation != wanted:
                result["execution"] = self._trade_refused(
                    goal, f"analysis recommends {recommendation}, not {wanted}")
                return Finish(result, reason="trade request conflicts with the analysis",
                              level=ActionLevel.PROPOSE, evidence=evidence)
            execution = state.last("execution")
            if execution is None:
                return CallTool("execution", {
                    "symbol": md.result.output["symbol"], "side": side,
                    "quantity": goal.trade.get("quantity"),
                    "reference_price": md.result.output["last_close"],
                    "reference_date": md.result.output["last_date"]},
                    reason=f"analysis supports {wanted}; prepare the paper order",
                    input_call_ids=[md.call_id, signal.call_id], draft_result=result)
            if not execution.result.ok:
                return self._on_error(state, execution, required=True)
            result["execution"] = execution.result.output
            evidence.append(execution.call_id)
            return Finish(result, reason="approved paper order executed",
                          level=ActionLevel.EXECUTE, evidence=evidence)

        return Finish(result, reason="analysis complete", level=ActionLevel.PROPOSE, evidence=evidence)

    @staticmethod
    def _decide(signal: Step, risk_level: str) -> Tuple[str, List[str]]:
        out = signal.result.output
        score = out["consensus_score"]
        why = [f"consensus of {len(out['signals'])} strategies = {score} ({out['consensus_label']})"]
        if out["unavailable"]:
            why.append(f"strategies without a signal: {sorted(out['unavailable'])}")
        if score >= 0.5:
            recommendation = "BUY"
        elif score <= -0.5:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        if recommendation == "BUY" and risk_level == "medium":
            why.append("risk level medium: size positions conservatively")
        return recommendation, why

    @staticmethod
    def _trade_refused(goal: Goal, why: str) -> Optional[Dict[str, Any]]:
        if not goal.trade:
            return None
        return {"status": "NOT_PROPOSED", "reason": why}

    @staticmethod
    def _symbol_result(md: Step, risk: Optional[Step], signal: Optional[Step],
                       factor: Optional[Step], state: LoopState, recommendation: str,
                       why: List[str], trade_status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        out = md.result.output
        result = {
            "symbol": out["symbol"],
            "as_of": out["last_date"],
            "last_close": out["last_close"],
            "bars": out["count"],
            "recommendation": recommendation,
            "recommendation_is_advice_only": True,
            "rationale": why,
            "risk": risk.result.output if risk else None,
            "signals": signal.result.output if signal else None,
            "factors": factor.result.output if factor else None,
            "skipped": list(state.notes),
            "decided_by": "rule-based planner (deterministic)",
        }
        if trade_status:
            result["execution"] = trade_status
        return result

    # ---- portfolio_risk -----------------------------------------------------------

    def _portfolio_risk(self, state: LoopState) -> Action:
        goal = state.goal
        portfolio = state.last("portfolio")
        if portfolio is None:
            return CallTool("portfolio", {"portfolio_id": goal.portfolio_id},
                            reason="check access and read the portfolio's positions")
        if not portfolio.result.ok:
            return self._on_error(state, portfolio, required=True)
        if portfolio.result.output["position_count"] == 0:
            return Stop(ToolErrorCode.INSUFFICIENT_DATA.value,
                        f"portfolio {goal.portfolio_id} has no positions; there is no risk to measure",
                        [portfolio.call_id])
        risk = state.last("risk")
        if risk is None:
            return CallTool("risk", {"portfolio_id": goal.portfolio_id},
                            reason=f"{portfolio.result.output['position_count']} positions; compute portfolio risk",
                            input_call_ids=[portfolio.call_id])
        if not risk.result.ok:
            return self._on_error(state, risk, required=True)
        out = risk.result.output
        result = {"portfolio_id": goal.portfolio_id,
                  "positions": portfolio.result.output["position_count"],
                  "risk_level": out["risk_level"], "metrics": out["metrics"],
                  "unavailable": out["unavailable"], "alerts": [], "proposal": None,
                  "decided_by": "rule-based planner (deterministic)"}
        level = ActionLevel.ANALYSIS
        if out["risk_level"] == "high":
            result["alerts"].append("portfolio risk level is high")
            result["proposal"] = {"action": "REVIEW_AND_REDUCE_EXPOSURE", "requires_human": True}
            level = ActionLevel.PROPOSE
        return Finish(result, reason="portfolio risk measured", level=level,
                      evidence=[portfolio.call_id, risk.call_id])

    # ---- strategy_suggestion ------------------------------------------------------

    def _strategy_suggestion(self, state: LoopState) -> Action:
        goal = state.goal
        action, md = self._market_data(state)
        if action:
            return action
        candidates = goal.strategies or DEFAULT_CANDIDATES
        for name in candidates:
            step = state.last("backtest", key=name)
            if step is None:
                return CallTool("backtest", self._bars_input(md, strategy=name), key=name,
                                reason=f"simulate candidate strategy {name}",
                                input_call_ids=[md.call_id])
            if not step.result.ok:
                action = self._on_error(state, step, required=False)
                if isinstance(action, (CallTool, AskUser)):
                    return action  # retry this candidate / the user named an unknown strategy
                if isinstance(action, Stop) and step.result.error_code in RETRYABLE_ERRORS:
                    return action  # backend down: stop instead of ranking a partial field
                # any other failure: this candidate is out, keep going
                if action is not None:
                    state.note(f"backtest/{name} excluded: {step.result.error_message}")

        tested = [state.last("backtest", key=n) for n in candidates]
        ok = [s for s in tested if s and s.result.ok]
        evidence = [md.call_id] + [s.call_id for s in tested if s]
        if not ok:
            return Stop(ToolErrorCode.INSUFFICIENT_DATA.value,
                        "no candidate strategy could be simulated on the available bars", evidence)
        traded = [s for s in ok if s.result.output["trade_count"] > 0
                  and s.result.output["metrics"].get("sharpe_ratio") is not None]
        ranking = sorted(traded, key=lambda s: s.result.output["metrics"]["sharpe_ratio"], reverse=True)

        action, risk = self._symbol_risk(state, md)
        if action:
            return action
        if risk:
            evidence.append(risk.call_id)
        risk_level = risk.result.output["risk_level"] if risk else "unknown"

        table = [{"strategy": s.key, "trade_count": s.result.output["trade_count"],
                  "sharpe_ratio": s.result.output["metrics"].get("sharpe_ratio"),
                  "total_return": s.result.output["metrics"].get("total_return"),
                  "max_drawdown": s.result.output["metrics"].get("max_drawdown")} for s in ok]
        result = {"symbol": md.result.output["symbol"],
                  "period": [md.result.output["first_date"], md.result.output["last_date"]],
                  "candidates": table, "skipped": list(state.notes), "risk_level": risk_level,
                  "suggestion": None, "suggestion_is_advice_only": True,
                  "decided_by": "rule-based planner (deterministic)"}
        if not ranking:
            result["reason"] = "no candidate traded in this window; nothing to rank"
            return Finish(result, reason="no strategy produced trades - no suggestion",
                          level=ActionLevel.SIMULATE, evidence=evidence)
        best = ranking[0]
        result["suggestion"] = {"strategy": best.key,
                                "why": f"highest Sharpe ({best.result.output['metrics']['sharpe_ratio']:.4f}) "
                                       f"among {len(ranking)} candidates that traded",
                                "caution": "risk level high" if risk_level == "high" else None}
        return Finish(result, reason="strategy suggestion ready", level=ActionLevel.PROPOSE,
                      evidence=evidence)
