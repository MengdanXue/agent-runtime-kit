# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal -> structured Plan -> validator.

Plan schema (what the LLM fills in, and what the rule planner produces)::

    {
      "goal": "...",
      "reasoning_summary": "...",
      "steps": [{"tool": "...", "purpose": "...",
                 "required_inputs": [{"name": "...", "value": "...", "from_step": 0}]}],
      "constraints": [{"key": "...", "value": "...", "source": "..."}],
      "clarification_question": ""
    }

``required_inputs`` name each input a step needs: a literal ``value`` for plain
parameters (symbol, days, strategy names), or ``from_step`` (1-based) when the
input is another step's output - bars always come from a market_data step, so
the planner can never supply prices.

The LLM only understands the goal, splits it into research steps and explains
why. It cannot decide to trade, skip the Risk Agent, reach the approval gate or
change key tool parameters. The validator enforces that for every plan, LLM or
rule:

* tools: research tools only - ``execution`` is rejected, ``risk`` is dropped
  (the Risk Agent always runs RiskTool itself);
* inputs: a per-tool whitelist (no commission / capital / window / price
  fields), bounded values, strategy and factor names from the registries the
  tools run (the planner sees them as ``allowed_values``), bars only by
  reference to a market_data step;
* necessary steps: what the goal type needs (e.g. a position review reads the
  portfolio, the symbol's bars and its signals), and what the goal text asks
  for when it can be read deterministically;
* goal facts the model may not change: the portfolio the user selected, the
  tickers the user named, and any trade the user asked about (those come from
  the text, never from the model);
* user constraints: the plan's constraints must match memory; a contradiction
  rejects the plan and missing ones are restored.

Without a usable model the rule planner builds the same schema and the plan
records ``planner: "rule"`` with the ``fallback_reason``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import ValidationError

from .llm import LLMError, StructuredLLM, llm_status
from .memory import PREFERENCE_KEYS

WORKFLOW_INTENTS = ("analyze_symbol", "position_review", "portfolio_risk", "strategy_suggestion")
RESEARCH_TOOLS = ("market_data", "portfolio", "signal", "factor", "backtest")
BARS_TOOLS = ("signal", "factor", "backtest")
BARS_REF = "$bars"
MAX_STEPS = 12
DEFAULT_CANDIDATES = ["momentum", "rsi_reversion", "bollinger_reversion", "macd"]
DEFAULT_FACTORS = ["momentum_20d", "volatility_20d", "rsi_14d"]
DEFAULT_DAYS = 180

# Inputs a plan may set, per tool. Everything else (commission, slippage,
# capital, strategy params, window, any price field) is a key constraint the
# planner may not touch.
INPUT_WHITELIST: Dict[str, Tuple[str, ...]] = {
    "market_data": ("symbol", "days"),
    "portfolio": ("portfolio_id",),
    "signal": ("symbol", "bars", "strategies"),
    "factor": ("symbol", "bars", "factors"),
    "backtest": ("symbol", "bars", "strategy"),
}
REQUIRED_INPUTS: Dict[str, Tuple[str, ...]] = {
    "market_data": ("symbol",),
    "portfolio": ("portfolio_id",),
    "signal": ("symbol", "bars"),
    "factor": ("symbol", "bars"),
    "backtest": ("symbol", "bars", "strategy"),
}
REQUIRED_STEPS: Dict[str, Tuple[str, ...]] = {
    "analyze_symbol": ("market_data", "signal"),
    "position_review": ("portfolio", "market_data", "signal"),
    "portfolio_risk": ("portfolio",),
    "strategy_suggestion": ("market_data", "backtest"),
}

_DUMMY_BARS = [{"date": "2000-01-01", "close": 1.0}, {"date": "2000-01-02", "close": 1.0}]
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9.=\-^]{0,31}$")


class PlanValidationError(ValueError):
    """A plan the runtime refuses to execute."""


@dataclass(frozen=True)
class PlanStep:
    tool: str
    purpose: str
    args: Tuple[Tuple[str, Any], ...]            # resolved inputs; bars as "$bars"
    required_inputs: Tuple[Tuple[Tuple[str, Any], ...], ...] = ()

    @property
    def arguments(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.args}

    @property
    def reason(self) -> str:
        return self.purpose

    def signature(self) -> str:
        return json.dumps({"tool": self.tool, "args": self.arguments}, sort_keys=True, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "purpose": self.purpose,
                "required_inputs": [dict(i) for i in self.required_inputs]}


def make_step(tool: str, purpose: str, required_inputs: List[Dict[str, Any]], **args: Any) -> PlanStep:
    return PlanStep(tool=tool, purpose=purpose,
                    args=tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in args.items())),
                    required_inputs=tuple(tuple(sorted(i.items())) for i in required_inputs))


@dataclass
class Plan:
    goal: str
    reasoning_summary: str
    intent: str
    symbol: Optional[str]
    portfolio_id: Optional[int]
    steps: List[PlanStep]
    constraints: List[Dict[str, Any]]
    planner: str                                   # "llm" | "rule"
    model: Optional[str] = None
    fallback_reason: Optional[str] = None
    trade: Optional[Dict[str, Any]] = None         # from the user's text, never from the model
    direction: Optional[str] = None
    direct_execution_requested: bool = False
    notes: List[str] = field(default_factory=list)
    revision: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "reasoning_summary": self.reasoning_summary,
                "steps": [s.to_dict() for s in self.steps],
                "constraints": [dict(c) for c in self.constraints],
                "planner": self.planner, "fallback_reason": self.fallback_reason,
                "model": self.model,
                "subject": {"intent": self.intent, "symbol": self.symbol,
                            "portfolio_id": self.portfolio_id},
                "trade": self.trade, "direction": self.direction,
                "direct_execution_requested": self.direct_execution_requested,
                "notes": list(self.notes), "revision": self.revision}


@dataclass(frozen=True)
class Clarification:
    question: str
    reason: str


# ------------------------------------------------------------------------ goal facts

_TICKER = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5}(?:[.=-][A-Z]{1,3})?)(?![A-Za-z0-9])")
_NOT_TICKERS = {"AI", "ETF", "VAR", "CVAR", "HOLD", "BUY", "SELL", "PNL", "ID", "API", "USD",
                "CNY", "CEO", "IPO", "MA", "RSI", "MACD", "EOD", "OK", "PAPER", "I", "A"}
_STRATEGY_WORDS = ("策略", "回测", "strategy", "backtest")
_PORTFOLIO_WORDS = ("组合", "仓位", "持仓", "portfolio", "position", "我的")
_RISK_WORDS = ("风险", "risk")
_BUY_WORDS = ("买入", "买", "加仓", "增加", "增持", "buy", "add")
_SELL_WORDS = ("卖出", "卖掉", "减仓", "减持", "sell", "trim")
_DIRECT_WORDS = ("直接", "自动", "立即", "马上", "execute", "place order", "下单")


@dataclass(frozen=True)
class GoalFacts:
    """What the user's text and selections say, read deterministically.

    The model may not change these: they bound what a plan may do.
    """

    text: str
    symbols: Tuple[str, ...]
    portfolio_id: Optional[int]
    trade: Optional[Dict[str, Any]]
    direction: Optional[str]
    direct_execution_requested: bool
    days: Optional[int]
    expected_intent: Optional[str]
    question: Optional[Clarification]

    def to_dict(self) -> Dict[str, Any]:
        return {"symbols": list(self.symbols), "portfolio_id": self.portfolio_id,
                "trade": self.trade, "direction": self.direction,
                "direct_execution_requested": self.direct_execution_requested,
                "days": self.days, "expected_intent": self.expected_intent}


def goal_facts(text: str, symbol_hint: Optional[str] = None,
               portfolio_hint: Optional[int] = None) -> GoalFacts:
    lowered = (text or "").lower()
    symbols = [t for t in _TICKER.findall(text or "") if t not in _NOT_TICKERS]
    if symbol_hint:
        symbols = [symbol_hint.upper()] + [s for s in symbols if s != symbol_hint.upper()]
    symbols = list(dict.fromkeys(symbols))
    symbol = symbols[0] if symbols else None
    mentions_portfolio = any(w in lowered for w in _PORTFOLIO_WORDS)

    if any(w in lowered for w in _STRATEGY_WORDS):
        intent: Optional[str] = "strategy_suggestion"
    elif symbol and mentions_portfolio:
        intent = "position_review"
    elif symbol:
        intent = "analyze_symbol"
    elif mentions_portfolio or any(w in lowered for w in _RISK_WORDS):
        intent = "portfolio_risk"
    else:
        intent = None

    question = None
    if intent is None:
        question = Clarification("请说明要分析的标的代码（例如 AAPL）或组合。", "goal names no symbol or portfolio")
    elif intent in ("position_review", "portfolio_risk") and portfolio_hint is None:
        question = Clarification("请选择要分析的组合（portfolio_id）。", f"{intent} needs a portfolio")
    elif intent in ("analyze_symbol", "strategy_suggestion", "position_review") and not symbol:
        question = Clarification("请说明要分析的标的代码（例如 AAPL）。", f"{intent} needs a symbol")

    direction = None
    if any(w in lowered for w in _SELL_WORDS):
        direction = "sell"
    elif any(w in lowered for w in _BUY_WORDS):
        direction = "buy"
    quantity = None
    match = re.search(r"(\d+(?:\.\d+)?)\s*(股|shares?)", lowered)
    if match:
        quantity = float(match.group(1))
    days = None
    match = re.search(r"(\d{1,4})\s*(天|日|days?)", lowered)
    if match:
        days = max(5, min(3650, int(match.group(1))))
    elif any(w in lowered for w in ("一年", "1年", "year")):
        days = 365
    return GoalFacts(
        text=text, symbols=tuple(symbols), portfolio_id=portfolio_hint,
        trade={"side": direction, "quantity": quantity} if direction and quantity else None,
        direction=direction,
        direct_execution_requested=any(w in lowered for w in _DIRECT_WORDS) and direction is not None,
        days=days, expected_intent=intent, question=question)


# ------------------------------------------------------------------------ validation

def _symbol(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SYMBOL.match(value.strip().upper()):
        raise PlanValidationError(f"{where}: invalid symbol {value!r}")
    return value.strip().upper()


def _as_list(value: Any, where: str) -> List[str]:
    items = value if isinstance(value, list) else str(value or "").split(",")
    items = [str(i).strip() for i in items if str(i).strip()]
    if not items or any(len(i) > 40 for i in items):
        raise PlanValidationError(f"{where}: expected a comma-separated list of names")
    return items


NAMED_INPUTS = ("strategies", "strategy", "factors")


def allowed_names(name: str) -> List[str]:
    """Names a strategies / strategy / factors input may take: the registries the tools run."""
    if name == "factors":
        from agentrt.quant.factor_agent import FactorAgent
        return sorted(FactorAgent.FACTOR_DEFINITIONS)
    from agentrt.ports.strategies import strategy_names
    return strategy_names()


def _known_names(name: str, items: List[str], where: str) -> List[str]:
    allowed = allowed_names(name)
    unknown = [i for i in items if i not in allowed]
    if unknown:
        raise PlanValidationError(
            f"{where}: unknown {name} {unknown}; use names from allowed_values: {', '.join(allowed)}")
    return items


def _parse_input(name: str, value: Any, where: str, facts: GoalFacts) -> Any:
    if name == "symbol":
        return _symbol(value, where)
    if name == "days":
        try:
            days = int(str(value).strip())
        except ValueError as exc:
            raise PlanValidationError(f"{where}: days must be an integer") from exc
        if not 5 <= days <= 3650:
            raise PlanValidationError(f"{where}: days must be within 5..3650")
        return days
    if name == "portfolio_id":
        try:
            pid = int(str(value).strip())
        except ValueError as exc:
            raise PlanValidationError(f"{where}: portfolio_id must be an integer") from exc
        if pid != facts.portfolio_id:
            raise PlanValidationError(
                f"{where}: portfolio steps may only read the portfolio the user selected ({facts.portfolio_id})")
        return pid
    if name in ("strategies", "factors"):
        return _known_names(name, _as_list(value, where), where)
    if name == "strategy":
        names = _as_list(value, where)
        if len(names) != 1:
            raise PlanValidationError(f"{where}: backtest takes exactly one strategy")
        return _known_names(name, names, where)[0]
    raise PlanValidationError(f"{where}: unknown input {name}")


def _derive_subject(steps: List[PlanStep]) -> Tuple[str, Optional[str]]:
    tools = [s.tool for s in steps]
    md_symbols = [s.arguments["symbol"] for s in steps if s.tool == "market_data"]
    symbol = md_symbols[0] if md_symbols else None
    if "backtest" in tools:
        return "strategy_suggestion", symbol
    if "portfolio" in tools and symbol:
        return "position_review", symbol
    if "portfolio" in tools:
        return "portfolio_risk", None
    return "analyze_symbol", symbol


def _norm(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    try:
        return f"{float(text):g}"
    except ValueError:
        return text


def _check_constraints(raw_constraints: Any, memory: Any, notes: List[str]) -> List[Dict[str, Any]]:
    prefs = getattr(memory, "preferences", {}) or {}
    given = raw_constraints if isinstance(raw_constraints, list) else []
    seen = set()
    for c in given:
        if not isinstance(c, dict):
            continue
        key = str(c.get("key", "")).strip()
        if key not in PREFERENCE_KEYS:
            notes.append(f"constraint {key or '?'} dropped: not a user constraint in memory")
            continue
        if key not in prefs:
            raise PlanValidationError(f"plan states user constraint {key}={c.get('value')!r}, "
                                      "but memory holds no such constraint")
        if _norm(c.get("value")) != _norm(prefs[key]["value"]):
            raise PlanValidationError(
                f"plan contradicts user constraint {key}={prefs[key]['value']} "
                f"(memory #{prefs[key]['memory_id']})")
        seen.add(key)
    out = []
    for key, entry in prefs.items():
        if key not in seen:
            notes.append(f"constraint {key} restored from memory #{entry['memory_id']}")
        out.append({"key": key, "value": entry["value"], "source": f"memory #{entry['memory_id']}"})
    return out


def validate_plan(raw: Any, tools: Any, planner: str, facts: GoalFacts, memory: Any = None,
                  model: Optional[str] = None) -> Union[Plan, Clarification]:
    """Validate a raw plan (LLM or rule) against tools, goal facts and user constraints."""
    if not isinstance(raw, dict):
        raise PlanValidationError("plan must be an object")
    question = str(raw.get("clarification_question") or "").strip()
    if question:
        return Clarification(question, "planner needs more information")
    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanValidationError("plan needs at least one step")
    if len(raw_steps) > MAX_STEPS:
        raise PlanValidationError(f"plan has more than {MAX_STEPS} steps")

    notes: List[str] = []
    steps: List[PlanStep] = []
    index_of: Dict[int, Optional[PlanStep]] = {}   # raw 1-based position -> kept step
    for i, raw_step in enumerate(raw_steps, 1):
        where = f"step {i}"
        if not isinstance(raw_step, dict):
            raise PlanValidationError(f"{where} must be an object")
        tool = raw_step.get("tool")
        purpose = str(raw_step.get("purpose") or "").strip()
        if tool == "execution":
            raise PlanValidationError(
                f"{where}: execution cannot be planned; it runs only after a human approves a proposal")
        if tool == "risk":
            notes.append(f"{where} (risk) dropped: risk is assessed by the Risk Agent through RiskTool")
            index_of[i] = None
            continue
        if tool not in RESEARCH_TOOLS:
            raise PlanValidationError(f"{where}: unknown or forbidden tool {tool!r}")
        if not purpose:
            raise PlanValidationError(f"{where}: every step needs a purpose")
        inputs = raw_step.get("required_inputs")
        if not isinstance(inputs, list):
            raise PlanValidationError(f"{where}: required_inputs must be a list")

        args: Dict[str, Any] = {}
        normalized: List[Dict[str, Any]] = []
        bars_source: Optional[PlanStep] = None
        for item in inputs:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise PlanValidationError(f"{where}: each required input needs a name")
            name = item["name"].strip()
            if name not in INPUT_WHITELIST[tool]:
                raise PlanValidationError(
                    f"{where}: the planner may not set {name!r} for {tool} "
                    f"(allowed: {', '.join(INPUT_WHITELIST[tool])})")
            source = item.get("from_step") or 0
            if name == "bars":
                if isinstance(source, bool) or not isinstance(source, int) or source <= 0:
                    raise PlanValidationError(
                        f"{where}: the planner may not supply market data (bars); "
                        "reference the market_data step with from_step")
                bars_source = index_of.get(source)
                if bars_source is None or bars_source.tool != "market_data":
                    raise PlanValidationError(f"{where}: bars must come from an earlier market_data step")
                args["bars"] = BARS_REF
                normalized.append({"name": "bars", "from_step": source})
                continue
            if source:
                raise PlanValidationError(f"{where}: only bars may reference another step")
            args[name] = _parse_input(name, item.get("value"), where, facts)
            shown = args[name]
            normalized.append({"name": name, "value": ",".join(shown) if isinstance(shown, list) else shown})

        if tool == "market_data":
            args.setdefault("days", facts.days or DEFAULT_DAYS)
        missing = [n for n in REQUIRED_INPUTS[tool] if n not in args]
        if missing:
            raise PlanValidationError(f"{where}: {tool} is missing required inputs {missing}")
        if bars_source is not None and bars_source.arguments["symbol"] != args["symbol"]:
            raise PlanValidationError(
                f"{where}: bars come from market_data for {bars_source.arguments['symbol']}, "
                f"not {args['symbol']}")
        if facts.symbols and "symbol" in args and args["symbol"] not in facts.symbols:
            raise PlanValidationError(
                f"{where}: the goal names {list(facts.symbols)}; the planner may not switch to {args['symbol']}")

        tool_obj = tools.get(tool) if tools is not None else None
        if tool_obj is not None:
            probe = dict(args)
            if probe.get("bars") == BARS_REF:
                probe["bars"] = _DUMMY_BARS
            try:
                tool_obj.Input.model_validate(probe)
            except ValidationError as exc:
                raise PlanValidationError(f"{where} ({tool}): {exc.errors()[0].get('msg')}") from exc
        step = make_step(tool, purpose, normalized, **args)
        steps.append(step)
        index_of[i] = step

    if not steps:
        raise PlanValidationError("plan has no research step")
    step_intent, symbol = _derive_subject(steps)
    # The goal type is a fact of the user's text, not of the steps the model chose: the
    # model may add analysis (a backtest, the portfolio, a symbol's data), and those steps
    # run as optional research, but they never turn the goal into another kind of goal.
    intent = facts.expected_intent or step_intent
    if intent != step_intent:
        notes.append(f"goal type {intent} kept from the goal text; the added steps are optional research")
    if symbol is None and intent != "portfolio_risk":
        raise PlanValidationError("plan reads no market data")
    if not facts.symbols and symbol:
        notes.append(f"symbol {symbol} inferred by the planner from the goal text")

    def has(tool: str) -> bool:
        return any(s.tool == tool and (tool == "portfolio" or s.arguments.get("symbol") == symbol)
                   for s in steps)

    required = set(REQUIRED_STEPS[step_intent]) | set(REQUIRED_STEPS[intent])
    if facts.expected_intent:
        required |= set(REQUIRED_STEPS[facts.expected_intent])
    if facts.portfolio_id is None:
        required.discard("portfolio")
    for tool in sorted(required):
        if not has(tool):
            raise PlanValidationError(
                f"plan is missing the required step {tool}"
                + (f" for {symbol}" if tool != "portfolio" and symbol else ""))

    constraints = _check_constraints(raw.get("constraints"), memory, notes)
    summary = str(raw.get("reasoning_summary") or "").strip()[:600]
    if not summary:
        raise PlanValidationError("plan needs a reasoning_summary")
    has_portfolio = any(s.tool == "portfolio" for s in steps)
    return Plan(goal=facts.text, reasoning_summary=summary, intent=intent, symbol=symbol,
                portfolio_id=facts.portfolio_id if has_portfolio else None,
                steps=steps, constraints=constraints, planner=planner, model=model,
                trade=facts.trade, direction=facts.direction,
                direct_execution_requested=facts.direct_execution_requested, notes=notes)


# ---------------------------------------------------------------------- rule planner

def _inp(name: str, value: Any = None, from_step: int = 0) -> Dict[str, Any]:
    return {"name": name, "value": "", "from_step": from_step} if from_step else \
        {"name": name, "value": str(value), "from_step": 0}


def rule_plan(facts: GoalFacts, memory: Any = None, strategies: Optional[List[str]] = None,
              days: Optional[int] = None, extra_factors: Optional[List[str]] = None) -> Dict[str, Any]:
    """The same plan schema, built by rules for the deterministic goal type."""
    intent = facts.expected_intent
    s = facts.symbols[0] if facts.symbols else None
    p = facts.portfolio_id
    days = days or facts.days or DEFAULT_DAYS
    steps: List[Dict[str, Any]] = []
    if intent in ("position_review", "portfolio_risk"):
        steps.append({"tool": "portfolio", "purpose": "获取组合持仓与当前暴露",
                      "required_inputs": [_inp("portfolio_id", p)]})
    md_index = 0
    if intent in ("analyze_symbol", "position_review", "strategy_suggestion"):
        steps.append({"tool": "market_data", "purpose": f"获取 {s} 近 {days} 天日线",
                      "required_inputs": [_inp("symbol", s), _inp("days", days)]})
        md_index = len(steps)
    if intent in ("analyze_symbol", "position_review"):
        steps.append({"tool": "signal", "purpose": f"计算 {s} 的策略信号",
                      "required_inputs": [_inp("symbol", s), _inp("bars", from_step=md_index)]})
    factors = list(DEFAULT_FACTORS) if intent == "analyze_symbol" else []
    for f in extra_factors or []:
        if f not in factors:
            factors.append(f)
    if factors and md_index:
        steps.append({"tool": "factor", "purpose": f"计算 {s} 的因子暴露",
                      "required_inputs": [_inp("symbol", s), _inp("bars", from_step=md_index),
                                          _inp("factors", ",".join(factors))]})
    if intent == "strategy_suggestion":
        for name in strategies or DEFAULT_CANDIDATES:
            steps.append({"tool": "backtest", "purpose": f"模拟候选策略 {name}",
                          "required_inputs": [_inp("symbol", s), _inp("bars", from_step=md_index),
                                              _inp("strategy", name)]})
    prefs = getattr(memory, "preferences", {}) or {}
    return {"goal": facts.text,
            "reasoning_summary": f"规则规划：目标类型 {intent}，按固定步骤读取数据并计算。",
            "steps": steps,
            "constraints": [{"key": k, "value": str(v["value"]), "source": f"memory #{v['memory_id']}"}
                            for k, v in prefs.items()],
            "clarification_question": ""}


# ---------------------------------------------------------------------- LLM planner

_INPUT_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["name", "value", "from_step"],
                 "properties": {"name": {"type": "string"}, "value": {"type": "string"},
                                "from_step": {"type": "integer"}}}
PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["goal", "reasoning_summary", "steps", "constraints", "clarification_question"],
    "properties": {
        "goal": {"type": "string"},
        "reasoning_summary": {"type": "string"},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["tool", "purpose", "required_inputs"],
            "properties": {"tool": {"type": "string", "enum": list(RESEARCH_TOOLS)},
                           "purpose": {"type": "string"},
                           "required_inputs": {"type": "array", "items": _INPUT_SCHEMA}}}},
        "constraints": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["key", "value", "source"],
            "properties": {"key": {"type": "string"}, "value": {"type": "string"},
                           "source": {"type": "string"}}}},
        "clarification_question": {"type": "string"},
    },
}

PLANNER_SYSTEM = """You are the planner of TraderX, a quantitative decision product.
Turn the user's goal into research steps. You only plan: you never answer the
question, state prices, recommend trades, or decide whether to buy or sell.

Rules:
- Tools: market_data, portfolio, signal, factor, backtest. Never plan execution or
  risk: a separate Risk Agent assesses risk, and execution only happens after a
  human approves a proposal.
- required_inputs: for each input give name, value and from_step. Plain inputs
  have a value and from_step 0. bars always have from_step set to the 1-based
  index of an earlier market_data step for the same symbol and an empty value.
  Never put prices, closes or bars as values.
- Allowed inputs: market_data(symbol, days) portfolio(portfolio_id)
  signal(symbol, bars, strategies) factor(symbol, bars, factors)
  backtest(symbol, bars, strategy). Do not set anything else.
- strategies, strategy and factors take names from the tool's allowed_values only
  (comma-separated for lists). Never use placeholders such as "all" or
  "registered"; omit strategies / factors to use the tool's default set.
- Only read the portfolio given in the context, and only the symbols the user named.
- constraints: restate the user's constraints exactly as given (key, value, source).
- If the goal is unclear, set clarification_question; otherwise leave it empty.
- reasoning_summary: two or three sentences on why these steps answer the goal."""


def _catalog(tools: Any) -> List[Dict[str, Any]]:
    out = []
    for name in RESEARCH_TOOLS:
        tool = tools.get(name) if tools is not None else None
        if tool is not None:
            entry = {"name": name, "description": tool.description,
                     "inputs": list(INPUT_WHITELIST[name])}
            named = {i: allowed_names(i) for i in INPUT_WHITELIST[name] if i in NAMED_INPUTS}
            if named:
                entry["allowed_values"] = named
            out.append(entry)
    return out


def _memory_lines(memory: Any) -> List[str]:
    lines = getattr(memory, "lines", None)
    return lines() if callable(lines) else []


class PlanningService:
    def __init__(self, tools: Any, llm: Optional[StructuredLLM] = None, llm_attempts: int = 2):
        self.tools = tools
        self.llm = llm
        self.llm_attempts = llm_attempts

    def llm_status(self) -> Tuple[bool, Optional[str]]:
        return llm_status(self.llm)

    def _prompt(self, facts: GoalFacts, memory: Any, **extra: Any) -> str:
        return json.dumps({
            "goal": facts.text,
            "context": {"portfolio_id": facts.portfolio_id, "symbols_named": list(facts.symbols)},
            "relevant_memories": _memory_lines(memory),
            "user_constraints": [{"key": k, "value": str(v["value"]), "source": f"memory #{v['memory_id']}"}
                                 for k, v in (getattr(memory, "preferences", {}) or {}).items()],
            "tools": _catalog(self.tools),
            **extra}, ensure_ascii=False, default=str)

    def _llm_attempt(self, n: int, outcome: str, raw: Optional[Dict[str, Any]] = None,
                     validation: Optional[Dict[str, Any]] = None,
                     detail: Optional[str] = None) -> Dict[str, Any]:
        """One LLM planner attempt as it goes into the trace: which model, what it
        returned, what the validator said."""
        call = dict(getattr(self.llm, "last_call", {}) or {})
        return {"planner": "llm", "attempt": n, "model": getattr(self.llm, "model", None),
                "provider": getattr(self.llm, "provider", None), "endpoint": getattr(self.llm, "endpoint", None),
                "served_model": call.get("model"), "outcome": outcome, "planner_output": raw,
                "validation": validation, "call": call or None, "detail": detail,
                "fallback": False}

    async def plan(self, goal_text: str, memory: Any = None, symbol_hint: Optional[str] = None,
                   portfolio_hint: Optional[int] = None,
                   strategies: Optional[List[str]] = None) -> Tuple[Union[Plan, Clarification], List[Dict[str, Any]]]:
        """Plan (or clarification) plus a log of planner attempts for the trace.

        Each attempt records the model, the planner's raw output, the validator's
        verdict and whether the rule planner had to take over (and why).
        """
        facts = goal_facts(goal_text, symbol_hint, portfolio_hint)
        attempts: List[Dict[str, Any]] = []
        ok, reason = self.llm_status()
        if ok:
            feedback = None
            for n in range(1, self.llm_attempts + 1):
                prompt = self._prompt(facts, memory, previous_plan_rejected_because=feedback)
                try:
                    raw = await self.llm.generate(PLANNER_SYSTEM, prompt, "plan", PLAN_SCHEMA)
                except LLMError as exc:
                    attempts.append(self._llm_attempt(n, "model_error", detail=str(exc)))
                    reason = f"model error: {exc}"
                    break
                try:
                    plan = validate_plan(raw, self.tools, "llm", facts, memory, model=self.llm.model)
                except PlanValidationError as exc:
                    feedback = str(exc)
                    attempts.append(self._llm_attempt(
                        n, "rejected", raw, {"valid": False, "errors": [feedback], "adjustments": []},
                        feedback))
                    reason = f"LLM plan rejected by the validator: {feedback}"
                    continue
                if isinstance(plan, Clarification):
                    attempts.append(self._llm_attempt(
                        n, "clarification", raw, {"valid": True, "errors": [], "adjustments": []},
                        plan.question))
                    return plan, attempts
                attempts.append(self._llm_attempt(
                    n, "accepted", raw, {"valid": True, "errors": [], "adjustments": list(plan.notes)}))
                return plan, attempts
        else:
            attempts.append({"planner": "llm", "attempt": 0, "model": None, "outcome": "unavailable",
                             "planner_output": None, "validation": None, "detail": reason,
                             "fallback": True})

        if facts.question is not None:
            attempts.append({"planner": "rule", "outcome": "clarification", "detail": facts.question.question,
                             "fallback": True, "fallback_reason": reason})
            return facts.question, attempts
        raw = rule_plan(facts, memory, strategies)
        plan = validate_plan(raw, self.tools, "rule", facts, memory)
        plan.fallback_reason = reason
        attempts.append({"planner": "rule", "attempt": 1, "model": None, "outcome": "accepted",
                         "planner_output": raw,
                         "validation": {"valid": True, "errors": [], "adjustments": list(plan.notes)},
                         "fallback": True, "fallback_reason": reason})
        return plan, attempts

    async def revise(self, plan: Plan, issue: str, remedy: Dict[str, Any],
                     observations: List[Dict[str, Any]],
                     memory: Any = None) -> Tuple[Optional[Plan], Dict[str, Any]]:
        """A complete revised plan for an issue raised during the run (e.g. by the Critic).

        The goal does not change on revision: symbol, portfolio and trade stay fixed.
        """
        facts = replace(goal_facts(plan.goal, plan.symbol, plan.portfolio_id),
                        expected_intent=plan.intent, question=None,
                        symbols=(plan.symbol,) if plan.symbol else ())
        ok, reason = self.llm_status()
        if ok and plan.planner == "llm":
            prompt = self._prompt(facts, memory, current_plan=plan.to_dict(), issue=issue,
                                  suggested_remedy=remedy, observations=observations,
                                  instruction="Return the complete revised plan (all steps) "
                                              "that addresses the issue.")
            raw_revision: Optional[Dict[str, Any]] = None
            try:
                raw_revision = await self.llm.generate(PLANNER_SYSTEM, prompt, "plan", PLAN_SCHEMA)
                revised = validate_plan(raw_revision, self.tools, "llm", facts, memory, model=self.llm.model)
                if isinstance(revised, Plan):
                    revised.revision = plan.revision + 1
                    revised.notes.append(f"revision {revised.revision}: {issue}")
                    return revised, {"planner": "llm", "outcome": "accepted", "model": self.llm.model,
                                     "planner_output": raw_revision,
                                     "validation": {"valid": True, "errors": [],
                                                    "adjustments": list(revised.notes)}}
                reason = "LLM asked a question instead of revising the plan"
            except (LLMError, PlanValidationError) as exc:
                reason = f"LLM revision failed: {exc}"
            llm_revision = {"model": self.llm.model, "planner_output": raw_revision,
                            "validation": None if raw_revision is None else
                            {"valid": False, "errors": [reason], "adjustments": []}}
        else:
            llm_revision = {}

        days, factors = remedy.get("days"), remedy.get("factors")
        if not days and not factors:
            return None, {"planner": "rule", "outcome": "no_remedy", "fallback_reason": reason,
                          "llm_attempt": llm_revision or None}
        current_days = next((s.arguments.get("days") for s in plan.steps
                             if s.tool == "market_data" and s.arguments.get("symbol") == plan.symbol),
                            DEFAULT_DAYS)
        strategies = [s.arguments["strategy"] for s in plan.steps if s.tool == "backtest"] or None
        raw = rule_plan(facts, memory, strategies, days=days or current_days, extra_factors=factors)
        if days:
            for st in raw["steps"]:
                if st["tool"] == "market_data":
                    st["purpose"] = f"扩大历史窗口至 {days} 天（{issue[:80]}）"
        revised = validate_plan(raw, self.tools, "rule", facts, memory)
        revised.revision = plan.revision + 1
        revised.fallback_reason = reason if plan.planner == "llm" else plan.fallback_reason
        revised.notes.append(f"revision {revised.revision}: {issue}")
        return revised, {"planner": "rule", "outcome": "accepted", "fallback_reason": reason,
                         "planner_output": raw,
                         "validation": {"valid": True, "errors": [], "adjustments": list(revised.notes)},
                         "llm_attempt": llm_revision or None}
