# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Decision Agent - AI-powered trading decisions with multi-model debate.
"""

from typing import Any, Dict, List, Optional
import logging

from ..base import BaseAgent, AgentTask, AgentResult, AgentStatus
from ..registry import register_agent
from ..debate.engine import DebateEngine, DebateTier
from .mock_guard import require_real_data

logger = logging.getLogger(__name__)


@register_agent("decision", task_types=["portfolio_decision", "strategy_selection", "risk_decision", "trade_decision"])
class DecisionAgent(BaseAgent):
    """
    Agent for AI-powered trading decisions.

    Integrates multi-model debate for important decisions.

    Task Payload:
        type: str - Decision type
        context: Dict - Decision context
        portfolio_id: str - Portfolio ID (optional)
        force_debate: bool - Force multi-model debate
        debate_tier: int - Override debate tier (0, 1, 2)
    """

    SUPPORTED_TASKS = {"portfolio_decision", "strategy_selection", "risk_decision", "trade_decision"}

    # Decision importance mapping
    DECISION_IMPORTANCE = {
        "portfolio_decision": DebateTier.TIER_2,
        "strategy_selection": DebateTier.TIER_1,
        "risk_decision": DebateTier.TIER_2,
        "trade_decision": DebateTier.TIER_1,
    }

    def __init__(
        self,
        agent_id: Optional[str] = None,
        debate_engine: Optional[DebateEngine] = None
    ):
        super().__init__(agent_id)
        self.debate_engine = debate_engine

    def can_handle(self, task_type: str) -> bool:
        return task_type in self.SUPPORTED_TASKS

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute decision task."""
        self.status = AgentStatus.RUNNING

        try:
            decision_type = task.payload.get("type", "portfolio_decision")
            context = task.payload.get("context", {})
            force_debate = task.payload.get("force_debate", False)

            # Determine debate tier
            tier = self._determine_tier(task)

            # Build decision question
            question = self._build_question(decision_type, context)

            # Execute debate or single-model decision
            if self.debate_engine and (force_debate or tier != DebateTier.TIER_0):
                decision_result = await self._debate_decision(question, context, tier)
            else:
                decision_result = await self._simple_decision(question, context)

            self.status = AgentStatus.COMPLETED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.COMPLETED,
                result={
                    "decision_type": decision_type,
                    "question": question,
                    "decision": decision_result["decision"],
                    "confidence": decision_result.get("confidence", 0.8),
                    "reasoning": decision_result.get("reasoning", ""),
                    "debate_used": decision_result.get("debate_used", False),
                    "debate_tier": tier.value if isinstance(tier, DebateTier) else tier,
                    "debate_result": decision_result.get("debate_result"),  # 完整辩论结果
                    "dissent": decision_result.get("dissent", []),
                    "actions": decision_result.get("actions", [])
                }
            )

        except Exception as e:
            logger.error(f"Decision failed: {e}")
            self.status = AgentStatus.FAILED
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                error=str(e)
            )

    def _determine_tier(self, task: AgentTask) -> DebateTier:
        """Determine appropriate debate tier."""
        # Check for explicit override
        if "debate_tier" in task.payload:
            return DebateTier(task.payload["debate_tier"])

        decision_type = task.payload.get("type", "portfolio_decision")
        context = task.payload.get("context", {})

        # Check default tier for decision type
        default_tier = self.DECISION_IMPORTANCE.get(decision_type, DebateTier.TIER_1)

        # Escalate based on context
        if context.get("trade_value", 0) > 100000:
            return DebateTier.TIER_2

        if context.get("risk_level", "").lower() == "high":
            return DebateTier.TIER_2

        if context.get("urgency", "").lower() == "low":
            # More time, use higher tier
            return max(default_tier, DebateTier.TIER_1)

        return default_tier

    def _build_question(self, decision_type: str, context: Dict) -> str:
        """Build decision question from type and context."""
        templates = {
            "portfolio_decision": "Given the current portfolio state and market conditions, what actions should be taken? Context: {context}",
            "strategy_selection": "Which trading strategy is most appropriate for {symbol} given current conditions? Context: {context}",
            "risk_decision": "How should portfolio risk be managed given current exposure and market volatility? Context: {context}",
            "trade_decision": "Should we execute a {direction} trade for {symbol}? Context: {context}",
        }

        template = templates.get(decision_type, "What is the recommended action? Context: {context}")

        # Format context for readability
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())

        return template.format(
            context=context_str,
            symbol=context.get("symbol", "unknown"),
            direction=context.get("direction", "unknown")
        )

    async def _debate_decision(
        self,
        question: str,
        context: Dict,
        tier: DebateTier
    ) -> Dict[str, Any]:
        """Make decision using multi-model debate."""
        debate_result = await self.debate_engine.run_debate(
            question=question,
            context=context,
            tier=tier
        )

        # A participant without credentials is swapped for MockAdapter, whose answer
        # is canned placeholder text. A debate built from those must not come back
        # as a decision.
        placeholder_models = sorted({
            response.model
            for debate_round in debate_result.rounds
            for response in debate_round.responses
            if (response.metadata or {}).get("mock")
        })
        if placeholder_models:
            require_real_data(
                "multi-model debate decision",
                f"{', '.join(placeholder_models)} answered with placeholder text instead "
                f"of a model; configure credentials for every debate participant.",
            )

        # 将 DebateResult 转换为可序列化的字典
        debate_result_dict = {
            "question": debate_result.question,
            "tier": debate_result.tier.value if hasattr(debate_result.tier, 'value') else debate_result.tier,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "round_type": r.round_type,
                    "responses": [
                        {
                            "model": resp.model,
                            "role": resp.role.value if hasattr(resp.role, 'value') else resp.role,
                            "response": resp.response,
                            "confidence": resp.confidence,
                            "timestamp": resp.timestamp,
                            "tokens_used": resp.tokens_used,
                            "latency_ms": resp.latency_ms,
                            "metadata": resp.metadata
                        }
                        for resp in r.responses
                    ],
                    "duration_ms": r.duration_ms,
                    "timestamp": r.timestamp
                }
                for r in debate_result.rounds
            ],
            "consensus": debate_result.consensus,
            "dissent": debate_result.dissent,
            "final_decision": debate_result.final_decision,
            "metadata": debate_result.metadata,
            "total_tokens": debate_result.total_tokens,
            "total_duration_ms": debate_result.total_duration_ms,
            "models_used": debate_result.models_used
        }

        return {
            "decision": debate_result.final_decision,
            "confidence": 0.9 if tier == DebateTier.TIER_2 else 0.8,
            "reasoning": debate_result.consensus,
            "debate_used": True,
            "debate_result": debate_result_dict,
            "dissent": debate_result.dissent,
            "actions": self._extract_actions(debate_result.final_decision)
        }

    async def _simple_decision(
        self,
        question: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Make simple single-model decision."""
        # In production, would call AI model
        # For now, return rule-based decision

        decision = self._rule_based_decision(context)

        return {
            "decision": decision["action"],
            "confidence": decision["confidence"],
            "reasoning": decision["reasoning"],
            "debate_used": False,
            "dissent": [],
            "actions": [decision["action"]]
        }

    def _rule_based_decision(self, context: Dict) -> Dict[str, Any]:
        """Make simple rule-based decision."""
        # Simple rules for demonstration
        risk_level = context.get("risk_level", "medium").lower()
        market_trend = context.get("market_trend", "neutral").lower()
        portfolio_beta = context.get("portfolio_beta", 1.0)

        if risk_level == "high":
            return {
                "action": "REDUCE_EXPOSURE",
                "confidence": 0.85,
                "reasoning": "High risk level detected, recommend reducing exposure"
            }

        if market_trend == "bearish" and portfolio_beta > 1.2:
            return {
                "action": "HEDGE",
                "confidence": 0.75,
                "reasoning": "Bearish trend with high beta, recommend hedging"
            }

        if market_trend == "bullish":
            return {
                "action": "HOLD_OR_ADD",
                "confidence": 0.7,
                "reasoning": "Bullish trend, maintain or increase positions"
            }

        return {
            "action": "HOLD",
            "confidence": 0.6,
            "reasoning": "Neutral conditions, maintain current positions"
        }

    def _extract_actions(self, decision_text: str) -> List[str]:
        """Extract actionable items from decision text."""
        actions = []

        # Simple keyword extraction
        action_keywords = {
            "buy": "BUY",
            "sell": "SELL",
            "hold": "HOLD",
            "hedge": "HEDGE",
            "reduce": "REDUCE_EXPOSURE",
            "increase": "INCREASE_EXPOSURE",
            "rebalance": "REBALANCE",
        }

        decision_lower = decision_text.lower()
        for keyword, action in action_keywords.items():
            if keyword in decision_lower:
                actions.append(action)

        return actions if actions else ["REVIEW"]
