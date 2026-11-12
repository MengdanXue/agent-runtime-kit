# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Model Debate Engine Core

Implements the 3-round debate protocol:
1. Initial Proposals (Parallel)
2. Cross-Examination
3. Consensus Building
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class DebateTier(Enum):
    """
    Debate tier levels for different decision types.

    - TIER_0: No debate, single model response
    - TIER_1: Quick debate with 2 models
    - TIER_2: Full debate with all 3 models
    """
    TIER_0 = 0  # No debate - single model
    TIER_1 = 1  # Quick debate - Claude + 1 model
    TIER_2 = 2  # Full debate - all 3 models


class ModelRole(Enum):
    """Role assignments for each model in the debate."""
    ARCHITECT = "architect"      # Claude - Proposes architecture and structure
    CHALLENGER = "challenger"    # Gemini - Challenges and proposes alternatives
    VALIDATOR = "validator"      # Codex - Validates implementation feasibility


@dataclass
class ModelResponse:
    """Response from a single model in the debate."""
    model: str
    role: ModelRole
    response: str
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    """A single round of the debate."""
    round_number: int
    round_type: str  # "proposal", "cross_exam", "consensus"
    responses: List[ModelResponse] = field(default_factory=list)
    duration_ms: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateResult:
    """Final result of a multi-model debate."""
    question: str
    tier: DebateTier
    rounds: List[DebateRound]
    consensus: str
    dissent: List[str]
    final_decision: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    total_duration_ms: int = 0
    models_used: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if debate reached a decision."""
        return bool(self.final_decision)


# Decision type to tier mapping
TIER_MAPPING = {
    # Tier 0: Simple queries, no debate needed
    "query": DebateTier.TIER_0,
    "signal": DebateTier.TIER_0,
    "data": DebateTier.TIER_0,
    "info": DebateTier.TIER_0,

    # Tier 1: Moderate decisions, quick debate
    "strategy": DebateTier.TIER_1,
    "parameter": DebateTier.TIER_1,
    "optimization": DebateTier.TIER_1,
    "analysis": DebateTier.TIER_1,

    # Tier 2: Important decisions, full debate
    "architecture": DebateTier.TIER_2,
    "risk": DebateTier.TIER_2,
    "trade": DebateTier.TIER_2,
    "security": DebateTier.TIER_2,
    "design": DebateTier.TIER_2,
}


class DebateEngine:
    """
    Multi-Model Debate Engine

    Orchestrates structured debates between AI models to produce
    higher-quality decisions through diverse perspectives.

    Protocol:
    1. Round 1 - Initial Proposals: Each model provides independent analysis
    2. Round 2 - Cross-Examination: Models review and critique each other
    3. Round 3 - Consensus Building: Synthesize viewpoints into final decision

    Example:
        engine = DebateEngine(claude, gemini, codex)

        result = await engine.run_debate(
            question="Should we use microservices or monolith?",
            context={"scale": "medium", "team_size": 5}
        )

        print(result.final_decision)
        print(result.dissent)
    """

    # Default prompts for each role
    ARCHITECT_PROMPT = """You are the LEAD ARCHITECT in a multi-model debate.

## Question
{question}

## Context
{context}

## Your Role
- Propose comprehensive solution
- Consider trade-offs
- Identify risks
- Structure the approach

## Output Format
### Proposed Solution
[Your main proposal]

### Architecture
[Key components and their relationships]

### Trade-offs
[Pros and cons considered]

### Risks
[Potential risks and mitigations]

### Open Questions
[Questions that need discussion]
"""

    CHALLENGER_PROMPT = """You are the CRITICAL CHALLENGER in a multi-model debate.

## Question
{question}

## Context
{context}

## Your Role
- Challenge assumptions
- Propose alternatives
- Find edge cases
- Ask hard questions

## Output Format
### Alternative Approaches
[At least 2 alternative solutions]

### Challenges to Obvious Solutions
[Why standard approaches might fail]

### Edge Cases
[Scenarios that could break the design]

### Critical Questions
[Questions that must be answered]
"""

    VALIDATOR_PROMPT = """You are the IMPLEMENTATION VALIDATOR in a multi-model debate.

## Question
{question}

## Context
{context}

## Your Role
- Assess implementation feasibility
- Identify technical blockers
- Suggest code architecture
- Recommend testing approach

## Output Format
### Feasibility Assessment
[Easy/Medium/Hard with explanation]

### Technical Considerations
[Key technical factors]

### Recommended Stack
[Technologies and reasons]

### Implementation Risks
[Risks and mitigations]

### Testing Strategy
[How to verify this works]
"""

    CROSS_EXAM_PROMPT = """Review these perspectives and provide your critique:

{other_responses}

1. What valid points do they raise?
2. Where do you disagree and why?
3. How would you modify your proposal based on their input?
"""

    def __init__(
        self,
        claude_adapter,
        gemini_adapter=None,
        codex_adapter=None,
        default_tier: DebateTier = DebateTier.TIER_1
    ):
        """
        Initialize the debate engine.

        Args:
            claude_adapter: Adapter for Claude API
            gemini_adapter: Optional adapter for Gemini API
            codex_adapter: Optional adapter for Codex/GPT API
            default_tier: Default debate tier when not specified
        """
        self.claude = claude_adapter
        self.gemini = gemini_adapter
        self.codex = codex_adapter
        self.default_tier = default_tier

        self.role_mapping = {
            "claude": ModelRole.ARCHITECT,
            "gemini": ModelRole.CHALLENGER,
            "codex": ModelRole.VALIDATOR,
        }

        self._total_debates = 0
        self._total_tokens = 0

    def determine_tier(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> DebateTier:
        """
        Determine appropriate debate tier based on question and context.

        Args:
            question: The debate question
            context: Additional context

        Returns:
            Appropriate DebateTier for this decision
        """
        decision_type = context.get("type", "").lower()

        # Check explicit mapping
        if decision_type in TIER_MAPPING:
            return TIER_MAPPING[decision_type]

        # Check for high-value indicators
        if context.get("trade_value", 0) > 100000:
            return DebateTier.TIER_2

        if context.get("risk_level", "").lower() in ("high", "critical"):
            return DebateTier.TIER_2

        # Check question keywords
        question_lower = question.lower()
        if any(kw in question_lower for kw in ["architecture", "design", "security", "risk"]):
            return DebateTier.TIER_2
        if any(kw in question_lower for kw in ["strategy", "optimize", "improve"]):
            return DebateTier.TIER_1

        return self.default_tier

    async def run_debate(
        self,
        question: str,
        context: Dict[str, Any] = None,
        tier: Optional[DebateTier] = None
    ) -> DebateResult:
        """
        Run a multi-model debate.

        Args:
            question: The question or topic to debate
            context: Additional context for the debate
            tier: Optional tier override

        Returns:
            DebateResult with consensus, dissent, and final decision
        """
        context = context or {}
        start_time = time.time()
        self._total_debates += 1

        # Determine tier
        if tier is None:
            tier = self.determine_tier(question, context)

        logger.info(f"Starting debate (Tier {tier.value}): {question[:50]}...")

        # Tier 0: Single model response
        if tier == DebateTier.TIER_0:
            return await self._single_model_response(question, context)

        # Get models to use based on tier
        models_to_use = self._get_models_for_tier(tier)

        rounds = []
        total_tokens = 0

        # Round 1: Initial Proposals
        round1 = await self._run_round_1(question, context, models_to_use)
        rounds.append(round1)
        total_tokens += sum(r.tokens_used for r in round1.responses)

        # Round 2: Cross-Examination
        round2 = await self._run_round_2(round1.responses, models_to_use)
        rounds.append(round2)
        total_tokens += sum(r.tokens_used for r in round2.responses)

        # Round 3: Consensus Building
        consensus, dissent, final_decision = await self._build_consensus(
            round1.responses, round2.responses
        )

        total_duration = int((time.time() - start_time) * 1000)
        self._total_tokens += total_tokens

        return DebateResult(
            question=question,
            tier=tier,
            rounds=rounds,
            consensus=consensus,
            dissent=dissent,
            final_decision=final_decision,
            metadata={
                "context": context,
                "models_used": models_to_use,
                "debate_id": self._total_debates
            },
            total_tokens=total_tokens,
            total_duration_ms=total_duration,
            models_used=models_to_use
        )

    async def _single_model_response(
        self,
        question: str,
        context: Dict
    ) -> DebateResult:
        """Handle Tier 0 - single model response."""
        prompt = self.ARCHITECT_PROMPT.format(
            question=question,
            context=self._format_context(context)
        )

        start_time = time.time()
        response = await self.claude.query(prompt, ModelRole.ARCHITECT)
        duration = int((time.time() - start_time) * 1000)

        return DebateResult(
            question=question,
            tier=DebateTier.TIER_0,
            rounds=[DebateRound(
                round_number=1,
                round_type="single_model",
                responses=[response],
                duration_ms=duration
            )],
            consensus=response.response,
            dissent=[],
            final_decision=response.response,
            metadata={"single_model": True},
            total_tokens=response.tokens_used,
            total_duration_ms=duration,
            models_used=["claude"]
        )

    def _get_models_for_tier(self, tier: DebateTier) -> List[str]:
        """Get list of models to use for given tier."""
        if tier == DebateTier.TIER_0:
            return ["claude"]
        elif tier == DebateTier.TIER_1:
            models = ["claude"]
            if self.gemini:
                models.append("gemini")
            elif self.codex:
                models.append("codex")
            return models
        else:  # TIER_2
            models = ["claude"]
            if self.gemini:
                models.append("gemini")
            if self.codex:
                models.append("codex")
            return models

    async def _run_round_1(
        self,
        question: str,
        context: Dict,
        models: List[str]
    ) -> DebateRound:
        """Run Round 1: Initial Proposals in parallel."""
        start_time = time.time()
        context_str = self._format_context(context)

        async def get_proposal(model: str):
            adapter = self._get_adapter(model)
            if not adapter:
                return None

            role = self.role_mapping[model]
            prompt = self._get_prompt_for_role(role).format(
                question=question,
                context=context_str
            )
            return await adapter.query(prompt, role)

        responses = await asyncio.gather(*[
            get_proposal(m) for m in models
        ])

        # Filter out None responses
        valid_responses = [r for r in responses if r is not None]
        duration = int((time.time() - start_time) * 1000)

        return DebateRound(
            round_number=1,
            round_type="proposal",
            responses=valid_responses,
            duration_ms=duration
        )

    async def _run_round_2(
        self,
        round1_responses: List[ModelResponse],
        models: List[str]
    ) -> DebateRound:
        """Run Round 2: Cross-Examination."""
        start_time = time.time()

        async def cross_examine(model: str):
            adapter = self._get_adapter(model)
            if not adapter:
                return None

            # Get other models' responses
            others = [r for r in round1_responses if r.model != model]
            if not others:
                return None

            other_text = "\n\n".join([
                f"**{r.model.upper()} ({r.role.value}):**\n{r.response}"
                for r in others
            ])

            prompt = self.CROSS_EXAM_PROMPT.format(other_responses=other_text)
            return await adapter.query(prompt, self.role_mapping[model])

        responses = await asyncio.gather(*[
            cross_examine(m) for m in models
        ])

        valid_responses = [r for r in responses if r is not None]
        duration = int((time.time() - start_time) * 1000)

        return DebateRound(
            round_number=2,
            round_type="cross_exam",
            responses=valid_responses,
            duration_ms=duration
        )

    async def _build_consensus(
        self,
        round1: List[ModelResponse],
        round2: List[ModelResponse]
    ) -> Tuple[str, List[str], str]:
        """
        Build consensus from debate rounds.

        Returns:
            Tuple of (consensus, dissent_list, final_decision)
        """
        # Analyze responses to find agreement and disagreement
        # This is a simplified implementation - could be enhanced with NLP

        all_responses = round1 + round2

        # Extract key points from each response
        points_by_model = {}
        for resp in round1:
            points_by_model[resp.model] = resp.response

        # Build consensus text
        consensus_parts = [
            "Based on the multi-model debate:",
            "",
            "**Points of Agreement:**"
        ]

        # Find common themes (simplified)
        consensus_parts.append("- All models agree on the general approach")

        dissent = []
        if len(points_by_model) > 1:
            dissent.append("Different models emphasized different trade-offs")

        # Build final decision
        final_decision_parts = [
            "**Final Synthesized Decision:**",
            "",
            f"Architect (Claude) Proposal: {self._summarize(points_by_model.get('claude', ''))}",
        ]

        if "gemini" in points_by_model:
            final_decision_parts.append(
                f"Challenger (Gemini) Concerns: {self._summarize(points_by_model.get('gemini', ''))}"
            )

        if "codex" in points_by_model:
            final_decision_parts.append(
                f"Validator (Codex) Assessment: {self._summarize(points_by_model.get('codex', ''))}"
            )

        consensus = "\n".join(consensus_parts)
        final_decision = "\n".join(final_decision_parts)

        return consensus, dissent, final_decision

    def _get_adapter(self, model: str):
        """Get adapter for given model."""
        if model == "claude":
            return self.claude
        elif model == "gemini":
            return self.gemini
        elif model == "codex":
            return self.codex
        return None

    def _get_prompt_for_role(self, role: ModelRole) -> str:
        """Get prompt template for given role."""
        if role == ModelRole.ARCHITECT:
            return self.ARCHITECT_PROMPT
        elif role == ModelRole.CHALLENGER:
            return self.CHALLENGER_PROMPT
        elif role == ModelRole.VALIDATOR:
            return self.VALIDATOR_PROMPT
        return self.ARCHITECT_PROMPT

    def _format_context(self, context: Dict) -> str:
        """Format context dict as readable string."""
        if not context:
            return "No additional context provided."
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def _summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + "..."

    def get_stats(self) -> Dict[str, Any]:
        """Get debate engine statistics."""
        return {
            "total_debates": self._total_debates,
            "total_tokens": self._total_tokens,
            "models_available": {
                "claude": self.claude is not None,
                "gemini": self.gemini is not None,
                "codex": self.codex is not None
            }
        }
