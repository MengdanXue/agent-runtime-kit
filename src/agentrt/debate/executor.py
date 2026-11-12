# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Model Debate Executor

Provides high-level interface for running multi-model debates with
parallel agent execution and consensus building.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import asyncio
import logging
import time
from datetime import datetime

from .engine import DebateEngine, DebateTier, ModelRole, ModelResponse, DebateResult
from .adapters import (
    ModelAdapter,
    ClaudeAdapter,
    GeminiAdapter,
    CodexAdapter,
    GeminiCLIAdapter,
    CodexCLIAdapter,
    MockAdapter,
    create_adapter_set,
)
from .consensus import ConsensusBuilder, VotingStrategy

logger = logging.getLogger(__name__)


class DebateMode(Enum):
    """Debate execution modes."""
    SEQUENTIAL = "sequential"  # Models respond one after another
    PARALLEL = "parallel"      # All models respond simultaneously
    HYBRID = "hybrid"          # Parallel proposals, sequential cross-exam


@dataclass
class DebateConfig:
    """Configuration for debate execution."""
    mode: DebateMode = DebateMode.PARALLEL
    tier: DebateTier = DebateTier.TIER_2
    max_rounds: int = 3
    timeout_per_model: int = 120  # seconds
    voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED
    use_cli: bool = True  # Prefer CLI tools over APIs
    include_dissent: bool = True
    verbose: bool = False


@dataclass
class ModelPerspective:
    """A model's perspective in the debate."""
    model: str
    role: ModelRole
    response: str
    confidence: float
    latency_ms: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateExecutionResult:
    """Result of debate execution."""
    question: str
    config: DebateConfig
    perspectives: List[ModelPerspective]
    rounds: List[Dict[str, Any]]
    consensus: str
    dissent: List[str]
    final_decision: str
    total_latency_ms: int
    models_used: List[str]
    started_at: datetime
    completed_at: datetime
    success: bool
    error: Optional[str] = None


class DebateExecutor:
    """
    High-level executor for multi-model debates.

    Orchestrates the debate process with support for parallel execution,
    CLI tools, and consensus building.
    """

    ARCHITECT_PROMPT = """You are the LEAD ARCHITECT in a multi-model debate.

## Question
{question}

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
| Option | Pros | Cons |
|--------|------|------|
| ... | ... | ... |

### Risks
1. [Risk 1]: [Mitigation]
2. [Risk 2]: [Mitigation]

### Open Questions
1. [Question that needs discussion]"""

    CHALLENGER_PROMPT = """You are the CRITICAL CHALLENGER in a multi-model debate.

## Question
{question}

## Your Role
- Challenge assumptions
- Propose alternatives
- Find edge cases
- Ask hard questions

## Output Format
### Alternative Approaches
1. [Alternative 1]: [Why it might be better]
2. [Alternative 2]: [Why it might be better]

### Challenges to Obvious Solutions
1. [Challenge 1]: [Why this is a concern]
2. [Challenge 2]: [Why this is a concern]

### Edge Cases
1. [Edge case that might break the design]

### Critical Questions
1. [Question that must be answered]"""

    VALIDATOR_PROMPT = """You are the IMPLEMENTATION VALIDATOR in a multi-model debate.

## Question
{question}

## Your Role
- Assess implementation feasibility
- Identify technical blockers
- Suggest code architecture
- Recommend testing approach

## Output Format
### Feasibility Assessment
[Easy/Medium/Hard] - [Explanation]

### Technical Considerations
1. [Consideration 1]
2. [Consideration 2]

### Recommended Stack
- [Technology]: [Reason]
- [Technology]: [Reason]

### Implementation Risks
1. [Risk]: [How to mitigate]

### Testing Strategy
[How to verify this works]"""

    CROSS_EXAM_PROMPT = """Review these perspectives and provide your critique:

## Your Original Response
{own_response}

## Other Perspectives
{other_responses}

## Your Task
1. What valid points do the other models raise?
2. Where do you disagree and why?
3. How would you modify your proposal based on this feedback?
4. What remains unresolved?"""

    SYNTHESIS_PROMPT = """Synthesize the debate into a final decision:

## Original Question
{question}

## Architect's Position
{architect_response}

## Challenger's Position
{challenger_response}

## Validator's Position
{validator_response}

## Cross-Examination Results
{cross_exam_results}

## Your Task
Provide:
1. **Points of Agreement** - What all models agree on
2. **Points of Disagreement** - Where views differ and why
3. **Final Decision** - The synthesized recommendation
4. **Dissenting Opinions** - Valid concerns that weren't fully addressed
5. **Next Steps** - Concrete actions to take"""

    def __init__(self, config: Optional[DebateConfig] = None):
        """
        Initialize debate executor.

        Args:
            config: Debate configuration
        """
        self.config = config or DebateConfig()
        self.adapters = create_adapter_set(use_cli=self.config.use_cli)
        self.consensus_builder = ConsensusBuilder(
            voting_strategy=self.config.voting_strategy
        )

    async def execute(self, question: str, context: Optional[Dict] = None) -> DebateExecutionResult:
        """
        Execute a full multi-model debate.

        Args:
            question: The question to debate
            context: Optional additional context

        Returns:
            DebateExecutionResult with full debate output
        """
        started_at = datetime.utcnow()
        start_time = time.time()

        perspectives = []
        rounds = []
        models_used = []

        try:
            if self.config.verbose:
                logger.info(f"🎭 Starting Multi-Model Debate")
                logger.info(f"📋 Question: {question}")
                logger.info(f"⚙️ Mode: {self.config.mode.value}")
                logger.info(f"🎚️ Tier: {self.config.tier.value}")

            # Round 1: Initial Proposals
            round1_results = await self._round1_proposals(question, context)
            rounds.append({"round": 1, "name": "Initial Proposals", "responses": round1_results})
            perspectives.extend(round1_results)

            for p in round1_results:
                if p.model not in models_used:
                    models_used.append(p.model)

            # Round 2: Cross-Examination (if tier >= 1)
            if self.config.tier.value >= DebateTier.TIER_1.value:
                round2_results = await self._round2_cross_exam(question, round1_results)
                rounds.append({"round": 2, "name": "Cross-Examination", "responses": round2_results})
                perspectives.extend(round2_results)

            # Round 3: Synthesis (if tier == 2)
            if self.config.tier == DebateTier.TIER_2:
                synthesis = await self._round3_synthesis(question, round1_results, rounds)
                rounds.append({"round": 3, "name": "Synthesis", "result": synthesis})

            # Build consensus
            consensus_result = self.consensus_builder.build(
                [p.response for p in round1_results],
                weights={p.model: p.confidence for p in round1_results}
            )

            total_latency = int((time.time() - start_time) * 1000)

            return DebateExecutionResult(
                question=question,
                config=self.config,
                perspectives=perspectives,
                rounds=rounds,
                consensus=consensus_result.get("consensus", ""),
                dissent=consensus_result.get("dissent", []),
                final_decision=synthesis if self.config.tier == DebateTier.TIER_2 else consensus_result.get("consensus", ""),
                total_latency_ms=total_latency,
                models_used=models_used,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                success=True
            )

        except Exception as e:
            logger.error(f"Debate execution failed: {e}")
            return DebateExecutionResult(
                question=question,
                config=self.config,
                perspectives=perspectives,
                rounds=rounds,
                consensus="",
                dissent=[],
                final_decision="",
                total_latency_ms=int((time.time() - start_time) * 1000),
                models_used=models_used,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                success=False,
                error=str(e)
            )

    async def _round1_proposals(
        self,
        question: str,
        context: Optional[Dict]
    ) -> List[ModelPerspective]:
        """Round 1: Get initial proposals from all models."""
        results = []

        if self.config.mode == DebateMode.PARALLEL:
            # Query all models in parallel
            tasks = [
                self._query_model("architect", self.ARCHITECT_PROMPT.format(question=question)),
                self._query_model("challenger", self.CHALLENGER_PROMPT.format(question=question)),
                self._query_model("validator", self.VALIDATOR_PROMPT.format(question=question)),
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            for resp in responses:
                if isinstance(resp, ModelPerspective):
                    results.append(resp)
                elif isinstance(resp, Exception):
                    logger.error(f"Model query failed: {resp}")

        else:
            # Sequential execution
            for role, prompt_template in [
                ("architect", self.ARCHITECT_PROMPT),
                ("challenger", self.CHALLENGER_PROMPT),
                ("validator", self.VALIDATOR_PROMPT),
            ]:
                resp = await self._query_model(role, prompt_template.format(question=question))
                if resp:
                    results.append(resp)

        return results

    async def _round2_cross_exam(
        self,
        question: str,
        round1_results: List[ModelPerspective]
    ) -> List[ModelPerspective]:
        """Round 2: Cross-examination between models."""
        results = []

        # Each model reviews the others
        for perspective in round1_results:
            other_responses = "\n\n".join([
                f"### {p.model.upper()} ({p.role.value}):\n{p.response}"
                for p in round1_results if p.model != perspective.model
            ])

            prompt = self.CROSS_EXAM_PROMPT.format(
                own_response=perspective.response,
                other_responses=other_responses
            )

            role_map = {
                "claude": "architect",
                "gemini": "challenger",
                "codex": "validator"
            }
            role = role_map.get(perspective.model, "architect")

            resp = await self._query_model(role, prompt)
            if resp:
                resp.role = ModelRole.REVIEWER
                results.append(resp)

        return results

    async def _round3_synthesis(
        self,
        question: str,
        round1_results: List[ModelPerspective],
        rounds: List[Dict]
    ) -> str:
        """Round 3: Synthesize all perspectives into final decision."""
        # Find responses by model
        architect_resp = next((p.response for p in round1_results if p.role == ModelRole.ARCHITECT), "")
        challenger_resp = next((p.response for p in round1_results if p.role == ModelRole.CHALLENGER), "")
        validator_resp = next((p.response for p in round1_results if p.role == ModelRole.VALIDATOR), "")

        # Get cross-exam results
        cross_exam = ""
        if len(rounds) > 1:
            cross_exam_round = rounds[1].get("responses", [])
            cross_exam = "\n\n".join([
                f"### {p.model} Review:\n{p.response}"
                for p in cross_exam_round
            ])

        prompt = self.SYNTHESIS_PROMPT.format(
            question=question,
            architect_response=architect_resp,
            challenger_response=challenger_resp,
            validator_response=validator_resp,
            cross_exam_results=cross_exam
        )

        # Use architect (Claude) for synthesis
        resp = await self._query_model("architect", prompt)
        return resp.response if resp else ""

    async def _query_model(self, role: str, prompt: str) -> Optional[ModelPerspective]:
        """Query a specific model."""
        adapter = self.adapters.get(role)
        if not adapter:
            logger.warning(f"No adapter found for role: {role}")
            return None

        role_enum = {
            "architect": ModelRole.ARCHITECT,
            "challenger": ModelRole.CHALLENGER,
            "validator": ModelRole.VALIDATOR,
        }.get(role, ModelRole.ARCHITECT)

        try:
            response = await asyncio.wait_for(
                adapter.query(prompt, role_enum),
                timeout=self.config.timeout_per_model
            )

            return ModelPerspective(
                model=response.model,
                role=role_enum,
                response=response.response,
                confidence=response.confidence,
                latency_ms=response.latency_ms,
                metadata=response.metadata
            )

        except asyncio.TimeoutError:
            logger.error(f"Timeout querying {role}")
            return ModelPerspective(
                model=role,
                role=role_enum,
                response=f"[Timeout after {self.config.timeout_per_model}s]",
                confidence=0.0,
                latency_ms=self.config.timeout_per_model * 1000,
                metadata={"error": "timeout"}
            )
        except Exception as e:
            logger.error(f"Error querying {role}: {e}")
            return ModelPerspective(
                model=role,
                role=role_enum,
                response=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=0,
                metadata={"error": str(e)}
            )


async def run_quick_debate(question: str) -> str:
    """
    Run a quick single-round debate.

    Args:
        question: The question to debate

    Returns:
        Summary of perspectives
    """
    config = DebateConfig(
        mode=DebateMode.PARALLEL,
        tier=DebateTier.TIER_1,
        max_rounds=1
    )
    executor = DebateExecutor(config)
    result = await executor.execute(question)

    if result.success:
        summary = f"## Debate Summary\n\n**Question:** {question}\n\n"
        for p in result.perspectives:
            summary += f"### {p.model.upper()} ({p.role.value})\n{p.response}\n\n"
        summary += f"## Consensus\n{result.consensus}"
        return summary
    else:
        return f"Debate failed: {result.error}"


async def run_full_debate(question: str, verbose: bool = True) -> DebateExecutionResult:
    """
    Run a full 3-round debate with all models.

    Args:
        question: The question to debate
        verbose: Print progress to console

    Returns:
        Complete debate result
    """
    config = DebateConfig(
        mode=DebateMode.PARALLEL,
        tier=DebateTier.TIER_2,
        max_rounds=3,
        verbose=verbose
    )
    executor = DebateExecutor(config)
    return await executor.execute(question)
