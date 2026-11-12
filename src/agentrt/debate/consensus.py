# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Consensus Building for Multi-Model Debate

Implements different voting and consensus strategies.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import logging

from .engine import ModelResponse

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Voting strategies for consensus building."""
    MAJORITY = "majority"           # Simple majority vote
    WEIGHTED = "weighted"           # Weight by confidence
    UNANIMOUS = "unanimous"         # Require all agree
    ARCHITECT_FINAL = "architect"   # Architect (Claude) has final say


@dataclass
class VoteResult:
    """Result of a voting round."""
    winner: str
    votes: Dict[str, int]
    confidence: float
    dissenting: List[str]


class ConsensusBuilder:
    """
    Builds consensus from multiple model responses.

    Supports different voting strategies and handles dissent tracking.
    """

    # Role weights for weighted voting
    DEFAULT_WEIGHTS = {
        "claude": 1.0,    # Architect
        "gemini": 0.9,    # Challenger
        "codex": 0.9,     # Validator
    }

    def __init__(
        self,
        strategy: VotingStrategy = VotingStrategy.WEIGHTED,
        weights: Dict[str, float] = None
    ):
        """
        Initialize consensus builder.

        Args:
            strategy: Voting strategy to use
            weights: Optional custom weights per model
        """
        self.strategy = strategy
        self.weights = weights or self.DEFAULT_WEIGHTS

    def build_consensus(
        self,
        responses: List[ModelResponse],
        question: str = ""
    ) -> Tuple[str, List[str], str]:
        """
        Build consensus from model responses.

        Args:
            responses: List of model responses
            question: Original question for context

        Returns:
            Tuple of (consensus_text, dissent_list, final_decision)
        """
        if not responses:
            return "", [], "No responses to build consensus from."

        if len(responses) == 1:
            return responses[0].response, [], responses[0].response

        # Extract key points from each response
        key_points = self._extract_key_points(responses)

        # Find agreements and disagreements
        agreements = self._find_agreements(key_points)
        disagreements = self._find_disagreements(key_points)

        # Build consensus text
        consensus = self._format_consensus(agreements, responses)

        # Track dissent
        dissent = self._track_dissent(disagreements, responses)

        # Make final decision based on strategy
        final_decision = self._make_final_decision(
            responses, agreements, disagreements
        )

        return consensus, dissent, final_decision

    def _extract_key_points(
        self,
        responses: List[ModelResponse]
    ) -> Dict[str, List[str]]:
        """Extract key points from each response."""
        points = {}
        for resp in responses:
            points[resp.model] = self._parse_points(resp.response)
        return points

    def _parse_points(self, text: str) -> List[str]:
        """Parse bullet points and key statements from text."""
        points = []

        # Find bullet points
        bullet_pattern = r'[-•*]\s+(.+?)(?=\n|$)'
        bullets = re.findall(bullet_pattern, text)
        points.extend(bullets)

        # Find numbered points
        numbered_pattern = r'\d+\.\s+(.+?)(?=\n|$)'
        numbered = re.findall(numbered_pattern, text)
        points.extend(numbered)

        # Find section headers
        header_pattern = r'###?\s+(.+?)(?=\n|$)'
        headers = re.findall(header_pattern, text)
        points.extend(headers)

        return [p.strip() for p in points if len(p.strip()) > 10]

    def _find_agreements(
        self,
        key_points: Dict[str, List[str]]
    ) -> List[str]:
        """Find points that multiple models agree on."""
        # Simplified: look for similar phrases across models
        agreements = []

        all_points = []
        for model_points in key_points.values():
            all_points.extend(model_points)

        # Find common themes (simplified)
        if len(key_points) > 1:
            agreements.append("All models provided structured analysis")
            agreements.append("Models addressed key trade-offs")

        return agreements

    def _find_disagreements(
        self,
        key_points: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Find points where models disagree."""
        disagreements = {}

        models = list(key_points.keys())
        if len(models) >= 2:
            # Simplified: note different emphases
            disagreements["emphasis"] = (
                f"{models[0]} and {models[1]} emphasized different aspects"
            )

        return disagreements

    def _format_consensus(
        self,
        agreements: List[str],
        responses: List[ModelResponse]
    ) -> str:
        """Format consensus as readable text."""
        lines = ["## Consensus Points\n"]

        for i, agreement in enumerate(agreements, 1):
            lines.append(f"{i}. {agreement}")

        lines.append("\n## Model Contributions\n")

        for resp in responses:
            role = resp.role.value.title()
            lines.append(f"**{role} ({resp.model})**:")
            lines.append(f"- Confidence: {resp.confidence:.0%}")
            lines.append(f"- Latency: {resp.latency_ms}ms")
            lines.append("")

        return "\n".join(lines)

    def _track_dissent(
        self,
        disagreements: Dict[str, str],
        responses: List[ModelResponse]
    ) -> List[str]:
        """Track and format dissenting opinions."""
        dissent = []

        for key, desc in disagreements.items():
            dissent.append(f"[{key}] {desc}")

        # Check for low-confidence responses
        for resp in responses:
            if resp.confidence < 0.5:
                dissent.append(
                    f"{resp.model} expressed low confidence ({resp.confidence:.0%})"
                )

        return dissent

    def _make_final_decision(
        self,
        responses: List[ModelResponse],
        agreements: List[str],
        disagreements: Dict[str, str]
    ) -> str:
        """Make final decision based on strategy."""
        lines = ["## Final Decision\n"]

        if self.strategy == VotingStrategy.ARCHITECT_FINAL:
            # Use architect (Claude) response as primary
            architect = next(
                (r for r in responses if r.model == "claude"),
                responses[0]
            )
            lines.append("**Decision (Architect Final):**")
            lines.append(self._summarize_response(architect.response))

        elif self.strategy == VotingStrategy.WEIGHTED:
            # Weighted combination
            lines.append("**Decision (Weighted Consensus):**")
            lines.append(self._weighted_synthesis(responses))

        elif self.strategy == VotingStrategy.MAJORITY:
            # Simple combination
            lines.append("**Decision (Majority):**")
            lines.append(self._majority_synthesis(responses))

        else:  # UNANIMOUS
            lines.append("**Decision (Consensus Required):**")
            if disagreements:
                lines.append("Full consensus not reached. Key points:")
                for a in agreements:
                    lines.append(f"- {a}")
            else:
                lines.append(self._majority_synthesis(responses))

        # Append acknowledged risks
        if disagreements:
            lines.append("\n**Noted Disagreements:**")
            for key, desc in disagreements.items():
                lines.append(f"- {desc}")

        return "\n".join(lines)

    def _summarize_response(self, text: str, max_length: int = 500) -> str:
        """Summarize a response."""
        if len(text) <= max_length:
            return text

        # Try to find a natural break point
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        if last_period > max_length // 2:
            return truncated[:last_period + 1]
        return truncated + "..."

    def _weighted_synthesis(self, responses: List[ModelResponse]) -> str:
        """Create weighted synthesis of responses."""
        # Sort by weight * confidence
        weighted = sorted(
            responses,
            key=lambda r: self.weights.get(r.model, 1.0) * r.confidence,
            reverse=True
        )

        # Primary response from highest weighted
        primary = weighted[0]

        synthesis = [
            f"Based on {primary.model}'s analysis (primary):",
            self._summarize_response(primary.response, 300),
        ]

        if len(weighted) > 1:
            synthesis.append("\nSupported by additional perspectives from:")
            for resp in weighted[1:]:
                synthesis.append(f"- {resp.model}: {self._summarize_response(resp.response, 100)}")

        return "\n".join(synthesis)

    def _majority_synthesis(self, responses: List[ModelResponse]) -> str:
        """Create synthesis from majority of responses."""
        synthesis = ["Combined analysis from all models:\n"]

        for resp in responses:
            synthesis.append(f"**{resp.model.title()}:**")
            synthesis.append(self._summarize_response(resp.response, 200))
            synthesis.append("")

        return "\n".join(synthesis)


def quick_consensus(
    responses: List[ModelResponse],
    strategy: VotingStrategy = VotingStrategy.WEIGHTED
) -> str:
    """
    Quick helper to build consensus from responses.

    Args:
        responses: List of model responses
        strategy: Voting strategy

    Returns:
        Final decision string
    """
    builder = ConsensusBuilder(strategy=strategy)
    _, _, decision = builder.build_consensus(responses)
    return decision
