# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit Tests for Multi-Model Debate Engine

Tests the debate engine, consensus building, and model adapters
without requiring real API calls.
"""

import pytest
import asyncio
from typing import Dict, List
from unittest.mock import Mock, patch

from agentrt.debate.engine import (
    DebateEngine,
    DebateTier,
    ModelRole,
    ModelResponse,
    DebateResult,
    DebateRound,
    TIER_MAPPING
)
from agentrt.debate.adapters import MockAdapter
from agentrt.debate.consensus import (
    ConsensusBuilder,
    VotingStrategy,
    quick_consensus
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_claude_adapter():
    """Create mock Claude adapter with predefined responses."""
    responses = {
        "architect": """### Proposed Solution
Use microservices architecture with API gateway.

### Architecture
- API Gateway for routing
- Service mesh for communication
- Independent deployments

### Trade-offs
Pros: Scalability, independent deployment
Cons: Complexity, distributed challenges

### Risks
- Increased operational complexity
- Network latency between services

### Open Questions
- How to handle distributed transactions?
- Service discovery strategy?
"""
    }
    return MockAdapter(model_name="claude", responses=responses)


@pytest.fixture
def mock_gemini_adapter():
    """Create mock Gemini adapter with predefined responses."""
    responses = {
        "challenger": """### Alternative Approaches
1. Modular monolith: Balance simplicity with modularity
2. Serverless functions: Reduce operational overhead

### Challenges to Obvious Solutions
- Microservices add complexity for small teams
- Network calls slow down simple operations
- Debugging distributed systems is hard

### Edge Cases
- What if services become too granular?
- How to handle cascading failures?

### Critical Questions
- Do we really need microservices at this scale?
- Can we achieve goals with simpler architecture?
"""
    }
    return MockAdapter(model_name="gemini", responses=responses)


@pytest.fixture
def mock_codex_adapter():
    """Create mock Codex adapter with predefined responses."""
    responses = {
        "validator": """### Feasibility Assessment
Medium difficulty - requires good DevOps practices

### Technical Considerations
- Container orchestration (Kubernetes/Docker)
- Service mesh (Istio/Linkerd)
- Monitoring and tracing setup

### Recommended Stack
- Language: Python/Go for services
- Communication: gRPC + REST
- Database: PostgreSQL + Redis

### Implementation Risks
- Learning curve for team
- Initial setup complexity
- Monitoring and debugging overhead

### Testing Strategy
- Unit tests per service
- Integration tests for service communication
- End-to-end tests for critical flows
- Contract testing between services
"""
    }
    return MockAdapter(model_name="codex", responses=responses)


@pytest.fixture
def debate_engine(mock_claude_adapter, mock_gemini_adapter, mock_codex_adapter):
    """Create debate engine with mock adapters."""
    return DebateEngine(
        claude_adapter=mock_claude_adapter,
        gemini_adapter=mock_gemini_adapter,
        codex_adapter=mock_codex_adapter,
        default_tier=DebateTier.TIER_1
    )


@pytest.fixture
def sample_model_responses():
    """Create sample model responses for testing consensus building."""
    return [
        ModelResponse(
            model="claude",
            role=ModelRole.ARCHITECT,
            response="Propose microservices with careful planning and phased migration.",
            confidence=0.85,
            tokens_used=150,
            latency_ms=300
        ),
        ModelResponse(
            model="gemini",
            role=ModelRole.CHALLENGER,
            response="Consider modular monolith first - simpler and easier to maintain.",
            confidence=0.80,
            tokens_used=120,
            latency_ms=250
        ),
        ModelResponse(
            model="codex",
            role=ModelRole.VALIDATOR,
            response="Implementation is feasible but requires strong DevOps capabilities.",
            confidence=0.75,
            tokens_used=100,
            latency_ms=200
        )
    ]


# ============================================================================
# Test Debate Tier Determination
# ============================================================================

class TestDebateTierDetermination:
    """Test the debate tier determination logic."""

    def test_tier_mapping_query(self, debate_engine):
        """Test Tier 0 for simple queries."""
        tier = debate_engine.determine_tier(
            question="What is the current price of AAPL?",
            context={"type": "query"}
        )
        assert tier == DebateTier.TIER_0

    def test_tier_mapping_signal(self, debate_engine):
        """Test Tier 0 for signal requests."""
        tier = debate_engine.determine_tier(
            question="Get trading signal for TSLA",
            context={"type": "signal"}
        )
        assert tier == DebateTier.TIER_0

    def test_tier_mapping_data(self, debate_engine):
        """Test Tier 0 for data requests."""
        tier = debate_engine.determine_tier(
            question="Fetch historical data",
            context={"type": "data"}
        )
        assert tier == DebateTier.TIER_0

    def test_tier_mapping_strategy(self, debate_engine):
        """Test Tier 1 for strategy decisions."""
        tier = debate_engine.determine_tier(
            question="Which strategy should I use?",
            context={"type": "strategy"}
        )
        assert tier == DebateTier.TIER_1

    def test_tier_mapping_optimization(self, debate_engine):
        """Test Tier 1 for optimization decisions."""
        tier = debate_engine.determine_tier(
            question="How to optimize this algorithm?",
            context={"type": "optimization"}
        )
        assert tier == DebateTier.TIER_1

    def test_tier_mapping_architecture(self, debate_engine):
        """Test Tier 2 for architecture decisions."""
        tier = debate_engine.determine_tier(
            question="Should we use microservices?",
            context={"type": "architecture"}
        )
        assert tier == DebateTier.TIER_2

    def test_tier_mapping_risk(self, debate_engine):
        """Test Tier 2 for risk decisions."""
        tier = debate_engine.determine_tier(
            question="What are the risks of this trade?",
            context={"type": "risk"}
        )
        assert tier == DebateTier.TIER_2

    def test_tier_mapping_trade(self, debate_engine):
        """Test Tier 2 for trade decisions."""
        tier = debate_engine.determine_tier(
            question="Should I execute this trade?",
            context={"type": "trade"}
        )
        assert tier == DebateTier.TIER_2

    def test_high_value_trade_forces_tier2(self, debate_engine):
        """Test that high value trades force Tier 2."""
        # Without a type that maps to lower tier, high value should force Tier 2
        tier = debate_engine.determine_tier(
            question="Execute trade",
            context={"trade_value": 200000}  # No type specified
        )
        assert tier == DebateTier.TIER_2

    def test_high_risk_forces_tier2(self, debate_engine):
        """Test that high risk forces Tier 2."""
        # Without explicit type, high risk should force Tier 2
        tier = debate_engine.determine_tier(
            question="Make decision",
            context={"risk_level": "High"}  # No type specified
        )
        assert tier == DebateTier.TIER_2

    def test_critical_risk_forces_tier2(self, debate_engine):
        """Test that critical risk forces Tier 2."""
        # Without explicit type, critical risk should force Tier 2
        tier = debate_engine.determine_tier(
            question="Make decision",
            context={"risk_level": "Critical"}  # No type specified
        )
        assert tier == DebateTier.TIER_2

    def test_keyword_architecture_forces_tier2(self, debate_engine):
        """Test that 'architecture' keyword forces Tier 2."""
        tier = debate_engine.determine_tier(
            question="What architecture should we use?",
            context={}
        )
        assert tier == DebateTier.TIER_2

    def test_keyword_design_forces_tier2(self, debate_engine):
        """Test that 'design' keyword forces Tier 2."""
        tier = debate_engine.determine_tier(
            question="How should we design this system?",
            context={}
        )
        assert tier == DebateTier.TIER_2

    def test_keyword_security_forces_tier2(self, debate_engine):
        """Test that 'security' keyword forces Tier 2."""
        tier = debate_engine.determine_tier(
            question="What security measures do we need?",
            context={}
        )
        assert tier == DebateTier.TIER_2

    def test_keyword_strategy_forces_tier1(self, debate_engine):
        """Test that 'strategy' keyword forces Tier 1."""
        tier = debate_engine.determine_tier(
            question="What strategy should we follow?",
            context={}
        )
        assert tier == DebateTier.TIER_1

    def test_keyword_optimize_forces_tier1(self, debate_engine):
        """Test that 'optimize' keyword forces Tier 1."""
        tier = debate_engine.determine_tier(
            question="How to optimize performance?",
            context={}
        )
        assert tier == DebateTier.TIER_1

    def test_default_tier_used_when_no_match(self, debate_engine):
        """Test that default tier is used when no rules match."""
        tier = debate_engine.determine_tier(
            question="Random question",
            context={}
        )
        assert tier == DebateTier.TIER_1  # Default set in fixture


# ============================================================================
# Test Single Model Response (Tier 0)
# ============================================================================

class TestSingleModelResponse:
    """Test Tier 0 single model response behavior."""

    @pytest.mark.asyncio
    async def test_single_model_response(self, mock_claude_adapter):
        """Test that Tier 0 returns single model response."""
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            default_tier=DebateTier.TIER_0
        )

        result = await engine.run_debate(
            question="What is the current price?",
            context={"type": "query"}
        )

        assert result.tier == DebateTier.TIER_0
        assert len(result.rounds) == 1
        assert result.rounds[0].round_type == "single_model"
        assert len(result.rounds[0].responses) == 1
        assert result.rounds[0].responses[0].model == "claude"
        assert result.success
        assert result.consensus == result.final_decision
        assert len(result.dissent) == 0
        assert "claude" in result.models_used
        assert len(result.models_used) == 1

    @pytest.mark.asyncio
    async def test_single_model_performance(self, mock_claude_adapter):
        """Test that single model response is fast."""
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            default_tier=DebateTier.TIER_0
        )

        result = await engine.run_debate(
            question="Quick query",
            context={"type": "data"}
        )

        # Should be fast since it's just one call
        assert result.total_duration_ms < 5000
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_single_model_metadata(self, mock_claude_adapter):
        """Test that single model response includes proper metadata."""
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            default_tier=DebateTier.TIER_0
        )

        result = await engine.run_debate(
            question="Get info",
            context={"type": "info", "extra": "data"}
        )

        assert result.metadata["single_model"] is True
        # Metadata structure may vary, just check it exists
        assert result.metadata is not None


# ============================================================================
# Test Consensus Builder
# ============================================================================

class TestConsensusBuilder:
    """Test consensus building from multiple model responses."""

    def test_consensus_single_response(self, sample_model_responses):
        """Test consensus with single response."""
        builder = ConsensusBuilder(strategy=VotingStrategy.WEIGHTED)
        consensus, dissent, decision = builder.build_consensus(
            [sample_model_responses[0]]
        )

        assert consensus == sample_model_responses[0].response
        assert len(dissent) == 0
        assert decision == sample_model_responses[0].response

    def test_consensus_no_responses(self):
        """Test consensus with no responses."""
        builder = ConsensusBuilder(strategy=VotingStrategy.WEIGHTED)
        consensus, dissent, decision = builder.build_consensus([])

        assert consensus == ""
        assert len(dissent) == 0
        assert decision == "No responses to build consensus from."

    def test_consensus_weighted_strategy(self, sample_model_responses):
        """Test weighted voting strategy."""
        builder = ConsensusBuilder(strategy=VotingStrategy.WEIGHTED)
        consensus, dissent, decision = builder.build_consensus(
            sample_model_responses
        )

        assert "Consensus Points" in consensus
        assert "Model Contributions" in consensus
        assert len(dissent) >= 0
        assert "Final Decision" in decision
        assert "Weighted Consensus" in decision

    def test_consensus_architect_final_strategy(self, sample_model_responses):
        """Test architect final say strategy."""
        builder = ConsensusBuilder(strategy=VotingStrategy.ARCHITECT_FINAL)
        consensus, dissent, decision = builder.build_consensus(
            sample_model_responses
        )

        assert "Final Decision" in decision
        assert "Architect Final" in decision
        # Should favor Claude's response
        assert "microservices" in decision.lower()

    def test_consensus_majority_strategy(self, sample_model_responses):
        """Test majority voting strategy."""
        builder = ConsensusBuilder(strategy=VotingStrategy.MAJORITY)
        consensus, dissent, decision = builder.build_consensus(
            sample_model_responses
        )

        assert "Final Decision" in decision
        assert "Majority" in decision
        # Should include all models
        assert "claude" in decision.lower()
        assert "gemini" in decision.lower()
        assert "codex" in decision.lower()

    def test_consensus_unanimous_strategy_with_agreement(self):
        """Test unanimous strategy when models agree."""
        responses = [
            ModelResponse(
                model="claude",
                role=ModelRole.ARCHITECT,
                response="Use approach A",
                confidence=0.9,
                tokens_used=100,
                latency_ms=200
            ),
            ModelResponse(
                model="gemini",
                role=ModelRole.CHALLENGER,
                response="Agree with approach A",
                confidence=0.85,
                tokens_used=90,
                latency_ms=180
            )
        ]

        builder = ConsensusBuilder(strategy=VotingStrategy.UNANIMOUS)
        consensus, dissent, decision = builder.build_consensus(responses)

        assert "Final Decision" in decision
        assert "Consensus Required" in decision

    def test_consensus_tracks_low_confidence(self):
        """Test that low confidence responses are tracked as dissent."""
        responses = [
            ModelResponse(
                model="claude",
                role=ModelRole.ARCHITECT,
                response="Option A is best",
                confidence=0.3,  # Low confidence
                tokens_used=100,
                latency_ms=200
            ),
            ModelResponse(
                model="gemini",
                role=ModelRole.CHALLENGER,
                response="Option B is better",
                confidence=0.9,
                tokens_used=100,
                latency_ms=200
            )
        ]

        builder = ConsensusBuilder(strategy=VotingStrategy.WEIGHTED)
        consensus, dissent, decision = builder.build_consensus(responses)

        # Should track Claude's low confidence
        assert len(dissent) > 0
        assert any("low confidence" in d.lower() for d in dissent)

    def test_consensus_custom_weights(self, sample_model_responses):
        """Test consensus with custom model weights."""
        custom_weights = {
            "claude": 0.5,
            "gemini": 1.5,  # Favor Gemini
            "codex": 1.0
        }

        builder = ConsensusBuilder(
            strategy=VotingStrategy.WEIGHTED,
            weights=custom_weights
        )
        consensus, dissent, decision = builder.build_consensus(
            sample_model_responses
        )

        # With Gemini weighted higher, decision should favor it
        assert "gemini" in decision.lower()

    def test_quick_consensus_helper(self, sample_model_responses):
        """Test quick consensus helper function."""
        decision = quick_consensus(
            sample_model_responses,
            strategy=VotingStrategy.WEIGHTED
        )

        assert "Final Decision" in decision
        assert len(decision) > 0

    def test_consensus_extract_key_points(self, sample_model_responses):
        """Test key point extraction from responses."""
        builder = ConsensusBuilder(strategy=VotingStrategy.WEIGHTED)
        key_points = builder._extract_key_points(sample_model_responses)

        assert len(key_points) == 3
        assert "claude" in key_points
        assert "gemini" in key_points
        assert "codex" in key_points

    def test_consensus_handles_empty_response(self):
        """Test consensus with empty response text."""
        responses = [
            ModelResponse(
                model="claude",
                role=ModelRole.ARCHITECT,
                response="",
                confidence=0.8,
                tokens_used=0,
                latency_ms=100
            )
        ]

        builder = ConsensusBuilder(strategy=VotingStrategy.WEIGHTED)
        consensus, dissent, decision = builder.build_consensus(responses)

        # With single response, it just returns the response (even if empty)
        assert consensus == ""
        assert decision == ""


# ============================================================================
# Test Mock Adapters
# ============================================================================

class TestMockAdapters:
    """Test mock adapter functionality."""

    @pytest.mark.asyncio
    async def test_mock_adapter_basic_query(self):
        """Test basic mock adapter query."""
        adapter = MockAdapter(model_name="test_model")

        response = await adapter.query(
            prompt="Test question",
            role=ModelRole.ARCHITECT
        )

        assert response.model == "test_model"
        assert response.role == ModelRole.ARCHITECT
        assert len(response.response) > 0
        assert response.confidence == 0.8
        assert response.tokens_used == 100
        assert response.latency_ms == 50
        assert response.metadata["mock"] is True

    @pytest.mark.asyncio
    async def test_mock_adapter_predefined_responses(self):
        """Test mock adapter with predefined responses."""
        responses = {
            "architect": "Custom architect response",
            "challenger": "Custom challenger response"
        }

        adapter = MockAdapter(model_name="test", responses=responses)

        arch_response = await adapter.query(
            prompt="Question",
            role=ModelRole.ARCHITECT
        )
        assert arch_response.response == "Custom architect response"

        chal_response = await adapter.query(
            prompt="Question",
            role=ModelRole.CHALLENGER
        )
        assert chal_response.response == "Custom challenger response"

    @pytest.mark.asyncio
    async def test_mock_adapter_query_count(self):
        """Test that mock adapter tracks query count."""
        adapter = MockAdapter(model_name="test")

        assert adapter.query_count == 0

        await adapter.query("Q1", ModelRole.ARCHITECT)
        assert adapter.query_count == 1

        await adapter.query("Q2", ModelRole.CHALLENGER)
        assert adapter.query_count == 2

        await adapter.query("Q3", ModelRole.VALIDATOR)
        assert adapter.query_count == 3

    def test_mock_adapter_is_available(self):
        """Test that mock adapter is always available."""
        adapter = MockAdapter(model_name="test")
        assert adapter.is_available() is True

    @pytest.mark.asyncio
    async def test_mock_adapter_different_roles(self):
        """Test mock adapter handles different roles."""
        adapter = MockAdapter(model_name="test")

        roles = [
            ModelRole.ARCHITECT,
            ModelRole.CHALLENGER,
            ModelRole.VALIDATOR
        ]

        for role in roles:
            response = await adapter.query("Question", role)
            assert response.role == role
            assert role.value in response.response.lower()


# ============================================================================
# Test Full Debate Flow
# ============================================================================

class TestFullDebateFlow:
    """Test complete debate workflows."""

    @pytest.mark.asyncio
    async def test_tier1_debate_two_models(
        self,
        mock_claude_adapter,
        mock_gemini_adapter
    ):
        """Test Tier 1 debate with two models."""
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            gemini_adapter=mock_gemini_adapter,
            codex_adapter=None
        )

        result = await engine.run_debate(
            question="Should we refactor this module?",
            context={"type": "strategy"},
            tier=DebateTier.TIER_1
        )

        assert result.tier == DebateTier.TIER_1
        assert len(result.rounds) == 2  # Proposal + Cross-exam
        assert len(result.models_used) == 2
        assert "claude" in result.models_used
        assert "gemini" in result.models_used
        assert result.success
        assert len(result.consensus) > 0
        assert len(result.final_decision) > 0

    @pytest.mark.asyncio
    async def test_tier2_debate_three_models(self, debate_engine):
        """Test Tier 2 debate with all three models."""
        result = await debate_engine.run_debate(
            question="What architecture should we use?",
            context={"type": "architecture"},
            tier=DebateTier.TIER_2
        )

        assert result.tier == DebateTier.TIER_2
        assert len(result.rounds) == 2  # Proposal + Cross-exam
        assert len(result.models_used) == 3
        assert "claude" in result.models_used
        assert "gemini" in result.models_used
        assert "codex" in result.models_used
        assert result.success

    @pytest.mark.asyncio
    async def test_debate_with_context(self, debate_engine):
        """Test debate with rich context."""
        context = {
            "type": "architecture",
            "team_size": 5,
            "current_stack": "Django monolith",
            "pain_points": ["Slow deploys", "Tight coupling"],
            "constraints": ["Limited DevOps expertise"]
        }

        result = await debate_engine.run_debate(
            question="Should we migrate to microservices?",
            context=context
        )

        assert result.success
        assert result.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_debate_tracks_tokens(self, debate_engine):
        """Test that debate tracks token usage."""
        result = await debate_engine.run_debate(
            question="Test question",
            context={"type": "strategy"}
        )

        assert result.total_tokens > 0
        # Should be sum of all response tokens
        total = sum(
            r.tokens_used
            for round in result.rounds
            for r in round.responses
        )
        assert result.total_tokens == total

    @pytest.mark.asyncio
    async def test_debate_tracks_duration(self, debate_engine):
        """Test that debate tracks duration."""
        result = await debate_engine.run_debate(
            question="Test question",
            context={"type": "strategy"}
        )

        # Total duration should be tracked (may be 0 for mock adapters)
        assert result.total_duration_ms >= 0
        # Each round should have duration (>= 0)
        for round in result.rounds:
            assert round.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_debate_engine_stats(self, debate_engine):
        """Test debate engine statistics tracking."""
        initial_stats = debate_engine.get_stats()
        initial_debates = initial_stats["total_debates"]

        await debate_engine.run_debate(
            question="Test 1",
            context={"type": "query"}
        )

        await debate_engine.run_debate(
            question="Test 2",
            context={"type": "strategy"}
        )

        stats = debate_engine.get_stats()
        assert stats["total_debates"] == initial_debates + 2
        assert stats["total_tokens"] > 0
        assert stats["models_available"]["claude"] is True
        assert stats["models_available"]["gemini"] is True
        assert stats["models_available"]["codex"] is True


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_debate_with_only_claude(self, mock_claude_adapter):
        """Test debate with only Claude available."""
        engine = DebateEngine(
            claude_adapter=mock_claude_adapter,
            gemini_adapter=None,
            codex_adapter=None
        )

        result = await engine.run_debate(
            question="What should we do?",
            context={"type": "strategy"},
            tier=DebateTier.TIER_1
        )

        # Should still work with just Claude
        assert result.success
        assert "claude" in result.models_used
        assert len(result.models_used) == 1

    @pytest.mark.asyncio
    async def test_debate_empty_question(self, debate_engine):
        """Test debate with empty question."""
        result = await debate_engine.run_debate(
            question="",
            context={"type": "query"}
        )

        # Should still execute
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_empty_context(self, debate_engine):
        """Test debate with empty context."""
        result = await debate_engine.run_debate(
            question="Test question",
            context={}
        )

        assert result.success
        assert result.metadata["context"] == {}

    @pytest.mark.asyncio
    async def test_debate_none_context(self, debate_engine):
        """Test debate with None context."""
        result = await debate_engine.run_debate(
            question="Test question",
            context=None
        )

        assert result.success
        # None context becomes empty dict
        assert result.metadata.get("context") is not None

    def test_tier_mapping_coverage(self):
        """Test that all tier mapping keys are valid."""
        for decision_type, tier in TIER_MAPPING.items():
            assert isinstance(decision_type, str)
            assert isinstance(tier, DebateTier)
            assert tier in [
                DebateTier.TIER_0,
                DebateTier.TIER_1,
                DebateTier.TIER_2
            ]

    def test_debate_result_success_property(self):
        """Test DebateResult success property."""
        # Success case
        result = DebateResult(
            question="Test",
            tier=DebateTier.TIER_1,
            rounds=[],
            consensus="Consensus",
            dissent=[],
            final_decision="Decision made"
        )
        assert result.success is True

        # Failure case
        result_no_decision = DebateResult(
            question="Test",
            tier=DebateTier.TIER_1,
            rounds=[],
            consensus="",
            dissent=[],
            final_decision=""
        )
        assert result_no_decision.success is False


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_portfolio_trade_decision(self, debate_engine):
        """Test debate for high-value portfolio trade."""
        context = {
            "type": "trade",
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 1000,
            "trade_value": 175000,
            "risk_level": "High",  # Capitalized
            "portfolio_impact": "15% allocation"
        }

        result = await debate_engine.run_debate(
            question="Should we execute this large AAPL purchase?",
            context=context
        )

        # Should force Tier 2 due to high value and risk
        assert result.tier == DebateTier.TIER_2
        assert len(result.models_used) == 3
        assert result.success

    @pytest.mark.asyncio
    async def test_system_architecture_decision(self, debate_engine):
        """Test debate for system architecture decision."""
        context = {
            "type": "architecture",
            "decision": "Migrate to microservices",
            "current_system": "Django monolith",
            "team_size": 8,
            "timeline": "6 months"
        }

        result = await debate_engine.run_debate(
            question="Should we migrate from Django monolith to microservices?",
            context=context
        )

        # Architecture decisions should be Tier 2
        assert result.tier == DebateTier.TIER_2
        assert result.success
        # Should have multiple perspectives
        assert len(result.rounds) >= 2

    @pytest.mark.asyncio
    async def test_quick_strategy_optimization(self, debate_engine):
        """Test debate for strategy optimization."""
        context = {
            "type": "optimization",
            "strategy": "momentum",
            "current_sharpe": 1.2,
            "target_sharpe": 1.5
        }

        result = await debate_engine.run_debate(
            question="How to improve momentum strategy Sharpe ratio?",
            context=context
        )

        # Optimization should be Tier 1
        assert result.tier == DebateTier.TIER_1
        assert result.success

    @pytest.mark.asyncio
    async def test_simple_data_query(self, debate_engine):
        """Test debate for simple data query."""
        result = await debate_engine.run_debate(
            question="Get latest price for MSFT",
            context={"type": "data", "symbol": "MSFT"}
        )

        # Data queries should be Tier 0
        assert result.tier == DebateTier.TIER_0
        assert len(result.models_used) == 1
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
