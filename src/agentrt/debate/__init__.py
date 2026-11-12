# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Model Debate Engine

Orchestrates structured debates between multiple AI models (Claude, Gemini, Codex)
to produce higher-quality design decisions through diverse perspectives.
"""

from .engine import (
    DebateTier,
    ModelRole,
    ModelResponse,
    DebateRound,
    DebateResult,
    DebateEngine,
)
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
from .consensus import (
    ConsensusBuilder,
    VotingStrategy,
)
from .executor import (
    DebateMode,
    DebateConfig,
    DebateExecutor,
    DebateExecutionResult,
    ModelPerspective,
    run_quick_debate,
    run_full_debate,
)

__all__ = [
    # Engine
    "DebateTier",
    "ModelRole",
    "ModelResponse",
    "DebateRound",
    "DebateResult",
    "DebateEngine",
    # Adapters
    "ModelAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
    "CodexAdapter",
    "GeminiCLIAdapter",
    "CodexCLIAdapter",
    "MockAdapter",
    "create_adapter_set",
    # Consensus
    "ConsensusBuilder",
    "VotingStrategy",
    # Executor
    "DebateMode",
    "DebateConfig",
    "DebateExecutor",
    "DebateExecutionResult",
    "ModelPerspective",
    "run_quick_debate",
    "run_full_debate",
]
