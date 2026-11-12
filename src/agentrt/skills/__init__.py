# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Skill engine module for dynamic skill discovery and execution.

This module provides the core skill management infrastructure:
- SkillEngine: Main skill management interface
- SkillDiscovery: Auto-discovery of skills from filesystem
- SkillMetadata: Skill metadata extracted from SKILL.md files
- SkillSandbox: Isolated execution environment for skills
- Multi-model debate integration
- Gemini/Codex CLI integration

Example:
    from agentrt.skills import get_skill_engine, initialize_skill_engine_with_handlers

    # Get global skill engine with all handlers
    engine = initialize_skill_engine_with_handlers()

    # List all skills
    skills = engine.list_skills()

    # Invoke a skill
    result = await engine.invoke("quant-signal", {
        "symbol": "AAPL",
        "timeframe": "1d"
    })

    # Run multi-model debate
    result = await engine.invoke("multi-model-debate", {
        "question": "What architecture should we use?",
        "tier": 2
    })

    # Call Gemini
    result = await engine.invoke("gemini", {
        "prompt": "Analyze this strategy",
        "role": "analyst"
    })

    # Call Codex
    result = await engine.invoke("codex", {
        "prompt": "Review this code",
        "task": "review"
    })
"""

from .engine import (
    SkillEngine,
    SkillDiscovery,
    SkillMetadata,
    SkillSandbox,
    SkillExecutionResult,
    SkillType,
    get_skill_engine,
    reset_skill_engine,
    DebateSkillHandler,
    register_debate_skill,
    setup_quant_skill_handlers,
    initialize_skill_engine_with_handlers,
)

from .gemini_skill import (
    GeminiSkillHandler,
    GeminiResponse,
    get_gemini_handler,
    register_gemini_skill,
    call_gemini,
)

from .codex_skill import (
    CodexSkillHandler,
    CodexResponse,
    get_codex_handler,
    register_codex_skill,
    call_codex,
)

__all__ = [
    # Core
    "SkillEngine",
    "SkillDiscovery",
    "SkillMetadata",
    "SkillSandbox",
    "SkillExecutionResult",
    "SkillType",
    # Global access
    "get_skill_engine",
    "reset_skill_engine",
    # Debate integration
    "DebateSkillHandler",
    "register_debate_skill",
    "setup_quant_skill_handlers",
    "initialize_skill_engine_with_handlers",
    # Gemini
    "GeminiSkillHandler",
    "GeminiResponse",
    "get_gemini_handler",
    "register_gemini_skill",
    "call_gemini",
    # Codex
    "CodexSkillHandler",
    "CodexResponse",
    "get_codex_handler",
    "register_codex_skill",
    "call_codex",
]
