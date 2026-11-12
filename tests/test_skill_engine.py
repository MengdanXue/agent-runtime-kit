# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Skill Engine Test Suite

Tests for the skill discovery and execution system.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from agentrt.skills import (
    SkillEngine,
    SkillDiscovery,
    SkillMetadata,
    SkillSandbox,
    SkillExecutionResult,
    SkillType,
    get_skill_engine,
    reset_skill_engine,
)


@pytest.fixture
def temp_skills_dir(tmp_path):
    """Create a temporary skills directory with sample skills."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    # Create a sample skill
    sample_skill = skills_dir / "test-skill"
    sample_skill.mkdir()

    skill_md = sample_skill / "SKILL.md"
    skill_md.write_text("""---
name: test-skill
description: A test skill for unit testing
version: 1.0.0
author: test-author
triggers:
  - "test trigger"
  - "run test"
dependencies:
  - pytest
type: test
location: managed
---

# Test Skill

This is a test skill for unit testing.
""")

    # Create a script
    scripts_dir = sample_skill / "scripts"
    scripts_dir.mkdir()
    main_script = scripts_dir / "main.py"
    main_script.write_text("""
import sys
print("Test skill executed successfully")
print(f"Args: {sys.argv[1:]}")
""")

    return skills_dir


class TestSkillDiscovery:
    """Test skill discovery functionality."""

    def test_discover_skills(self, temp_skills_dir):
        """Test discovering skills from directory."""
        discovery = SkillDiscovery(str(temp_skills_dir))
        skills = discovery.discover()

        assert len(skills) == 1
        skill = skills[0]
        assert skill.name == "test-skill"
        assert skill.description == "A test skill for unit testing"
        assert skill.version == "1.0.0"
        assert skill.author == "test-author"
        assert "test trigger" in skill.triggers
        assert "pytest" in skill.dependencies
        assert skill.skill_type == SkillType.TEST

    def test_discover_no_skills(self, tmp_path):
        """Test discovery with no skills present."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        discovery = SkillDiscovery(str(empty_dir))
        skills = discovery.discover()

        assert len(skills) == 0

    def test_discover_nonexistent_directory(self, tmp_path):
        """Test discovery with nonexistent directory."""
        nonexistent = tmp_path / "nonexistent"

        discovery = SkillDiscovery(str(nonexistent))
        skills = discovery.discover()

        assert len(skills) == 0


class TestSkillMetadata:
    """Test skill metadata handling."""

    def test_metadata_creation(self):
        """Test creating skill metadata."""
        metadata = SkillMetadata(
            name="test-skill",
            description="Test description",
            version="1.0.0",
            triggers=["trigger1", "trigger2"],
            skill_type="quant"
        )

        assert metadata.name == "test-skill"
        assert metadata.skill_type == SkillType.QUANT
        assert len(metadata.triggers) == 2

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = SkillMetadata(
            name="test-skill",
            description="Test description",
            version="1.0.0",
            skill_type=SkillType.QUANT
        )

        data = metadata.to_dict()

        assert data["name"] == "test-skill"
        assert data["skill_type"] == "quant"
        assert data["version"] == "1.0.0"


class TestSkillSandbox:
    """Test skill sandbox execution."""

    def test_sandbox_creation(self):
        """Test creating sandbox."""
        sandbox = SkillSandbox(timeout=30, max_memory_mb=256)

        assert sandbox.timeout == 30
        assert sandbox.max_memory_mb == 256

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent / ".claude" / "skills",
        reason="Skills directory not found"
    )
    def test_execute_success(self, temp_skills_dir):
        """Test successful script execution."""
        sandbox = SkillSandbox(timeout=5)
        script_path = temp_skills_dir / "test-skill" / "scripts" / "main.py"

        result = sandbox.execute(
            skill_name="test-skill",
            script_path=str(script_path),
            args=["--param1=value1"]
        )

        assert result.success is True
        assert result.skill_name == "test-skill"
        assert "Test skill executed successfully" in result.output
        assert result.error is None

    def test_execute_nonexistent_script(self):
        """Test execution with nonexistent script."""
        sandbox = SkillSandbox(timeout=5)

        result = sandbox.execute(
            skill_name="nonexistent",
            script_path="/nonexistent/script.py",
            args=[]
        )

        assert result.success is False
        assert "Script not found" in result.error


class TestSkillEngine:
    """Test skill engine functionality."""

    def test_engine_creation(self, temp_skills_dir):
        """Test creating skill engine."""
        engine = SkillEngine(
            skills_dir=str(temp_skills_dir),
            auto_discover=True
        )

        assert len(engine._skills) == 1
        assert "test-skill" in engine._skills

    def test_list_skills(self, temp_skills_dir):
        """Test listing skills."""
        engine = SkillEngine(str(temp_skills_dir))

        skills = engine.list_skills()

        assert "test-skill" in skills

    def test_get_skill(self, temp_skills_dir):
        """Test getting skill metadata."""
        engine = SkillEngine(str(temp_skills_dir))

        skill = engine.get_skill("test-skill")

        assert skill is not None
        assert skill.name == "test-skill"

    def test_get_nonexistent_skill(self, temp_skills_dir):
        """Test getting nonexistent skill."""
        engine = SkillEngine(str(temp_skills_dir))

        skill = engine.get_skill("nonexistent")

        assert skill is None

    def test_find_by_trigger(self, temp_skills_dir):
        """Test finding skills by trigger."""
        engine = SkillEngine(str(temp_skills_dir))

        skills = engine.find_by_trigger("test trigger")

        assert len(skills) == 1
        assert skills[0].name == "test-skill"

    def test_register_handler(self, temp_skills_dir):
        """Test registering Python handler."""
        engine = SkillEngine(str(temp_skills_dir))

        async def test_handler(params):
            return {"result": "success", "params": params}

        engine.register_handler("test-skill", test_handler)

        assert "test-skill" in engine._handlers

    @pytest.mark.asyncio
    async def test_invoke_with_handler(self, temp_skills_dir):
        """Test invoking skill with Python handler."""
        engine = SkillEngine(str(temp_skills_dir))

        async def test_handler(params):
            return {"result": "success", "symbol": params.get("symbol")}

        engine.register_handler("test-skill", test_handler)

        result = await engine.invoke("test-skill", {"symbol": "AAPL"})

        assert result.success is True
        assert result.output["result"] == "success"
        assert result.output["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_skill(self, temp_skills_dir):
        """Test invoking nonexistent skill."""
        engine = SkillEngine(str(temp_skills_dir))

        result = await engine.invoke("nonexistent", {})

        assert result.success is False
        assert "Skill not found" in result.error

    def test_get_skill_info(self, temp_skills_dir):
        """Test getting skill info."""
        engine = SkillEngine(str(temp_skills_dir))

        info = engine.get_skill_info("test-skill")

        assert info is not None
        assert info["name"] == "test-skill"
        assert info["has_handler"] is False
        assert info["script_exists"] is True

    def test_get_stats(self, temp_skills_dir):
        """Test getting engine stats."""
        engine = SkillEngine(str(temp_skills_dir))

        stats = engine.get_stats()

        assert stats["total_skills"] == 1
        assert "skill_types" in stats
        assert "total_triggers" in stats

    def test_refresh_skills(self, temp_skills_dir):
        """Test refreshing skills."""
        engine = SkillEngine(str(temp_skills_dir), auto_discover=False)

        assert len(engine._skills) == 0

        count = engine.refresh_skills()

        assert count == 1
        assert len(engine._skills) == 1


class TestGlobalSkillEngine:
    """Test global skill engine singleton."""

    def test_get_skill_engine(self):
        """Test getting global engine."""
        reset_skill_engine()

        engine1 = get_skill_engine()
        engine2 = get_skill_engine()

        assert engine1 is engine2

    def test_reset_skill_engine(self):
        """Test resetting global engine."""
        engine1 = get_skill_engine()
        reset_skill_engine()
        engine2 = get_skill_engine()

        assert engine1 is not engine2


class TestSkillExecutionResult:
    """Test skill execution result."""

    def test_result_creation(self):
        """Test creating execution result."""
        result = SkillExecutionResult(
            skill_name="test-skill",
            success=True,
            output={"key": "value"},
            execution_time_ms=100
        )

        assert result.skill_name == "test-skill"
        assert result.success is True
        assert result.output["key"] == "value"

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = SkillExecutionResult(
            skill_name="test-skill",
            success=True,
            output={"key": "value"}
        )

        data = result.to_dict()

        assert data["skill_name"] == "test-skill"
        assert data["success"] is True
        assert data["output"]["key"] == "value"


class TestSkillIntegration:
    """Integration tests for skill engine with real skills."""

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent.parent.parent / ".claude" / "skills").exists(),
        reason="Skills directory not found"
    )
    def test_discover_real_skills(self):
        """Test discovering real skills from .claude/skills."""
        engine = get_skill_engine()

        skills = engine.list_skills()

        # Should have at least some skills
        assert len(skills) > 0

        # Check for quant skills
        quant_skills = engine.list_skills(skill_type=SkillType.QUANT)
        expected_quant_skills = ["quant-signal", "quant-backtest", "quant-risk", "quant-decision"]

        for expected in expected_quant_skills:
            assert expected in quant_skills, f"Expected skill {expected} not found"

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent.parent.parent / ".claude" / "skills").exists(),
        reason="Skills directory not found"
    )
    def test_get_quant_skill_info(self):
        """Test getting info for quant skills."""
        engine = get_skill_engine()

        skill = engine.get_skill("quant-signal")

        assert skill is not None
        assert skill.name == "quant-signal"
        assert skill.skill_type == SkillType.QUANT
        assert len(skill.triggers) > 0
        assert "generate signal" in [t.lower() for t in skill.triggers]

    @pytest.mark.skipif(
        not (Path(__file__).parent.parent.parent.parent.parent / ".claude" / "skills").exists(),
        reason="Skills directory not found"
    )
    def test_search_by_trigger(self):
        """Test searching skills by trigger phrase."""
        engine = get_skill_engine()

        # Search for signal-related skills
        skills = engine.find_by_trigger("signal")

        assert len(skills) > 0

        # Should find quant-signal
        skill_names = [s.name for s in skills]
        assert "quant-signal" in skill_names
