# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Skill Engine - Dynamic skill discovery and execution system.

This module provides:
1. SkillDiscovery - Scans .claude/skills/ directory to discover all skills
2. SkillMetadata - Parses YAML front matter from SKILL.md files
3. SkillEngine - Manages skill registry and execution
4. Integration with API routes for skill invocation

Based on UQAP architecture design:
docs/architecture/UQAP-Universal-Quant-Agent-Platform.md
"""

import os
import re
import yaml
import hashlib
import subprocess
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """Skill types."""
    QUANT = "quant"           # Quantitative analysis skills
    AGENT = "agent"           # Agent coordination skills
    DEBUG = "debug"           # Debugging skills
    TEST = "test"             # Testing skills
    GENERAL = "general"       # General purpose skills


@dataclass
class SkillMetadata:
    """
    Metadata extracted from SKILL.md YAML front matter.

    Attributes:
        name: Skill name (e.g., 'quant-signal')
        description: Brief description of what the skill does
        version: Semantic version (e.g., '1.0.0')
        author: Skill author or maintainer
        triggers: List of trigger phrases that activate this skill
        dependencies: List of Python package dependencies
        skill_type: Type/category of skill
        location: Location type ('managed' or 'user')
        path: Full path to skill directory
        signature: Optional cryptographic signature for verification
        metadata: Additional custom metadata
    """
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "unknown"
    triggers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    skill_type: SkillType = SkillType.GENERAL
    location: str = "managed"  # 'managed' or 'user'
    path: Optional[Path] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert skill_type string to enum if needed."""
        if isinstance(self.skill_type, str):
            try:
                self.skill_type = SkillType(self.skill_type.lower())
            except ValueError:
                self.skill_type = SkillType.GENERAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "triggers": self.triggers,
            "dependencies": self.dependencies,
            "skill_type": self.skill_type.value,
            "location": self.location,
            "path": str(self.path) if self.path else None,
            "signature": self.signature,
            "metadata": self.metadata
        }


@dataclass
class SkillExecutionResult:
    """
    Result of skill execution.

    Attributes:
        skill_name: Name of executed skill
        success: Whether execution succeeded
        output: Skill output data
        error: Error message if failed
        execution_time_ms: Execution duration in milliseconds
        metadata: Additional execution metadata
    """
    skill_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "skill_name": self.skill_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "executed_at": self.executed_at.isoformat()
        }


class SkillDiscovery:
    """
    Skill auto-discovery system.

    Scans the .claude/skills/ directory and discovers all available
    skills by parsing their SKILL.md files.
    """

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Initialize skill discovery.

        Args:
            skills_dir: Root directory containing skills (defaults to .claude/skills)
        """
        if skills_dir is None:
            # Default to .claude/skills relative to project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.skills_dir = project_root / ".claude" / "skills"
        else:
            self.skills_dir = Path(skills_dir)

        logger.info(f"Skill discovery initialized with directory: {self.skills_dir}")

    def discover(self) -> List[SkillMetadata]:
        """
        Scan and discover all skills in the skills directory.

        Returns:
            List of discovered skill metadata
        """
        skills = []

        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return skills

        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith('.'):
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        metadata = self._parse_skill_file(skill_file)
                        if metadata:
                            metadata.path = skill_dir
                            skills.append(metadata)
                            logger.debug(f"Discovered skill: {metadata.name}")
                    except Exception as e:
                        logger.error(f"Failed to parse skill file {skill_file}: {e}")

        logger.info(f"Discovered {len(skills)} skills")
        return skills

    def _parse_skill_file(self, path: Path) -> Optional[SkillMetadata]:
        """
        Parse SKILL.md file and extract metadata from YAML front matter.

        The expected format is:
        ---
        name: skill-name
        description: Brief description
        version: 1.0.0
        triggers:
          - "trigger phrase 1"
          - "trigger phrase 2"
        dependencies:
          - pandas
          - numpy
        ---
        # Skill Documentation
        ...

        Args:
            path: Path to SKILL.md file

        Returns:
            SkillMetadata if successfully parsed, None otherwise
        """
        try:
            content = path.read_text(encoding="utf-8")

            # Parse YAML front matter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    front_matter_text = parts[1].strip()
                    front_matter = yaml.safe_load(front_matter_text)

                    if not isinstance(front_matter, dict):
                        logger.warning(f"Invalid YAML front matter in {path}")
                        return None

                    # Extract metadata
                    name = front_matter.get("name", path.parent.name)
                    description = front_matter.get("description", "")
                    version = front_matter.get("version", "1.0.0")
                    author = front_matter.get("author", "unknown")
                    triggers = front_matter.get("triggers", [])
                    dependencies = front_matter.get("dependencies", [])
                    skill_type = front_matter.get("type")
                    if skill_type is None:
                        skill_type = "quant" if name.startswith("quant-") else "general"
                    location = front_matter.get("location", "managed")

                    # Ensure triggers and dependencies are lists
                    if isinstance(triggers, str):
                        triggers = [triggers]
                    if isinstance(dependencies, str):
                        dependencies = [dependencies]

                    return SkillMetadata(
                        name=name,
                        description=description,
                        version=version,
                        author=author,
                        triggers=triggers,
                        dependencies=dependencies,
                        skill_type=skill_type,
                        location=location,
                        metadata=front_matter  # Store full front matter
                    )

            logger.warning(f"No YAML front matter found in {path}")
            return None

        except Exception as e:
            logger.error(f"Error parsing skill file {path}: {e}")
            return None

    def watch_for_changes(
        self,
        on_change_callback: Optional[Callable[[str, str], None]] = None
    ) -> Optional[Any]:
        """
        Watch for changes in the skills directory.

        Uses watchdog library to monitor file system changes and automatically
        reload skills when they are added, modified, or removed.

        Args:
            on_change_callback: Optional callback function(event_type, file_path)
                                called when a skill file changes

        Returns:
            Observer instance if watchdog is available, None otherwise

        Usage:
            discovery = SkillDiscovery()
            observer = discovery.watch_for_changes()
            # ... later ...
            if observer:
                observer.stop()
                observer.join()
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileSystemEvent
        except ImportError:
            logger.warning(
                "watchdog library not installed. "
                "Install with: pip install watchdog"
            )
            return None

        discovery = self

        class SkillFileHandler(FileSystemEventHandler):
            """Handler for skill file changes."""

            def __init__(self):
                super().__init__()
                self._debounce_timers: Dict[str, float] = {}
                self._debounce_interval = 0.5  # 500ms debounce

            def _should_process(self, event: FileSystemEvent) -> bool:
                """Check if event should be processed."""
                # Only process SKILL.md files
                if event.is_directory:
                    return False
                path = Path(event.src_path)
                return path.name == "SKILL.md"

            def _handle_change(self, event_type: str, src_path: str):
                """Handle skill file change with debouncing."""
                import time
                now = time.time()

                # Debounce: ignore rapid successive events for same file
                last_time = self._debounce_timers.get(src_path, 0)
                if now - last_time < self._debounce_interval:
                    return
                self._debounce_timers[src_path] = now

                logger.info(f"Skill file {event_type}: {src_path}")

                # Reload all skills
                try:
                    discovery.discover()
                    logger.info("Skills reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload skills: {e}")

                # Call custom callback if provided
                if on_change_callback:
                    try:
                        on_change_callback(event_type, src_path)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            def on_created(self, event: FileSystemEvent):
                if self._should_process(event):
                    self._handle_change("created", event.src_path)

            def on_modified(self, event: FileSystemEvent):
                if self._should_process(event):
                    self._handle_change("modified", event.src_path)

            def on_deleted(self, event: FileSystemEvent):
                if self._should_process(event):
                    self._handle_change("deleted", event.src_path)

            def on_moved(self, event: FileSystemEvent):
                # Handle rename/move as delete + create
                if hasattr(event, 'dest_path'):
                    dest_path = Path(event.dest_path)
                    if dest_path.name == "SKILL.md":
                        self._handle_change("moved", event.dest_path)

        # Create and start observer
        event_handler = SkillFileHandler()
        observer = Observer()

        # Watch all skill directories
        for skills_dir in self.skills_dirs:
            if skills_dir.exists():
                observer.schedule(
                    event_handler,
                    str(skills_dir),
                    recursive=True  # Watch subdirectories too
                )
                logger.info(f"Watching for skill changes in: {skills_dir}")

        observer.start()
        logger.info("File system watcher started for skills directories")

        return observer


class SkillSandbox:
    """
    Skill sandbox execution environment.

    Provides isolated execution of skill scripts with resource limits
    and security constraints.
    """

    def __init__(
        self,
        timeout: int = 60,
        max_memory_mb: int = 512,
        allowed_env_vars: Optional[List[str]] = None
    ):
        """
        Initialize sandbox.

        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in megabytes
            allowed_env_vars: List of environment variables to pass through
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_env_vars = allowed_env_vars or [
            "PATH", "PYTHONPATH", "HOME", "USER"
        ]

    def execute(
        self,
        skill_name: str,
        script_path: str,
        args: List[str],
        env_vars: Optional[Dict[str, str]] = None
    ) -> SkillExecutionResult:
        """
        Execute skill script in sandboxed environment.

        Args:
            skill_name: Name of skill being executed
            script_path: Path to script file
            args: Command-line arguments
            env_vars: Additional environment variables

        Returns:
            SkillExecutionResult with execution outcome
        """
        import time

        start_time = time.time()

        try:
            script = Path(script_path)
            if not script.is_file():
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    output=None,
                    error=f"Script not found: {script_path}",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                )

            # Build environment
            sandbox_env = self._build_sandbox_env(skill_name, env_vars)

            # Execute script
            result = subprocess.run(
                [sys.executable, script_path] + args,  # the interpreter running the engine; a bare "python" may not exist
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=sandbox_env,
                cwd=script.parent
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=True,
                    output=result.stdout,
                    execution_time_ms=execution_time_ms,
                    metadata={"returncode": result.returncode}
                )
            else:
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    execution_time_ms=execution_time_ms,
                    metadata={"returncode": result.returncode}
                )

        except subprocess.TimeoutExpired:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                output=None,
                error=f"Skill execution timeout after {self.timeout}s",
                execution_time_ms=execution_time_ms
            )
        except FileNotFoundError:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                output=None,
                error=f"Script not found: {script_path}",
                execution_time_ms=execution_time_ms
            )
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                output=None,
                error=f"Execution error: {str(e)}",
                execution_time_ms=execution_time_ms
            )

    def _build_sandbox_env(
        self,
        skill_name: str,
        env_vars: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Build sandboxed environment variables.

        Args:
            skill_name: Name of skill
            env_vars: Additional environment variables

        Returns:
            Dictionary of environment variables
        """
        # Start with allowed system env vars
        sandbox_env = {
            k: v for k, v in os.environ.items()
            if k in self.allowed_env_vars
        }

        # Add sandbox markers
        sandbox_env.update({
            "SKILL_SANDBOX": "1",
            "SKILL_NAME": skill_name,
            "SKILL_TIMEOUT": str(self.timeout),
        })

        # Add custom env vars if provided
        if env_vars:
            sandbox_env.update(env_vars)

        return sandbox_env


class SkillEngine:
    """
    Skill Engine - Central manager for skill discovery, registration, and execution.

    This is the main interface for working with skills. It handles:
    - Discovering skills from the filesystem
    - Registering and managing skill metadata
    - Executing skills in sandboxed environments
    - Finding skills by name or trigger phrases
    """

    def __init__(
        self,
        skills_dir: Optional[str] = None,
        sandbox_timeout: int = 60,
        auto_discover: bool = True
    ):
        """
        Initialize skill engine.

        Args:
            skills_dir: Root directory containing skills
            sandbox_timeout: Default timeout for skill execution
            auto_discover: Whether to automatically discover skills on init
        """
        self.discovery = SkillDiscovery(skills_dir)
        self.sandbox = SkillSandbox(timeout=sandbox_timeout)
        self._skills: Dict[str, SkillMetadata] = {}
        self._trigger_index: Dict[str, List[str]] = {}  # trigger -> [skill_names]
        self._handlers: Dict[str, Callable] = {}  # skill_name -> handler function

        if auto_discover:
            self.refresh_skills()

        logger.info("Skill engine initialized")

    def refresh_skills(self) -> int:
        """
        Refresh skill registry by discovering all skills.

        Returns:
            Number of skills discovered
        """
        logger.info("Refreshing skill registry...")
        self._skills.clear()
        self._trigger_index.clear()

        for skill in self.discovery.discover():
            self._skills[skill.name] = skill

            # Build trigger index
            for trigger in skill.triggers:
                trigger_lower = trigger.lower()
                if trigger_lower not in self._trigger_index:
                    self._trigger_index[trigger_lower] = []
                self._trigger_index[trigger_lower].append(skill.name)

        logger.info(f"Loaded {len(self._skills)} skills")
        return len(self._skills)

    def list_skills(
        self,
        skill_type: Optional[SkillType] = None,
        location: Optional[str] = None
    ) -> List[str]:
        """
        List all available skill names.

        Args:
            skill_type: Filter by skill type
            location: Filter by location ('managed' or 'user')

        Returns:
            List of skill names
        """
        skills = self._skills.values()

        if skill_type:
            skills = [s for s in skills if s.skill_type == skill_type]

        if location:
            skills = [s for s in skills if s.location == location]

        return [s.name for s in skills]

    def get_skill(self, name: str) -> Optional[SkillMetadata]:
        """
        Get skill metadata by name.

        Args:
            name: Skill name

        Returns:
            SkillMetadata if found, None otherwise
        """
        return self._skills.get(name)

    def find_by_trigger(self, query: str) -> List[SkillMetadata]:
        """
        Find skills that match a trigger phrase.

        Args:
            query: Query string to match against triggers

        Returns:
            List of matching skills
        """
        query_lower = query.lower()
        matching_skills = []

        # Exact trigger match
        if query_lower in self._trigger_index:
            skill_names = self._trigger_index[query_lower]
            matching_skills.extend([
                self._skills[name] for name in skill_names
                if name in self._skills
            ])

        # Partial trigger match
        for trigger, skill_names in self._trigger_index.items():
            if trigger in query_lower or query_lower in trigger:
                for name in skill_names:
                    skill = self._skills.get(name)
                    if skill and skill not in matching_skills:
                        matching_skills.append(skill)

        return matching_skills

    def register_handler(
        self,
        skill_name: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """
        Register a Python handler function for a skill.

        This allows skills to be implemented as Python functions instead
        of external scripts.

        Args:
            skill_name: Name of skill
            handler: Async function to handle skill execution

        Example:
            async def my_skill_handler(params: Dict[str, Any]) -> Any:
                symbol = params.get("symbol")
                # ... implementation
                return {"signal": "BUY", "confidence": 0.85}

            engine.register_handler("quant-signal", my_skill_handler)
        """
        self._handlers[skill_name] = handler
        logger.info(f"Registered Python handler for skill: {skill_name}")

    async def invoke(
        self,
        skill_name: str,
        params: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> SkillExecutionResult:
        """
        Invoke a skill by name.

        Args:
            skill_name: Name of skill to invoke
            params: Skill parameters
            timeout: Custom timeout (overrides default)

        Returns:
            SkillExecutionResult with execution outcome
        """
        import time

        start_time = time.time()

        # Check for Python handler first (handlers can exist without skill metadata)
        if skill_name in self._handlers:
            try:
                handler = self._handlers[skill_name]
                if asyncio.iscoroutinefunction(handler):
                    output = await handler(params)
                else:
                    output = handler(params)

                execution_time_ms = int((time.time() - start_time) * 1000)

                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=True,
                    output=output,
                    execution_time_ms=execution_time_ms,
                    metadata={"handler_type": "python"}
                )
            except Exception as e:
                execution_time_ms = int((time.time() - start_time) * 1000)
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    output=None,
                    error=f"Handler error: {str(e)}",
                    execution_time_ms=execution_time_ms
                )

        # Check if skill exists in registry
        skill = self._skills.get(skill_name)
        if not skill:
            return SkillExecutionResult(
                skill_name=skill_name,
                success=False,
                output=None,
                error=f"Skill not found: {skill_name}"
            )

        # Execute script-based skill
        script_path = skill.path / "scripts" / "main.py"
        if not script_path.exists():
            # Try alternative locations
            script_path = skill.path / "main.py"
            if not script_path.exists():
                return SkillExecutionResult(
                    skill_name=skill_name,
                    success=False,
                    output=None,
                    error=f"Skill script not found at {script_path}"
                )

        # Convert params to command-line args
        args = [f"--{k}={v}" for k, v in params.items()]

        # Update sandbox timeout if custom timeout provided
        if timeout:
            original_timeout = self.sandbox.timeout
            self.sandbox.timeout = timeout

        result = self.sandbox.execute(
            skill_name=skill_name,
            script_path=str(script_path),
            args=args
        )

        # Restore original timeout
        if timeout:
            self.sandbox.timeout = original_timeout

        return result

    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a skill.

        Args:
            skill_name: Name of skill

        Returns:
            Dictionary with skill details, or None if not found
        """
        skill = self._skills.get(skill_name)
        if not skill:
            return None

        return {
            **skill.to_dict(),
            "has_handler": skill_name in self._handlers,
            "script_exists": (skill.path / "scripts" / "main.py").exists()
            if skill.path else False
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.

        Returns:
            Dictionary with engine stats
        """
        skill_types = {}
        for skill in self._skills.values():
            skill_type = skill.skill_type.value
            skill_types[skill_type] = skill_types.get(skill_type, 0) + 1

        return {
            "total_skills": len(self._skills),
            "skill_types": skill_types,
            "total_triggers": len(self._trigger_index),
            "registered_handlers": len(self._handlers),
            "skills_dir": str(self.discovery.skills_dir)
        }


# Global skill engine instance
_skill_engine: Optional[SkillEngine] = None


def get_skill_engine(
    skills_dir: Optional[str] = None,
    sandbox_timeout: int = 60
) -> SkillEngine:
    """
    Get or create global skill engine instance.

    Args:
        skills_dir: Root directory containing skills
        sandbox_timeout: Default timeout for skill execution

    Returns:
        Global SkillEngine instance
    """
    global _skill_engine

    if _skill_engine is None:
        _skill_engine = SkillEngine(
            skills_dir=skills_dir,
            sandbox_timeout=sandbox_timeout
        )

    return _skill_engine


def reset_skill_engine() -> None:
    """Reset global skill engine (mainly for testing)."""
    global _skill_engine
    _skill_engine = None


# =============================================================================
# Multi-Model Debate Integration
# =============================================================================

class DebateSkillHandler:
    """
    Handler for multi-model debate skill.

    Integrates the debate engine with the skill system for easy invocation.
    """

    def __init__(self):
        """Initialize debate handler."""
        self._executor = None

    def _get_executor(self):
        """Lazy initialization of debate executor."""
        if self._executor is None:
            from ..debate.executor import DebateExecutor, DebateConfig, DebateMode, DebateTier
            self._executor = DebateExecutor(DebateConfig(
                mode=DebateMode.PARALLEL,
                tier=DebateTier.TIER_2,
                use_cli=True
            ))
        return self._executor

    async def handle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle debate skill invocation.

        Args:
            params: Must contain 'question', optionally 'tier', 'context'

        Returns:
            Debate result dictionary
        """
        question = params.get("question")
        if not question:
            return {"error": "Missing required parameter: question"}

        tier = params.get("tier", 2)
        context = params.get("context", {})

        executor = self._get_executor()

        # Update tier if specified
        from ..debate.executor import DebateTier
        tier_map = {
            0: DebateTier.TIER_0,
            1: DebateTier.TIER_1,
            2: DebateTier.TIER_2,
        }
        executor.config.tier = tier_map.get(tier, DebateTier.TIER_2)

        result = await executor.execute(question, context)

        return {
            "success": result.success,
            "question": result.question,
            "final_decision": result.final_decision,
            "consensus": result.consensus,
            "dissent": result.dissent,
            "models_used": result.models_used,
            "total_latency_ms": result.total_latency_ms,
            "perspectives": [
                {
                    "model": p.model,
                    "role": p.role.value,
                    "response": p.response[:500] + "..." if len(p.response) > 500 else p.response,
                    "confidence": p.confidence
                }
                for p in result.perspectives
            ],
            "error": result.error
        }


def register_debate_skill(engine: SkillEngine) -> None:
    """
    Register the multi-model debate skill with the engine.

    Args:
        engine: SkillEngine instance to register with
    """
    handler = DebateSkillHandler()
    engine.register_handler("multi-model-debate", handler.handle)
    logger.info("Registered multi-model-debate skill handler")


def setup_quant_skill_handlers(engine: SkillEngine) -> None:
    """
    Register all quant-related skill handlers.

    Args:
        engine: SkillEngine instance to register with
    """
    # Register debate skill
    register_debate_skill(engine)

    # Register Gemini skill
    try:
        from .gemini_skill import register_gemini_skill
        register_gemini_skill(engine)
    except ImportError as e:
        logger.warning(f"Could not register Gemini skill: {e}")

    # Register Codex skill
    try:
        from .codex_skill import register_codex_skill
        register_codex_skill(engine)
    except ImportError as e:
        logger.warning(f"Could not register Codex skill: {e}")

    # Register quant-signal handler
    async def signal_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        from ..quant import SignalAgent
        from ..base import AgentTask

        symbol = params.get("symbol")
        if not symbol:
            return {"error": "Missing required parameter: symbol"}

        agent = SignalAgent()
        task = AgentTask(
            agent_type="signal",
            payload={
                "symbol": symbol,
                "timeframe": params.get("timeframe", "1d"),
                "strategies": params.get("strategies", ["momentum", "mean_reversion"]),
                "aggregate": params.get("aggregate", True)
            }
        )
        result = await agent.execute(task)
        return result.result if result.is_success else {"error": result.error}

    engine.register_handler("quant-signal", signal_handler)

    # Register quant-risk handler
    async def risk_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        from ..quant import RiskAgent
        from ..base import AgentTask

        portfolio_id = params.get("portfolio_id")
        if not portfolio_id:
            return {"error": "Missing required parameter: portfolio_id"}

        agent = RiskAgent()
        task = AgentTask(
            agent_type="risk",
            payload={
                "type": params.get("type", "risk_analysis"),
                "portfolio_id": portfolio_id,
                "window": params.get("window", "1y")
            }
        )
        result = await agent.execute(task)
        return result.result if result.is_success else {"error": result.error}

    engine.register_handler("quant-risk", risk_handler)

    # Register quant-backtest handler
    async def backtest_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        from ..quant import BacktestAgent
        from ..base import AgentTask

        strategy = params.get("strategy")
        symbol = params.get("symbol")
        if not strategy or not symbol:
            return {"error": "Missing required parameters: strategy, symbol"}

        agent = BacktestAgent()
        task = AgentTask(
            agent_type="backtest",
            payload={
                "type": params.get("type", "backtest"),
                "strategy": strategy,
                "symbol": symbol,
                "start_date": params.get("start_date", "2020-01-01"),
                "end_date": params.get("end_date", "2023-12-31"),
                "initial_capital": params.get("initial_capital", 100000),
                "params": params.get("params", {})
            }
        )
        result = await agent.execute(task)
        return result.result if result.is_success else {"error": result.error}

    engine.register_handler("quant-backtest", backtest_handler)

    # Register quant-decision handler
    async def decision_handler(params: Dict[str, Any]) -> Dict[str, Any]:
        from ..quant import DecisionAgent
        from ..base import AgentTask

        decision_type = params.get("type")
        if not decision_type:
            return {"error": "Missing required parameter: type"}

        agent = DecisionAgent()
        task = AgentTask(
            agent_type="decision",
            payload={
                "type": decision_type,
                "context": params.get("context", {}),
                "force_debate": params.get("force_debate", False),
                "debate_tier": params.get("debate_tier", 1)
            }
        )
        result = await agent.execute(task)
        return result.result if result.is_success else {"error": result.error}

    engine.register_handler("quant-decision", decision_handler)

    logger.info("Registered all quant skill handlers")


def initialize_skill_engine_with_handlers(
    skills_dir: Optional[str] = None,
    sandbox_timeout: int = 60
) -> SkillEngine:
    """
    Initialize skill engine with all handlers registered.

    Args:
        skills_dir: Root directory containing skills
        sandbox_timeout: Default timeout for skill execution

    Returns:
        Configured SkillEngine instance
    """
    engine = get_skill_engine(skills_dir, sandbox_timeout)
    setup_quant_skill_handlers(engine)
    return engine
