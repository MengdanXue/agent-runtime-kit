# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Model Adapters for Multi-Model Debate Engine

Provides unified interface to different AI model APIs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import asyncio
import time
import logging
import os

from .engine import ModelRole, ModelResponse

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    All model adapters must implement the query() method to provide
    a unified interface for the debate engine.
    """

    @abstractmethod
    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """
        Query the model with a prompt.

        Args:
            prompt: The prompt to send
            role: The role for this query
            context: Optional additional context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            ModelResponse with the result
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model API is available."""
        pass


class ClaudeAdapter(ModelAdapter):
    """
    Adapter for Anthropic Claude API.

    The primary model in debates, serving as the Architect.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229"
    ):
        """
        Initialize Claude adapter.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic package not installed")
                raise
        return self._client

    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Query Claude API."""
        start_time = time.time()

        try:
            client = self._get_client()

            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"

            # Run sync API in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=temperature
                )
            )

            latency = int((time.time() - start_time) * 1000)

            return ModelResponse(
                model="claude",
                role=role,
                response=response.content[0].text,
                confidence=0.9,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency,
                metadata={"model_version": self.model}
            )

        except Exception as e:
            logger.error(f"Claude query failed: {e}")
            return ModelResponse(
                model="claude",
                role=role,
                response=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return bool(self.api_key)


class GeminiAdapter(ModelAdapter):
    """
    Adapter for Google Gemini API.

    Serves as the Challenger in debates.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro"
    ):
        """
        Initialize Gemini adapter.

        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                logger.error("google-generativeai package not installed")
                raise
        return self._client

    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Query Gemini API."""
        start_time = time.time()

        try:
            model = self._get_client()

            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"

            # Gemini's async API
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    full_prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature
                    }
                )
            )

            latency = int((time.time() - start_time) * 1000)

            return ModelResponse(
                model="gemini",
                role=role,
                response=response.text,
                confidence=0.85,
                latency_ms=latency,
                metadata={"model_version": self.model}
            )

        except Exception as e:
            logger.error(f"Gemini query failed: {e}")
            return ModelResponse(
                model="gemini",
                role=role,
                response=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )

    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        return bool(self.api_key)


class CodexAdapter(ModelAdapter):
    """
    Adapter for OpenAI Codex/GPT API.

    Serves as the Validator in debates.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4"
    ):
        """
        Initialize Codex adapter.

        Args:
            api_key: OpenAI API key (or from OPENAI_API_KEY env)
            model: Model to use (gpt-4 or codex variants)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client

    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Query OpenAI API."""
        start_time = time.time()

        try:
            client = self._get_client()

            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )

            latency = int((time.time() - start_time) * 1000)

            return ModelResponse(
                model="codex",
                role=role,
                response=response.choices[0].message.content,
                confidence=0.85,
                tokens_used=response.usage.total_tokens,
                latency_ms=latency,
                metadata={"model_version": self.model}
            )

        except Exception as e:
            logger.error(f"Codex query failed: {e}")
            return ModelResponse(
                model="codex",
                role=role,
                response=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )

    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)


class MockAdapter(ModelAdapter):
    """
    Mock adapter for testing.

    Returns predefined responses without making API calls.
    """

    def __init__(self, model_name: str = "mock", responses: Dict[str, str] = None):
        self.model_name = model_name
        self.responses = responses or {}
        self.query_count = 0

    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Return mock response."""
        self.query_count += 1

        # Check for preset response
        response_text = self.responses.get(
            role.value,
            f"Mock {role.value} response for: {prompt[:50]}..."
        )

        return ModelResponse(
            model=self.model_name,
            role=role,
            response=response_text,
            confidence=0.8,
            tokens_used=100,
            latency_ms=50,
            metadata={"mock": True}
        )

    def is_available(self) -> bool:
        """Mock is always available."""
        return True


class GeminiCLIAdapter(ModelAdapter):
    """
    Adapter for Gemini CLI tool.

    Uses the `gemini` CLI command for model queries.
    """

    def __init__(self, timeout: int = 120):
        """
        Initialize Gemini CLI adapter.

        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout
        self._available = None

    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Query Gemini via CLI."""
        import subprocess
        import shlex

        start_time = time.time()

        try:
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"

            # Escape prompt for shell
            # Use subprocess with proper quoting
            cmd = ["gemini", full_prompt]

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    shell=False
                )
            )

            latency = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return ModelResponse(
                    model="gemini",
                    role=role,
                    response=result.stdout.strip(),
                    confidence=0.85,
                    latency_ms=latency,
                    metadata={"source": "cli", "returncode": result.returncode}
                )
            else:
                return ModelResponse(
                    model="gemini",
                    role=role,
                    response=f"[CLI Error: {result.stderr}]",
                    confidence=0.0,
                    latency_ms=latency,
                    metadata={"error": result.stderr, "returncode": result.returncode}
                )

        except subprocess.TimeoutExpired:
            logger.error("Gemini CLI timeout")
            return ModelResponse(
                model="gemini",
                role=role,
                response="[Error: CLI timeout]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "timeout"}
            )
        except FileNotFoundError:
            logger.error("Gemini CLI not found")
            return ModelResponse(
                model="gemini",
                role=role,
                response="[Error: gemini CLI not found in PATH]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "not_found"}
            )
        except Exception as e:
            logger.error(f"Gemini CLI failed: {e}")
            return ModelResponse(
                model="gemini",
                role=role,
                response=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )

    def is_available(self) -> bool:
        """Check if Gemini CLI is available."""
        if self._available is None:
            import shutil
            self._available = shutil.which("gemini") is not None
        return self._available


class CodexCLIAdapter(ModelAdapter):
    """
    Adapter for Codex CLI tool.

    Uses the `codex` CLI command for model queries.
    """

    def __init__(self, timeout: int = 120):
        """
        Initialize Codex CLI adapter.

        Args:
            timeout: Command timeout in seconds
        """
        self.timeout = timeout
        self._available = None

    async def query(
        self,
        prompt: str,
        role: ModelRole,
        context: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> ModelResponse:
        """Query Codex via CLI."""
        import subprocess

        start_time = time.time()

        try:
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"

            cmd = ["codex", full_prompt]

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    shell=False
                )
            )

            latency = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return ModelResponse(
                    model="codex",
                    role=role,
                    response=result.stdout.strip(),
                    confidence=0.85,
                    latency_ms=latency,
                    metadata={"source": "cli", "returncode": result.returncode}
                )
            else:
                return ModelResponse(
                    model="codex",
                    role=role,
                    response=f"[CLI Error: {result.stderr}]",
                    confidence=0.0,
                    latency_ms=latency,
                    metadata={"error": result.stderr, "returncode": result.returncode}
                )

        except subprocess.TimeoutExpired:
            logger.error("Codex CLI timeout")
            return ModelResponse(
                model="codex",
                role=role,
                response="[Error: CLI timeout]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "timeout"}
            )
        except FileNotFoundError:
            logger.error("Codex CLI not found")
            return ModelResponse(
                model="codex",
                role=role,
                response="[Error: codex CLI not found in PATH]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": "not_found"}
            )
        except Exception as e:
            logger.error(f"Codex CLI failed: {e}")
            return ModelResponse(
                model="codex",
                role=role,
                response=f"[Error: {str(e)}]",
                confidence=0.0,
                latency_ms=int((time.time() - start_time) * 1000),
                metadata={"error": str(e)}
            )

    def is_available(self) -> bool:
        """Check if Codex CLI is available."""
        if self._available is None:
            import shutil
            self._available = shutil.which("codex") is not None
        return self._available


def create_adapter_set(use_cli: bool = True) -> Dict[str, ModelAdapter]:
    """
    Create a set of adapters for multi-model debate.

    Args:
        use_cli: If True, prefer CLI adapters over API adapters

    Returns:
        Dict mapping role to adapter
    """
    adapters = {}

    # Claude - always use API (we're running inside Claude)
    claude = ClaudeAdapter()
    adapters["architect"] = claude

    # Gemini - prefer CLI if available and requested
    if use_cli:
        gemini_cli = GeminiCLIAdapter()
        if gemini_cli.is_available():
            adapters["challenger"] = gemini_cli
        else:
            gemini_api = GeminiAdapter()
            if gemini_api.is_available():
                adapters["challenger"] = gemini_api
            else:
                adapters["challenger"] = MockAdapter("gemini_mock")
    else:
        gemini_api = GeminiAdapter()
        adapters["challenger"] = gemini_api if gemini_api.is_available() else MockAdapter("gemini_mock")

    # Codex - prefer CLI if available and requested
    if use_cli:
        codex_cli = CodexCLIAdapter()
        if codex_cli.is_available():
            adapters["validator"] = codex_cli
        else:
            codex_api = CodexAdapter()
            if codex_api.is_available():
                adapters["validator"] = codex_api
            else:
                adapters["validator"] = MockAdapter("codex_mock")
    else:
        codex_api = CodexAdapter()
        adapters["validator"] = codex_api if codex_api.is_available() else MockAdapter("codex_mock")

    return adapters
