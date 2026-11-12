# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured LLM client for the planner, the critic and the decision agent.

The model is only ever asked to fill a JSON schema (structured outputs,
``output_config.format``), so its answer is parsed, validated by the caller and
either accepted or rejected; free text never reaches the runtime, and nothing
the model returns is executed without the caller's validator.

Uses the official ``anthropic`` SDK against the Messages API of a known
provider. Configuration:

* ``AGENT_LLM_PROVIDER``   - ``anthropic`` (default) or ``deepseek``. Any other
                             value is an unknown provider: the model is not used.
* ``AGENT_LLM_API_KEY``    - the provider's key. For Anthropic ``ANTHROPIC_API_KEY``
                             is also read; it is never sent to another provider.
                             Without a key every caller uses its rule-based path.
* ``AGENT_LLM_MODEL``      - model id (default per provider: ``claude-opus-5`` /
                             ``deepseek-chat``)
* ``AGENT_LLM_BASE_URL``   - API base (default per provider: https://api.anthropic.com,
                             https://api.deepseek.com/anthropic). Its host must be
                             the provider's official host, otherwise the provider is
                             reported as ``unknown:<host>`` or ``mismatch:...`` and
                             the model is not used - a compatible endpoint can never
                             pass for another provider. The generic
                             ``ANTHROPIC_BASE_URL`` is deliberately not read: in
                             developer sandboxes it points at a tool's own proxy.
* ``AGENT_LLM_EFFORT``     - low | medium | high (default medium; Anthropic only)

On Anthropic, answers use native structured outputs and declined requests are
re-run server-side on the recommended fallback model (``fallbacks: "default"``).
Other providers get the JSON schema in the system prompt; either way the answer
must parse as a JSON object, and every caller validates it before use.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Protocol, Tuple

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BASE_URL = "https://api.anthropic.com"
# Known providers: their official API host, default endpoint and model, and the
# model family a served model must belong to.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "anthropic": {"hosts": ("api.anthropic.com",), "base_url": DEFAULT_BASE_URL,
                  "model": DEFAULT_MODEL, "family": "claude", "native_structured_output": True,
                  "max_tokens": 16000},
    "deepseek": {"hosts": ("api.deepseek.com",), "base_url": "https://api.deepseek.com/anthropic",
                 "model": "deepseek-chat", "family": "deepseek", "native_structured_output": False,
                 "max_tokens": 8000},
}
DEFAULT_EFFORT = "medium"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000


class LLMError(Exception):
    """The model could not produce a structured answer."""


class LLMOutputError(LLMError):
    """The model answered, but not with a usable JSON object (truncated, no JSON,
    invalid JSON, not an object) - an invalid answer rather than an unavailable model."""


class StructuredLLM(Protocol):
    model: str

    def available(self) -> Tuple[bool, Optional[str]]: ...

    async def generate(self, system: str, prompt: str, schema_name: str,
                       schema: Dict[str, Any]) -> Dict[str, Any]: ...


class AnthropicStructuredLLM:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None, effort: Optional[str] = None,
                 http_client: Any = None, timeout: float = 90.0, max_retries: int = 2,
                 provider: Optional[str] = None):
        declared = (provider or os.getenv("AGENT_LLM_PROVIDER") or "").strip().lower()
        self.declared_provider = declared or None
        spec = PROVIDERS.get(declared or "anthropic", {})
        self.model = model or os.getenv("AGENT_LLM_MODEL") or spec.get("model") or DEFAULT_MODEL
        self.base_url = (base_url or os.getenv("AGENT_LLM_BASE_URL") or spec.get("base_url")
                         or DEFAULT_BASE_URL).rstrip("/")
        if api_key is None:
            api_key = os.getenv("AGENT_LLM_API_KEY")
            if not api_key and self.provider == "anthropic":  # an Anthropic key only ever goes to Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api_key = api_key
        self.effort = effort or os.getenv("AGENT_LLM_EFFORT") or DEFAULT_EFFORT
        self._http_client = http_client
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Any = None
        self.last_call: Dict[str, Any] = {}

    @property
    def endpoint(self) -> str:
        return self.base_url

    @property
    def provider(self) -> str:
        """Who answers, from the endpoint's host: a known provider only on its
        official host and only if it is the provider that was declared."""
        at_endpoint = provider_of(self.base_url)
        declared = self.declared_provider
        if declared and declared not in PROVIDERS:
            return f"unknown:{declared}"
        if declared and at_endpoint in PROVIDERS and at_endpoint != declared:
            return f"mismatch:{declared}@{_host(self.base_url)}"
        return at_endpoint

    def provider_problem(self) -> Optional[str]:
        provider = self.provider
        if provider.startswith("unknown:"):
            return f"unknown provider {provider.split(':', 1)[1]!r}"
        if provider.startswith("mismatch:"):
            return (f"provider {self.declared_provider!r} does not match the endpoint "
                    f"{_host(self.base_url)!r}")
        return None

    def available(self) -> Tuple[bool, Optional[str]]:
        problem = self.provider_problem()
        if problem:
            return False, f"model unavailable: {problem}"
        if not self.api_key:
            key = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "AGENT_LLM_API_KEY"
            return False, f"model unavailable: {key} is not configured"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "model unavailable: the anthropic SDK is not installed"
        return True, None

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            kwargs: Dict[str, Any] = {"api_key": self.api_key, "base_url": self.base_url,
                                      "timeout": self._timeout, "max_retries": self._max_retries}
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def generate(self, system: str, prompt: str, schema_name: str,
                       schema: Dict[str, Any]) -> Dict[str, Any]:
        ok, reason = self.available()
        if not ok:
            raise LLMError(reason)
        import anthropic

        spec = PROVIDERS[self.provider]
        self.last_call = {"model": self.model, "provider": self.provider, "endpoint": self.base_url,
                          "request_id": None, "stop_reason": None, "usage": None}
        try:
            if spec["native_structured_output"]:
                response = await self._get_client().beta.messages.create(
                    model=self.model,
                    max_tokens=spec["max_tokens"],
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"effort": self.effort,
                                   "format": {"type": "json_schema", "schema": schema}},
                    betas=[FALLBACK_BETA],
                    fallbacks="default",
                )
            else:  # the schema goes into the instructions; the answer is still parsed and validated
                response = await self._get_client().messages.create(
                    model=self.model,
                    max_tokens=spec["max_tokens"],
                    system=system + "\n\nRespond with only one JSON object (no prose, no code fence) "
                                    "that matches this JSON Schema:\n" + json.dumps(schema),
                    messages=[{"role": "user", "content": prompt}],
                )
        except anthropic.APITimeoutError as exc:
            raise LLMError(f"model call timed out ({schema_name})") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"model endpoint unreachable: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("model rate limited (429)") from exc
        except anthropic.APIStatusError as exc:
            self.last_call["request_id"] = getattr(exc, "request_id", None)
            raise LLMError(f"model call failed: HTTP {exc.status_code} {str(exc)[:200]}") from exc

        self.last_call = {"model": getattr(response, "model", self.model),
                          "provider": self.provider, "endpoint": self.base_url,
                          "request_id": getattr(response, "_request_id", None),
                          "stop_reason": response.stop_reason,
                          "usage": _usage(response)}
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"model declined the {schema_name} request (refusal: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMOutputError(f"model output truncated ({schema_name})")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMOutputError(f"model returned no {schema_name} JSON")
        text = text.strip()
        if text.startswith("```"):  # tolerate a fenced answer; the content is still validated
            text = text.strip("`").removeprefix("json").strip()
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise LLMOutputError(f"model returned invalid JSON for {schema_name}") from exc
        if not isinstance(data, dict):
            raise LLMOutputError(f"model returned a non-object {schema_name}")
        return data


def _usage(response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {"input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None)}


def _host(base_url: str) -> str:
    from urllib.parse import urlparse

    return (urlparse(base_url).hostname or "").lower() or base_url


def provider_of(base_url: str) -> str:
    """The known provider whose official host this is, else ``unknown:<host>``."""
    host = _host(base_url)
    for name, spec in PROVIDERS.items():
        if host in spec["hosts"]:
            return name
    return f"unknown:{host}"


def default_llm() -> Optional[AnthropicStructuredLLM]:
    """The configured model client, or None when no model is configured at all.

    A client that is configured but unusable (unknown provider, missing key)
    is returned too, so the status endpoint and every fallback carry its reason.
    """
    client = AnthropicStructuredLLM()
    configured = client.api_key or os.getenv("AGENT_LLM_PROVIDER") or os.getenv("AGENT_LLM_BASE_URL")
    return client if client.available()[0] or configured else None


def llm_status(llm: Optional[StructuredLLM]) -> Tuple[bool, Optional[str]]:
    """Availability of an optional model, with the reason the rule path is used."""
    if llm is None:
        return False, "model unavailable: ANTHROPIC_API_KEY is not configured"
    return llm.available()
