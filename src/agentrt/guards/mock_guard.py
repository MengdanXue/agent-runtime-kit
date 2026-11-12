# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard against silently serving fabricated numbers anywhere in the system.

Several code paths - quant agents, the decision API, the report generator, and a
few routers that never got a real implementation - used to answer with invented
numbers when a real data source was missing. That is the worst failure mode a
trading system can have: the endpoint still answers 200 with a correctly shaped
payload, so nothing downstream can tell the numbers were made up.

This module is the single place that decides whether synthetic data may be
served, so the rule cannot drift between call sites:

* Default (``ALLOW_MOCK_AGENT_DATA`` unset or false) - any attempt to serve
  synthetic numbers raises :class:`MockDataDisabledError`, which callers turn
  into a failed task or an HTTP error.
* Explicitly enabled - synthetic results are still produced for local
  development and tests, but every payload carries ``data_source: "mock"`` so a
  caller can reject it.

``agentrt.quant.mock_guard`` re-exports these names; it was the original home
of the guard, before the same defect was found outside the agents package.
"""

from __future__ import annotations

import os
from typing import Any, Dict

MOCK_FLAG_ENV_VAR = "ALLOW_MOCK_AGENT_DATA"

_TRUE_VALUES = {"1", "true", "yes", "on"}


class MockDataDisabledError(RuntimeError):
    """Raised when a code path would have to fabricate data to answer."""


def mock_data_allowed() -> bool:
    """Whether synthetic data may be substituted for a real measurement.

    The environment is read on every call so tests can flip the flag without
    re-importing the module.
    """
    return os.getenv(MOCK_FLAG_ENV_VAR, "false").strip().lower() in _TRUE_VALUES


def require_real_data(context: str, remedy: str) -> None:
    """Refuse to fabricate ``context`` unless mock data is explicitly allowed.

    Args:
        context: What would have been fabricated, e.g. ``"portfolio VaR"``.
        remedy: How to get the real thing, e.g. ``"Point the agent at a database
            that holds this portfolio's positions."``

    Raises:
        MockDataDisabledError: Unless ``ALLOW_MOCK_AGENT_DATA`` is enabled.
    """
    if mock_data_allowed():
        return
    raise MockDataDisabledError(
        f"Refusing to return fabricated {context}: no real data source is available. "
        f"{remedy} "
        f"Set {MOCK_FLAG_ENV_VAR}=true to allow synthetic values for local development "
        f"only - the numbers are random and must never reach a trading decision."
    )


def mock_metadata(context: str) -> Dict[str, Any]:
    """Marker to attach to any payload that was built from synthetic data."""
    return {
        "data_source": "mock",
        "warning": f"{context} is randomly generated, not a real measurement",
    }


__all__ = [
    "MOCK_FLAG_ENV_VAR",
    "MockDataDisabledError",
    "mock_data_allowed",
    "mock_metadata",
    "require_real_data",
]
