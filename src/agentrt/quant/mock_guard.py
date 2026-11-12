# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-export of the shared mock guard.

The guard started here, when the only known fabricated-data defect was in the
quant agents. The same defect was later found in the decision API, the report
generator and two routers that never had a real implementation, so the
implementation moved to :mod:`agentrt.guards.mock_guard` and this module forwards to
it. Importing either path enforces the same rule from the same state.
"""

from __future__ import annotations

from agentrt.guards.mock_guard import (
    MOCK_FLAG_ENV_VAR,
    MockDataDisabledError,
    mock_data_allowed,
    mock_metadata,
    require_real_data,
)

__all__ = [
    "MOCK_FLAG_ENV_VAR",
    "MockDataDisabledError",
    "mock_data_allowed",
    "mock_metadata",
    "require_real_data",
]
