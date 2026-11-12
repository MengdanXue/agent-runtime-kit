# Copyright 2026 Mengdan Xue
# SPDX-License-Identifier: Apache-2.0
"""Optional dependency handling for the S2c suite.

Only tests marked ``needs_strategies`` (they run the decision workflow, whose research step
needs named strategies from ``agentrt.ports.strategies``, i.e. quantrisk-core) are skipped
when no strategy is available, each with the reason. Every other test always runs.

Set ``AGENTRT_REQUIRE_STRATEGIES=1`` in an environment that is meant to be complete (CI's
quantrisk job): a missing provider then stops the run instead of skipping silently.
"""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_strategies: runs the decision workflow, which needs named strategies from "
        "agentrt.ports.strategies (quantrisk-core); skipped with the reason when none is available")


def pytest_collection_modifyitems(config, items):
    from agentrt.ports.strategies import strategy_names

    if strategy_names():
        return
    if os.getenv("AGENTRT_REQUIRE_STRATEGIES", "").strip().lower() in ("1", "true", "yes"):
        raise pytest.UsageError("AGENTRT_REQUIRE_STRATEGIES=1 but agentrt.ports.strategies offers no "
                                "strategies: install quantrisk-core")
    skip = pytest.mark.skip(reason="needs quantrisk-core: the decision workflow's research step uses named "
                                   "strategies and agentrt.ports.strategies offers none")
    for item in items:
        if "needs_strategies" in item.keywords:
            item.add_marker(skip)
