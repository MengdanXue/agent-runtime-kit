# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/demo/agent_scenario.py checks the contract the workflow really returns."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from .test_proposal_flow import over_limit
from .workflow_fixtures import calm_uptrend, make_workflow, owner_ctx  # noqa: F401

pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

_PATH = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "agent_scenario.py"
_spec = importlib.util.spec_from_file_location("agent_scenario", _PATH)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


@pytest.mark.asyncio
async def test_demo_checks_pass_on_the_real_demo_run(make_workflow):
    h = await over_limit(make_workflow)
    goal = demo.DEMO_GOAL.format(symbol="AAPL", quantity=5.0)
    run = (await h.run(goal, portfolio_id=7)).to_dict()
    checks = demo.check_run(run, "AAPL", 5.0)
    assert [name for name, ok, _ in checks if not ok] == []

    order = run["result"]["trade"]["proposal"]["action"]
    approved = await h.loop.execute_approved(run["result"]["trade"]["proposal"]["proposal_id"], owner_ctx())
    assert [n for n, ok, _ in demo.check_approval(200, approved.to_dict(), order) if not ok] == []


@pytest.mark.asyncio
async def test_demo_checks_fail_when_the_scenario_does_not_hold(make_workflow):
    h = await over_limit(make_workflow)
    run = (await h.run(demo.DEMO_GOAL.format(symbol="AAPL", quantity=50.0), portfolio_id=7)).to_dict()  # short sale
    failed = [name for name, ok, _ in demo.check_run(run, "AAPL", 50.0) if not ok]
    assert "proposal awaits approval (PAPER)" in failed
    assert [n for n, ok, _ in demo.check_preferences({"preferences": {}}) if not ok] == [
        "single-position limit in memory", "auto trading forbidden in memory"]
