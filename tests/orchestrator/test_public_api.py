"""Public API contract tests for the orchestrator package."""


def test_agent_result_is_exported_from_orchestrator_package():
    import agentrt.orchestrator as orchestrator
    from agentrt.orchestrator.agent_base import AgentResult

    assert orchestrator.AgentResult is AgentResult
    assert "AgentResult" in orchestrator.__all__
