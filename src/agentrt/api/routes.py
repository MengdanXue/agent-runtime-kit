# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agent API Routes

Exposes the agent orchestration framework as REST endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
import asyncio
import logging

from ..ports.db import get_async_db
from ..ports.portfolio import PortfolioForbidden, PortfolioNotFound, provider as portfolio_provider

from .. import (
    AgentOrchestrator,
    AgentRegistry,
    AgentPool,
    AgentTask,
    AgentResult,
    AgentStatus,
    AgentPriority,
)
from ..quant import (
    SignalAgent,
    BacktestAgent,
    RiskAgent,
    DecisionAgent,
    OptimizationAgent,
)
from ..debate import DebateEngine, DebateTier, ClaudeAdapter, GeminiAdapter, CodexAdapter, MockAdapter, create_adapter_set
from ..skills import get_skill_engine, SkillEngine
from ..runtime.approval import PROPOSAL_STATUSES, ProposalStoreError, SqlProposalStore
from ..runtime.contract import AgentRun, RunStatus
from ..runtime.llm import default_llm
from ..runtime.loop import AgentLoop
from ..runtime.memory import MemoryStore, MemoryValidationError, PREFERENCE_KEYS, SqlMemoryStore
from ..runtime.planner import Goal
from ..runtime.planning import PlanningService
from ..runtime.recorder import RunLog, SqlRunArchive, legacy_run
from ..runtime.workflow import DecisionWorkflow
from ..tools import ToolContext, default_tools

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Global instances (would be dependency injected in production)
_registry: Optional[AgentRegistry] = None
_pool: Optional[AgentPool] = None
_orchestrator: Optional[AgentOrchestrator] = None
_debate_engine: Optional[DebateEngine] = None


def get_debate_engine() -> DebateEngine:
    """Get or create debate engine with available adapters."""
    global _debate_engine
    if _debate_engine is None:
        # 创建适配器集合
        adapters = create_adapter_set(use_cli=True)

        # 获取各角色的适配器
        claude_adapter = adapters.get("architect")
        gemini_adapter = adapters.get("challenger")
        codex_adapter = adapters.get("validator")

        # 如果 Claude 适配器不可用，使用 Mock
        if not claude_adapter or not claude_adapter.is_available():
            claude_adapter = MockAdapter("claude_mock")

        _debate_engine = DebateEngine(
            claude_adapter=claude_adapter,
            gemini_adapter=gemini_adapter,
            codex_adapter=codex_adapter
        )
    return _debate_engine


def get_registry() -> AgentRegistry:
    """Get or create agent registry."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
        # Register all quant agents
        _registry.register("signal", SignalAgent, ["generate_signal", "multi_signal"])
        _registry.register("backtest", BacktestAgent, ["backtest", "monte_carlo", "walk_forward"])
        _registry.register("risk", RiskAgent, ["risk_analysis", "var_calculation", "stress_test"])
        _registry.register("decision", DecisionAgent, ["portfolio_decision", "strategy_selection"])
        _registry.register("optimization", OptimizationAgent, ["parameter_optimization"])
    return _registry


def get_pool() -> AgentPool:
    """Get or create agent pool."""
    global _pool
    if _pool is None:
        _pool = AgentPool(max_concurrent=10, max_queue_size=100)
    return _pool


def get_orchestrator() -> AgentOrchestrator:
    """Get or create orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(
            registry=get_registry(),
            pool=get_pool()
        )
    return _orchestrator


def get_skills() -> SkillEngine:
    """Get or create skill engine."""
    return get_skill_engine()


_run_log: Optional[RunLog] = None
_runtime_loop: Optional[AgentLoop] = None


def get_run_log() -> RunLog:
    """Contract records of every agent call made through this router."""
    global _run_log
    if _run_log is None:
        _run_log = RunLog()
    return _run_log


_proposal_store: Optional[SqlProposalStore] = None


def get_proposal_store() -> SqlProposalStore:
    """Trade proposals awaiting or past human approval (agent_proposals, Flyway V34)."""
    global _proposal_store
    if _proposal_store is None:
        _proposal_store = SqlProposalStore()
    return _proposal_store


def get_runtime_loop() -> AgentLoop:
    """Observation-driven agent loop over the tool layer."""
    global _runtime_loop
    if _runtime_loop is None:
        _runtime_loop = AgentLoop(default_tools(), run_log=get_run_log(), proposals=get_proposal_store())
    return _runtime_loop


_memory_store: Optional[MemoryStore] = None
_run_archive: Optional[SqlRunArchive] = None


def get_memory_store() -> MemoryStore:
    """Agent memory (agent_memory, Flyway V32)."""
    global _memory_store
    if _memory_store is None:
        _memory_store = SqlMemoryStore()
    return _memory_store


def get_run_archive() -> SqlRunArchive:
    """Persistent run contracts (agent_runs, Flyway V32)."""
    global _run_archive
    if _run_archive is None:
        _run_archive = SqlRunArchive()
    return _run_archive


def get_workflow(loop: AgentLoop = Depends(get_runtime_loop),
                 memory: MemoryStore = Depends(get_memory_store),
                 archive: SqlRunArchive = Depends(get_run_archive)) -> DecisionWorkflow:
    """Multi-agent workflow; shares the loop's tools and proposal store, so its
    proposals are approved through the same ``/runtime/run`` path."""
    llm = default_llm()  # one model client for planner, critic and decision (None: rule paths)
    return DecisionWorkflow(loop.tools, PlanningService(loop.tools, llm), memory,
                            proposals=loop.proposals, run_log=loop.run_log, archive=archive, llm=llm)


def _caller_id(http_request: Optional[Request]) -> Optional[int]:
    return getattr(getattr(http_request, "state", None), "user_id", None)


def record_legacy(http_request: Optional[Request], task: AgentTask, result: Any) -> AgentRun:
    """Record a legacy agent call in the Agent Runtime Contract (behaviour unchanged)."""
    registry = get_registry()
    agent_class = registry.get(task.agent_type) or registry.get_for_task(task.agent_type)
    run = legacy_run(task, result, agent_class.__name__ if agent_class else None,
                     _caller_id(http_request))
    get_run_log().add(run)
    return run


# Request/Response Models

class TaskRequest(BaseModel):
    """Request to dispatch an agent task."""
    agent_type: str = Field(..., description="Type of agent to dispatch")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload")
    priority: int = Field(default=1, ge=0, le=3, description="Task priority (0-3)")
    timeout: int = Field(default=300, ge=1, le=3600, description="Timeout in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_type": "signal",
                "payload": {
                    "symbol": "AAPL",
                    "timeframe": "1d",
                    "strategies": ["momentum", "mean_reversion"]
                },
                "priority": 1,
                "timeout": 60
            }
        }


class TaskResponse(BaseModel):
    """Response from task dispatch."""
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    # Agent Runtime Contract record of this call (additive; existing fields unchanged)
    run: Optional[Dict[str, Any]] = None


class ParallelTaskRequest(BaseModel):
    """Request to dispatch multiple tasks in parallel."""
    tasks: List[TaskRequest]
    max_concurrent: Optional[int] = None


class DebateRequest(BaseModel):
    """Request for multi-model debate."""
    question: str = Field(..., description="Question to debate")
    context: Dict[str, Any] = Field(default_factory=dict)
    tier: Optional[int] = Field(default=None, ge=0, le=2)


class QuantDecisionRequest(BaseModel):
    """Request for quant decision with optional debate."""
    decision_type: str = Field(..., description="Type of decision (portfolio_decision, strategy_selection, etc.)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Decision context")
    portfolio_id: Optional[str] = Field(default=None, description="Portfolio ID")
    force_debate: bool = Field(default=False, description="Force multi-model debate")
    debate_tier: Optional[int] = Field(default=None, ge=0, le=2, description="Override debate tier")

    class Config:
        json_schema_extra = {
            "example": {
                "decision_type": "portfolio_decision",
                "context": {
                    "symbol": "AAPL",
                    "risk_level": "medium",
                    "market_trend": "bullish"
                },
                "force_debate": True,
                "debate_tier": 2
            }
        }


class AgentInfo(BaseModel):
    """Information about a registered agent."""
    agent_type: str
    class_name: str
    task_types: List[str]


class OrchestratorStats(BaseModel):
    """Orchestrator statistics."""
    dispatch_count: int
    success_count: int
    failure_count: int
    success_rate: float
    circuit_breaker_state: str
    pool_stats: Dict[str, Any]


class SkillInfo(BaseModel):
    """Information about a skill."""
    name: str
    description: str
    version: str
    author: str
    triggers: List[str]
    dependencies: List[str]
    skill_type: str
    location: str
    has_handler: bool
    script_exists: bool


class SkillInvokeRequest(BaseModel):
    """Request to invoke a skill."""
    skill_name: str = Field(..., description="Name of skill to invoke")
    params: Dict[str, Any] = Field(default_factory=dict, description="Skill parameters")
    timeout: Optional[int] = Field(default=None, description="Custom timeout in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "skill_name": "quant-signal",
                "params": {
                    "symbol": "AAPL",
                    "timeframe": "1d",
                    "strategies": ["momentum", "mean_reversion"]
                },
                "timeout": 60
            }
        }


class SkillInvokeResponse(BaseModel):
    """Response from skill invocation."""
    skill_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int
    executed_at: str


# API Endpoints

@router.get("/registry", response_model=List[AgentInfo])
async def list_agents(registry: AgentRegistry = Depends(get_registry)):
    """List all registered agent types."""
    return registry.list_with_info()


@router.get("/registry/{agent_type}")
async def get_agent_info(
    agent_type: str,
    registry: AgentRegistry = Depends(get_registry)
):
    """Get information about a specific agent type."""
    agent_class = registry.get(agent_type)
    if not agent_class:
        raise HTTPException(status_code=404, detail=f"Agent type not found: {agent_type}")

    return {
        "agent_type": agent_type,
        "class_name": agent_class.__name__,
        "docstring": agent_class.__doc__
    }


async def authorize_task(
    http_request: Request,
    task_request: TaskRequest,
    db: Any,
) -> None:
    """Apply the portfolio ownership rule to generically dispatched tasks.

    ``/quant/risk/{portfolio_id}`` checks ownership, but ``/dispatch`` routes by
    ``agent_type`` alone, so naming the risk agent directly reached RiskAgent with
    any ``portfolio_id`` and returned another user's positions. The agent class is
    resolved exactly as the orchestrator resolves it, so aliases such as
    ``var_calculation`` cannot slip past.
    """
    registry = get_registry()
    agent_class = registry.get(task_request.agent_type) or registry.get_for_task(
        task_request.agent_type
    )
    if agent_class is not RiskAgent:
        return
    try:
        portfolio_id = RiskAgent._parse_portfolio_id(task_request.payload.get("portfolio_id"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await require_portfolio_owner(http_request, portfolio_id, db)


@router.post("/dispatch", response_model=TaskResponse)
async def dispatch_task(
    request: TaskRequest,
    http_request: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: Any = Depends(get_async_db),
):
    """
    Dispatch a task to an agent.

    The task will be routed to the appropriate agent based on agent_type.
    """
    await authorize_task(http_request, request, db)

    task = AgentTask(
        agent_type=request.agent_type,
        payload=request.payload,
        priority=AgentPriority(request.priority),
        timeout=request.timeout
    )

    try:
        result = await orchestrator.dispatch(task)
        run = record_legacy(http_request, task, result)
        return TaskResponse(
            task_id=result.task_id,
            status=result.status.value,
            result=result.result,
            error=result.error,
            duration_ms=result.duration_ms,
            run=run.to_dict(),
        )
    except Exception as e:
        logger.error(f"Task dispatch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dispatch/parallel", response_model=List[TaskResponse])
async def dispatch_parallel(
    request: ParallelTaskRequest,
    http_request: Request,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: Any = Depends(get_async_db),
):
    """
    Dispatch multiple tasks in parallel.

    All tasks will be executed concurrently, with optional concurrency limit.
    """
    # Every task is authorized before any of them runs.
    for task_request in request.tasks:
        await authorize_task(http_request, task_request, db)

    tasks = [
        AgentTask(
            agent_type=t.agent_type,
            payload=t.payload,
            priority=AgentPriority(t.priority),
            timeout=t.timeout
        )
        for t in request.tasks
    ]

    try:
        results = await orchestrator.dispatch_parallel(
            tasks,
            max_concurrent=request.max_concurrent
        )
        return [
            TaskResponse(
                task_id=r.task_id,
                status=r.status.value,
                result=r.result,
                error=r.error,
                duration_ms=r.duration_ms,
                run=record_legacy(http_request, t, r).to_dict(),
            )
            for t, r in zip(tasks, results)
        ]
    except Exception as e:
        logger.error(f"Parallel dispatch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=OrchestratorStats)
async def get_stats(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Get orchestrator statistics."""
    return orchestrator.get_metrics()


@router.get("/pool/status")
async def get_pool_status(pool: AgentPool = Depends(get_pool)):
    """Get agent pool status."""
    stats = pool.get_stats()
    return {
        "total_capacity": stats.total_capacity,
        "active_count": stats.active_count,
        "idle_count": stats.idle_count,
        "pending_count": stats.pending_count,
        # 转换百分比为小数 (前端期望 0-1 范围)
        "utilization": stats.utilization / 100.0,
        "total_executed": stats.total_executed,
        "total_failed": stats.total_failed,
        "active_agents": pool.get_active_agents()
    }


# Quant-specific endpoints

@router.post("/quant/signal")
async def generate_signal(
    symbol: str,
    http_request: Request,
    timeframe: str = "1d",
    strategies: List[str] = None,
    aggregate: bool = True,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Generate trading signals for a symbol."""
    task = AgentTask(
        agent_type="signal",
        payload={
            "symbol": symbol,
            "timeframe": timeframe,
            "strategies": strategies or ["momentum", "mean_reversion", "macd"],
            "aggregate": aggregate
        }
    )

    result = await orchestrator.dispatch(task)
    record_legacy(http_request, task, result)
    if result.status != AgentStatus.COMPLETED:
        raise HTTPException(status_code=500, detail=result.error)

    return result.result


@router.post("/quant/backtest")
async def run_backtest(
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
    http_request: Request,
    initial_capital: float = 100000,
    backtest_type: str = "backtest",
    params: Dict[str, Any] = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Run a strategy backtest."""
    task = AgentTask(
        agent_type="backtest",
        payload={
            "type": backtest_type,
            "strategy": strategy,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "params": params or {}
        },
        timeout=600  # Backtests may take longer
    )

    result = await orchestrator.dispatch(task)
    record_legacy(http_request, task, result)
    if result.status != AgentStatus.COMPLETED:
        raise HTTPException(status_code=500, detail=result.error)

    return result.result


async def require_portfolio_owner(
    request: Request,
    portfolio_id: int,
    db: Any,
) -> int:
    """Refuse to work on a portfolio the caller does not own.

    Reuses ``portfolio_analytics.verify_portfolio_access`` so the ownership rule
    lives in one place: 404 when the portfolio does not exist, 403 when it
    belongs to someone else.
    """
    user_id = getattr(getattr(request, "state", None), "user_id", None)
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required to analyse a portfolio.",
        )
    try:
        await portfolio_provider().verify_access(portfolio_id, user_id, db)
    except PortfolioNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found") from exc
    except PortfolioForbidden as exc:
        raise HTTPException(status_code=403, detail=f"Access to portfolio {portfolio_id} denied") from exc
    return user_id


@router.post("/quant/risk/{portfolio_id}")
async def analyze_risk(
    portfolio_id: int,
    request: Request,
    analysis_type: str = "risk_analysis",
    confidence_level: float = 0.95,
    window: str = "1y",
    benchmark: str = "SPY",
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: Any = Depends(get_async_db),
):
    """Analyze portfolio risk.

    Metrics are computed by PortfolioAnalyticsService from the portfolio's stored
    positions and price history. Metrics that cannot be computed are listed under
    `unavailable` with the reason rather than substituted.

    The caller must own ``portfolio_id``: the response exposes that portfolio's
    real exposure and per-position risk contribution, so an unchecked id here
    would let any authenticated user read anyone else's book.

    Args:
        portfolio_id: Portfolio to analyze.
        analysis_type: risk_analysis | var_calculation | stress_test | correlation
        confidence_level: VaR confidence, e.g. 0.95 or 0.99.
        window: Historical lookback window (1m/3m/1y/3y).
        benchmark: Benchmark symbol for beta and correlation.
    """
    await require_portfolio_owner(request, portfolio_id, db)

    task = AgentTask(
        agent_type="risk",
        payload={
            "type": analysis_type,
            "portfolio_id": portfolio_id,
            "confidence_level": confidence_level,
            "window": window,
            "benchmark": benchmark
        }
    )

    result = await orchestrator.dispatch(task)
    record_legacy(request, task, result)
    if result.status != AgentStatus.COMPLETED:
        raise HTTPException(status_code=500, detail=result.error)

    return result.result


@router.post("/quant/decision")
async def make_decision(
    request: QuantDecisionRequest,
    http_request: Request,
    debate_engine: DebateEngine = Depends(get_debate_engine)
):
    """Make an AI-powered trading decision with optional multi-model debate."""
    # 直接创建带有 debate_engine 的 DecisionAgent
    agent = DecisionAgent(
        agent_id=f"decision_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        debate_engine=debate_engine
    )

    task = AgentTask(
        agent_type="decision",
        payload={
            "type": request.decision_type,
            "context": request.context,
            "portfolio_id": request.portfolio_id,
            "force_debate": request.force_debate,
            "debate_tier": request.debate_tier
        },
        priority=AgentPriority.HIGH if request.force_debate else AgentPriority.NORMAL
    )

    try:
        result = await agent._execute_with_tracking(task)
        record_legacy(http_request, task, result)
        if result.status != AgentStatus.COMPLETED:
            raise HTTPException(status_code=500, detail=result.error)
        return result.result
    except Exception as e:
        logger.error(f"Decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quant/optimize")
async def optimize_strategy(
    strategy: str,
    symbol: str,
    param_ranges: Dict[str, List[float]],
    http_request: Request,
    objective: str = "sharpe",
    method: str = "bayesian",
    max_iterations: int = 100,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Optimize strategy parameters."""
    task = AgentTask(
        agent_type="optimization",
        payload={
            "type": "parameter_optimization",
            "strategy": strategy,
            "symbol": symbol,
            "param_ranges": param_ranges,
            "objective": objective,
            "method": method,
            "max_iterations": max_iterations
        },
        timeout=600  # Optimization may take longer
    )

    result = await orchestrator.dispatch(task)
    record_legacy(http_request, task, result)
    if result.status != AgentStatus.COMPLETED:
        raise HTTPException(status_code=500, detail=result.error)

    return result.result


# Multi-symbol parallel analysis

@router.post("/quant/multi-signal")
async def multi_symbol_signals(
    symbols: List[str],
    http_request: Request,
    timeframe: str = "1d",
    strategies: List[str] = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Generate signals for multiple symbols in parallel."""
    tasks = [
        AgentTask(
            agent_type="signal",
            payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "strategies": strategies or ["momentum", "mean_reversion"],
                "aggregate": True
            }
        )
        for symbol in symbols
    ]

    results = await orchestrator.dispatch_parallel(tasks)
    for task, result in zip(tasks, results):
        record_legacy(http_request, task, result)

    return {
        "symbols": symbols,
        "results": [
            {
                "symbol": symbols[i],
                "status": r.status.value,
                "signal": r.result.get("aggregated") if r.result else None,
                "error": r.error
            }
            for i, r in enumerate(results)
        ]
    }


# Skill Engine Endpoints

@router.get("/skills", response_model=List[str])
async def list_skills(
    skill_type: Optional[str] = None,
    location: Optional[str] = None,
    skills: SkillEngine = Depends(get_skills)
):
    """
    List all available skills.

    Args:
        skill_type: Filter by skill type (quant, agent, debug, test, general)
        location: Filter by location (managed, user)
    """
    from ..skills import SkillType

    type_filter = None
    if skill_type:
        try:
            type_filter = SkillType(skill_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid skill_type: {skill_type}")

    return skills.list_skills(skill_type=type_filter, location=location)


@router.get("/skills/{skill_name}", response_model=SkillInfo)
async def get_skill_info(
    skill_name: str,
    skills: SkillEngine = Depends(get_skills)
):
    """
    Get detailed information about a skill.

    Args:
        skill_name: Name of the skill
    """
    info = skills.get_skill_info(skill_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")

    return SkillInfo(**info)


@router.post("/skills/invoke", response_model=SkillInvokeResponse)
async def invoke_skill(
    request: SkillInvokeRequest,
    skills: SkillEngine = Depends(get_skills)
):
    """
    Invoke a skill by name.

    This endpoint executes the specified skill with the provided parameters.
    Skills can be implemented as Python handlers or external scripts.

    Example:
        POST /agents/skills/invoke
        {
            "skill_name": "quant-signal",
            "params": {
                "symbol": "AAPL",
                "timeframe": "1d"
            }
        }
    """
    try:
        result = await skills.invoke(
            skill_name=request.skill_name,
            params=request.params,
            timeout=request.timeout
        )

        return SkillInvokeResponse(
            skill_name=result.skill_name,
            success=result.success,
            output=result.output,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            executed_at=result.executed_at.isoformat()
        )
    except Exception as e:
        logger.error(f"Skill invocation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/search")
async def search_skills_by_trigger(
    query: str,
    skills: SkillEngine = Depends(get_skills)
):
    """
    Search for skills by trigger phrase.

    Args:
        query: Search query to match against skill triggers

    Returns:
        List of matching skills with their metadata
    """
    matching_skills = skills.find_by_trigger(query)

    return {
        "query": query,
        "count": len(matching_skills),
        "skills": [
            {
                "name": skill.name,
                "description": skill.description,
                "triggers": skill.triggers,
                "skill_type": skill.skill_type.value
            }
            for skill in matching_skills
        ]
    }


@router.post("/skills/refresh")
async def refresh_skills(skills: SkillEngine = Depends(get_skills)):
    """
    Refresh the skill registry by re-scanning the skills directory.

    This is useful after adding, modifying, or removing skills.
    """
    try:
        count = skills.refresh_skills()
        return {
            "status": "success",
            "message": f"Refreshed skill registry",
            "skills_loaded": count
        }
    except Exception as e:
        logger.error(f"Failed to refresh skills: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills/stats")
async def get_skill_stats(skills: SkillEngine = Depends(get_skills)):
    """
    Get skill engine statistics.

    Returns information about the number of skills, types, and configuration.
    """
    return skills.get_stats()



# Agent Runtime Endpoints
#
# POST /runtime/run              run a goal through the observation-driven loop, or
#                                carry out a proposal the caller approves (PAPER, once)
# GET  /runtime/runs             the caller's recent runs (runtime and legacy)
# GET  /runtime/runs/{id}        one run's full contract (owner only)
# GET  /runtime/proposals        the caller's proposals (agent_proposals), ?status=PENDING...
# GET  /runtime/proposals/{id}   one proposal with its status and linked runs (owner only)
# GET  /runtime/tools            tools with input/output schemas and action levels

class RuntimeTrade(BaseModel):
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0, le=1_000_000)


class RuntimeGoal(BaseModel):
    intent: Literal["analyze_symbol", "portfolio_risk", "strategy_suggestion"]
    symbol: Optional[str] = Field(default=None, min_length=1, max_length=32)
    portfolio_id: Optional[int] = Field(default=None, gt=0)
    days: int = Field(default=180, ge=5, le=3650)
    strategies: Optional[List[str]] = None
    factors: Optional[List[str]] = None
    trade: Optional[RuntimeTrade] = None


class RuntimeRunRequest(BaseModel):
    goal: Optional[RuntimeGoal] = None
    approve_proposal_id: Optional[str] = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "RuntimeRunRequest":
        if (self.goal is None) == (self.approve_proposal_id is None):
            raise ValueError("give exactly one of goal or approve_proposal_id")
        return self


# A FAILED run is not an HTTP success: the status says why it failed and the
# body is still the full contract, so the trace is never lost.
_FAILED_RUN_STATUS = {
    "UNAUTHENTICATED": 401,
    "FORBIDDEN": 403,
    "APPROVAL_FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "APPROVAL_NOT_FOUND": 404,
    "APPROVAL_CONSUMED": 409,
    "APPROVAL_EXPIRED": 410,
    "INVALID_INPUT": 422,
    "INSUFFICIENT_DATA": 422,
    "NEEDS_USER_INPUT": 422,
    "UNAVAILABLE": 503,
    "TIMEOUT": 504,
}


def run_http_status(run: AgentRun) -> int:
    if run.status == RunStatus.SUCCESS:
        return 200
    if run.status == RunStatus.RUNNING:
        return 202
    return _FAILED_RUN_STATUS.get(run.error_code or "", 500)


async def _remember_approval(run: AgentRun, user_id: int, memory: MemoryStore) -> None:
    """An approved (PAPER) proposal becomes a proposal memory; failures are only logged."""
    proposal = (run.data_provenance or {}).get("approved_proposal")
    if run.status != RunStatus.SUCCESS or not proposal or proposal.get("status") != "EXECUTED":
        return
    action = proposal.get("action") or {}
    try:
        await memory.add(user_id, "proposal", {
            "subject": str(action.get("symbol")), "side": action.get("side"),
            "quantity": action.get("quantity"), "status": "APPROVED_PAPER_EXECUTED",
            "proposal_id": proposal.get("proposal_id")},
            source_task_id=proposal.get("source_task_id") or run.task_id, importance=4)
    except Exception as exc:  # memory is context; the execution record stands on its own
        logger.warning("approved proposal not remembered: %s", exc)


@router.post("/runtime/run")
async def runtime_run(
    body: RuntimeRunRequest,
    http_request: Request,
    loop: AgentLoop = Depends(get_runtime_loop),
    archive: SqlRunArchive = Depends(get_run_archive),
    memory: MemoryStore = Depends(get_memory_store),
):
    """Run a goal (Goal -> Plan -> Tool -> Observation -> Decision -> Final).

    EXECUTE actions are never carried out on the agent's own decision: the run
    returns a proposal, and only a later request from the same user naming
    ``approve_proposal_id`` carries it out (PAPER ONLY).
    """
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required to run an agent.")
    ctx = ToolContext(user_id=user_id, authorization=http_request.headers.get("Authorization"))
    if body.approve_proposal_id:
        run = await loop.execute_approved(body.approve_proposal_id, ctx)
        await _remember_approval(run, user_id, memory)
    else:
        goal = body.goal
        run = await loop.run(Goal(
            intent=goal.intent, symbol=goal.symbol, portfolio_id=goal.portfolio_id,
            days=goal.days, strategies=goal.strategies, factors=goal.factors,
            trade=goal.trade.model_dump() if goal.trade else None), ctx)
    await archive.save(run)
    return JSONResponse(status_code=run_http_status(run), content=run.to_dict())


@router.get("/runtime/runs")
async def runtime_runs(http_request: Request, limit: int = 20):
    """The caller's recent runs, newest first."""
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return [{"task_id": r.task_id, "agent": r.agent, "status": r.status.value,
             "stop_reason": r.stop_reason, "action_level": r.action_level.value,
             "started_at": r.started_at}
            for r in get_run_log().recent(user_id, max(1, min(limit, 100)))]


@router.get("/runtime/runs/{task_id}")
async def runtime_run_detail(task_id: str, http_request: Request,
                             archive: SqlRunArchive = Depends(get_run_archive)):
    """Full contract of one run. Runs of other users are indistinguishable from missing ones.

    Recent runs come from memory; older ones (or after a restart) from agent_runs.
    """
    user_id = _caller_id(http_request)
    run = get_run_log().get(task_id, user_id)
    if run is not None:
        return run.to_dict()
    try:
        stored = await archive.load(task_id, user_id)
    except Exception as exc:
        logger.error("run archive unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Run archive unavailable") from exc
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {task_id}")
    return stored


@router.get("/runtime/proposals")
async def runtime_proposals(http_request: Request, status: Optional[str] = None, limit: int = 20,
                            proposals: SqlProposalStore = Depends(get_proposal_store)):
    """The caller's trade proposals, newest first (optionally one status)."""
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    if status is not None and status not in PROPOSAL_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {list(PROPOSAL_STATUSES)}")
    try:
        items = await proposals.list(user_id, status, max(1, min(limit, 100)))
    except ProposalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return [p.to_dict() for p in items]


@router.get("/runtime/proposals/{proposal_id}")
async def runtime_proposal(proposal_id: str, http_request: Request,
                           proposals: SqlProposalStore = Depends(get_proposal_store)):
    """One of the caller's proposals; other users' proposals read as missing."""
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        proposal = await proposals.get(proposal_id, user_id)
    except ProposalStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal not found: {proposal_id}")
    return proposal.to_dict()


@router.get("/runtime/tools")
async def runtime_tools(loop: AgentLoop = Depends(get_runtime_loop)):
    """Tools available to the runtime, with schemas, action levels and error codes."""
    return loop.tools.describe()


# Multi-agent workflow and agent memory
#
# POST /runtime/workflow               goal text -> Memory -> Planner -> Research -> Risk -> Critic -> Decision
#                                      ("async": true returns 202 at once; poll GET /runtime/runs/{id})
# GET  /runtime/workflow/status        which planner/critic/decision path is live, and why
# GET  /runtime/memory                 the caller's constraints and recent memories
# PUT  /runtime/memory/preferences     set one preference (validated whitelist)
# POST /runtime/memory/feedback        the caller's verdict on one of their runs

class WorkflowRequest(BaseModel):
    goal_text: str = Field(..., min_length=1, max_length=500)
    portfolio_id: Optional[int] = Field(default=None, gt=0)
    symbol: Optional[str] = Field(default=None, min_length=1, max_length=32)
    run_async: bool = Field(default=False, alias="async")

    model_config = {"populate_by_name": True}


@router.post("/runtime/workflow")
async def runtime_workflow(body: WorkflowRequest, http_request: Request,
                           workflow: DecisionWorkflow = Depends(get_workflow)):
    """Run a goal through the multi-agent workflow.

    The result is at most PROPOSE / SIMULATE. A requested trade comes back as a
    proposal; it is carried out only if the same user approves it through
    ``/runtime/run`` with ``approve_proposal_id`` (PAPER ONLY). With
    ``"async": true`` the call returns 202 and the contract (status RUNNING) at
    once; ``GET /runtime/runs/{task_id}`` shows it growing until it finishes.
    """
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required to run an agent.")
    ctx = ToolContext(user_id=user_id, authorization=http_request.headers.get("Authorization"))
    if body.run_async:
        run = workflow.start(body.goal_text, ctx, symbol=body.symbol, portfolio_id=body.portfolio_id)
    else:
        run = await workflow.run(body.goal_text, ctx, symbol=body.symbol, portfolio_id=body.portfolio_id)
    return JSONResponse(status_code=run_http_status(run), content=run.to_dict())


@router.get("/runtime/workflow/status")
async def runtime_workflow_status(workflow: DecisionWorkflow = Depends(get_workflow)):
    """Which planner / critic / decision path the workflow will use, and why."""
    ok, reason = workflow.planning.llm_status()
    return {"llm_available": ok,
            "llm_planner_available": ok,
            "llm_model": workflow.planning.llm.model if workflow.planning.llm else None,
            "llm_provider": getattr(workflow.planning.llm, "provider", None),
            "llm_endpoint": getattr(workflow.planning.llm, "endpoint", None),
            "planner": "llm" if ok else "rule",
            "critic": "rule+llm" if ok else "rule",
            "decision": "llm (validated)" if ok else "rule",
            "reason": reason,
            "agents": ["research", "risk", "critic", "decision"],
            "proposal_store": type(workflow.proposals).__name__,
            "execution_mode": "PAPER ONLY",
            "max_action_level": "PROPOSE/SIMULATE (EXECUTE only via human approval)"}


class PreferenceRequest(BaseModel):
    key: Literal["risk_tolerance", "max_single_position_pct", "forbid_auto_trading"]
    value: Any


class FeedbackRequest(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=64)
    verdict: Literal["agree", "disagree"]
    comment: Optional[str] = Field(default=None, max_length=500)


@router.get("/runtime/memory")
async def runtime_memory(http_request: Request, memory: MemoryStore = Depends(get_memory_store)):
    """The caller's agent memory: constraints and the most recent memories."""
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        prefs = await memory.preferences(user_id)
        recent = [m for m in await memory.list(user_id, limit=30) if m.memory_type != "preference"]
    except Exception as exc:
        logger.warning("agent memory unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=f"Agent memory unavailable: {exc}") from exc
    return {"available": True, "preferences": prefs,
            "recent": [m.to_dict() for m in recent[:20]],
            "preference_keys": list(PREFERENCE_KEYS),
            "note": "context only; never used as market data, risk figures or tool results"}


@router.put("/runtime/memory/preferences")
async def runtime_set_preference(body: PreferenceRequest, http_request: Request,
                                 memory: MemoryStore = Depends(get_memory_store)):
    """Set one preference. Values are validated; forbid_auto_trading can only be true."""
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        item = await memory.set_preference(user_id, body.key, body.value)
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return item.to_dict()


@router.post("/runtime/memory/feedback")
async def runtime_feedback(body: FeedbackRequest, http_request: Request,
                           memory: MemoryStore = Depends(get_memory_store),
                           archive: SqlRunArchive = Depends(get_run_archive)):
    """Record the caller's verdict on one of their own workflow runs."""
    user_id = _caller_id(http_request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    run = get_run_log().get(body.task_id, user_id)
    contract = run.to_dict() if run is not None else await archive.load(body.task_id, user_id)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {body.task_id}")
    result = contract.get("result") or {}
    if contract.get("status") != "SUCCESS" or not result.get("recommendation"):
        raise HTTPException(status_code=422, detail="Only a finished run with a conclusion can be judged.")
    subject = result.get("symbol") or f"portfolio:{result.get('portfolio_id')}"
    content = {"subject": subject, "verdict": body.verdict,
               "about_recommendation": result["recommendation"]}
    if body.comment:
        content["comment"] = body.comment
    try:
        item = await memory.add(user_id, "feedback", content, source_task_id=body.task_id, importance=4)
    except MemoryValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return item.to_dict()
