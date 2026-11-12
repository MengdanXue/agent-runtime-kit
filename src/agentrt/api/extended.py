# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Agents Extended API - 代理扩展API端点
Provides additional agent-related endpoints for registry, tasks, and debate history
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import functools
import logging

from agentrt.guards.mock_guard import MOCK_FLAG_ENV_VAR, mock_data_allowed, mock_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["Agents Extended"])


# ==================== 真实数据守卫 ====================
#
# 这个路由从未有过真实实现：下面的 MOCK_* 常量是它唯一的数据来源，而端点却以
# ``success: true`` 返回，声称四个 agent 正在运行、有任务在跑、有辩论历史。
# 真实的 agent 注册表由 ``src/api/agent_routes.py`` 在 ``/api/v1/agents`` 提供，
# 前端用的也是它。
#
# 端点保留挂载（不隐藏入口），但默认明确回答"不可用"并指向真实入口，
# 而不是继续伪装成功。

_REAL_ENDPOINT_HINT = (
    "The real agent registry is served by /api/v1/agents (src/api/agent_routes.py); "
    "this router never had an implementation behind it."
)


def requires_real_agent_data(resource: str):
    """让端点在没有真实数据源时明确失败，而不是返回硬编码样例。"""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not mock_data_allowed():
                raise HTTPException(
                    status_code=501,
                    detail=(
                        f"{resource} is not available: this endpoint has no real data "
                        f"source. {_REAL_ENDPOINT_HINT} "
                        f"Set {MOCK_FLAG_ENV_VAR}=true to return the hardcoded sample "
                        f"for local development only."
                    ),
                )
            result = await func(*args, **kwargs)
            if isinstance(result, dict):
                result.setdefault("meta", {}).update(mock_metadata(resource))
            return result

        return wrapper

    return decorator


# ==================== Response Models ====================

class AgentInfo(BaseModel):
    """代理信息"""
    id: str
    name: str
    type: str
    status: str
    description: str
    capabilities: List[str]


class TaskInfo(BaseModel):
    """任务信息"""
    id: str
    agent_id: str
    task_type: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class DebateEntry(BaseModel):
    """辩论记录"""
    id: str
    topic: str
    participants: List[str]
    started_at: str
    ended_at: Optional[str] = None
    rounds: int
    status: str
    conclusion: Optional[str] = None


# ==================== Mock Data ====================

MOCK_AGENTS = [
    {
        "id": "agent_claude",
        "name": "Claude Decision Agent",
        "type": "decision",
        "status": "active",
        "description": "使用Claude进行交易决策分析",
        "capabilities": ["analysis", "decision_making", "risk_assessment"]
    },
    {
        "id": "agent_gemini",
        "name": "Gemini Analysis Agent",
        "type": "analysis",
        "status": "active",
        "description": "使用Gemini进行市场分析",
        "capabilities": ["market_analysis", "trend_prediction", "sentiment_analysis"]
    },
    {
        "id": "agent_executor",
        "name": "Trade Executor Agent",
        "type": "execution",
        "status": "active",
        "description": "执行交易指令",
        "capabilities": ["order_execution", "position_management", "risk_control"]
    },
    {
        "id": "agent_monitor",
        "name": "Market Monitor Agent",
        "type": "monitoring",
        "status": "active",
        "description": "实时监控市场状态",
        "capabilities": ["market_monitoring", "alert_generation", "data_collection"]
    },
]

MOCK_TASKS = [
    {
        "id": "task_001",
        "agent_id": "agent_claude",
        "task_type": "analyze_symbol",
        "status": "completed",
        "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        "completed_at": (datetime.now() - timedelta(hours=1, minutes=50)).isoformat(),
        "result": {"decision": "buy", "confidence": 0.85}
    },
    {
        "id": "task_002",
        "agent_id": "agent_gemini",
        "task_type": "market_analysis",
        "status": "in_progress",
        "created_at": (datetime.now() - timedelta(minutes=30)).isoformat(),
        "completed_at": None,
        "result": None
    },
    {
        "id": "task_003",
        "agent_id": "agent_executor",
        "task_type": "execute_trade",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result": None
    },
]

MOCK_DEBATES = [
    {
        "id": "debate_001",
        "topic": "BTCUSDT 交易决策",
        "participants": ["agent_claude", "agent_gemini"],
        "started_at": (datetime.now() - timedelta(hours=5)).isoformat(),
        "ended_at": (datetime.now() - timedelta(hours=4, minutes=45)).isoformat(),
        "rounds": 3,
        "status": "completed",
        "conclusion": "建议买入，两个模型均认为风险可控"
    },
    {
        "id": "debate_002",
        "topic": "ETHUSDT 风险评估",
        "participants": ["agent_claude", "agent_gemini"],
        "started_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        "ended_at": (datetime.now() - timedelta(hours=1, minutes=30)).isoformat(),
        "rounds": 4,
        "status": "completed",
        "conclusion": "风险较高，建议观望"
    },
    {
        "id": "debate_003",
        "topic": "投资组合再平衡策略",
        "participants": ["agent_claude", "agent_gemini"],
        "started_at": (datetime.now() - timedelta(minutes=45)).isoformat(),
        "ended_at": None,
        "rounds": 2,
        "status": "in_progress",
        "conclusion": None
    },
]


# ==================== API Endpoints ====================

@router.get("/registry")
@requires_real_agent_data("agent registry")
async def get_agents_registry():
    """
    获取代理注册表

    Returns:
        {
            "success": true,
            "data": {
                "agents": [...],
                "count": 4
            }
        }
    """
    try:
        logger.info("获取代理注册表")

        return {
            "success": True,
            "data": {
                "agents": MOCK_AGENTS,
                "count": len(MOCK_AGENTS),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"获取代理注册表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry/{agent_id}")
@requires_real_agent_data("agent detail")
async def get_agent_info(agent_id: str):
    """
    获取特定代理信息

    Args:
        agent_id: 代理ID

    Returns:
        {
            "success": true,
            "data": {...}
        }
    """
    try:
        agent = next((a for a in MOCK_AGENTS if a["id"] == agent_id), None)

        if not agent:
            raise HTTPException(status_code=404, detail=f"代理 {agent_id} 不存在")

        return {
            "success": True,
            "data": agent
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取代理信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
@requires_real_agent_data("agent task list")
async def get_agent_tasks(
    agent_id: Optional[str] = Query(None, description="代理ID过滤"),
    status: Optional[str] = Query(None, description="任务状态过滤"),
    limit: int = Query(50, ge=1, le=200, description="返回数量")
):
    """
    获取代理任务列表

    Args:
        agent_id: 代理ID过滤（可选）
        status: 任务状态过滤（可选）
        limit: 返回数量

    Returns:
        {
            "success": true,
            "data": {
                "tasks": [...],
                "count": 3
            }
        }
    """
    try:
        logger.info(f"获取代理任务列表: agent_id={agent_id}, status={status}")

        tasks = MOCK_TASKS.copy()

        # 过滤
        if agent_id:
            tasks = [t for t in tasks if t["agent_id"] == agent_id]
        if status:
            tasks = [t for t in tasks if t["status"] == status]

        # 限制数量
        tasks = tasks[:limit]

        return {
            "success": True,
            "data": {
                "tasks": tasks,
                "count": len(tasks),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
@requires_real_agent_data("agent task detail")
async def get_task_detail(task_id: str):
    """
    获取任务详情

    Args:
        task_id: 任务ID

    Returns:
        {
            "success": true,
            "data": {...}
        }
    """
    try:
        task = next((t for t in MOCK_TASKS if t["id"] == task_id), None)

        if not task:
            raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

        return {
            "success": True,
            "data": task
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debate/history")
@requires_real_agent_data("debate history")
async def get_debate_history(
    status: Optional[str] = Query(None, description="状态过滤: completed, in_progress"),
    limit: int = Query(50, ge=1, le=200, description="返回数量")
):
    """
    获取辩论历史记录

    Args:
        status: 状态过滤（可选）
        limit: 返回数量

    Returns:
        {
            "success": true,
            "data": {
                "debates": [...],
                "count": 3
            }
        }
    """
    try:
        logger.info(f"获取辩论历史: status={status}")

        debates = MOCK_DEBATES.copy()

        # 过滤
        if status:
            debates = [d for d in debates if d["status"] == status]

        # 限制数量
        debates = debates[:limit]

        return {
            "success": True,
            "data": {
                "debates": debates,
                "count": len(debates),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"获取辩论历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debate/history/{debate_id}")
@requires_real_agent_data("debate detail")
async def get_debate_detail(debate_id: str):
    """
    获取辩论详情

    Args:
        debate_id: 辩论ID

    Returns:
        {
            "success": true,
            "data": {...}
        }
    """
    try:
        debate = next((d for d in MOCK_DEBATES if d["id"] == debate_id), None)

        if not debate:
            raise HTTPException(status_code=404, detail=f"辩论 {debate_id} 不存在")

        # 添加详细的辩论轮次信息
        detail = {
            **debate,
            "rounds_detail": [
                {
                    "round": 1,
                    "speaker": "agent_claude",
                    "content": "基于技术分析，建议买入",
                    "timestamp": debate["started_at"]
                },
                {
                    "round": 1,
                    "speaker": "agent_gemini",
                    "content": "同意，市场情绪积极",
                    "timestamp": debate["started_at"]
                },
            ] if debate["status"] == "completed" else []
        }

        return {
            "success": True,
            "data": detail
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取辩论详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
@requires_real_agent_data("agent statistics")
async def get_agent_statistics():
    """
    获取代理统计信息

    Returns:
        {
            "success": true,
            "data": {...}
        }
    """
    try:
        stats = {
            "total_agents": len(MOCK_AGENTS),
            "active_agents": len([a for a in MOCK_AGENTS if a["status"] == "active"]),
            "total_tasks": len(MOCK_TASKS),
            "pending_tasks": len([t for t in MOCK_TASKS if t["status"] == "pending"]),
            "in_progress_tasks": len([t for t in MOCK_TASKS if t["status"] == "in_progress"]),
            "completed_tasks": len([t for t in MOCK_TASKS if t["status"] == "completed"]),
            "total_debates": len(MOCK_DEBATES),
            "active_debates": len([d for d in MOCK_DEBATES if d["status"] == "in_progress"]),
            "completed_debates": len([d for d in MOCK_DEBATES if d["status"] == "completed"]),
        }

        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """代理模块健康检查"""
    return {
        "status": "healthy",
        "module": "agents_extended",
        "timestamp": datetime.now().isoformat()
    }
