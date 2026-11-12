# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Gemini CLI Skill

通过Skill引擎调用Gemini CLI进行AI分析。
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
import subprocess
import json
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Gemini响应"""
    success: bool
    response: str
    model: str = "gemini"
    latency_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response": self.response,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class GeminiSkillHandler:
    """Gemini Skill处理器"""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """检查Gemini CLI是否可用"""
        if self._available is None:
            self._available = shutil.which("gemini") is not None
        return self._available

    async def handle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理Gemini Skill调用

        Args:
            params: {
                "prompt": str,           # 提示词
                "context": str,          # 可选上下文
                "role": str,             # 角色 (analyst, critic, etc.)
                "max_tokens": int,       # 最大输出token
            }

        Returns:
            响应字典
        """
        prompt = params.get("prompt", "")
        context = params.get("context", "")
        role = params.get("role", "analyst")
        max_tokens = params.get("max_tokens", 2000)

        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided",
            }

        # 构建完整提示
        full_prompt = self._build_prompt(prompt, context, role)

        # 调用Gemini
        response = await self._call_gemini(full_prompt)

        return response.to_dict()

    def _build_prompt(self, prompt: str, context: str, role: str) -> str:
        """构建完整提示"""
        role_prompts = {
            "analyst": "You are a senior quantitative analyst. Analyze the following:",
            "critic": "You are a critical reviewer. Challenge and critique the following:",
            "architect": "You are a system architect. Design a solution for:",
            "validator": "You are a technical validator. Assess the feasibility of:",
        }

        role_intro = role_prompts.get(role, "Analyze the following:")

        if context:
            return f"{role_intro}\n\n## Context\n{context}\n\n## Task\n{prompt}"
        else:
            return f"{role_intro}\n\n{prompt}"

    async def _call_gemini(self, prompt: str) -> GeminiResponse:
        """调用Gemini CLI"""
        import time
        start_time = time.time()

        if not self.is_available():
            return GeminiResponse(
                success=False,
                response="",
                error="Gemini CLI not available",
            )

        try:
            # 使用subprocess调用gemini命令
            loop = asyncio.get_event_loop()

            cmd = ["gemini", prompt]

            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
            )

            latency = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return GeminiResponse(
                    success=True,
                    response=result.stdout.strip(),
                    latency_ms=latency,
                )
            else:
                return GeminiResponse(
                    success=False,
                    response="",
                    latency_ms=latency,
                    error=result.stderr or f"Exit code: {result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return GeminiResponse(
                success=False,
                response="",
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return GeminiResponse(
                success=False,
                response="",
                error=str(e),
            )


# ============ Skill注册 ============

_gemini_handler: Optional[GeminiSkillHandler] = None


def get_gemini_handler() -> GeminiSkillHandler:
    """获取Gemini处理器单例"""
    global _gemini_handler
    if _gemini_handler is None:
        _gemini_handler = GeminiSkillHandler()
    return _gemini_handler


def register_gemini_skill(engine) -> None:
    """
    注册Gemini Skill到引擎

    Args:
        engine: SkillEngine实例
    """
    handler = get_gemini_handler()
    engine.register_handler("gemini", handler.handle)
    logger.info("Registered Gemini skill")


# ============ 便捷函数 ============

async def call_gemini(
    prompt: str,
    context: str = "",
    role: str = "analyst",
) -> GeminiResponse:
    """
    便捷函数: 调用Gemini

    Args:
        prompt: 提示词
        context: 上下文
        role: 角色

    Returns:
        GeminiResponse
    """
    handler = get_gemini_handler()
    result = await handler.handle({
        "prompt": prompt,
        "context": context,
        "role": role,
    })
    return GeminiResponse(**result)
