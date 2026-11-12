# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Codex CLI Skill

通过Skill引擎调用Codex CLI进行代码分析和生成。
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
class CodexResponse:
    """Codex响应"""
    success: bool
    response: str
    model: str = "codex"
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


class CodexSkillHandler:
    """Codex Skill处理器"""

    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        self._available = None

    def is_available(self) -> bool:
        """检查Codex CLI是否可用"""
        if self._available is None:
            self._available = shutil.which("codex") is not None
        return self._available

    async def handle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理Codex Skill调用

        Args:
            params: {
                "prompt": str,           # 提示词
                "code_context": str,     # 代码上下文
                "language": str,         # 编程语言
                "task": str,             # 任务类型 (analyze, generate, review, fix)
                "files": List[str],      # 相关文件路径
            }

        Returns:
            响应字典
        """
        prompt = params.get("prompt", "")
        code_context = params.get("code_context", "")
        language = params.get("language", "python")
        task = params.get("task", "analyze")
        files = params.get("files", [])

        if not prompt:
            return {
                "success": False,
                "error": "No prompt provided",
            }

        # 构建完整提示
        full_prompt = self._build_prompt(prompt, code_context, language, task, files)

        # 调用Codex
        response = await self._call_codex(full_prompt)

        return response.to_dict()

    def _build_prompt(
        self,
        prompt: str,
        code_context: str,
        language: str,
        task: str,
        files: List[str],
    ) -> str:
        """构建完整提示"""
        task_prompts = {
            "analyze": f"Analyze the following {language} code:",
            "generate": f"Generate {language} code for:",
            "review": f"Review the following {language} code for issues and improvements:",
            "fix": f"Fix the bugs in the following {language} code:",
            "optimize": f"Optimize the following {language} code for performance:",
            "document": f"Generate documentation for the following {language} code:",
            "test": f"Generate unit tests for the following {language} code:",
        }

        task_intro = task_prompts.get(task, f"Analyze the following {language} code:")

        parts = [task_intro]

        if code_context:
            parts.append(f"\n```{language}\n{code_context}\n```")

        if files:
            parts.append(f"\n## Related Files\n" + "\n".join(f"- {f}" for f in files))

        parts.append(f"\n## Task\n{prompt}")

        return "\n".join(parts)

    async def _call_codex(self, prompt: str) -> CodexResponse:
        """调用Codex CLI"""
        import time
        start_time = time.time()

        if not self.is_available():
            return CodexResponse(
                success=False,
                response="",
                error="Codex CLI not available",
            )

        try:
            loop = asyncio.get_event_loop()

            cmd = ["codex", prompt]

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
                return CodexResponse(
                    success=True,
                    response=result.stdout.strip(),
                    latency_ms=latency,
                )
            else:
                return CodexResponse(
                    success=False,
                    response="",
                    latency_ms=latency,
                    error=result.stderr or f"Exit code: {result.returncode}",
                )

        except subprocess.TimeoutExpired:
            return CodexResponse(
                success=False,
                response="",
                error=f"Timeout after {self.timeout}s",
            )
        except Exception as e:
            return CodexResponse(
                success=False,
                response="",
                error=str(e),
            )


# ============ Skill注册 ============

_codex_handler: Optional[CodexSkillHandler] = None


def get_codex_handler() -> CodexSkillHandler:
    """获取Codex处理器单例"""
    global _codex_handler
    if _codex_handler is None:
        _codex_handler = CodexSkillHandler()
    return _codex_handler


def register_codex_skill(engine) -> None:
    """
    注册Codex Skill到引擎

    Args:
        engine: SkillEngine实例
    """
    handler = get_codex_handler()
    engine.register_handler("codex", handler.handle)
    logger.info("Registered Codex skill")


# ============ 便捷函数 ============

async def call_codex(
    prompt: str,
    code_context: str = "",
    language: str = "python",
    task: str = "analyze",
) -> CodexResponse:
    """
    便捷函数: 调用Codex

    Args:
        prompt: 提示词
        code_context: 代码上下文
        language: 编程语言
        task: 任务类型

    Returns:
        CodexResponse
    """
    handler = get_codex_handler()
    result = await handler.handle({
        "prompt": prompt,
        "code_context": code_context,
        "language": language,
        "task": task,
    })
    return CodexResponse(**result)
