# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent reports: each workflow agent's structured, immutable output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AgentReport:
    """One agent's structured output. Immutable: findings are handed out as copies,
    so no agent can change another agent's report."""

    agent: str
    status: str   # research: OK|FAILED  risk: OK|VETO|UNASSESSED  critic: APPROVE|REQUEST_REVISION  decision: DECIDED
    summary: str
    evidence: Tuple[str, ...]
    findings_json: str
    error_code: Optional[str] = None
    round: int = 1

    @property
    def findings(self) -> Dict[str, Any]:
        return json.loads(self.findings_json)

    def to_dict(self) -> Dict[str, Any]:
        return {"agent": self.agent, "status": self.status, "summary": self.summary,
                "evidence": list(self.evidence), "findings": self.findings,
                "error_code": self.error_code, "round": self.round}


def make_report(agent: str, status: str, summary: str, findings: Dict[str, Any],
                evidence: List[str], error_code: Optional[str] = None, round: int = 1) -> AgentReport:
    return AgentReport(agent=agent, status=status, summary=summary,
                       evidence=tuple(dict.fromkeys(evidence)),
                       findings_json=json.dumps(findings, ensure_ascii=False, default=str),
                       error_code=error_code, round=round)
