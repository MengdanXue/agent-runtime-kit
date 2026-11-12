# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal agent loop:  Goal -> Plan -> Tool call -> Observation -> Decision -> Final.

The loop owns the mechanics - calling tools, the approval gate, recording the
contract - and asks the planner for the next step after every observation. It
never decides what to do itself, and the planner never touches a tool directly.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Set

from ..tools.base import ToolContext, ToolErrorCode, ToolRegistry, ToolResult
from ..tools.execution import order_fingerprint
from .approval import ApprovalError, ProposalStore, ProposalStoreError, gate
from .contract import ActionLevel, AgentRun, Observation, RunStatus, ToolCall
from .planner import AskUser, CallTool, Finish, Goal, LoopState, RuleBasedPlanner, Step, Stop
from .recorder import RunLog

AGENT_NAME = "traderx.runtime.agent"
MAX_STEPS = 20


def summarize_input(payload: Dict[str, Any], input_call_ids: List[str]) -> Dict[str, Any]:
    """Tool input for the contract; bar series are referenced, not copied."""
    out = dict(payload)
    if isinstance(out.get("bars"), list):
        out["bars"] = {"count": len(out["bars"]), "from_calls": list(input_call_ids)}
    return out


def summarize_output(tool: str, output: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if output is None:
        return None
    if tool == "market_data":
        return {k: v for k, v in output.items() if k != "bars"}
    return output


class AgentLoop:
    def __init__(self, tools: ToolRegistry, planner: Optional[RuleBasedPlanner] = None,
                 proposals: Optional[ProposalStore] = None, run_log: Optional[RunLog] = None,
                 max_steps: int = MAX_STEPS):
        self.tools = tools
        self.planner = planner or RuleBasedPlanner()
        self.proposals = proposals or ProposalStore()
        self.run_log = run_log or RunLog()
        self.max_steps = max_steps

    # ------------------------------------------------------------------------ run

    async def run(self, goal: Goal, ctx: ToolContext) -> AgentRun:
        run = AgentRun(agent=AGENT_NAME, input={"goal": goal.to_dict()}, owner_user_id=ctx.user_id)
        ctx = replace(ctx, task_id=run.task_id, approval=None)  # a goal never carries an approval
        run.event("goal", f"goal received: {goal.intent}", goal=goal.to_dict())
        state = LoopState(goal=goal)
        try:
            for _ in range(self.max_steps):
                action = self.planner.next_action(state)
                if isinstance(action, CallTool):
                    done = await self._call_tool(run, state, action, ctx)
                    if done:
                        return done
                    continue
                return self._conclude(run, state, action)
            return run.finish(RunStatus.FAILED, {"error": "step budget exhausted"},
                              f"no conclusion within {self.max_steps} steps", "STEP_BUDGET_EXHAUSTED")
        finally:
            self.run_log.add(run)

    async def _call_tool(self, run: AgentRun, state: LoopState, action: CallTool,
                         ctx: ToolContext) -> Optional[AgentRun]:
        tool = self.tools.get(action.tool)
        if tool is None:
            return run.finish(RunStatus.FAILED, {"error": f"planner chose unknown tool {action.tool}"},
                              "unknown tool", ToolErrorCode.INTERNAL.value)
        run.event("plan", action.reason, tool=tool.name, key=action.key,
                  input_from=action.input_call_ids)
        call_id = f"call-{len(run.tool_calls) + 1}"
        attempt = state.attempts(tool.name, action.key) + 1

        blocked = gate(tool.action_level, ctx.approval)
        if blocked:
            return await self._block_for_approval(run, action, tool.name, tool.action_level, call_id,
                                                  attempt, blocked, ctx)

        started = time.monotonic()
        result = await tool.invoke(action.input, replace(ctx, call_id=call_id,
                                                          input_call_ids=list(action.input_call_ids)))
        self._record(run, tool.name, tool.action_level, call_id, attempt, action.input,
                     action.input_call_ids, result, int((time.monotonic() - started) * 1000))
        state.steps.append(Step(tool=tool.name, key=action.key, call_id=call_id,
                                input=action.input, result=result,
                                input_call_ids=list(action.input_call_ids)))
        return None

    async def _block_for_approval(self, run: AgentRun, action: CallTool, tool: str, level: ActionLevel,
                                  call_id: str, attempt: int, why: str, ctx: ToolContext) -> AgentRun:
        run.tool_calls.append(ToolCall(
            call_id=call_id, tool=tool, action_level=level.value,
            input=summarize_input(action.input, action.input_call_ids), status="BLOCKED",
            attempt=attempt, error={"code": ToolErrorCode.APPROVAL_REQUIRED.value, "message": why,
                                    "retryable": False}))
        if ctx.user_id is None:
            return run.finish(RunStatus.FAILED, {"error": "EXECUTE needs an authenticated owner"},
                              why, ToolErrorCode.UNAUTHENTICATED.value)
        try:
            proposal = await self.proposals.create(
                owner_user_id=ctx.user_id, tool=tool, tool_input=dict(action.input),
                rationale=action.reason, evidence={"call_ids": list(action.input_call_ids)},
                source_task_id=run.task_id, input_call_ids=list(action.input_call_ids))
        except ProposalStoreError as exc:
            run.event("approval", f"{tool} blocked and no proposal could be stored: {exc}")
            return run.finish(RunStatus.FAILED, {"error": str(exc)},
                              "proposal store unavailable (nothing was executed)",
                              ToolErrorCode.UNAVAILABLE.value)
        run.event("approval", f"{tool} blocked: {why}; proposal stored for human review",
                  proposal_id=proposal.proposal_id)
        result = dict(action.draft_result or {})
        result["execution"] = {"status": "AWAITING_APPROVAL", "proposal": proposal.to_dict()}
        run.raise_level(ActionLevel.PROPOSE)
        run.data_provenance = self._provenance(run, action.input_call_ids)
        return run.finish(RunStatus.SUCCESS, result,
                          "awaiting human approval for EXECUTE (nothing was executed)")

    def _record(self, run: AgentRun, tool: str, level: ActionLevel, call_id: str, attempt: int,
                payload: Dict[str, Any], input_call_ids: List[str], result: ToolResult,
                duration_ms: int) -> None:
        summary = summarize_output(tool, result.output)
        run.tool_calls.append(ToolCall(
            call_id=call_id, tool=tool, action_level=level.value,
            input=summarize_input(payload, input_call_ids),
            status="OK" if result.ok else "ERROR", attempt=attempt, error=result.error_dict(),
            output_summary=summary,
            provenance=result.provenance.to_dict() if result.provenance else None,
            duration_ms=duration_ms))
        observed = summary if result.ok else result.error_dict()
        run.observations.append(Observation(
            call_id=call_id, tool=tool, ok=result.ok, summary=observed or {},
            error_code=None if result.ok else result.error_code.value))
        run.event("tool_call", f"{tool} -> {'OK' if result.ok else result.error_code.value}",
                  call_id=call_id, attempt=attempt, duration_ms=duration_ms)
        run.event("observation", _observation_text(tool, result), call_id=call_id)
        if result.ok:
            run.raise_level(level)

    def _conclude(self, run: AgentRun, state: LoopState, action: Any) -> AgentRun:
        if isinstance(action, Finish):
            run.event("decision", action.reason, evidence=action.evidence)
            run.raise_level(action.level)
            run.data_provenance = self._provenance(run, action.evidence)
            return run.finish(RunStatus.SUCCESS, action.result, action.reason)
        if isinstance(action, Stop):
            run.event("decision", f"stop: {action.reason}", evidence=action.evidence)
            run.data_provenance = self._provenance(run, action.evidence)
            return run.finish(RunStatus.FAILED,
                              {"error": action.reason, "skipped": list(state.notes)},
                              action.reason, action.error_code)
        if isinstance(action, AskUser):
            run.event("decision", f"ask user: {action.question}")
            return run.finish(RunStatus.FAILED, {"question": action.question},
                              action.reason, "NEEDS_USER_INPUT")
        return run.finish(RunStatus.FAILED, {"error": f"planner returned {action!r}"},
                          "invalid planner action", ToolErrorCode.INTERNAL.value)

    def _provenance(self, run: AgentRun, evidence: List[str]) -> Dict[str, Any]:
        return provenance_for(run, evidence)

    # -------------------------------------------------------------- approved execute

    async def execute_approved(self, proposal_id: str, ctx: ToolContext) -> AgentRun:
        run = AgentRun(agent=AGENT_NAME, input={"approve_proposal_id": proposal_id},
                       owner_user_id=ctx.user_id)
        run.event("goal", "human approval submitted", proposal_id=proposal_id)
        try:
            try:
                proposal = await self.proposals.approve(proposal_id, ctx.user_id)
            except ApprovalError as exc:
                run.event("approval", f"approval rejected: {exc}", code=exc.code, proposal_id=proposal_id)
                return run.finish(RunStatus.FAILED, {"error": str(exc), "proposal_id": proposal_id},
                                  str(exc), f"APPROVAL_{exc.code}")
            except ProposalStoreError as exc:
                run.event("approval", f"approval not recorded: {exc}", proposal_id=proposal_id)
                return run.finish(RunStatus.FAILED, {"error": str(exc), "proposal_id": proposal_id},
                                  "proposal store unavailable (nothing was executed)",
                                  ToolErrorCode.UNAVAILABLE.value)
            run.event("approval", f"proposal approved by user {ctx.user_id}",
                      proposal_id=proposal_id, source_task_id=proposal.source_task_id)
            tool = self.tools.get(proposal.tool)
            approval = {"approved_by": ctx.user_id, "proposal_id": proposal_id,
                        "order": order_fingerprint(proposal.tool_input)}
            call_id = "call-1"
            run.event("plan", f"carry out the approved {proposal.tool} action exactly as proposed",
                      tool=proposal.tool)
            started = time.monotonic()
            result = await tool.invoke(proposal.tool_input, replace(
                ctx, task_id=run.task_id, approval=approval, call_id=call_id,
                input_call_ids=[f"{proposal.source_task_id}:{c}" for c in proposal.input_call_ids]))
            self._record(run, tool.name, tool.action_level, call_id, 1, proposal.tool_input, [],
                         result, int((time.monotonic() - started) * 1000))
            try:  # APPROVED -> EXECUTED / FAILED; a proposal is never carried out twice
                proposal = await self.proposals.finish(
                    proposal_id, run.task_id, result.ok,
                    execution=result.output if result.ok else None,
                    error=None if result.ok else result.error_message)
                run.event("approval", f"proposal {proposal_id} is now {proposal.status}",
                          proposal_id=proposal_id, status=proposal.status,
                          source_task_id=proposal.source_task_id)
            except ProposalStoreError as exc:
                run.event("approval", f"proposal status not updated: {exc} (it stays APPROVED and "
                                      f"cannot be approved again)", proposal_id=proposal_id)
            run.data_provenance = self._provenance(run, [call_id])
            run.data_provenance["approved_proposal"] = proposal.to_dict()
            if not result.ok:
                return run.finish(RunStatus.FAILED, {"error": result.error_message,
                                                     "proposal": proposal.to_dict()},
                                  f"{tool.name} failed", result.error_code.value)
            run.event("decision", "approved paper order recorded", evidence=[call_id])
            return run.finish(RunStatus.SUCCESS, {"execution": result.output, "proposal": proposal.to_dict()},
                              "approved action executed (PAPER)")
        finally:
            self.run_log.add(run)


def provenance_for(run: AgentRun, evidence: List[str]) -> Dict[str, Any]:
    """Sources a result rests on: the evidence calls and what they were derived from."""
    by_id = {c.call_id: c for c in run.tool_calls}
    seen: Set[str] = set()
    queue = list(evidence)
    while queue:
        call_id = queue.pop()
        if call_id in seen or call_id not in by_id:
            continue
        seen.add(call_id)
        prov = by_id[call_id].provenance or {}
        queue.extend(prov.get("derived_from", []))
    sources = [{"call_id": cid, "tool": by_id[cid].tool, **(by_id[cid].provenance or {})}
               for cid in sorted(seen, key=lambda c: int(c.split("-")[1]))
               if by_id[cid].status == "OK"]
    kinds = {s.get("kind") for s in sources}
    return {
        "evidence_call_ids": list(evidence),
        "sources": sources,
        "source_kinds": sorted(k for k in kinds if k),
        "synthetic_data_used": bool(kinds - {"database", "derived", "paper_simulation"}),
    }


def _observation_text(tool: str, result: ToolResult) -> str:
    if not result.ok:
        return f"{tool} failed: {result.error_message}"
    out = result.output or {}
    if tool == "market_data":
        return f"{out['count']} bars {out['first_date']}..{out['last_date']}, last close {out['last_close']}"
    if tool == "risk":
        return f"risk level {out['risk_level']} ({out['scope']} {out['subject']})"
    if tool == "signal":
        return (f"consensus {out['consensus_label']} ({out['consensus_score']}) from "
                f"{len(out['signals'])} strategies; unavailable: {sorted(out['unavailable'])}")
    if tool == "backtest":
        return (f"{out['strategy']}: {out['trade_count']} trades, "
                f"sharpe {out['metrics'].get('sharpe_ratio')}")
    if tool == "portfolio":
        return f"portfolio {out['portfolio_id']} has {out['position_count']} positions"
    if tool == "factor":
        return f"factors {sorted(out['values'])}; unavailable {sorted(out['unavailable'])}"
    if tool == "execution":
        return f"{out['mode']} {out['side']} {out['quantity']} {out['symbol']} @ {out['fill_price']}"
    return "ok"
