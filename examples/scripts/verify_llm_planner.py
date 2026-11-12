#!/usr/bin/env python3
# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the multi-provider LLM planner end to end through the product API.

    User goal -> LLM planner -> structured plan -> validator -> tool execution -> trace

Runs one real analysis goal (never a trade) through ``POST /agents/runtime/workflow``
(via the Java proxy by default) and reports, from the workflow status and the
returned run contract::

    provider, model, endpoint, planner_mode, verification_status

plus the raw planner output, the validator's verdict, the fallback reason, the
tool calls and the trace.

verification_status (exit code):
  VERIFIED  (0) a known provider on its official endpoint, a served model of
                that provider's family, an LLM plan that passed the validator
                and tools that ran
  FAILED    (1) unknown or mismatched provider, a model outside the provider's
                family, or a run that did not complete - a compatible endpoint
                never passes for another provider
  PARTIAL   (2) no usable model (no key, timeout, endpoint unavailable, plan
                rejected): the rule planner ran instead and the run completed
  TEST_ONLY (3) a mock / scripted model - never evidence of a real model
  EXPECTED_INSUFFICIENT_DATA (4)
                every provider and plan check passed, but the run failed only
                because tools reported INSUFFICIENT_DATA (e.g. a strategy needs
                more history than the store holds). Not VERIFIED - the run did
                not succeed - and not a provider or planner defect

    AGENT_VERIFY_PASSWORD=... python scripts/testing/verify_llm_planner.py \\
        --email e2e@example.com --goal "分析我的AAPL仓位风险" --portfolio-id 2 --out evidence.json

The model key lives only in the Python service's environment
(AGENT_LLM_API_KEY / ANTHROPIC_API_KEY); this script never reads or prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

VERIFIED, FAILED, PARTIAL, TEST_ONLY, EXPECTED_INSUFFICIENT_DATA = 0, 1, 2, 3, 4
STATUS_NAMES = {VERIFIED: "VERIFIED", FAILED: "FAILED", PARTIAL: "PARTIAL", TEST_ONLY: "TEST_ONLY",
                EXPECTED_INSUFFICIENT_DATA: "EXPECTED_INSUFFICIENT_DATA"}
# Official API hosts and model families (kept equal to src/agents/runtime/llm.py PROVIDERS by a test)
OFFICIAL = {"anthropic": {"hosts": ("api.anthropic.com",), "family": "claude"},
            "deepseek": {"hosts": ("api.deepseek.com",), "family": "deepseek"}}
MOCK_PROVIDERS = ("mock",)


def _host(url: Optional[str]) -> str:
    return (urlparse(url or "").hostname or "").lower()


def assess(status: Dict[str, Any], run: Optional[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """verification_status (as exit code) and the report, from the status and one run."""
    result = (run or {}).get("result") or {}
    planner = result.get("planner") or {}
    attempts: List[Dict[str, Any]] = planner.get("attempts") or []
    accepted = [a for a in attempts if a.get("planner") == "llm" and a.get("outcome") == "accepted"]
    last_llm = accepted[-1] if accepted else next((a for a in reversed(attempts) if a.get("planner") == "llm"), {})
    provider = (last_llm.get("provider") if accepted else None) or status.get("llm_provider")
    endpoint = (last_llm.get("endpoint") if accepted else None) or status.get("llm_endpoint")
    llm_mode = planner.get("planner") == "llm" and not planner.get("fallback")
    calls = [{"call_id": c["call_id"], "agent": c.get("agent"), "tool": c["tool"], "status": c["status"],
              "error_code": (c.get("error") or {}).get("code"), "error": (c.get("error") or {}).get("message")}
             for c in (run or {}).get("tool_calls") or []]
    report: Dict[str, Any] = {
        "provider": provider,
        "model": (last_llm.get("served_model") if accepted else None) or status.get("llm_model"),
        "endpoint": endpoint,
        "planner_mode": "llm" if llm_mode else ("rule" if planner else None),
        "fallback_reason": None if llm_mode else (planner.get("fallback_reason") or status.get("reason")),
        "llm_available": status.get("llm_available"),
        "status_reason": status.get("reason"),
        "task_id": (run or {}).get("task_id"), "run_status": (run or {}).get("status"),
        "attempts": [{"attempt": a.get("attempt"), "planner": a.get("planner"), "provider": a.get("provider"),
                      "endpoint": a.get("endpoint"), "model": a.get("model"),
                      "served_model": a.get("served_model"), "outcome": a.get("outcome"),
                      "planner_output": a.get("planner_output"), "validation": a.get("validation"),
                      "request_id": (a.get("call") or {}).get("request_id"), "detail": a.get("detail")}
                     for a in attempts],
        "tool_calls": calls,
        "trace": [{"seq": e.get("seq"), "agent": (e.get("data") or {}).get("agent"), "kind": e.get("kind"),
                   "message": e.get("message")} for e in (run or {}).get("trace") or []],
    }
    problems: List[str] = []

    def done(code: int) -> Tuple[int, Dict[str, Any]]:
        report["verification_status"] = STATUS_NAMES[code]
        report["problems"] = problems
        return code, report

    if provider in MOCK_PROVIDERS:
        problems.append("mock model: test evidence only, never a real provider")
        return done(TEST_ONLY)
    if provider and provider not in OFFICIAL:
        problems.append(f"unknown provider {provider!r}" if not str(provider).startswith("mismatch:")
                        else f"provider does not match its endpoint ({provider})")
        return done(FAILED)
    completed = (run or {}).get("status") == "SUCCESS" and any(c["status"] == "OK" for c in calls)
    if not llm_mode:
        if not completed:
            problems.append("the run did not complete")
            return done(FAILED)
        problems.append(f"no usable model; rule planner ran: {report['fallback_reason']}")
        return done(PARTIAL)
    # the model planned: every claim must check out
    spec = OFFICIAL.get(provider or "")
    if spec is None:
        problems.append("provider not reported")
    else:
        if _host(endpoint) not in spec["hosts"]:
            problems.append(f"endpoint {_host(endpoint)!r} is not {provider}'s official API")
        if not str(report["model"] or "").startswith(spec["family"]):
            problems.append(f"the plan was served by {report['model']!r}, not a {spec['family']} model")
    if not accepted or not (accepted[-1].get("validation") or {}).get("valid"):
        problems.append("the accepted LLM plan has no passing validation result")
    elif not accepted[-1].get("planner_output"):
        problems.append("the accepted LLM attempt carries no planner output")
    if completed:
        return done(FAILED if problems else VERIFIED)
    failed = [c for c in calls if c["status"] != "OK"]
    if not problems and failed and all(c["error_code"] == "INSUFFICIENT_DATA" for c in failed):
        problems.extend(f"{c['tool']}: INSUFFICIENT_DATA - {c['error']}" for c in failed)
        return done(EXPECTED_INSUFFICIENT_DATA)
    problems.append(f"the run ended {report['run_status']}; failed tool calls: "
                    + (", ".join(f"{c['tool']} {c['error_code'] or c['status']}" for c in failed) or "none"))
    return done(FAILED)


def main(argv: List[str] | None = None) -> int:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api",
                        help="Java API base (the workflow is reached through /python/v1/...)")
    parser.add_argument("--email", required=True)
    parser.add_argument("--goal", default="分析我的AAPL仓位风险")
    parser.add_argument("--portfolio-id", type=int)
    parser.add_argument("--out", help="write the evidence JSON here")
    args = parser.parse_args(argv)
    password = os.environ.get("AGENT_VERIFY_PASSWORD")
    if not password:
        parser.error("set AGENT_VERIFY_PASSWORD for the login")

    runtime = f"{args.base_url}/python/v1/agents/runtime"
    with httpx.Client(timeout=300) as client:
        login = client.post(f"{args.base_url}/users/login",
                            json={"username": args.email, "password": password})
        login.raise_for_status()
        body = login.json()
        data = body.get("data") or body
        headers = {"Authorization": f"Bearer {data.get('accessToken') or data.get('token')}"}
        status = client.get(f"{runtime}/workflow/status", headers=headers).json()
        request = {"goal_text": args.goal}  # an analysis goal: the run never executes anything
        if args.portfolio_id:
            request["portfolio_id"] = args.portfolio_id
        run = client.post(f"{runtime}/workflow", json=request, headers=headers).json()
    code, report = assess(status, run)
    report["goal"] = args.goal
    text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    print(STATUS_NAMES[code], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
