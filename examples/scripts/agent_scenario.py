#!/usr/bin/env python3
# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent demo scenario, fixed and repeatable (docs/demo/agent-demo.md).

    Risk detection -> Decision -> Proposal -> Owner approval -> PAPER execution

Runs the demo goal through the product API (Java proxy -> Python) and checks
every step against what the demo shows. PAPER ONLY: the approval writes a
paper order ticket; positions, cash and orders are not changed, so the demo
can be run again and gives the same result.

    AGENT_DEMO_PASSWORD=... [AGENT_DEMO_OTHER_PASSWORD=...] \\
      python scripts/demo/agent_scenario.py --email e2e@example.com \\
          [--other-email other@example.com] [--no-approve] [--out demo.json]

Exit code 0 when every check passed, 1 otherwise. Nothing is set up or
changed silently: missing preconditions are reported as failed checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

DEMO_GOAL = "我的{symbol}仓位超过上限，减持 {quantity:g} 股 {symbol}"
Check = Tuple[str, bool, str]


def check_status(status: Dict[str, Any]) -> List[Check]:
    return [
        ("execution is PAPER ONLY", status.get("execution_mode") == "PAPER ONLY", str(status.get("execution_mode"))),
        ("proposals are persisted", status.get("proposal_store") == "SqlProposalStore", str(status.get("proposal_store"))),
        ("planner path reported", status.get("planner") in ("llm", "rule"),
         f"{status.get('planner')} ({status.get('llm_model') or status.get('reason')})"),
    ]


def check_preferences(memory: Dict[str, Any]) -> List[Check]:
    prefs = {k: v.get("value") for k, v in (memory.get("preferences") or {}).items()}
    return [
        ("single-position limit in memory", prefs.get("max_single_position_pct") is not None,
         f"max_single_position_pct={prefs.get('max_single_position_pct')}"),
        ("auto trading forbidden in memory", prefs.get("forbid_auto_trading") is True,
         f"forbid_auto_trading={prefs.get('forbid_auto_trading')}"),
    ]


def check_run(run: Dict[str, Any], symbol: str, quantity: float) -> List[Check]:
    r = run.get("result") or {}
    risk = r.get("risk") or {}
    trade = r.get("trade") or {}
    proposal = trade.get("proposal") or {}
    order = proposal.get("action") or {}
    planner = r.get("planner") or {}
    tools = [c["tool"] for c in run.get("tool_calls") or []]
    return [
        ("run finished", run.get("status") == "SUCCESS", f"{run.get('status')} {run.get('error_code') or ''}".strip()),
        ("planner recorded", bool(planner.get("attempts")),
         f"{planner.get('planner')}; fallback={planner.get('fallback')} ({planner.get('fallback_reason')})"),
        ("memory retrieved", any(e["kind"] == "memory" for e in run.get("trace") or []), "memory event in trace"),
        ("risk detected: position above the limit", risk.get("status") == "VETO"
         and any(f"{symbol} is" in v and "limit" in v for v in risk.get("vetoes") or []),
         "; ".join(risk.get("vetoes") or [])[:160]),
        ("decision is to reduce", r.get("recommendation") in ("REDUCE_EXPOSURE", "REDUCE_CONCENTRATION", "SELL"),
         f"{r.get('recommendation')} ({r.get('confidence')}, {(r.get('decision') or {}).get('decided_by')})"),
        ("critic reviewed", (r.get("critic") or {}).get("verdict") in ("approve", "request_revision"),
         ", ".join(f"{i['code']}/{i['severity']}" for i in (r.get("critic") or {}).get("issues") or [])),
        ("proposal awaits approval (PAPER)", trade.get("status") == "AWAITING_APPROVAL" and trade.get("mode") == "PAPER",
         trade.get("reason") or proposal.get("proposal_id", "")),
        ("order is the requested sale at the observed close",
         order.get("side") == "sell" and order.get("symbol") == symbol and order.get("quantity") == quantity
         and order.get("reference_price") == ((r.get("research") or {}).get("market") or {}).get(symbol, {}).get("last_close"),
         f"{order.get('side')} {order.get('quantity')} {order.get('symbol')} @ {order.get('reference_price')}"),
        ("the run executed nothing", "execution" not in tools and run.get("action_level") != "EXECUTE",
         f"level {run.get('action_level')}"),
    ]


def check_approval(code: int, run: Dict[str, Any], order: Dict[str, Any]) -> List[Check]:
    fill = (run.get("result") or {}).get("execution") or {}
    return [
        ("owner approval accepted", code == 200 and run.get("status") == "SUCCESS", f"HTTP {code}"),
        ("paper fill of exactly the proposed order",
         fill.get("mode") == "PAPER" and fill.get("status") == "PAPER_FILLED" and fill.get("side") == order.get("side")
         and fill.get("quantity") == order.get("quantity") and fill.get("fill_price") == order.get("reference_price"),
         f"{fill.get('status')} {fill.get('side')} {fill.get('quantity')} @ {fill.get('fill_price')} "
         f"order {fill.get('order_id')}"),
    ]


def run_demo(http: Any, base_url: str, email: str, password: str, portfolio_id: int, symbol: str,
             quantity: float, other: Optional[Tuple[str, str]] = None, approve: bool = True,
             log: Callable[[str], None] = print) -> Tuple[List[Check], Dict[str, Any]]:
    runtime = f"{base_url}/python/v1/agents/runtime"

    def login(user: str, secret: str) -> Dict[str, str]:
        resp = http.post(f"{base_url}/users/login", json={"username": user, "password": secret})
        resp.raise_for_status()
        data = resp.json().get("data") or resp.json()
        return {"Authorization": f"Bearer {data.get('accessToken') or data.get('token')}"}

    owner = login(email, password)
    evidence: Dict[str, Any] = {}
    checks: List[Check] = []
    evidence["status"] = http.get(f"{runtime}/workflow/status", headers=owner).json()
    checks += check_status(evidence["status"])
    evidence["memory"] = http.get(f"{runtime}/memory", headers=owner).json()
    checks += check_preferences(evidence["memory"])

    goal = DEMO_GOAL.format(symbol=symbol, quantity=quantity)
    log(f"goal: {goal} (portfolio {portfolio_id})")
    start = http.post(f"{runtime}/workflow", json={"goal_text": goal, "portfolio_id": portfolio_id, "async": True},
                      headers=owner)
    run = start.json()
    checks.append(("run started in the background", start.status_code == 202 and run.get("status") == "RUNNING",
                   f"HTTP {start.status_code}"))
    deadline = time.monotonic() + 300
    while run.get("status") == "RUNNING" and time.monotonic() < deadline:
        time.sleep(0.2)
        run = http.get(f"{runtime}/runs/{run['task_id']}", headers=owner).json()
    evidence["run"] = run
    checks += check_run(run, symbol, quantity)
    proposal = ((run.get("result") or {}).get("trade") or {}).get("proposal")
    if not proposal:
        return checks, evidence
    pid, order = proposal["proposal_id"], proposal["action"]

    pending = http.get(f"{runtime}/proposals", params={"status": "PENDING"}, headers=owner).json()
    checks.append(("proposal listed as PENDING from the database",
                   pid in [p["proposal_id"] for p in pending], f"{len(pending)} pending"))
    if other:
        stranger = login(*other)
        read = http.get(f"{runtime}/proposals/{pid}", headers=stranger)
        denied = http.post(f"{runtime}/run", json={"approve_proposal_id": pid}, headers=stranger)
        checks.append(("another user cannot see it", read.status_code == 404, f"HTTP {read.status_code}"))
        checks.append(("another user cannot approve it", denied.status_code == 403
                       and denied.json().get("error_code") == "APPROVAL_FORBIDDEN", f"HTTP {denied.status_code}"))
    if not approve:
        evidence["proposal"] = http.get(f"{runtime}/proposals/{pid}", headers=owner).json()
        return checks, evidence

    approved = http.post(f"{runtime}/run", json={"approve_proposal_id": pid}, headers=owner)
    evidence["approval"] = approved.json()
    checks += check_approval(approved.status_code, evidence["approval"], order)
    again = http.post(f"{runtime}/run", json={"approve_proposal_id": pid}, headers=owner)
    checks.append(("second approval refused (executed at most once)",
                   again.status_code == 409 and again.json().get("error_code") == "APPROVAL_CONSUMED",
                   f"HTTP {again.status_code} {again.json().get('stop_reason')}"))
    final = http.get(f"{runtime}/proposals/{pid}", headers=owner).json()
    evidence["proposal"] = final
    checks.append(("proposal is EXECUTED and linked to both runs",
                   final.get("status") == "EXECUTED" and final.get("source_task_id") == run["task_id"]
                   and final.get("execution_task_id") == evidence["approval"].get("task_id"),
                   f"{final.get('status')} source {final.get('source_task_id')} exec {final.get('execution_task_id')}"))
    return checks, evidence


def main(argv: Optional[List[str]] = None) -> int:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api")
    parser.add_argument("--email", required=True)
    parser.add_argument("--other-email", help="a second user, to show the proposal is private to its owner")
    parser.add_argument("--portfolio-id", type=int, default=2)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--quantity", type=float, default=32)
    parser.add_argument("--no-approve", action="store_true", help="stop at the pending proposal (approve in the UI)")
    parser.add_argument("--out", help="write the evidence JSON here")
    args = parser.parse_args(argv)
    password = os.environ.get("AGENT_DEMO_PASSWORD")
    if not password:
        parser.error("set AGENT_DEMO_PASSWORD")
    other = None
    if args.other_email:
        other_password = os.environ.get("AGENT_DEMO_OTHER_PASSWORD") or password
        other = (args.other_email, other_password)

    with httpx.Client(timeout=300) as http:
        checks, evidence = run_demo(http, args.base_url, args.email, password, args.portfolio_id,
                                    args.symbol.upper(), args.quantity, other, not args.no_approve)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  -  {detail}")
    passed = all(ok for _, ok, _ in checks)
    print(f"\n{sum(ok for _, ok, _ in checks)}/{len(checks)} checks passed")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in checks],
                       "evidence": evidence}, fh, ensure_ascii=False, indent=1, default=str)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
