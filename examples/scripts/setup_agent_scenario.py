#!/usr/bin/env python3
# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the agent demo (docs/demo/agent-demo.md) through the product API.

Creates, only if missing:
  * the demo user (and optionally a second user, to show proposals are private),
  * a portfolio holding AAPL x50 and MSFT x30 - with the seeded EOD closes
    (Flyway V8) AAPL is about 41% of it, above the demo's 20% limit,
  * the demo preferences in agent memory: risk_tolerance=medium,
    max_single_position_pct=20, forbid_auto_trading=true.

Idempotent and non-destructive: existing users, portfolios and positions are
reused, never changed or deleted; a position with a different quantity is
reported and the script exits 1 instead of "fixing" it. It never writes to the
database directly. Prices in the demo always come from eod_data; the cost
basis given here only fills the position form.

    AGENT_DEMO_PASSWORD=... python scripts/demo/setup_agent_scenario.py \\
        --email agent-demo@example.com [--other-email agent-demo-other@example.com]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

DEMO_POSITIONS = [("AAPL", 50.0, 200.0), ("MSFT", 30.0, 500.0)]
DEMO_PREFERENCES = {"risk_tolerance": "medium", "max_single_position_pct": 20, "forbid_auto_trading": True}


def _data(resp: Any) -> Any:
    body = resp.json()
    return body.get("data", body) if isinstance(body, dict) else body


class Api:
    def __init__(self, http: Any, base_url: str):
        self.http, self.base = http, base_url
        self.headers: Dict[str, str] = {}

    def login(self, email: str, password: str) -> bool:
        resp = self.http.post(f"{self.base}/users/login", json={"username": email, "password": password})
        if resp.status_code != 200 or not (_data(resp) or {}).get("accessToken"):
            return False
        self.headers = {"Authorization": f"Bearer {_data(resp)['accessToken']}"}
        return True

    def ensure_user(self, email: str, password: str, log) -> None:
        if self.login(email, password):
            log(f"user {email}: exists")
            return
        resp = self.http.post(f"{self.base}/users/register", json={
            "username": email.split("@")[0], "email": email, "password": password,
            "displayName": "Agent Demo"})
        if resp.status_code != 200 or not self.login(email, password):
            raise SystemExit(f"cannot register or log in {email}: HTTP {resp.status_code} {resp.text[:200]}")
        log(f"user {email}: registered")


def ensure_portfolio(api: Api, name: str, log) -> int:
    existing = _data(api.http.get(f"{api.base}/portfolios", headers=api.headers)) or []
    for p in existing:
        if p.get("portfolioName") == name:
            log(f"portfolio '{name}': exists (#{p['portfolioId']})")
            return int(p["portfolioId"])
    resp = api.http.post(f"{api.base}/portfolios", headers=api.headers, json={
        "portfolioName": name, "description": "Agent demo: AAPL above the 20% single-position limit",
        "baseCurrency": "USD", "initialCapital": 100000})
    created = _data(resp)
    if resp.status_code != 200 or not created or "portfolioId" not in created:
        raise SystemExit(f"cannot create portfolio: HTTP {resp.status_code} {resp.text[:200]}")
    log(f"portfolio '{name}': created (#{created['portfolioId']})")
    return int(created["portfolioId"])


def ensure_positions(api: Api, portfolio_id: int, log) -> List[str]:
    """Create missing demo positions; report (never change) ones that differ."""
    rows = _data(api.http.get(f"{api.base}/positions/portfolio/{portfolio_id}", headers=api.headers)) or []
    if isinstance(rows, dict):
        rows = rows.get("positions") or rows.get("content") or []
    held: Dict[str, float] = {}
    for r in rows:
        held[str(r.get("symbol")).upper()] = held.get(str(r.get("symbol")).upper(), 0.0) + float(r.get("quantity") or 0)
    problems = []
    for symbol, quantity, cost in DEMO_POSITIONS:
        if symbol in held:
            if abs(held[symbol] - quantity) > 1e-9:
                problems.append(f"{symbol}: holds {held[symbol]:g}, the demo expects {quantity:g} (left unchanged)")
            else:
                log(f"position {symbol} x{quantity:g}: exists")
            continue
        resp = api.http.post(f"{api.base}/positions", headers=api.headers, json={
            "portfolioId": portfolio_id, "symbol": symbol, "symbolName": symbol, "assetType": "STOCK",
            "quantity": quantity, "averagePrice": cost, "currency": "USD", "notes": "agent demo"})
        if resp.status_code != 200 or (resp.json().get("code") not in (None, 200)):
            problems.append(f"{symbol}: not created: HTTP {resp.status_code} {resp.text[:200]}")
        else:
            log(f"position {symbol} x{quantity:g}: created")
    others = sorted(set(held) - {s for s, _, _ in DEMO_POSITIONS})
    if others:
        problems.append(f"other holdings {others} change the weights the demo expects (left unchanged)")
    return problems


def ensure_preferences(api: Api, log) -> None:
    runtime = f"{api.base}/python/v1/agents/runtime"
    current = {k: v.get("value") for k, v in
               (api.http.get(f"{runtime}/memory", headers=api.headers).json().get("preferences") or {}).items()}
    for key, value in DEMO_PREFERENCES.items():
        if current.get(key) == value or (isinstance(value, int) and current.get(key) == float(value)):
            log(f"preference {key}={value}: set")
            continue
        resp = api.http.put(f"{runtime}/memory/preferences", headers=api.headers, json={"key": key, "value": value})
        if resp.status_code != 200:
            raise SystemExit(f"cannot set {key}: HTTP {resp.status_code} {resp.text[:200]}")
        log(f"preference {key}={value}: written to agent memory")


def main(argv: Optional[List[str]] = None) -> int:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/api")
    parser.add_argument("--email", required=True)
    parser.add_argument("--other-email")
    parser.add_argument("--portfolio-name", default="Agent Demo")
    args = parser.parse_args(argv)
    password = os.environ.get("AGENT_DEMO_PASSWORD")
    if not password:
        parser.error("set AGENT_DEMO_PASSWORD (at least 8 characters)")

    with httpx.Client(timeout=60) as http:
        if args.other_email:
            Api(http, args.base_url).ensure_user(args.other_email,
                                                 os.environ.get("AGENT_DEMO_OTHER_PASSWORD") or password, print)
        api = Api(http, args.base_url)
        api.ensure_user(args.email, password, print)
        portfolio_id = ensure_portfolio(api, args.portfolio_name, print)
        problems = ensure_positions(api, portfolio_id, print)
        ensure_preferences(api, print)
    for p in problems:
        print(f"PROBLEM  {p}")
    print(f"\ndemo portfolio id: {portfolio_id}")
    print(f"run: python scripts/demo/agent_scenario.py --email {args.email} --portfolio-id {portfolio_id}"
          + (f" --other-email {args.other_email}" if args.other_email else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
