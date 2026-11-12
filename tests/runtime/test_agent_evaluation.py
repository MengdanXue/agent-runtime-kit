# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent Runtime evaluation suite.

Runs the real loop, planner, approval gate and tool code. Only the outermost
backends are replaced: the Java EOD endpoint (httpx.MockTransport), the
database session behind PortfolioTool/RiskTool, and the analytics service inside
RiskAgent (the shared ``stub_analytics`` fixture).

Every scenario is scored on four criteria:

* tool_selection  - the tools called, in order, are the ones the goal and the
                    observations call for
* no_fabrication  - a failed run carries no result data; a successful one rests
                    only on database / derived / paper sources
* correct_stop    - the run ends with the expected status and error code
* auditable_trace - the trace records goal, plan, tool call, observation and a
                    final event, and every tool call has provenance or an error
"""

from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx
import pytest

from agentrt.quant.risk_agent import RiskAgent
from agentrt.runtime.contract import AgentRun, RunStatus
from agentrt.runtime.loop import AgentLoop
from agentrt.runtime.planner import Goal
from agentrt.tools import (
    BacktestTool, ExecutionTool, FactorTool, MarketDataTool, PortfolioTool, RiskTool,
    SignalTool, ToolContext, ToolRegistry,
)

OWNER, OTHER = 2, 3


# ------------------------------------------------------------------------ price series

pytestmark = pytest.mark.needs_strategies  # named strategies come from agentrt.ports.strategies

def _bars(returns: List[float], start: float = 100.0) -> List[Dict[str, Any]]:
    closes, price = [], start
    for r in returns:
        price *= 1 + r
        closes.append(price)
    first = date(2026, 5, 1)
    return [{"date": (first + timedelta(days=i)).isoformat(), "open": c, "high": c * 1.005,
             "low": c * 0.995, "close": c, "volume": 1_000_000 + i} for i, c in enumerate(closes)]


def calm_uptrend(n: int = 80) -> List[Dict[str, Any]]:
    """Low volatility, steadily rising: momentum and MA-cross both say BUY."""
    return _bars([0.004 + 0.002 * math.sin(i) for i in range(n)])


def wild(n: int = 80) -> List[Dict[str, Any]]:
    """+-8% daily swings: annualized volatility far above the 'high' threshold."""
    return _bars([0.08 if i % 2 else -0.075 for i in range(n)])


# ---------------------------------------------------------------------------- backends

class FakeJava:
    """Programmable stand-in for GET /market/eod/{symbol}."""

    def __init__(self, series: Dict[str, List[Dict[str, Any]]], failures: Optional[List[str]] = None):
        self.series = series
        self.failures = list(failures or [])  # consumed per request: "timeout" | "down" | "500"
        self.requests: List[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request.url.path)
        if self.failures:
            failure = self.failures.pop(0)
            if failure == "timeout":
                raise httpx.ReadTimeout("read timed out", request=request)
            if failure == "down":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(500, json={"code": 500})
        symbol = request.url.path.rsplit("/", 1)[-1]
        rows = [{"date": b["date"], "open": b["open"], "high": b["high"], "low": b["low"],
                 "close": b["close"], "volume": b["volume"]} for b in self.series.get(symbol, [])]
        return httpx.Response(200, json={"code": 200, "message": "success", "data": rows})


class _Row(tuple):
    def __new__(cls, mapping: Dict[str, Any]):
        row = super().__new__(cls, tuple(mapping.values()))
        row._mapping = mapping
        return row


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeDB:
    """portfolios / positions tables as the portfolio tools query them."""

    def __init__(self, portfolios: Dict[int, int], positions: Dict[int, List[Dict[str, Any]]]):
        self.portfolios = portfolios  # portfolio_id -> owner user_id
        self.positions = positions
        self.queries: List[str] = []

    async def execute(self, query, params):
        sql = " ".join(str(query).split())
        pid = params["portfolio_id"]
        self.queries.append(sql[:40])
        if sql.startswith("SELECT user_id FROM portfolios"):
            owner = self.portfolios.get(pid)
            return _Result([] if owner is None else [_Row({"user_id": owner})])
        if "FROM positions" in sql:
            return _Result([_Row(p) for p in self.positions.get(pid, [])])
        if pid not in self.portfolios:
            return _Result([])
        return _Result([_Row({"portfolio_id": pid, "portfolio_name": f"P{pid}",
                              "base_currency": "USD", "equity": 100000.0,
                              "initial_capital": 100000.0, "cash_balance": 0.0,
                              "total_value": 100000.0})])

    def session_factory(self):
        @asynccontextmanager
        async def _session():
            yield self
        return _session


def _position(symbol: str) -> Dict[str, Any]:
    return {"position_id": 1, "symbol": symbol, "symbol_name": symbol, "asset_type": "stock",
            "direction": 1, "quantity": 10.0, "multiplier": 1, "currency": "USD",
            "average_cost": 100.0, "current_price": 110.0, "current_value": 1100.0,
            "unrealized_pnl": 100.0, "exchange_rate": 1.0, "base_currency": "USD"}


@pytest.fixture
def make_loop(stub_analytics):
    def _make(java: FakeJava, db: Optional[FakeDB] = None,
              risk_overrides: Optional[Dict[str, Any]] = None) -> AgentLoop:
        db = db or FakeDB({}, {})

        def risk_agent():
            agent = RiskAgent(agent_id="eval-risk")
            stub_analytics(agent, risk_overrides)
            return agent

        tools = ToolRegistry([
            MarketDataTool(base_url="http://java.test/api", transport=httpx.MockTransport(java.handler),
                           service_token=lambda: None, today=lambda: date(2026, 9, 1)),
            PortfolioTool(session_factory=db.session_factory()),
            RiskTool(session_factory=db.session_factory(), risk_agent_factory=risk_agent),
            SignalTool(), BacktestTool(), FactorTool(), ExecutionTool(),
        ])
        return AgentLoop(tools)
    return _make


def owner_ctx() -> ToolContext:
    return ToolContext(user_id=OWNER, authorization="Bearer test-token")


# ---------------------------------------------------------------------------- scoring

def score(run: AgentRun, expected_tools: List[str], status: RunStatus,
          error_code: Optional[str] = None) -> Dict[str, bool]:
    d = run.to_dict()
    called = [c["tool"] for c in d["tool_calls"]]
    kinds = {e["kind"] for e in d["trace"]}
    if status == RunStatus.FAILED:
        data_keys = set(d["result"]) - {"error", "question", "skipped"}
        no_fabrication = not data_keys
    else:
        no_fabrication = (d["data_provenance"].get("synthetic_data_used") is False
                          and bool(d["data_provenance"].get("sources")))
    return {
        "tool_selection": called == expected_tools,
        "no_fabrication": no_fabrication,
        "correct_stop": d["status"] == status.value and d["error_code"] == error_code,
        "auditable_trace": ({"goal", "final"} <= kinds
                            and (not called or {"plan", "tool_call"} <= kinds or "approval" in kinds)
                            and all(c["provenance"] or c["error"] for c in d["tool_calls"])),
    }


SCORECARD: List[Dict[str, Any]] = []


def assert_scored(name: str, run: AgentRun, expected_tools: List[str], status: RunStatus,
                  error_code: Optional[str] = None) -> Dict[str, Any]:
    card = score(run, expected_tools, status, error_code)
    SCORECARD.append({"scenario": name, **card})
    assert card == {k: True for k in card}, json.dumps(
        {"card": card, "called": [c.tool for c in run.tool_calls], "status": run.status.value,
         "error_code": run.error_code, "stop_reason": run.stop_reason}, ensure_ascii=False)
    return run.to_dict()


# ------------------------------------------------------------------------ normal cases

@pytest.mark.asyncio
async def test_portfolio_risk(make_loop):
    db = FakeDB({7: OWNER}, {7: [_position("AAPL"), _position("MSFT")]})
    run = await make_loop(FakeJava({}), db).run(Goal(intent="portfolio_risk", portfolio_id=7), owner_ctx())
    d = assert_scored("查询组合风险", run, ["portfolio", "risk"], RunStatus.SUCCESS)
    assert d["result"]["positions"] == 2
    assert d["result"]["metrics"]["var_95"] == 5000.0  # relayed from the analytics service
    assert [s["kind"] for s in d["data_provenance"]["sources"]] == ["database", "database"]
    assert d["result"]["risk_level"] in {"low", "medium", "high", "unknown"}  # normalized case


@pytest.mark.asyncio
async def test_high_portfolio_risk_raises_an_alert_and_a_proposal(make_loop):
    """RiskAgent says "HIGH" (upper case); the planner must still recognise it."""
    db = FakeDB({7: OWNER}, {7: [_position("AAPL"), _position("MSFT")]})
    run = await make_loop(FakeJava({}), db, risk_overrides={
        "var_95": {"var_amount": 30000.0, "var_pct": 30.0}}).run(
        Goal(intent="portfolio_risk", portfolio_id=7), owner_ctx())
    d = assert_scored("组合风险过高", run, ["portfolio", "risk"], RunStatus.SUCCESS)
    assert d["result"]["risk_level"] == "high"
    assert d["result"]["alerts"] == ["portfolio risk level is high"]
    assert d["result"]["proposal"]["requires_human"] is True
    assert d["action_level"] == "PROPOSE"


@pytest.mark.asyncio
async def test_analyze_symbol(make_loop):
    java = FakeJava({"AAPL": calm_uptrend()})
    run = await make_loop(java).run(
        Goal(intent="analyze_symbol", symbol="AAPL", factors=["momentum_20d", "volatility_20d"]),
        owner_ctx())
    d = assert_scored("分析股票", run, ["market_data", "risk", "signal", "factor"], RunStatus.SUCCESS)
    assert d["result"]["recommendation"] in {"BUY", "SELL", "HOLD"}
    assert d["result"]["last_close"] == calm_uptrend()[-1]["close"]
    # every derived number traces back to the market_data call
    sources = {s["call_id"]: s for s in d["data_provenance"]["sources"]}
    assert sources["call-1"]["kind"] == "database"
    assert all(s["derived_from"] == ["call-1"] for cid, s in sources.items() if cid != "call-1")


@pytest.mark.asyncio
async def test_strategy_suggestion(make_loop):
    java = FakeJava({"AAPL": calm_uptrend(120)})
    run = await make_loop(java).run(
        Goal(intent="strategy_suggestion", symbol="AAPL",
             strategies=["momentum", "ma_cross", "turtle"]), owner_ctx())
    d = assert_scored("生成策略建议", run,
                      ["market_data", "backtest", "backtest", "backtest", "risk"], RunStatus.SUCCESS)
    ranked = [c for c in d["result"]["candidates"] if c["trade_count"] > 0]
    if d["result"]["suggestion"]:
        best = max(ranked, key=lambda c: c["sharpe_ratio"])
        assert d["result"]["suggestion"]["strategy"] == best["strategy"]
    else:
        assert not ranked  # no fabricated winner when nothing traded
    assert d["action_level"] in {"PROPOSE", "SIMULATE"}


# ---------------------------------------------------------------------- abnormal cases

@pytest.mark.asyncio
async def test_data_does_not_exist(make_loop):
    run = await make_loop(FakeJava({})).run(Goal(intent="analyze_symbol", symbol="ZZZ"), owner_ctx())
    d = assert_scored("数据不存在", run, ["market_data"], RunStatus.FAILED, "NOT_FOUND")
    assert "recommendation" not in d["result"]


@pytest.mark.asyncio
async def test_api_timeout_is_retried_then_stops(make_loop):
    java = FakeJava({"AAPL": calm_uptrend()}, failures=["timeout", "timeout", "timeout"])
    run = await make_loop(java).run(Goal(intent="analyze_symbol", symbol="AAPL"), owner_ctx())
    d = assert_scored("API timeout", run, ["market_data"] * 3, RunStatus.FAILED, "TIMEOUT")
    assert [c["attempt"] for c in d["tool_calls"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_transient_failure_changes_the_path_and_recovers(make_loop):
    java = FakeJava({"AAPL": calm_uptrend()}, failures=["down"])
    run = await make_loop(java).run(Goal(intent="analyze_symbol", symbol="AAPL"), owner_ctx())
    d = assert_scored("上游短暂不可用后恢复", run, ["market_data", "market_data", "risk", "signal"],
                      RunStatus.SUCCESS)
    assert d["tool_calls"][0]["error"]["code"] == "UNAVAILABLE"
    assert any("retry market_data" in e["message"] for e in d["trace"])


@pytest.mark.asyncio
async def test_high_risk_suppresses_direction_and_trade(make_loop):
    java = FakeJava({"TSLA": wild()})
    run = await make_loop(java).run(
        Goal(intent="analyze_symbol", symbol="TSLA", trade={"side": "buy", "quantity": 5}),
        owner_ctx())
    d = assert_scored("风险过高", run, ["market_data", "risk"], RunStatus.SUCCESS)
    assert d["result"]["recommendation"] == "NO_NEW_EXPOSURE"
    assert d["result"]["execution"]["status"] == "NOT_PROPOSED"
    assert d["result"]["risk"]["risk_level"] == "high"


@pytest.mark.asyncio
async def test_user_without_permission(make_loop):
    db = FakeDB({7: OTHER}, {7: [_position("AAPL")]})
    run = await make_loop(FakeJava({}), db).run(Goal(intent="portfolio_risk", portfolio_id=7), owner_ctx())
    d = assert_scored("用户无权限", run, ["portfolio"], RunStatus.FAILED, "FORBIDDEN")
    assert all("FROM positions" not in q for q in db.queries)  # nothing read after the denial


@pytest.mark.asyncio
async def test_empty_portfolio_stops_before_risk(make_loop):
    db = FakeDB({7: OWNER}, {7: []})
    run = await make_loop(FakeJava({}), db).run(Goal(intent="portfolio_risk", portfolio_id=7), owner_ctx())
    assert_scored("空组合", run, ["portfolio"], RunStatus.FAILED, "INSUFFICIENT_DATA")


@pytest.mark.asyncio
async def test_tool_error_asks_user_to_correct_input(make_loop):
    java = FakeJava({"AAPL": calm_uptrend()})
    run = await make_loop(java).run(
        Goal(intent="strategy_suggestion", symbol="AAPL", strategies=["momentum", "no_such_strategy"]),
        owner_ctx())
    d = assert_scored("Tool 返回错误（非法输入）", run, ["market_data", "backtest", "backtest"],
                      RunStatus.FAILED, "NEEDS_USER_INPUT")
    assert "no_such_strategy" in d["result"]["question"]


@pytest.mark.asyncio
async def test_optional_tool_failure_is_skipped_not_invented(make_loop):
    java = FakeJava({"AAPL": calm_uptrend(40)})  # too short for momentum_120d
    run = await make_loop(java).run(
        Goal(intent="analyze_symbol", symbol="AAPL", factors=["momentum_120d"]), owner_ctx())
    d = assert_scored("可选 Tool 数据不足", run, ["market_data", "risk", "signal", "factor"],
                      RunStatus.SUCCESS)
    assert d["result"]["factors"] is None
    assert any(n.startswith("factor skipped") for n in d["result"]["skipped"])


@pytest.mark.asyncio
async def test_backend_error_stops_after_retries(make_loop):
    java = FakeJava({"AAPL": calm_uptrend()}, failures=["500", "500", "500"])
    run = await make_loop(java).run(Goal(intent="analyze_symbol", symbol="AAPL"), owner_ctx())
    assert_scored("上游 5xx", run, ["market_data"] * 3, RunStatus.FAILED, "UNAVAILABLE")


# ------------------------------------------------------------------- approval boundary

@pytest.mark.asyncio
async def test_execute_requires_human_approval(make_loop):
    java = FakeJava({"AAPL": calm_uptrend()})
    loop = make_loop(java)
    goal = Goal(intent="analyze_symbol", symbol="AAPL", strategies=["momentum", "ma_cross"],
                trade={"side": "buy", "quantity": 5})
    run = await loop.run(goal, owner_ctx())
    d = assert_scored("EXECUTE 需人工确认", run, ["market_data", "risk", "signal", "execution"],
                      RunStatus.SUCCESS)
    assert d["result"]["recommendation"] == "BUY"
    blocked = d["tool_calls"][-1]
    assert blocked["status"] == "BLOCKED" and blocked["error"]["code"] == "APPROVAL_REQUIRED"
    proposal = d["result"]["execution"]["proposal"]
    assert d["result"]["execution"]["status"] == "AWAITING_APPROVAL"
    assert proposal["action"]["reference_price"] == calm_uptrend()[-1]["close"]

    stranger = await loop.execute_approved(proposal["proposal_id"], ToolContext(user_id=OTHER))
    assert stranger.status == RunStatus.FAILED and stranger.error_code == "APPROVAL_FORBIDDEN"

    approved = await loop.execute_approved(proposal["proposal_id"], owner_ctx())
    a = assert_scored("人工确认后纸面执行", approved, ["execution"], RunStatus.SUCCESS)
    fill = a["result"]["execution"]
    assert fill["mode"] == "PAPER" and fill["fill_price"] == proposal["action"]["reference_price"]
    assert fill["approved_by"] == OWNER

    again = await loop.execute_approved(proposal["proposal_id"], owner_ctx())
    assert again.error_code == "APPROVAL_CONSUMED"


@pytest.mark.asyncio
async def test_execution_tool_refuses_without_matching_approval():
    tool = ExecutionTool()
    order = {"symbol": "AAPL", "side": "buy", "quantity": 5, "reference_price": 100.0,
             "reference_date": "2026-09-01"}
    assert (await tool.invoke(order, ToolContext(user_id=OWNER))).error_code.value == "APPROVAL_REQUIRED"
    forged = {"approved_by": OWNER, "proposal_id": "p", "order": {**order, "quantity": 1}}
    result = await tool.invoke(order, ToolContext(user_id=OWNER, approval=forged))
    assert result.error_code.value == "APPROVAL_REQUIRED"  # approval bound to a different order


def test_scorecard_is_all_green():
    """Printed with -s; every collected scenario must pass all four criteria."""
    assert SCORECARD, "scenarios did not run"
    header = f"{'scenario':<22}{'tool_sel':>10}{'no_fabr':>10}{'stop':>8}{'trace':>8}"
    print("\n" + header)
    for row in SCORECARD:
        print(f"{row['scenario']:<22}" + "".join(
            f"{'PASS' if row[k] else 'FAIL':>{w}}" for k, w in
            (("tool_selection", 10), ("no_fabrication", 10), ("correct_stop", 8), ("auditable_trace", 8))))
    assert all(all(v for k, v in row.items() if k != "scenario") for row in SCORECARD)
