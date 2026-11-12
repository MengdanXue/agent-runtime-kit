# agent-runtime-kit

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

An LLM agent runtime for decisions that have consequences, built so that every run can be audited after the fact and nothing is executed without a human.

**The problem it addresses.** Agent demos usually let a model pick tools freely, fill gaps with plausible numbers, and act on its own conclusion. For anything touching money that is backwards. This runtime inverts it:

- **Every run is an audit record.** One `AgentRun` per invocation: which tools were called with what, where each number came from, the plan / observation / decision trace, and exactly where it stopped. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Rules first, model second.** A deterministic planner handles every tool outcome (retry, stop, ask, skip). An LLM may propose a plan or a decision, but each one is validated against the tool catalogue and the rule findings; anything that does not validate falls back to the rules, and the trace says so.
- **No fabricated data.** Tools return output *with provenance* or a typed error. A single guard decides whether synthetic numbers may ever be served; a test scans the package to prove every random draw sits behind it.
- **Humans approve execution.** EXECUTE actions are never run on the agent's decision. They become proposals that only the owning user can approve, bound to the exact stored action, executed at most once — and in this package execution is PAPER only: there is no exchange or broker client.

Extracted from the author's private trading-system prototype (TraderX / QuantRiskPro). What came from where, and what changed on the way out, is recorded in [NOTICE](NOTICE) and [docs/PROVENANCE.md](docs/PROVENANCE.md). The companion library for strategies and backtesting is [quantrisk-core](https://github.com/MengdanXue/quantrisk-core).

> For study and research. Nothing here is investment advice, and the quant agents' numbers are not validated (see below).

## Install

```bash
pip install -e .                      # core: pydantic, anthropic, httpx, PyYAML
pip install -e ".[dev]"               # + pytest and the optional numeric / API deps used by tests
pip install "git+https://github.com/MengdanXue/quantrisk-core.git"   # strategy registry for signal / backtest tools
```

Extras: `openai`, `gemini` (debate adapters), `quant` (numpy / pandas / scipy for the quant agents), `api` (FastAPI routes), `skills` (file-watching skill discovery).

No API key, database or network is needed to import the package or run the tests. With `ANTHROPIC_API_KEY` set, the LLM planner / critic / decision paths are used (with validation and rule fallback); without it, the rule planner runs alone.

## A run in a few lines

```python
import asyncio
from agentrt.runtime.loop import AgentLoop
from agentrt.runtime.planner import Goal
from agentrt.tools import ToolContext, ToolRegistry, MarketDataTool, SignalTool, BacktestTool

loop = AgentLoop(ToolRegistry([MarketDataTool(request_timeout=2.0), SignalTool(), BacktestTool()]))
run = asyncio.run(loop.run(Goal(intent="analyze_symbol", symbol="AAPL"), ToolContext(user_id=1)))
print(run.status, "|", run.stop_reason)
print([e["kind"] + ": " + e["message"] for e in run.to_dict()["trace"]])
```

With no EOD service running this prints, verbatim:

```
RunStatus.FAILED | market_data still failing after 3 attempts: EOD backend timed out: ReadTimeout('')
['goal: goal received: analyze_symbol',
 'plan: need price history for AAPL',
 'tool_call: market_data -> TIMEOUT',
 "observation: market_data failed: EOD backend timed out: ReadTimeout('')",
 'plan: retry market_data after TIMEOUT (attempt 2 of 3)', ...]
```

That is the behaviour the package exists for: a typed error, a bounded retry policy, a trace of every attempt, and no invented prices. Point `MarketDataTool` at a real EOD endpoint (`AGENTRT_EOD_BASE_URL`, see `agentrt/ports/market.py`) or inject an `httpx` transport as the tests do, and the same run continues into signals and a backtest.

### The full decision workflow

```python
from agentrt.runtime.workflow import DecisionWorkflow
```

`User goal → memory → planner → research → risk → critic → decision → (PAPER proposal → human approval)`. Each agent returns a frozen report; the critic can add issues but not remove rule findings; a trade becomes a proposal that only its owner can approve. `tests/runtime/test_workflow.py` and `test_proposal_flow.py` walk every path with fake tools and a scripted LLM.

### Ports: plugging in your own data

`agentrt.ports` is where the runtime touches things it does not own — a database session, a strategy registry, portfolio storage, a price service. Each has a default that needs nothing external, or raises `PortNotConfigured` naming the one call that configures it:

```python
from agentrt.ports import db, portfolio, strategies
db.configure_session_factory(my_async_sessionmaker)   # enables the SQL-backed proposal / memory stores
portfolio.configure(MyPortfolioProvider())            # verify_access / positions / analytics
strategies.configure(my_registry)                     # get_strategy(name, params) / names()
```

## Other pieces

| Package | What it is |
|---|---|
| `agentrt.debate` | Three-round multi-model debate (proposals → cross-examination → consensus) over Claude / Gemini / OpenAI adapters, plus a mock adapter |
| `agentrt.orchestrator` | Registry, pool and orchestrator for `BaseAgent`s (priorities, dependency DAG, retries, circuit breaker); a four-agent trading pipeline |
| `agentrt.skills` | `SKILL.md`-defined skills with YAML front matter, discovered from a directory and run in a subprocess sandbox; examples in `examples/skills/` |
| `agentrt.quant` | Signal / factor / risk / decision / optimisation / backtest agents — the agent *pattern*; numbers not validated |
| `agentrt.api` | Optional FastAPI routes (`/runtime/run`, `/runtime/workflow`, proposals, memory, dispatch, debate, skills) |
| `examples/scripts/` | Scenario and planner-verification scripts that drive a running host over HTTP; two tests check them against the real workflow contract |

## Tests

```bash
pytest -q -rs
```

517 tests pass and 9 skip with quantrisk-core installed (2026-11-12, Python 3.12). The skips are stated: 6 tests exercise the legacy `BacktestAgent` against a host-provided event-driven engine that is not part of this package; 3 integration tests look for a developer's `.claude/skills` directory. Without quantrisk-core the modules that need named strategies skip as a block instead of failing; CI runs both configurations on Python 3.10–3.12.

All tests run with fake tools, a scripted LLM and in-memory stores: no API key, database or network.

## What is and is not verified

| | Verified (tests in this repository) | Not verified |
|---|---|---|
| Runtime | approval gate (owner-only, bound action, at-most-once under concurrency), typed tool errors and their planner handling, trace completeness, memory never used as data, LLM plan / decision validation and fallback, PAPER-only execution | behaviour against a **real** model end-to-end: the host's 2026-09 release audit attempted it and was blocked for lack of credentials |
| Tools | schema enforcement, provenance on every success, ownership checks, EOD client against a mocked transport | any live EOD service |
| Guards | every random draw in the package is behind `require_real_data` (AST scan) | — |
| Quant agents | they run and refuse to fabricate | **numerical correctness** of signals, factors, risk and optimisation |
| Debate / skills / orchestrator | protocol mechanics with mock adapters, sandbox execution, scheduling and retries | quality of model output |

No return figures appear in this repository, and nothing here places real orders.

## License

Apache-2.0. Source headers carry the original project name (TraderX); see [NOTICE](NOTICE).
