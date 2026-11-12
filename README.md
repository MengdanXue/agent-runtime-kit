# agent-runtime-kit

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

An LLM agent runtime for decisions that have consequences, built so that every run can be audited after the fact and nothing is executed without a human.

**The problem it addresses.** Agent demos usually let a model pick tools freely, fill gaps with plausible numbers, and act on its own conclusion. For anything touching money that is backwards. This runtime inverts it:

- **Every run is an audit record.** One `AgentRun` per invocation: which tools were called with what, where each number came from, the plan / observation / decision trace, and exactly where it stopped. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- **Rules first, model second.** A deterministic planner handles every tool outcome (retry, stop, ask, skip). An LLM may propose a plan or a decision, but each one is validated against the tool catalogue and the rule findings; anything that does not validate falls back to the rules, and the trace says so. The one exception is the model's *risk review* of a trade: when that review is required, a review that is invalid, times out or fails blocks the trade instead of falling back (see below).
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

Extras: `openai`, `gemini` (debate adapters), `quant` (numpy / pandas / scipy for the quant agents), `api` (FastAPI routes), `sql` (SQLAlchemy for the SQL-backed stores), `skills` (file-watching skill discovery).

No API key, database or network is needed to import the package or run the tests.

**Two modes.** The model client is configured with `ANTHROPIC_API_KEY`, or with `AGENT_LLM_PROVIDER=deepseek` and `AGENT_LLM_API_KEY` (see `agentrt/runtime/llm.py`).

| | Rule-only mode | Required model review |
|---|---|---|
| When | no model client is configured | default whenever a model client is configured — also one that is configured but unusable (missing key, wrong provider); or `AGENT_LLM_REVIEW_REQUIRED=1` |
| Planner / decision | rules | model, validated; rule fallback on any invalid answer |
| Critic | rule review | rule review + model review (the model can add issues, never remove the rule's) |
| Model review invalid, timed out or failed | — | **the requested trade is not proposed**: a non-downgradable `LLM_REVIEW_INVALID` / `LLM_REVIEW_UNAVAILABLE` issue; the raw answer and the failing field are kept in the run record |

`AGENT_LLM_REVIEW_REQUIRED=0` switches the requirement off explicitly; every run records the policy it ran under (`result.review_policy`). In both modes every trade is only a PAPER proposal that its owner must approve.

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

Two narrow exceptions exist for a sale that reduces a position above the user's single-position limit — the one trade that cannot add exposure. Both are decided by code from the reports, never by the model:

- **One re-analysis** may run although the Risk Agent vetoed new exposure, if that concentration limit is the only veto and the sale is no larger than the long holding. Every blocking rule still applies afterwards.
- **Research-quality issues may become WARNINGs** (sample length, factor coverage, backtest coverage, historical performance, signal disagreement — a closed list) only if all eleven conditions in `review.research_quality_waiver` hold: among them, the sale lowers the measured weight and creates no new limit breach, the risk levels recompute from their own metrics and policy, no hard-rule issue exists at any severity, and the model review passed schema and semantic validation. Data contradictions, risk-assessment, permission and unknown issues always block. The proposal still needs the owner's approval and executes PAPER only.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#decision-workflow).

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

The S2c sale / approval validation has its own deterministic suite, which replays recorded DeepSeek answers instead of calling a model (no key, no network):

```bash
AGENTRT_REQUIRE_STRATEGIES=1 pytest -q -rs validation/real_llm/test_s2c.py
```

| Environment | `pytest -q -rs` (tests/) | `validation/real_llm/test_s2c.py` |
|---|---|---|
| `.[dev]` + quantrisk-core | 517 passed, 9 skipped | 116 passed, 0 skipped |
| `.[dev]` only | 317 passed, 209 skipped | 53 passed, 63 skipped |

(2026-11-12, fresh clone, Python 3.12.) Without quantrisk-core the tests that run the decision workflow are **skipped, not passed** — its research step needs named strategies — and each skip states that reason; the schema checks, the required-review block (run on the critic directly), the research-quality gate and the review policy still run. `AGENTRT_REQUIRE_STRATEGIES=1` turns a missing strategy provider into an error, so a complete environment cannot skip silently. CI runs the suite in both configurations.

## What is and is not verified

| | Verified (tests in this repository) | Not verified |
|---|---|---|
| Runtime | approval gate (owner-only, bound action, at-most-once under concurrency), typed tool errors and their planner handling, trace completeness, memory never used as data, LLM plan / decision validation and fallback, required-review fail-closed, PAPER-only execution | — |
| Real model (DeepSeek, S2c sandbox) | wired in and run end-to-end on sandbox data; observed: schema validation, rejection of reviews citing numbers not in the reports, the required-review block, risk blocks, complete audit records ([docs/VALIDATION.md](docs/VALIDATION.md)) | a real-model run in which a legitimate sale passes the research-quality rule to approval and a PAPER fill: **not observed** — the real reviews always raised at least one issue that must block. The research-quality rule's acceptance path is shown by deterministic tests only |
| Tools | schema enforcement, provenance on every success, ownership checks, EOD client against a mocked transport | any live EOD service |
| Guards | every random draw in the package is behind `require_real_data` (AST scan) | — |
| Quant agents | they run and refuse to fabricate | **numerical correctness** of signals, factors, risk and optimisation |
| Metric definitions | stated in every report (`basis`, `level_basis`): backtest Sharpe = (mean daily return × 252 − 0.03) / (daily std × √252), i.e. net of a fixed 3% risk-free rate; risk levels use VaR (portfolio) or volatility and drawdown (symbol) only, never returns | the S2c sandbox's portfolio Sharpe uses no risk-free rate, unlike the backtest Sharpe; a host analytics service's definitions are its own |
| Debate / skills / orchestrator | protocol mechanics with mock adapters, sandbox execution, scheduling and retries | quality of model output |

No return figures appear in this repository, and nothing here places real orders.

## License

Apache-2.0. Source headers carry the original project name (TraderX); see [NOTICE](NOTICE).
