# Architecture

## One run, one audit record

Every agent invocation — a runtime loop run or a legacy `/dispatch` call — produces one `AgentRun` (`agentrt/runtime/contract.py`). It answers four questions from data, not prose:

| Question | Field |
|---|---|
| what the agent called | `tool_calls` (each with its own provenance) |
| what data it used | `data_provenance` |
| why it reached its result | `trace` — plan / observation / decision events |
| where it failed | `status`, `stop_reason`, the failing call |

Serialized shape:

```json
{"task_id": "", "agent": "", "input": {},
 "tool_calls": [], "observations": [], "result": {},
 "status": "RUNNING|SUCCESS|FAILED",
 "data_provenance": {}, "trace": [],
 "action_level": "...", "stop_reason": "...", "started_at": "...", "finished_at": "..."}
```

## The loop

```
Goal -> Plan -> Tool call -> Observation -> Decision -> Final
```

`runtime/loop.py` owns the mechanics — calling tools, the approval gate, recording the contract — and asks the planner for the next step after every observation. The loop never decides what to do itself; the planner never touches a tool directly.

### Planner: rules first, model second

`runtime/planner.py` is an observation-driven, deterministic, rule-based policy: `next_action(state)` is called after every observation and picks the single next step. Retryable tool errors (TIMEOUT / UNAVAILABLE) are retried then stop the run; NOT_FOUND / FORBIDDEN / UNAUTHENTICATED stop immediately; INVALID_INPUT asks the user to correct the goal; INSUFFICIENT_DATA stops a required step and skips (and reports) an optional one; a "high" risk observation ends directional analysis and suppresses any trade proposal.

`runtime/planning.py` adds an LLM planner on top: the model proposes a plan, every step is validated against the tool catalogue (tool names, input names, allowed values such as the strategy list from `ports.strategies`), and anything that does not validate falls back to the rule planner. The trace records the model, the raw planner output, the validation result and whether a fallback happened. Without model credentials the rule planner is the only path.

### Tools, not services

`tools/base.py`: agents never read a database, call an internal service or generate data themselves. Every tool declares a pydantic input and output schema (enforced on each call — output that drifts from its schema fails the call), returns either output *with provenance* or a typed error (no "worked, sort of" state), and declares an `ActionLevel` the approval gate checks before the call is made.

Built-in tools: `market_data` (HTTP EOD service via `ports.market`), `portfolio` and `risk` (via `ports.portfolio`), `signal`, `backtest`, `factor` (strategies via `ports.strategies`), `execution` (PAPER only — there is no exchange client in this package).

### Approval boundary

`runtime/approval.py`:

```
ANALYSIS  runs automatically
PROPOSE   output only (no tool is PROPOSE; it is what a run produces)
SIMULATE  runs automatically - simulations have no side effects
EXECUTE   never runs on the agent's own decision
```

When a run reaches an EXECUTE action, the gate blocks it and stores the exact action as a `Proposal`. It can only be carried out by a later request from the same authenticated user naming the proposal id; the approval is bound to the stored action, so a caller cannot approve one order and submit another, and the agent has no way to produce an approval itself. Transitions (`PENDING → APPROVED → EXECUTED | FAILED`, `PENDING → EXPIRED`) are compare-and-set, so a proposal executes at most once even under concurrent approvals.

### Decision workflow

`runtime/workflow.py` composes agents with single duties:

```
User goal -> Memory retrieval -> Planner (LLM, rule fallback) -> Research Agent
          -> Risk Agent -> Critic Agent -> (one re-analysis on request_revision)
          -> Portfolio Decision Agent -> (PAPER proposal, human approval)
```

Each agent returns a frozen `AgentReport`; no agent can change another's report or the plan. Agents reach data only through `ToolRunner`, which records each call in the contract tagged with the agent, resolves `"$bars"` to bars a `market_data` call actually returned, and refuses any tool outside the agent's allowance — the workflow's ceiling is PROPOSE / SIMULATE. `runtime/review.py` (critic) can add issues but cannot remove rule findings; an LLM decision that cites numbers not present in the reports, or an action level the rules do not allow, is discarded in favour of the rule decision.

### Memory

`runtime/memory.py` stores user preferences and run conclusions (`MemoryStore`; in-memory and SQL implementations). Retrieved memory is context only: a price smuggled into memory is never used as data (tested).

## Other subsystems

- **`debate/`** — three-round multi-model debate (parallel proposals → cross-examination → consensus) over `ModelAdapter`s for Claude, Gemini, OpenAI and their CLIs, plus a `MockAdapter` for tests.
- **`orchestrator/`** — registry, pool and orchestrator for `BaseAgent`s (priority scheduling, dependency DAG, retries, circuit breaker), and four domain agents (data source, signal aggregation, risk assessment, execution) wired into a `TradingPipeline`.
- **`skills/`** — discovery and execution of `SKILL.md`-defined skills (YAML front matter, optional scripts run in a subprocess sandbox with an environment allow-list).
- **`quant/`** — signal / factor / risk / decision / optimisation / backtest agents. The `BacktestAgent` needs a host-provided event-driven engine (`ports.backtest`); the runtime's `backtest` tool does not.
- **`guards/mock_guard.py`** — the one place that decides whether synthetic numbers may be served (`ALLOW_MOCK_AGENT_DATA`); by default any attempt raises, and when enabled every payload is tagged as mock. A test scans the package to prove every random draw sits behind it.
- **`api/`** (optional, FastAPI) — `/runtime/run`, `/runtime/workflow`, proposals, memory, dispatch, debate and skills endpoints. Ownership of a portfolio is checked through `ports.portfolio`; unauthenticated requests get 401.

## Ports

Everything the original host supplied is behind `agentrt.ports` (database session, strategy registry, portfolio storage, EOD price service, backtest engine). `import agentrt` builds no engine, opens no connection and reads no credentials. See `docs/PROVENANCE.md` for the mapping.
