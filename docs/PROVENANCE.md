# Provenance: what was copied, what was changed

agent-runtime-kit is an extraction, not a rewrite. This page records exactly what came from where and every change made while taking it out of its host, so a reader can judge what is original work and what is packaging.

## Source

- **Host project**: TraderX (product name "QuantRiskPro V2.0"), the author's private prototype of a trading-management system (2025–2026; Java + FastAPI + React). Written alone; prompted by a remote freelance enquiry that was never concluded (no contract, payment or delivery), so no rights were transferred. The repository stays private because it holds the enquirer's requirement documents and data files; none of that is here.
- **Copied on**: 2026-11-12 from TraderX revision `156d265e`, into a fresh git history (the host's history is not carried over).
- **Copied from** `backend-python/`:

| Host path | Here | Notes |
|---|---|---|
| `src/agents/**` except `browser/` and `orchestrator/realtime/` | `src/agentrt/**` | 49 modules, ~16.5k lines |
| `src/core/mock_guard.py` | `src/agentrt/guards/mock_guard.py` | 83 lines; the single rule for when synthetic data may be served |
| `src/api/agent_routes.py`, `src/api/agents_extended_api.py` | `src/agentrt/api/routes.py`, `extended.py` | optional FastAPI routes |
| `tests/agents/**` (Python only), `tests/test_skill_engine.py` | `tests/**` | 444 test functions; the host's Markdown test notes were not copied |
| `scripts/demo/agent_scenario.py`, `setup_agent_scenario.py`, `scripts/testing/verify_llm_planner.py` | `examples/scripts/` | talk to a running host over HTTP; kept because two tests check them against the real workflow contract |
| `.claude/skills/{quant-signal,quant-factor,quant-risk,quant-decision,quant-backtest,codex,gemini}` | `examples/skills/` | the author's own skill definitions; skill packs from other projects that sat next to them were **not** copied |

**Not copied**: `src/agents/browser/` (Playwright browser agent), `src/agents/orchestrator/realtime/` (WebSocket manager, market-data cache, data-stream agent), `src/api/agents_router.py` (depends on the realtime pipeline), the React `AgentWorkbench`, database migrations, docker-compose files, the host's release audit and demo documents.

## Changes made during extraction

All changes are mechanical or boundary-related; no agent logic, planner rule, approval rule or tool schema was altered.

### 1. Package rename
`src.agents.*` → `agentrt.*`; `src.core.mock_guard` → `agentrt.guards.mock_guard`; `src.api.agent_routes` → `agentrt.api.routes`. Relative imports inside the package were unchanged.

### 2. Ports (new package `agentrt.ports`)
The host provided five things the agents reached by importing host modules. Each is now a small port with a default that needs no external service:

| Port | Replaces | Default |
|---|---|---|
| `ports.db` | `src.database.session.db_session` / `get_async_db` | none — raises `PortNotConfigured` until `configure_session_factory(...)` is called; the in-memory stores need nothing |
| `ports.strategies` | `src.strategies.get_strategy` / `STRATEGY_REGISTRY` | binds to `quantrisk.strategies` when the optional dependency quantrisk-core is installed; otherwise no names are offered |
| `ports.portfolio` | `src.api.portfolio_analytics.verify_portfolio_access`, `src.services.portfolio_analytics.PortfolioAnalyticsService` | `SqlPortfolioProvider`, which issues the same three SQL statements the host did against `portfolios` / `positions` through whatever session the tool was given; the analytics service for RiskAgent is host-specific and must be configured |
| `ports.market` | `src.config.system_parameters.JAVA_API_BASE_URL`, `src.api.market_indicators.generate_service_account_token` | environment variables `AGENTRT_EOD_BASE_URL`, `AGENTRT_EOD_TOKEN` |
| `ports.backtest` | `src.engine.backtest.BacktestEngine` / `MonteCarloSimulator`, `src.datasources.DataSourceManager` | none — the legacy `BacktestAgent` fails the task with a clear error; the runtime's `backtest` tool does not use this port |

Files touched for this: `runtime/approval.py`, `runtime/memory.py`, `runtime/planning.py`, `tools/portfolio.py`, `tools/market_data.py`, `tools/analysis.py`, `quant/backtest_agent.py`, `quant/risk_agent.py`, `api/routes.py` (also dropped the SQLAlchemy type import; `db` parameters are typed `Any`).

### 3. Two behavioural fixes found by the tests
- `skills/engine.py`: the sandbox ran skill scripts with a bare `python`, and a missing interpreter was reported as "Script not found". It now runs `sys.executable`.
- `ports/backtest.py` error text names the runtime's `backtest` tool, which the legacy agent's failure message already promised.

### 4. Tests
- Two tests that load example scripts by path were repointed from the host layout to `examples/scripts/`.
- The six tests that exercised the host's event-driven backtest engine are marked `needs_backtest_engine` and skip when `ports.backtest` is not configured.
- Modules that need named strategies are marked `needs_strategies` and skip when the strategy port offers none (i.e. quantrisk-core is not installed). CI runs both configurations.
- One test scans the package for unguarded random draws; its root path was updated.

### 5. Licensing and headers
49 copied files already carried `SPDX-License-Identifier: Apache-2.0`; the same header was added to the 10 that had none (`orchestrator/*`, `api/*`). The repository license is Apache-2.0 to match. See `NOTICE`.

## What is *not* verified

- The quant agents (`agentrt.quant`) compute signals, factors, risk and optimisation; their numerical correctness was never independently checked in the host either. Treat them as examples of the agent pattern, not as validated analytics.
- The LLM planner / critic / decision paths were exercised in the host with scripted fakes; a run against a real model was attempted in the host's 2026-09 release audit and blocked for lack of credentials. After the extraction, the sale / approval scenario was run against a real model (DeepSeek, sandbox data, 2026-11-12); see [VALIDATION.md](VALIDATION.md) and README "What is and is not verified".
