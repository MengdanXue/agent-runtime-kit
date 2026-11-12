---
name: quant-decision
description: AI-powered trading decisions using multi-model debate. Use when you need to make important portfolio decisions, strategy selections, or risk management choices with AI collaboration.
version: 1.0.0
triggers:
  - "portfolio decision"
  - "should I buy"
  - "strategy decision"
  - "trading decision"
  - "ai decision"
---

# Quant Decision Skill

## Overview

Make AI-powered trading decisions using multi-model debate (Claude, Gemini, Codex) for important choices. Integrates signal generation, risk analysis, and strategic reasoning.

## When to Use

```
✅ Use for:
  • Portfolio allocation decisions
  • Strategy selection
  • Risk management decisions
  • High-value trade decisions
  • Complex market situations

❌ Don't use for:
  • Simple signal queries (use quant-signal)
  • Historical analysis (use quant-backtest)
  • Basic risk metrics (use quant-risk)
```

## Decision Types

| Type | Debate Tier | Description |
|------|-------------|-------------|
| portfolio_decision | Tier 2 (Full) | Portfolio-level changes |
| strategy_selection | Tier 1 (Quick) | Choose trading strategy |
| risk_decision | Tier 2 (Full) | Risk management actions |
| trade_decision | Tier 1 (Quick) | Individual trade decisions |

## Debate Tiers

| Tier | Models | Latency | Use Case |
|------|--------|---------|----------|
| 0 | Claude only | <1s | Simple decisions |
| 1 | Claude + Gemini | 2-5s | Moderate importance |
| 2 | Claude + Gemini + Codex | 5-15s | Critical decisions |

## Usage

### Portfolio Decision

```python
from agentrt.quant import DecisionAgent
from agentrt.debate import DebateEngine, ClaudeAdapter
from agentrt import AgentTask

# Setup with debate engine
claude = ClaudeAdapter(api_key="...")
debate_engine = DebateEngine(claude)
agent = DecisionAgent(debate_engine=debate_engine)

task = AgentTask(
    agent_type="decision",
    payload={
        "type": "portfolio_decision",
        "portfolio_id": "portfolio_123",
        "context": {
            "market_trend": "bearish",
            "portfolio_beta": 1.3,
            "risk_level": "high",
            "cash_available": 50000
        }
    }
)

result = await agent.execute(task)
print(f"Decision: {result.result['decision']}")
print(f"Confidence: {result.result['confidence']:.0%}")
print(f"Actions: {result.result['actions']}")
```

### Strategy Selection

```python
task = AgentTask(
    agent_type="decision",
    payload={
        "type": "strategy_selection",
        "context": {
            "symbol": "TSLA",
            "market_regime": "volatile",
            "available_strategies": ["momentum", "mean_reversion", "volatility"]
        }
    }
)

result = await agent.execute(task)
print(f"Recommended strategy: {result.result['decision']}")
```

### Force Full Debate

```python
task = AgentTask(
    agent_type="decision",
    payload={
        "type": "trade_decision",
        "context": {
            "symbol": "NVDA",
            "direction": "buy",
            "size": 100000
        },
        "force_debate": True,  # Force Tier 2 debate
        "debate_tier": 2
    }
)
```

## Response Format

```json
{
  "decision_type": "portfolio_decision",
  "question": "Given the current portfolio state and market conditions...",
  "decision": "REDUCE_EXPOSURE - Recommend reducing portfolio beta by selling high-beta positions",
  "confidence": 0.85,
  "reasoning": "## Consensus Points\n1. All models agree on reducing risk...",
  "debate_used": true,
  "debate_tier": 2,
  "dissent": [
    "Gemini suggests considering hedging instead of selling"
  ],
  "actions": ["REDUCE_EXPOSURE", "HEDGE"]
}
```

## Possible Actions

| Action | Description |
|--------|-------------|
| HOLD | Maintain current positions |
| BUY | Add to positions |
| SELL | Reduce positions |
| HEDGE | Add hedging positions |
| REDUCE_EXPOSURE | Lower overall risk |
| INCREASE_EXPOSURE | Add risk |
| REBALANCE | Adjust allocation |
| REVIEW | Manual review needed |

## Automatic Tier Escalation

The decision agent automatically escalates to higher debate tiers based on:

1. **Trade Value**: >$100k → Tier 2
2. **Risk Level**: "high" or "critical" → Tier 2
3. **Decision Type**: portfolio/risk decisions → Tier 2
4. **Explicit Override**: `debate_tier` parameter

## Multi-Model Debate Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Decision Request                          │
└─────────────────────────────────────────────────────────────┘
                            │
                   Determine Debate Tier
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
      Tier 0             Tier 1             Tier 2
    (Claude)      (Claude + Gemini)   (All 3 Models)
         │                  │                  │
         │                  │                  │
    Single Model       Quick Debate       Full Debate
    Decision           (2 rounds)         (3 rounds)
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                   Build Consensus
                            │
                   Return Decision + Actions
```

## Integration Example

```python
# Complete decision flow with all agents
async def make_informed_decision(symbol: str, portfolio_id: str):
    # 1. Generate signals
    signals = await signal_agent.execute(AgentTask(
        agent_type="signal",
        payload={"symbol": symbol, "strategies": ["momentum", "mean_reversion"]}
    ))

    # 2. Analyze risk
    risk = await risk_agent.execute(AgentTask(
        agent_type="risk",
        payload={"portfolio_id": portfolio_id}
    ))

    # 3. Make decision with all context
    decision = await decision_agent.execute(AgentTask(
        agent_type="decision",
        payload={
            "type": "trade_decision",
            "context": {
                "symbol": symbol,
                "signals": signals.result,
                "risk_metrics": risk.result["metrics"],
                "risk_level": risk.result["risk_level"]
            }
        }
    ))

    return decision.result
```

## Related Skills

- `quant-signal` - Generate input signals
- `quant-risk` - Provide risk context
- `quant-backtest` - Validate decisions historically
- `multi-model-debate` - Underlying debate mechanism
