---
name: quant-signal
description: Generate multi-strategy trading signals with AI-powered analysis. Use when you need to analyze symbols, generate buy/sell signals, or aggregate multiple strategy outputs.
version: 1.0.0
triggers:
  - "generate signal"
  - "trading signal"
  - "signal for"
  - "buy or sell"
  - "analyze symbol"
---

# Quant Signal Skill

## Overview

Generate trading signals using multiple quantitative strategies with optional AI-powered analysis and multi-model debate for important decisions.

## When to Use

```
✅ Use for:
  • Generating trading signals for symbols
  • Multi-strategy signal aggregation
  • Technical indicator analysis
  • Entry/exit point identification

❌ Don't use for:
  • Historical backtesting (use quant-backtest)
  • Risk analysis (use quant-risk)
  • Portfolio decisions (use quant-decision)
```

## Available Strategies

| Category | Strategies |
|----------|-----------|
| Momentum | MomentumStrategy, RelativeStrengthStrategy, BreakoutStrategy, DualMomentumStrategy |
| Mean Reversion | BollingerReversionStrategy, RSIReversionStrategy, ZScoreReversionStrategy |
| Trend Following | MovingAverageCrossStrategy, MACDStrategy, TurtleTradingStrategy, TripleMAStrategy |
| Volatility | VolatilityBreakoutStrategy, ATRChannelStrategy |

## Signal Values

- `STRONG_BUY`: High confidence buy signal
- `BUY`: Buy signal
- `HOLD`: No action recommended
- `SELL`: Sell signal
- `STRONG_SELL`: High confidence sell signal

## Usage

### Basic Signal Generation

```python
from agentrt.quant import SignalAgent
from agentrt import AgentTask

agent = SignalAgent()
task = AgentTask(
    agent_type="signal",
    payload={
        "symbol": "AAPL",
        "timeframe": "1d",
        "strategies": ["momentum", "mean_reversion", "macd"],
        "aggregate": True
    }
)

result = await agent.execute(task)
print(result.result["aggregated"])
# {'signal': 'BUY', 'score': 0.75, 'confidence': 0.82}
```

### Via Orchestrator

```python
from agentrt import AgentOrchestrator, AgentRegistry, AgentPool, AgentTask
from agentrt.quant import SignalAgent

registry = AgentRegistry()
registry.register("signal", SignalAgent)

pool = AgentPool(max_concurrent=5)
orchestrator = AgentOrchestrator(registry, pool)

task = AgentTask(
    agent_type="signal",
    payload={
        "symbol": "TSLA",
        "strategies": ["momentum", "breakout"]
    }
)

result = await orchestrator.dispatch(task)
```

## Response Format

```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "signals": [
    {
      "strategy": "momentum",
      "signal": "BUY",
      "confidence": 0.85,
      "metadata": {}
    },
    {
      "strategy": "mean_reversion",
      "signal": "HOLD",
      "confidence": 0.72,
      "metadata": {}
    }
  ],
  "aggregated": {
    "signal": "BUY",
    "score": 0.65,
    "confidence": 0.78,
    "contributing_signals": 2
  },
  "strategies_used": ["momentum", "mean_reversion"]
}
```

## Integration with Multi-Model Debate

For important signals, you can escalate to multi-model debate:

```python
from agentrt.debate import DebateEngine, DebateTier

# Use debate for high-value decisions
debate_engine = DebateEngine(claude_adapter)
result = await debate_engine.run_debate(
    question=f"Should we buy {symbol} based on these signals: {signals}",
    context={"signals": signals, "trade_value": 50000},
    tier=DebateTier.TIER_1
)
```

## Related Skills

- `quant-backtest` - Test signal performance historically
- `quant-risk` - Analyze risk of signal-based trades
- `quant-decision` - Make portfolio-level decisions
