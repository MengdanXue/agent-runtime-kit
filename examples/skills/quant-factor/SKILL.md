---
name: quant-factor
description: Factor analysis and evaluation for quantitative trading. Use when you need to analyze factor performance, calculate IC/IR, run factor attribution, or build multi-factor models.
version: 1.0.0
triggers:
  - "factor analysis"
  - "IC IR"
  - "factor attribution"
  - "multi-factor"
  - "alpha factor"
---

# Quant Factor Skill

## Overview

Comprehensive factor analysis toolkit for quantitative research. Analyze factor performance, calculate information coefficients, run factor attribution, and build multi-factor models.

## When to Use

```
✅ Use for:
  • Factor performance analysis (IC, IR, turnover)
  • Factor attribution and decomposition
  • Multi-factor model construction
  • Factor correlation analysis
  • Factor decay analysis
  • Cross-sectional factor ranking

❌ Don't use for:
  • Strategy backtesting (use quant-backtest)
  • Real-time signals (use quant-signal)
  • Portfolio risk analysis (use quant-risk)
```

## Factor Types Supported

| Category | Factors |
|----------|---------|
| **Momentum** | Price momentum, earnings momentum, volume momentum |
| **Value** | P/E, P/B, EV/EBITDA, dividend yield |
| **Quality** | ROE, profit margin, debt ratio, earnings stability |
| **Volatility** | Historical vol, implied vol, beta, idiosyncratic risk |
| **Technical** | RSI, MACD deviation, moving average distance |
| **Custom** | User-defined factors via expression language |

## Usage

### Factor Performance Analysis

```python
from agentrt.quant import FactorAgent
from agentrt import AgentTask

agent = FactorAgent()
task = AgentTask(
    agent_type="factor",
    payload={
        "type": "performance_analysis",
        "factor": "momentum_20d",
        "universe": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "holding_period": 20  # days
    }
)

result = await agent.execute(task)
print(f"IC Mean: {result.result['ic_mean']}")
print(f"IC IR: {result.result['ic_ir']}")
print(f"Turnover: {result.result['turnover']}")
```

### Factor Attribution

```python
task = AgentTask(
    agent_type="factor",
    payload={
        "type": "attribution",
        "portfolio_id": "my_portfolio",
        "factors": ["momentum", "value", "quality", "volatility"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
)

result = await agent.execute(task)
# Returns factor exposures and contribution to returns
```

### Multi-Factor Model

```python
task = AgentTask(
    agent_type="factor",
    payload={
        "type": "multi_factor_model",
        "factors": {
            "momentum": {"weight": 0.3, "lookback": 20},
            "value": {"weight": 0.25, "metric": "pe_ratio"},
            "quality": {"weight": 0.25, "metric": "roe"},
            "volatility": {"weight": 0.2, "lookback": 60}
        },
        "universe": "SP500",
        "rebalance_frequency": "monthly"
    }
)

result = await agent.execute(task)
print(f"Composite scores: {result.result['scores']}")
```

### Factor Correlation Matrix

```python
task = AgentTask(
    agent_type="factor",
    payload={
        "type": "correlation_analysis",
        "factors": ["momentum", "value", "quality", "size", "volatility"],
        "universe": "SP500",
        "lookback_days": 252
    }
)

result = await agent.execute(task)
# Returns correlation matrix and cluster analysis
```

## Response Formats

### Performance Analysis

```json
{
  "type": "performance_analysis",
  "factor": "momentum_20d",
  "period": "2022-01-01 to 2023-12-31",
  "metrics": {
    "ic_mean": 0.045,
    "ic_std": 0.12,
    "ic_ir": 0.375,
    "ic_hit_rate": 0.62,
    "turnover": 0.35,
    "quintile_spread": 0.08,
    "t_stat": 2.85
  },
  "quintile_returns": {
    "Q1": -0.02,
    "Q2": 0.01,
    "Q3": 0.03,
    "Q4": 0.05,
    "Q5": 0.06
  },
  "ic_series": [...],
  "decay_analysis": {
    "half_life_days": 15,
    "decay_rate": 0.046
  }
}
```

### Attribution

```json
{
  "type": "attribution",
  "portfolio_id": "my_portfolio",
  "total_return": 0.18,
  "factor_contributions": {
    "momentum": 0.05,
    "value": 0.03,
    "quality": 0.04,
    "volatility": -0.02,
    "specific": 0.08
  },
  "factor_exposures": {
    "momentum": 0.8,
    "value": -0.3,
    "quality": 1.2,
    "volatility": 0.5
  }
}
```

## Key Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| **IC** | Information Coefficient (rank correlation) | > 0.03 |
| **IR** | Information Ratio (IC mean / IC std) | > 0.3 |
| **Turnover** | Portfolio turnover rate | < 0.5 |
| **T-stat** | Statistical significance | > 2.0 |
| **Quintile Spread** | Q5 - Q1 return | > 0.05 |
| **Decay Half-life** | Days until IC halves | > 10 days |

## Integration with Other Skills

```
quant-factor → quant-backtest
   │              │
   │              ▼
   │         Test factor-based strategy
   │
   └──→ quant-risk
              │
              ▼
         Analyze factor risk exposure
```

## Related Skills

- `quant-backtest` - Test factor-based strategies
- `quant-risk` - Analyze factor risk exposures
- `quant-signal` - Generate signals from factors
- `quant-decision` - Factor-informed decisions
