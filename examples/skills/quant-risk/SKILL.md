---
name: quant-risk
description: Comprehensive portfolio risk analysis including VaR, stress tests, and correlation analysis. Use when you need to assess portfolio risk, run stress scenarios, or analyze position correlations.
version: 1.0.0
triggers:
  - "risk analysis"
  - "calculate var"
  - "portfolio risk"
  - "stress test"
  - "correlation matrix"
---

# Quant Risk Skill

## Overview

Perform comprehensive portfolio risk analysis including Value at Risk (VaR), Conditional VaR, stress testing, and correlation analysis.

## When to Use

```
✅ Use for:
  • Portfolio risk assessment
  • VaR and CVaR calculation
  • Stress test scenarios
  • Correlation matrix analysis
  • Risk contribution by position

❌ Don't use for:
  • Signal generation (use quant-signal)
  • Strategy backtesting (use quant-backtest)
  • Trade decisions (use quant-decision)
```

## Analysis Types

| Type | Description |
|------|-------------|
| Full Risk Analysis | Comprehensive risk metrics |
| VaR Calculation | Value at Risk with multiple methods |
| Stress Test | Scenario-based impact analysis |
| Correlation | Position correlation matrix |

## Usage

### Full Risk Analysis

```python
from agentrt.quant import RiskAgent
from agentrt import AgentTask

agent = RiskAgent()
task = AgentTask(
    agent_type="risk",
    payload={
        "type": "risk_analysis",
        "portfolio_id": "portfolio_123",
        "lookback_days": 252
    }
)

result = await agent.execute(task)
metrics = result.result["metrics"]
print(f"VaR 95%: ${metrics['var_95']:,.0f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
print(f"Risk Level: {result.result['risk_level']}")
```

### VaR Calculation

```python
task = AgentTask(
    agent_type="risk",
    payload={
        "type": "var_calculation",
        "portfolio_id": "portfolio_123",
        "confidence_level": 0.99,
        "method": "monte_carlo"  # historical, parametric, monte_carlo
    }
)

result = await agent.execute(task)
var_results = result.result["var_results"]
print(f"Historical VaR: ${var_results['historical_var']:,.0f}")
print(f"Monte Carlo VaR: ${var_results['monte_carlo_var']:,.0f}")
```

### Stress Testing

```python
task = AgentTask(
    agent_type="risk",
    payload={
        "type": "stress_test",
        "portfolio_id": "portfolio_123",
        "scenarios": [
            {"name": "Market Crash", "shock": -0.2},
            {"name": "Rate Hike", "shock": -0.05},
            {"name": "Black Swan", "shock": -0.35}
        ]
    }
)

result = await agent.execute(task)
for scenario in result.result["scenarios"]:
    print(f"{scenario['scenario']}: {scenario['impact_pct']:.1f}%")
```

### Correlation Analysis

```python
task = AgentTask(
    agent_type="risk",
    payload={
        "type": "correlation",
        "portfolio_id": "portfolio_123"
    }
)

result = await agent.execute(task)
matrix = result.result["correlation_matrix"]
```

## Response Formats

### Full Risk Analysis

```json
{
  "portfolio_id": "portfolio_123",
  "analysis_type": "full_risk_analysis",
  "lookback_days": 252,
  "metrics": {
    "var_95": -5000,
    "var_99": -8000,
    "cvar_95": -6500,
    "max_drawdown": -0.15,
    "sharpe_ratio": 1.2,
    "sortino_ratio": 1.5,
    "beta": 1.1,
    "volatility": 0.18,
    "correlation_to_spy": 0.85
  },
  "risk_contribution": [
    {"symbol": "AAPL", "weight": 0.2, "risk_contribution_pct": 25},
    {"symbol": "GOOGL", "weight": 0.15, "risk_contribution_pct": 18}
  ],
  "risk_level": "MEDIUM"
}
```

### Stress Test

```json
{
  "portfolio_id": "portfolio_123",
  "analysis_type": "stress_test",
  "scenarios": [
    {
      "scenario": "Black Swan",
      "description": "Extreme tail event",
      "portfolio_impact": -0.32,
      "impact_pct": -32.0
    },
    {
      "scenario": "Market Crash",
      "description": "20% market decline",
      "portfolio_impact": -0.18,
      "impact_pct": -18.0
    }
  ],
  "worst_case": {
    "scenario": "Black Swan",
    "portfolio_impact": -0.32
  }
}
```

## Risk Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| VaR 95% | Max loss at 95% confidence | 5% chance of exceeding |
| CVaR (ES) | Expected loss beyond VaR | Average worst-case loss |
| Max Drawdown | Largest peak-to-trough | Historical worst decline |
| Beta | Market sensitivity | >1 = more volatile than market |
| Volatility | Standard deviation | Higher = more risky |

## Default Stress Scenarios

1. **Market Crash**: 20% market decline
2. **Rate Hike**: 100bp interest rate increase
3. **Recession**: Economic recession scenario
4. **Tech Selloff**: 30% tech sector decline
5. **Black Swan**: Extreme tail event (-35%)

## Integration with Decision Agent

```python
# Risk analysis feeds into decision making
risk_result = await risk_agent.execute(risk_task)

if risk_result.result["risk_level"] == "HIGH":
    # Escalate to decision agent with full debate
    decision_task = AgentTask(
        agent_type="decision",
        payload={
            "type": "risk_decision",
            "context": {
                "risk_metrics": risk_result.result["metrics"],
                "risk_level": "HIGH"
            },
            "force_debate": True
        }
    )
```

## Related Skills

- `quant-signal` - Generate signals with risk context
- `quant-backtest` - Backtest with risk constraints
- `quant-decision` - Risk-informed decisions
