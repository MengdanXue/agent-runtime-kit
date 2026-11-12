---
name: quant-backtest
description: Run strategy backtests with Monte Carlo simulation and walk-forward optimization. Use when you need to test strategy performance, optimize parameters, or validate trading ideas.
version: 1.0.0
triggers:
  - "backtest"
  - "test strategy"
  - "historical performance"
  - "monte carlo"
  - "walk forward"
---

# Quant Backtest Skill

## Overview

Run comprehensive strategy backtests including standard backtests, Monte Carlo simulations for risk assessment, and walk-forward optimization for parameter validation.

## When to Use

```
✅ Use for:
  • Testing strategy performance on historical data
  • Monte Carlo risk simulation
  • Walk-forward optimization
  • Parameter sensitivity analysis
  • Comparing multiple strategies

❌ Don't use for:
  • Real-time signal generation (use quant-signal)
  • Live risk monitoring (use quant-risk)
  • Trade execution decisions (use quant-decision)
```

## Backtest Types

| Type | Description | Use Case |
|------|-------------|----------|
| Standard | Run strategy on historical data | Basic performance assessment |
| Monte Carlo | Simulate many possible outcomes | Risk estimation, confidence intervals |
| Walk Forward | Rolling train/test optimization | Parameter robustness validation |

## Usage

### Standard Backtest

```python
from agentrt.quant import BacktestAgent
from agentrt import AgentTask

agent = BacktestAgent()
task = AgentTask(
    agent_type="backtest",
    payload={
        "type": "backtest",
        "strategy": "momentum",
        "symbol": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000,
        "params": {
            "lookback": 20,
            "threshold": 1.5
        }
    }
)

result = await agent.execute(task)
print(result.result["metrics"]["sharpe_ratio"])
```

### Monte Carlo Simulation

```python
task = AgentTask(
    agent_type="backtest",
    payload={
        "type": "monte_carlo",
        "strategy": "momentum",
        "symbol": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "initial_capital": 100000,
        "num_simulations": 1000
    }
)

result = await agent.execute(task)
distribution = result.result["distribution"]
print(f"VaR 95%: {distribution['var_95']}")
print(f"Probability of profit: {distribution['probability_profit']}")
```

### Walk-Forward Optimization

```python
task = AgentTask(
    agent_type="backtest",
    payload={
        "type": "walk_forward",
        "strategy": "momentum",
        "symbol": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "num_folds": 5,
        "train_pct": 0.8
    }
)

result = await agent.execute(task)
print(f"Average test Sharpe: {result.result['avg_test_sharpe']}")
```

## Response Formats

### Standard Backtest

```json
{
  "type": "backtest",
  "strategy": "momentum",
  "symbol": "AAPL",
  "period": "2020-01-01 to 2023-12-31",
  "metrics": {
    "initial_capital": 100000,
    "final_equity": 145000,
    "total_return": 0.45,
    "total_return_pct": 45.0,
    "sharpe_ratio": 1.85,
    "sortino_ratio": 2.1,
    "calmar_ratio": 1.2,
    "max_drawdown": -0.15,
    "win_rate": 0.58,
    "profit_factor": 1.8,
    "total_trades": 120
  }
}
```

### Monte Carlo

```json
{
  "type": "monte_carlo",
  "simulations": 1000,
  "distribution": {
    "mean_return": 0.42,
    "std_dev": 0.18,
    "var_95": -0.12,
    "var_99": -0.25,
    "percentile_5": -0.12,
    "percentile_50": 0.40,
    "percentile_95": 0.85,
    "probability_profit": 0.78
  }
}
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (vs risk-free rate) |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | Return / Max Drawdown |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | Percentage of winning trades |
| Profit Factor | Gross profit / Gross loss |

## Related Skills

- `quant-signal` - Generate signals to test
- `quant-risk` - Analyze strategy risk
- `quant-decision` - Decide on strategy deployment
