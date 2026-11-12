# Copyright 2026 Mengdan Xue
# SPDX-License-Identifier: Apache-2.0
"""Internally consistent sandbox data for the S2 (sell / approval) scenarios.

The earlier S2 set-up reused one price series for AAPL and MSFT, fed the portfolio
risk from fixed stub payloads (equity 100 000) and overrode VaR to 1 %, so nothing
agreed with the 10 AAPL + 30 MSFT holding. A real critic rightly flagged that.

Here everything derives from one seeded set of series and one holding list:

* each symbol has its own one-factor price path (market x beta + own noise) on
  weekdays only, ending on the last weekday before TODAY;
* position rows (current_price, current_value, average_cost, unrealized_pnl) use the
  last close of that symbol's own series;
* the portfolio row's total_value / equity = position values + cash;
* every portfolio-risk payload (volatility, VaR 95/99, CVaR, Sharpe, drawdown, beta,
  risk contribution, correlations) is computed from the holding-weighted daily returns
  of those same series over the 1y window. Nothing is chosen to make a check pass.

Only the test data lives here; risk rules, schemas and validators are not touched.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEED = 20261112
BENCHMARK = "SPY"
WINDOW_RETURNS = 252          # "1y" window of daily returns for portfolio analytics
N_BARS = 320                  # > 253, so tsmom and the 1y window both have enough history

# symbol: (start price, beta to the market, own annual volatility, annual drift)
UNIVERSE: Dict[str, Tuple[float, float, float, float]] = {
    "SPY": (520.0, 1.0, 0.0, 0.08),
    "AAPL": (190.0, 1.15, 0.20, 0.10),
    "MSFT": (410.0, 1.05, 0.17, 0.09),
    "JNJ": (155.0, 0.55, 0.13, 0.04),
    "XOM": (110.0, 0.85, 0.22, 0.05),
    "KO": (62.0, 0.60, 0.12, 0.04),
    "JPM": (200.0, 1.10, 0.19, 0.07),
}
MARKET_VOL = 0.16

# Holding of portfolio 7 (owner OWNER). Quantities give AAPL ~23 % (above the 20 % cap)
# and every other position below the cap, before and after selling 5 AAPL (AAPL -> ~19 %).
HOLDINGS: Dict[str, float] = {"AAPL": 25, "MSFT": 6, "JNJ": 19, "XOM": 26, "KO": 47, "JPM": 14}
CASH = 1500.0


def weekdays_until(last: date, n: int) -> List[date]:
    days: List[date] = []
    d = last
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return days[::-1]


def last_weekday_before(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def make_series(today: date, n: int = N_BARS, end: Optional[date] = None,
                seed: int = SEED) -> Dict[str, List[Dict[str, Any]]]:
    """OHLCV bars per symbol: one shared market factor, own noise per symbol."""
    rng = random.Random(seed)
    days = weekdays_until(end or last_weekday_before(today), n)
    dt = 1 / 252
    market = [rng.gauss(0.0, MARKET_VOL * math.sqrt(dt)) for _ in range(n - 1)]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym, (p0, beta, own_vol, drift) in UNIVERSE.items():
        own = random.Random(f"{seed}:{sym}")
        closes = [p0]
        for m in market:
            r = drift * dt + beta * m + own.gauss(0.0, own_vol * math.sqrt(dt))
            closes.append(closes[-1] * (1 + r))
        bars = []
        prev = closes[0]
        for i, (d, c) in enumerate(zip(days, closes)):
            o = prev * (1 + own.gauss(0.0, 0.002))
            hi = max(o, c) * (1 + abs(own.gauss(0.0, 0.004)))
            lo = min(o, c) * (1 - abs(own.gauss(0.0, 0.004)))
            bars.append({"date": d.isoformat(), "open": round(o, 4), "high": round(hi, 4),
                         "low": round(lo, 4), "close": round(c, 4),
                         "volume": int(5_000_000 * (1 + abs(own.gauss(0.0, 0.3))))})
            prev = c
        out[sym] = bars
    return out


def position_rows(series: Dict[str, List[Dict[str, Any]]],
                  holdings: Dict[str, float] = HOLDINGS) -> List[Dict[str, Any]]:
    """Position rows priced at each symbol's own last close; cost = close 120 bars earlier."""
    rows = []
    for i, (sym, qty) in enumerate(holdings.items(), start=1):
        bars = series[sym]
        last, cost = bars[-1]["close"], bars[-121]["close"]
        rows.append({"position_id": i, "symbol": sym, "symbol_name": sym, "asset_type": "stock",
                     "direction": 1, "quantity": float(qty), "multiplier": 1, "currency": "USD",
                     "average_cost": cost, "current_price": last,
                     "current_value": round(qty * last, 2),
                     "unrealized_pnl": round(qty * (last - cost), 2),
                     "exchange_rate": 1.0, "base_currency": "USD"})
    return rows


def portfolio_row(pid: int, rows: List[Dict[str, Any]], cash: float = CASH) -> Dict[str, Any]:
    invested = sum(r["current_value"] for r in rows)
    total = round(invested + cash, 2)
    initial = round(sum(r["quantity"] * r["average_cost"] for r in rows) + cash, 2)
    return {"portfolio_id": pid, "portfolio_name": f"P{pid}", "base_currency": "USD",
            "equity": total, "initial_capital": initial, "cash_balance": cash, "total_value": total}


class ConsistentDB:
    """The FakeDB query surface, answering with the rows above (not fixed 100 000 rows)."""

    def __init__(self, owners: Dict[int, int], positions: Dict[int, List[Dict[str, Any]]],
                 cash: float = CASH):
        from tests.runtime.test_agent_evaluation import _Result, _Row
        self._Result, self._Row = _Result, _Row
        self.portfolios, self.positions, self.cash = owners, positions, cash
        self.queries: List[str] = []

    async def execute(self, query, params):
        sql = " ".join(str(query).split())
        pid = params["portfolio_id"]
        self.queries.append(sql[:40])
        if sql.startswith("SELECT user_id FROM portfolios"):
            owner = self.portfolios.get(pid)
            return self._Result([] if owner is None else [self._Row({"user_id": owner})])
        if "FROM positions" in sql:
            return self._Result([self._Row(p) for p in self.positions.get(pid, [])])
        if pid not in self.portfolios:
            return self._Result([])
        return self._Result([self._Row(portfolio_row(pid, self.positions.get(pid, []), self.cash))])

    def session_factory(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _session():
            yield self
        return _session


class ConsistentAnalytics:
    """PortfolioAnalyticsService stand-in that computes every payload from the data."""

    def __init__(self, series: Dict[str, List[Dict[str, Any]]], rows: List[Dict[str, Any]],
                 cash: float = CASH):
        self.calls: List[str] = []
        self.cash = cash
        self.symbols = [r["symbol"] for r in rows]
        closes = {s: np.array([b["close"] for b in series[s]], dtype=float)
                  for s in self.symbols + [BENCHMARK]}
        rets = {s: np.diff(c)[-WINDOW_RETURNS:] / c[:-1][-WINDOW_RETURNS:] for s, c in closes.items()}
        values = np.array([r["current_value"] for r in rows], dtype=float)
        self.invested = float(values.sum())
        self.value = self.invested + cash                         # equity incl. cash
        self.weights = values / self.value                        # cash earns 0
        self.R = np.vstack([rets[s] for s in self.symbols])       # symbols x days
        self.port = self.weights @ self.R                         # daily portfolio returns
        self.bench = rets[BENCHMARK]
        self.n = int(len(self.port))

    # helpers
    def _var(self, alpha: float) -> float:
        return float(-np.quantile(self.port, alpha))

    def _amount(self, frac: float) -> Dict[str, float]:
        return {"amount": round(frac * self.value, 2), "pct": round(frac * 100, 4)}

    # service surface used by RiskAgent._full_risk_analysis
    async def generate_portfolio_static(self, portfolio_id, window="1y"):
        self.calls.append("portfolio_static")
        mu, sd = float(self.port.mean()), float(self.port.std(ddof=1))
        eq = np.cumprod(1 + self.port)
        dd = float(-np.min(eq / np.maximum.accumulate(eq) - 1))
        return {"sharpe_ratio": round(mu / sd * math.sqrt(252), 4), "volatility": round(sd * math.sqrt(252), 4),
                "max_drawdown": round(dd, 4), "cumulative_return": round(float(eq[-1] - 1), 4)}

    async def calculate_portfolio_volatility(self, portfolio_id, **kwargs):
        self.calls.append("volatility")
        sd = float(self.port.std(ddof=1))
        return {"portfolio_volatility": round(sd, 6), "annualized_volatility": round(sd * math.sqrt(252), 4)}

    async def calculate_var(self, portfolio_id, confidence_level=0.05, **kwargs):
        key = "var_99" if confidence_level <= 0.01 else "var_95"
        self.calls.append(key)
        a = self._amount(self._var(0.01 if key == "var_99" else 0.05))
        return {"var_amount": a["amount"], "var_absolute": a["amount"], "var_pct": a["pct"],
                "equity_var": a["amount"], "equity_var_pct": a["pct"],
                "abs_portfolio_value": round(self.value, 2), "net_portfolio_value": round(self.value, 2),
                "equity": round(self.value, 2), "window": kwargs.get("window", "1y"),
                "var_window": 1, "sample_size": self.n}

    async def calculate_cvar(self, portfolio_id, confidence_level=0.05, **kwargs):
        self.calls.append("cvar")
        var = self._var(0.05)
        tail = self.port[self.port <= -var]
        cvar = float(-tail.mean())
        a = self._amount(cvar)
        return {"cvar_amount": a["amount"], "cvar_pct": a["pct"], "cvar_var_ratio": round(cvar / var, 4)}

    async def calculate_beta(self, portfolio_id, benchmark_symbol="SPY", window="1y"):
        self.calls.append("beta")
        cov = np.cov(self.port, self.bench, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1])
        alpha = float((self.port.mean() - beta * self.bench.mean()) * 252)
        corr = float(np.corrcoef(self.port, self.bench)[0, 1])
        return {"beta": round(beta, 4), "alpha": round(alpha, 4), "correlation": round(corr, 4)}

    async def calculate_marginal_risk_contribution(self, portfolio_id, **kwargs):
        self.calls.append("risk_contribution")
        cov = np.cov(self.R, ddof=1)
        sd = float(math.sqrt(self.weights @ cov @ self.weights))
        mrc = cov @ self.weights / sd
        contrib = self.weights * mrc / sd * 100
        w_inv = self.weights / self.weights.sum()
        hhi = float((w_inv ** 2).sum())
        return {"portfolio_volatility": round(sd, 6),
                "positions": [{"symbol": s, "weight": round(float(w), 4), "mrc": round(float(m), 6),
                               "risk_contribution_pct": round(float(c), 2)}
                              for s, w, m, c in zip(self.symbols, self.weights, mrc, contrib)],
                "concentration_metrics": {"herfindahl_index": round(hhi, 4), "effective_n": round(1 / hhi, 2)}}

    async def calculate_correlation_matrix(self, portfolio_id, window="1y"):
        self.calls.append("correlation")
        return {"symbols": self.symbols, "correlation_matrix": np.round(np.corrcoef(self.R), 4).tolist(),
                "window": window}
