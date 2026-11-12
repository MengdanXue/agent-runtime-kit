# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portfolio storage port.

The portfolio / risk tools, RiskAgent and the optional API need three things
from whoever stores portfolios: an ownership check, the portfolio's positions,
and an analytics object with the method set RiskAgent calls
(``generate_portfolio_static``, ``calculate_portfolio_volatility``, ``calculate_var``, ...;
see tests/conftest.py ``_StubAnalytics`` for the full shape).

The default, :class:`SqlPortfolioProvider`, issues the same three SQL statements
the original host used against ``portfolios`` / ``positions`` tables through
whatever session the tool was given. It has no analytics service: RiskAgent's
portfolio-scope risk needs one from the host.

    from agentrt.ports import portfolio
    portfolio.configure(my_provider)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from . import PortNotConfigured


class PortfolioNotFound(LookupError):
    """The portfolio does not exist."""


class PortfolioForbidden(PermissionError):
    """The portfolio exists but belongs to someone else."""


class PortfolioProvider(Protocol):
    async def verify_access(self, portfolio_id: int, user_id: int, session: Any) -> None:
        """Return normally when ``user_id`` owns the portfolio; raise PortfolioNotFound / PortfolioForbidden."""

    async def portfolio_info(self, portfolio_id: int, session: Any) -> Dict[str, Any]:
        """At least ``portfolio_name`` and ``base_currency``."""

    async def positions(self, portfolio_id: int, session: Any) -> List[Dict[str, Any]]:
        """Rows with ``symbol, quantity, direction, current_value, currency``."""

    def analytics(self, session: Any) -> Any:
        """Analytics service bound to ``session`` (the methods RiskAgent uses)."""


def _sql(statement: str) -> Any:
    try:
        from sqlalchemy import text
    except ImportError:  # the session is something else (a test double, another driver)
        return statement
    return text(statement)


def _mapping(row: Any) -> Dict[str, Any]:
    m = getattr(row, "_mapping", None)
    return dict(m) if m is not None else dict(row)


class SqlPortfolioProvider:
    """``portfolios`` / ``positions`` tables, queried exactly as the original host did."""

    OWNER_SQL = "SELECT user_id FROM portfolios WHERE portfolio_id = :portfolio_id"
    INFO_SQL = ("SELECT portfolio_id, portfolio_name, base_currency, equity, initial_capital, "
                "cash_balance, total_value FROM portfolios WHERE portfolio_id = :portfolio_id")
    POSITIONS_SQL = ("SELECT p.position_id, p.symbol, p.symbol_name, p.asset_type, p.direction, p.quantity, "
                     "p.multiplier, p.currency, p.average_cost, p.current_price, p.current_value, "
                     "p.unrealized_pnl, p.exchange_rate, pf.base_currency "
                     "FROM positions p JOIN portfolios pf ON p.portfolio_id = pf.portfolio_id "
                     "WHERE p.portfolio_id = :portfolio_id ORDER BY p.symbol")

    async def verify_access(self, portfolio_id: int, user_id: int, session: Any) -> None:
        result = await session.execute(_sql(self.OWNER_SQL), {"portfolio_id": portfolio_id})
        row = result.fetchone()
        if not row:
            raise PortfolioNotFound(portfolio_id)
        owner = _mapping(row).get("user_id")
        if owner != user_id:
            raise PortfolioForbidden(portfolio_id)

    async def portfolio_info(self, portfolio_id: int, session: Any) -> Dict[str, Any]:
        result = await session.execute(_sql(self.INFO_SQL), {"portfolio_id": portfolio_id})
        row = result.fetchone()
        return _mapping(row) if row else {}

    async def positions(self, portfolio_id: int, session: Any) -> List[Dict[str, Any]]:
        result = await session.execute(_sql(self.POSITIONS_SQL), {"portfolio_id": portfolio_id})
        return [_mapping(r) for r in result.fetchall()]

    def analytics(self, session: Any) -> Any:
        raise PortNotConfigured(
            "portfolio analytics is host-specific; configure a provider whose analytics(session) "
            "returns an object with RiskAgent's method set (see tests/conftest.py _StubAnalytics)")


_provider: Optional[PortfolioProvider] = None
_default = SqlPortfolioProvider()


def configure(provider: Optional[PortfolioProvider]) -> None:
    global _provider
    _provider = provider


def provider() -> PortfolioProvider:
    return _provider if _provider is not None else _default
