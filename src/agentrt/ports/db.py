# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async database session port.

Used only by the SQL-backed stores (``SqlProposalStore``, ``SqlMemoryStore``,
``SqlRunArchive``) and the optional FastAPI routes. The in-memory stores need
nothing here.

    from agentrt.ports import db
    db.configure_session_factory(my_async_sessionmaker)   # returns an async context manager
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional

from . import PortNotConfigured

_session_factory: Optional[Callable[[], Any]] = None


def configure_session_factory(factory: Optional[Callable[[], Any]]) -> None:
    """``factory()`` must return an async context manager yielding a session that commits on exit."""
    global _session_factory
    _session_factory = factory


def session_factory() -> Optional[Callable[[], Any]]:
    return _session_factory


@asynccontextmanager
async def open_session() -> AsyncIterator[Any]:
    if _session_factory is None:
        raise PortNotConfigured(
            "no database session configured; call agentrt.ports.db.configure_session_factory(...) "
            "or use the in-memory stores")
    async with _session_factory() as session:
        yield session


async def get_async_db() -> AsyncIterator[Any]:
    """FastAPI dependency form of :func:`open_session`."""
    async with open_session() as session:
        yield session
