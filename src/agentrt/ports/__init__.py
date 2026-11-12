# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ports: the only places where agentrt touches things it does not own.

agentrt was extracted from a larger trading system. Everything that system
provided - a PostgreSQL session, a strategy registry, portfolio storage, an
EOD price service, an event-driven backtest engine - is reached through the
small modules in this package. Each port has a default that works without any
external service (in-memory, environment variable, or an optional dependency
such as quantrisk-core); when there is no sensible default, the port raises
:class:`PortNotConfigured` with the call that configures it.

Nothing in agentrt imports a database driver, builds an engine or reads
credentials as a side effect of ``import agentrt``.
"""


class PortNotConfigured(RuntimeError):
    """A port was used before an implementation was configured."""
