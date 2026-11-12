# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EOD price service port.

MarketDataTool talks HTTP to an EOD endpoint
(``GET {base_url}/market/eod/{symbol}?startDate&endDate``). The base URL and an
optional bearer token come from the environment; tests inject an httpx
transport instead.

    AGENTRT_EOD_BASE_URL   default http://127.0.0.1:8081/api
    AGENTRT_EOD_TOKEN      optional, sent as ``Authorization: Bearer ...`` when the caller has none
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8081/api"


def base_url() -> str:
    return os.environ.get("AGENTRT_EOD_BASE_URL", DEFAULT_BASE_URL)


def service_token() -> Optional[str]:
    return os.environ.get("AGENTRT_EOD_TOKEN") or None
