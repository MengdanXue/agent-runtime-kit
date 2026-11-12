# Copyright 2025 TraderX. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Data Source Agent - 数据源代理
负责从多个数据源并行获取数据

功能：
1. 多数据源并行获取
2. 数据源故障切换
3. 数据缓存和去重
4. 实时/历史数据分离
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import logging

from agentrt.guards.mock_guard import require_real_data
from .agent_base import BaseAgent, AgentResult, AgentPriority

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型"""
    EXCHANGE = "exchange"           # 交易所
    MARKET_DATA = "market_data"     # 市场数据
    NEWS = "news"                   # 新闻
    SENTIMENT = "sentiment"         # 情绪
    MACRO = "macro"                 # 宏观经济
    ALTERNATIVE = "alternative"     # 另类数据


class DataFreshness(Enum):
    """数据新鲜度"""
    REALTIME = "realtime"       # 实时 (<1秒)
    NEAR_REALTIME = "near_rt"   # 近实时 (<1分钟)
    DELAYED = "delayed"         # 延迟 (<15分钟)
    EOD = "eod"                 # 日末
    HISTORICAL = "historical"   # 历史


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    type: DataSourceType
    freshness: DataFreshness
    priority: int = 1                   # 优先级，1最高
    rate_limit: int = 60                # 每分钟请求限制
    timeout: float = 10.0               # 超时时间
    retry_count: int = 3                # 重试次数
    fallback: Optional[str] = None      # 备用数据源
    enabled: bool = True
    cache_ttl: int = 60                 # 缓存TTL（秒）


@dataclass
class MarketData:
    """市场数据结构"""
    symbol: str
    timestamp: datetime
    source: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    change_pct: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "change_pct": self.change_pct,
            **self.extra,
        }


class DataSourceAgent(BaseAgent[Dict[str, MarketData]]):
    """
    数据源代理

    核心功能：
    1. 管理多个数据源
    2. 并行获取数据
    3. 数据源故障切换
    4. 缓存管理
    """

    def __init__(
        self,
        name: str = "data_source_agent",
        timeout: float = 30.0,
    ):
        super().__init__(name, AgentPriority.NORMAL, timeout)

        # 数据源配置
        self._source_configs: Dict[str, DataSourceConfig] = {}

        # 数据获取器
        self._fetchers: Dict[str, Any] = {}

        # 缓存
        self._cache: Dict[str, Dict[str, MarketData]] = {}  # source -> symbol -> data
        self._cache_timestamps: Dict[str, datetime] = {}

        # 速率限制追踪
        self._rate_limits: Dict[str, List[datetime]] = {}

        # 订阅主题
        self.subscribe("market_data.request")
        self.subscribe("market_data.subscribe")

        # 注册消息处理器
        self.register_handler("market_data.request", self._handle_data_request)

        logger.info(f"DataSourceAgent {name} initialized")

    def register_source(
        self,
        config: DataSourceConfig,
        fetcher: Any = None,
    ) -> None:
        """注册数据源"""
        self._source_configs[config.name] = config
        if fetcher:
            self._fetchers[config.name] = fetcher
        self._cache[config.name] = {}
        self._rate_limits[config.name] = []
        logger.info(f"Registered data source: {config.name}")

    def configure_default_sources(self) -> None:
        """配置默认数据源"""
        default_configs = [
            DataSourceConfig(
                name="binance",
                type=DataSourceType.EXCHANGE,
                freshness=DataFreshness.REALTIME,
                priority=1,
                rate_limit=1200,
                timeout=5.0,
            ),
            DataSourceConfig(
                name="yahoo_finance",
                type=DataSourceType.MARKET_DATA,
                freshness=DataFreshness.DELAYED,
                priority=3,
                rate_limit=60,
                timeout=10.0,
                fallback="finnhub",
            ),
            DataSourceConfig(
                name="finnhub",
                type=DataSourceType.MARKET_DATA,
                freshness=DataFreshness.REALTIME,
                priority=2,
                rate_limit=60,
                timeout=5.0,
            ),
            DataSourceConfig(
                name="coingecko",
                type=DataSourceType.MARKET_DATA,
                freshness=DataFreshness.NEAR_REALTIME,
                priority=2,
                rate_limit=50,
                timeout=10.0,
                fallback="binance",
            ),
            DataSourceConfig(
                name="fred",
                type=DataSourceType.MACRO,
                freshness=DataFreshness.EOD,
                priority=1,
                rate_limit=120,
                timeout=15.0,
            ),
            DataSourceConfig(
                name="fear_greed",
                type=DataSourceType.SENTIMENT,
                freshness=DataFreshness.REALTIME,
                priority=1,
                rate_limit=30,
                timeout=10.0,
            ),
        ]

        for config in default_configs:
            self.register_source(config)

    async def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行数据获取

        context 参数:
        - symbols: List[str] - 要获取的标的列表
        - sources: Optional[List[str]] - 指定数据源（可选）
        - data_type: str - 数据类型 (ohlcv, quote, ticker)
        - freshness: Optional[str] - 要求的数据新鲜度
        - timeframe: Optional[str] - 时间周期 (1m, 5m, 1h, 1d)
        - start_date: Optional[str] - 历史数据开始时间
        - end_date: Optional[str] - 历史数据结束时间
        """
        symbols = context.get("symbols", [])
        requested_sources = context.get("sources")
        data_type = context.get("data_type", "quote")
        required_freshness = context.get("freshness")

        if not symbols:
            return AgentResult(
                success=False,
                error="No symbols provided"
            )

        # 确定要使用的数据源
        sources = self._select_sources(requested_sources, required_freshness)

        if not sources:
            return AgentResult(
                success=False,
                error="No available data sources"
            )

        # 并行从多个数据源获取数据
        tasks = []
        for source_name in sources:
            task = self._fetch_from_source(source_name, symbols, data_type, context)
            tasks.append((source_name, task))

        # 执行并收集结果
        results: Dict[str, Dict[str, MarketData]] = {}
        errors: Dict[str, str] = {}

        for source_name, task in tasks:
            try:
                source_data = await asyncio.wait_for(task, timeout=self.timeout)
                results[source_name] = source_data
            except asyncio.TimeoutError:
                errors[source_name] = f"Timeout fetching from {source_name}"
            except Exception as e:
                errors[source_name] = str(e)
                logger.warning(f"Error fetching from {source_name}: {e}")

                # 尝试备用数据源
                config = self._source_configs.get(source_name)
                if config and config.fallback and config.fallback not in results:
                    try:
                        fallback_data = await self._fetch_from_source(
                            config.fallback, symbols, data_type, context
                        )
                        results[config.fallback] = fallback_data
                    except Exception as fe:
                        errors[config.fallback] = str(fe)

        if not results:
            return AgentResult(
                success=False,
                error=f"All data sources failed: {errors}",
                data={"errors": errors}
            )

        # 合并数据（按优先级）
        merged_data = self._merge_data(results, symbols)

        # 广播数据更新
        await self.broadcast("market_data.updated", {
            "symbols": symbols,
            "data": {s: d.to_dict() for s, d in merged_data.items()},
            "sources_used": list(results.keys()),
        })

        return AgentResult(
            success=True,
            data={
                "merged": merged_data,
                "by_source": results,
                "errors": errors,
            },
            metadata={
                "sources_attempted": sources,
                "sources_succeeded": list(results.keys()),
                "sources_failed": list(errors.keys()),
                "symbols_requested": len(symbols),
                "symbols_returned": len(merged_data),
            }
        )

    def _select_sources(
        self,
        requested: Optional[List[str]],
        required_freshness: Optional[str]
    ) -> List[str]:
        """选择要使用的数据源"""
        sources = []

        for name, config in self._source_configs.items():
            if not config.enabled:
                continue

            # 如果指定了数据源，只使用指定的
            if requested and name not in requested:
                continue

            # 检查新鲜度要求
            if required_freshness:
                freshness_order = [
                    DataFreshness.REALTIME,
                    DataFreshness.NEAR_REALTIME,
                    DataFreshness.DELAYED,
                    DataFreshness.EOD,
                    DataFreshness.HISTORICAL,
                ]
                required_idx = next(
                    (i for i, f in enumerate(freshness_order) if f.value == required_freshness),
                    len(freshness_order)
                )
                current_idx = freshness_order.index(config.freshness)
                if current_idx > required_idx:
                    continue

            sources.append(name)

        # 按优先级排序
        sources.sort(key=lambda x: self._source_configs[x].priority)

        return sources

    async def _fetch_from_source(
        self,
        source_name: str,
        symbols: List[str],
        data_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, MarketData]:
        """从指定数据源获取数据"""
        config = self._source_configs.get(source_name)
        if not config:
            raise ValueError(f"Unknown data source: {source_name}")

        # 检查速率限制
        await self._check_rate_limit(source_name)

        # 检查缓存
        cached_data = self._get_cached_data(source_name, symbols)
        symbols_to_fetch = [s for s in symbols if s not in cached_data]

        if not symbols_to_fetch:
            return cached_data

        # 获取数据（这里应该调用实际的数据获取器）
        fetched_data = await self._do_fetch(source_name, symbols_to_fetch, data_type, context)

        # 更新缓存
        self._update_cache(source_name, fetched_data)

        # 合并缓存和新获取的数据
        return {**cached_data, **fetched_data}

    async def _do_fetch(
        self,
        source_name: str,
        symbols: List[str],
        data_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, MarketData]:
        """实际执行数据获取（可扩展点）"""
        # 如果有注册的fetcher，使用它
        if source_name in self._fetchers:
            fetcher = self._fetchers[source_name]
            return await fetcher.fetch(symbols, data_type, context)

        # 没有注册 fetcher 的数据源没有真实数据。此前这里会用随机价格填充并标上该数据源的名字
        # （例如 source="binance" 的 AAPL 报价），调用方无法分辨。现在默认拒绝。
        require_real_data(
            f"market data from source '{source_name}'",
            f"No fetcher is registered for '{source_name}'; register one (e.g. the "
            f"MarketDataTool-backed 'traderx_eod' source) instead of reading random prices.",
        )
        return await self._mock_fetch(source_name, symbols, data_type)

    async def _mock_fetch(
        self,
        source_name: str,
        symbols: List[str],
        data_type: str
    ) -> Dict[str, MarketData]:
        """模拟数据获取（开发用）"""
        require_real_data("random draws in _mock_fetch",
                          "synthetic data for local development only (ALLOW_MOCK_AGENT_DATA=true)")
        import random

        result = {}
        for symbol in symbols:
            base_price = 100.0 if "BTC" not in symbol else 40000.0
            price = base_price * (1 + random.uniform(-0.01, 0.01))

            result[symbol] = MarketData(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                source=source_name,
                open=price * 0.995,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=random.uniform(1000, 10000),
                last=price,
                bid=price * 0.999,
                ask=price * 1.001,
                change_pct=random.uniform(-2, 2),
            )

        # 模拟网络延迟
        await asyncio.sleep(random.uniform(0.1, 0.5))

        return result

    async def _check_rate_limit(self, source_name: str) -> None:
        """检查速率限制"""
        config = self._source_configs.get(source_name)
        if not config:
            return

        now = datetime.utcnow()
        one_minute_ago = now - timedelta(minutes=1)

        # 清理旧记录
        self._rate_limits[source_name] = [
            t for t in self._rate_limits[source_name]
            if t > one_minute_ago
        ]

        # 检查是否超过限制
        if len(self._rate_limits[source_name]) >= config.rate_limit:
            # 等待直到可以请求
            oldest = self._rate_limits[source_name][0]
            wait_time = (oldest + timedelta(minutes=1) - now).total_seconds()
            if wait_time > 0:
                logger.debug(f"Rate limited for {source_name}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)

        # 记录请求
        self._rate_limits[source_name].append(now)

    def _get_cached_data(
        self,
        source_name: str,
        symbols: List[str]
    ) -> Dict[str, MarketData]:
        """获取缓存数据"""
        config = self._source_configs.get(source_name)
        if not config:
            return {}

        cached = {}
        source_cache = self._cache.get(source_name, {})
        now = datetime.utcnow()

        for symbol in symbols:
            if symbol in source_cache:
                data = source_cache[symbol]
                cache_age = (now - data.timestamp).total_seconds()
                if cache_age < config.cache_ttl:
                    cached[symbol] = data

        return cached

    def _update_cache(
        self,
        source_name: str,
        data: Dict[str, MarketData]
    ) -> None:
        """更新缓存"""
        if source_name not in self._cache:
            self._cache[source_name] = {}

        self._cache[source_name].update(data)
        self._cache_timestamps[source_name] = datetime.utcnow()

    def _merge_data(
        self,
        results: Dict[str, Dict[str, MarketData]],
        symbols: List[str]
    ) -> Dict[str, MarketData]:
        """按优先级合并数据"""
        merged = {}

        # 按数据源优先级排序
        sorted_sources = sorted(
            results.keys(),
            key=lambda x: self._source_configs[x].priority if x in self._source_configs else 999
        )

        for symbol in symbols:
            for source_name in sorted_sources:
                if symbol in results.get(source_name, {}):
                    merged[symbol] = results[source_name][symbol]
                    break

        return merged

    async def _handle_data_request(self, message: Any) -> None:
        """处理数据请求消息"""
        payload = message.payload
        context = {
            "symbols": payload.get("symbols", []),
            "sources": payload.get("sources"),
            "data_type": payload.get("data_type", "quote"),
        }

        result = await self.execute(context)

        await self.respond(message, {
            "success": result.success,
            "data": result.data if result.success else None,
            "error": result.error,
        })

    # === 便捷方法 ===

    async def get_quotes(
        self,
        symbols: List[str],
        sources: Optional[List[str]] = None
    ) -> Dict[str, MarketData]:
        """获取报价数据"""
        result = await self.execute({
            "symbols": symbols,
            "sources": sources,
            "data_type": "quote",
        })
        if result.success:
            return result.data.get("merged", {})
        return {}

    async def get_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取OHLCV数据"""
        result = await self.execute({
            "symbols": symbols,
            "data_type": "ohlcv",
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
        })
        if result.success:
            return result.data.get("merged", {})
        return {}

    def get_source_status(self) -> Dict[str, Any]:
        """获取所有数据源状态"""
        status = {}
        for name, config in self._source_configs.items():
            status[name] = {
                "enabled": config.enabled,
                "type": config.type.value,
                "freshness": config.freshness.value,
                "priority": config.priority,
                "rate_limit": config.rate_limit,
                "requests_last_minute": len(self._rate_limits.get(name, [])),
                "cache_size": len(self._cache.get(name, {})),
            }
        return status
