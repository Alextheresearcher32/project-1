"""
Redis cache + pub/sub. Hot path for ticker quotes, dedup, rate limiting.
Strategies read tickers/book from here, not directly from exchange WS.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import redis.asyncio as redis

from glitz_quant.data.types import OrderBook, Ticker, Venue
from glitz_quant.settings import get_app_config, get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


def _json_default(o: Any) -> Any:
    if isinstance(o, Decimal):
        return str(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "value"):
        return o.value
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class RedisCache:
    def __init__(self) -> None:
        self.url = get_settings().redis_url
        self._client: redis.Redis | None = None
        self._ttl = get_app_config().get("data", {}).get("cache_ttl_seconds", {})

    async def connect(self) -> None:
        if self._client is None:
            self._client = redis.from_url(
                self.url,
                decode_responses=True,
                socket_keepalive=True,
                health_check_interval=30,
            )
            await self._client.ping()
            log.info("redis_connected", url=self.url.split("@")[-1])

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            raise RuntimeError("RedisCache not connected — call .connect() first")
        return self._client

    # -------- Tickers --------
    async def set_ticker(self, ticker: Ticker) -> None:
        key = f"ticker:{ticker.venue.value}:{ticker.symbol}"
        ttl = int(self._ttl.get("ticker", 5))
        await self.client.set(key, ticker.model_dump_json(), ex=ttl)
        # also publish for live subscribers
        await self.client.publish(f"ch:ticker:{ticker.symbol}", ticker.model_dump_json())

    async def get_ticker(self, venue: Venue, symbol: str) -> Ticker | None:
        key = f"ticker:{venue.value}:{symbol}"
        raw = await self.client.get(key)
        if not raw:
            return None
        return Ticker.model_validate_json(raw)

    async def get_best_ticker(self, symbol: str) -> Ticker | None:
        """Return the ticker with tightest spread across all venues."""
        keys = await self.client.keys(f"ticker:*:{symbol}")
        best: Ticker | None = None
        best_spread: Decimal | None = None
        for key in keys:
            raw = await self.client.get(key)
            if not raw:
                continue
            t = Ticker.model_validate_json(raw)
            sp = t.spread_bps
            if best is None or sp < (best_spread or Decimal("9999999")):
                best = t
                best_spread = sp
        return best

    # -------- Order books --------
    async def set_orderbook(self, book: OrderBook) -> None:
        key = f"book:{book.venue.value}:{book.symbol}"
        ttl = int(self._ttl.get("orderbook", 1))
        await self.client.set(key, book.model_dump_json(), ex=ttl)

    async def get_orderbook(self, venue: Venue, symbol: str) -> OrderBook | None:
        key = f"book:{venue.value}:{symbol}"
        raw = await self.client.get(key)
        if not raw:
            return None
        return OrderBook.model_validate_json(raw)

    # -------- Rate limiting (token bucket via Redis) --------
    async def rate_limit_check(self, key: str, max_per_minute: int) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        full_key = f"rl:{key}"
        pipe = self.client.pipeline()
        pipe.incr(full_key)
        pipe.expire(full_key, 60)
        count, _ = await pipe.execute()
        return int(count) <= max_per_minute

    # -------- Deduplication --------
    async def seen(self, key: str, ttl_seconds: int = 300) -> bool:
        """Returns True if we've seen this key in the last ttl_seconds."""
        full = f"seen:{key}"
        added = await self.client.set(full, "1", ex=ttl_seconds, nx=True)
        return added is None  # None means key already existed

    # -------- Generic JSON kv --------
    async def set_json(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value, default=_json_default)
        await self.client.set(key, payload, ex=ttl)

    async def get_json(self, key: str) -> Any:
        raw = await self.client.get(key)
        return json.loads(raw) if raw else None
