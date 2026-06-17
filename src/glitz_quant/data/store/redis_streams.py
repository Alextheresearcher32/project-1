"""
Redis Streams with msgpack binary serialization. ~0.1ms per write.

Write path (ThetaData/Oanda connectors):
    push(event_type, data) → XADD mkt * t <type> d <msgpack bytes>

Read path (WolfPack consumers):
    read(consumer, count) → XREADGROUP wolves <wolf-N> mkt > COUNT n BLOCK 500
    ack(*msg_ids)         → XACK mkt wolves <ids>

A separate Redis client (decode_responses=False) is required so we can
write raw bytes. The existing RedisCache uses decode_responses=True and
continues to serve the OMS/ticker-lookup path unchanged.
"""
from __future__ import annotations

from typing import Any

import msgpack  # type: ignore[import-untyped]
import redis.asyncio as redis

from glitz_quant.settings import get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)

MARKET_STREAM = b"mkt"
_GROUP = b"wolves"
_MAX_LEN = 100_000


def _mp_default(obj: Any) -> Any:
    from decimal import Decimal
    from datetime import datetime
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"msgpack: cannot pack {type(obj)}")


class RedisStreams:
    """Binary Redis Streams layer. One instance shared across the process."""

    def __init__(self) -> None:
        self._client: redis.Redis | None = None
        self._available: bool = False

    async def connect(self) -> None:
        url = get_settings().redis_url
        try:
            self._client = redis.from_url(
                url,
                decode_responses=False,
                socket_keepalive=True,
                health_check_interval=30,
            )
            await self._client.ping()
            # Create consumer group idempotently; mkstream=True creates the stream key
            try:
                await self._client.xgroup_create(
                    MARKET_STREAM, _GROUP, id=b"$", mkstream=True
                )
            except Exception:
                pass  # BUSYGROUP — group already exists
            self._available = True
            log.info("redis_streams_connected", stream=MARKET_STREAM.decode())
        except Exception as e:
            self._available = False
            log.warning("redis_streams_unavailable", reason=str(e)[:120])
            log.warning("redis_streams_degraded_mode", msg="market stream disabled — strategies will run on polled candles only")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def _r(self) -> redis.Redis:
        if self._client is None or not self._available:
            raise RuntimeError("RedisStreams not connected")
        return self._client

    # ---- Write side (connectors call these) ----

    async def push(self, event_type: str, data: dict) -> None:
        if not self._available:
            return
        payload = msgpack.packb(data, default=_mp_default, use_bin_type=True)
        try:
            await self._r.xadd(
                MARKET_STREAM,
                {b"t": event_type.encode(), b"d": payload},
                maxlen=_MAX_LEN,
                approximate=True,
            )
        except Exception:
            pass

    async def push_ticker(self, ticker_dict: dict) -> None:
        await self.push("ticker", ticker_dict)

    async def push_candle(self, candle_dict: dict) -> None:
        await self.push("candle", candle_dict)

    # ---- Read side (WolfPack calls these) ----

    async def read(self, consumer: str, count: int = 50) -> list[tuple[bytes, str, dict]]:
        """
        Read new messages for this consumer. Blocks ≤500ms if stream is empty.
        Returns [(msg_id, event_type, data_dict), ...].
        Call ack() after processing to advance the group offset.
        """
        if not self._available:
            return []
        try:
            result = await self._r.xreadgroup(
                _GROUP,
                consumer.encode(),
                {MARKET_STREAM: b">"},
                count=count,
                block=500,
            )
        except Exception:
            return []
        if not result:
            return []
        out: list[tuple[bytes, str, dict]] = []
        for _stream, messages in result:
            for msg_id, fields in messages:
                evt = fields.get(b"t", b"").decode()
                raw = fields.get(b"d", b"")
                data: dict = msgpack.unpackb(raw, raw=False) if raw else {}
                out.append((msg_id, evt, data))
        return out

    async def ack(self, *msg_ids: bytes) -> None:
        if not self._available or not msg_ids:
            return
        try:
            await self._r.xack(MARKET_STREAM, _GROUP, *msg_ids)
        except Exception:
            pass

    async def stream_len(self) -> int:
        if not self._available:
            return 0
        try:
            return await self._r.xlen(MARKET_STREAM)
        except Exception:
            return 0
