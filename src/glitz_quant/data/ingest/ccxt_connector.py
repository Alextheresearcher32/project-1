"""
CCXT-based exchange ingest. One connector per venue, async via ccxt.async_support.
Pulls tickers + candles + recent trades, normalizes to canonical types, writes to
Redis and (optionally) Supabase.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential

from glitz_quant.data.store.redis_cache import RedisCache
from glitz_quant.data.types import Candle, Side, Ticker, Trade, Venue
from glitz_quant.settings import get_exchanges_config, get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


VENUE_TO_CCXT_ID = {
    Venue.COINBASE: "coinbaseadvanced",
    Venue.KRAKEN: "kraken",
    Venue.BINANCE_US: "binanceus",
    Venue.BINANCE: "binance",
    Venue.GEMINI: "gemini",
    Venue.HYPERLIQUID: "hyperliquid",
}


def _build_client(venue: Venue) -> Any:
    s = get_settings()
    ccxt_id = VENUE_TO_CCXT_ID[venue]
    klass = getattr(ccxt, ccxt_id)

    creds: dict[str, Any] = {}
    if venue == Venue.COINBASE and s.coinbase_api_key and s.coinbase_api_secret:
        creds = {
            "apiKey": s.coinbase_api_key.get_secret_value(),
            "secret": s.coinbase_api_secret.get_secret_value(),
        }
    elif venue == Venue.KRAKEN and s.kraken_api_key and s.kraken_api_secret:
        creds = {
            "apiKey": s.kraken_api_key.get_secret_value(),
            "secret": s.kraken_api_secret.get_secret_value(),
        }
    elif venue == Venue.BINANCE_US and s.binance_us_api_key and s.binance_us_api_secret:
        creds = {
            "apiKey": s.binance_us_api_key.get_secret_value(),
            "secret": s.binance_us_api_secret.get_secret_value(),
        }
    elif venue == Venue.GEMINI and s.gemini_api_key and s.gemini_api_secret:
        creds = {
            "apiKey": s.gemini_api_key.get_secret_value(),
            "secret": s.gemini_api_secret.get_secret_value(),
        }
    elif venue == Venue.HYPERLIQUID and s.hyperliquid_api_private_key:
        # Hyperliquid uses private keys directly in CCXT
        creds = {
            "secret": s.hyperliquid_api_private_key.get_secret_value(),
            "address": s.hyperliquid_account_address,
        }

    return klass({**creds, "enableRateLimit": True})


def _canonical_to_venue_symbol(venue: Venue, canonical: str) -> str:
    cfg = get_exchanges_config()
    sym = cfg.get("symbols", {}).get(canonical, {})
    venues_map = sym.get("venues", {})
    return venues_map.get(venue.value, canonical)


class CCXTIngest:
    """One ingest worker per venue. Polls tickers/candles, writes to Redis."""

    def __init__(self, venue: Venue, cache: RedisCache) -> None:
        self.venue = venue
        self.cache = cache
        self._client: Any = None
        self._running = False

    async def start(self) -> None:
        self._client = _build_client(self.venue)
        await self._client.load_markets()
        self._running = True
        log.info("ccxt_ingest_started", venue=self.venue.value)

    async def stop(self) -> None:
        self._running = False
        if self._client is not None:
            await self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def fetch_ticker(self, canonical_symbol: str) -> Ticker | None:
        venue_sym = _canonical_to_venue_symbol(self.venue, canonical_symbol)
        try:
            raw = await self._client.fetch_ticker(venue_sym)
        except Exception as e:
            log.warning("fetch_ticker_failed", venue=self.venue.value, symbol=canonical_symbol, err=str(e))
            return None
        bid = raw.get("bid")
        ask = raw.get("ask")
        last = raw.get("last") or raw.get("close")
        if bid is None or ask is None or last is None:
            return None
        return Ticker(
            venue=self.venue,
            symbol=canonical_symbol,
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            last=Decimal(str(last)),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def fetch_candles(
        self, canonical_symbol: str, timeframe: str = "15m", limit: int = 500
    ) -> list[Candle]:
        venue_sym = _canonical_to_venue_symbol(self.venue, canonical_symbol)
        try:
            raw = await self._client.fetch_ohlcv(venue_sym, timeframe=timeframe, limit=limit)
        except Exception as e:
            log.warning("fetch_candles_failed", venue=self.venue.value, symbol=canonical_symbol, err=str(e))
            return []

        out: list[Candle] = []
        from datetime import datetime, timezone

        for ts_ms, o, h, l, c, v in raw:
            out.append(
                Candle(
                    venue=self.venue,
                    symbol=canonical_symbol,
                    timeframe=timeframe,
                    open=Decimal(str(o)),
                    high=Decimal(str(h)),
                    low=Decimal(str(l)),
                    close=Decimal(str(c)),
                    volume=Decimal(str(v)),
                    ts=datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc),
                )
            )
        return out

    async def fetch_recent_trades(
        self, canonical_symbol: str, limit: int = 100
    ) -> list[Trade]:
        venue_sym = _canonical_to_venue_symbol(self.venue, canonical_symbol)
        try:
            raw = await self._client.fetch_trades(venue_sym, limit=limit)
        except Exception as e:
            log.warning("fetch_trades_failed", err=str(e))
            return []
        from datetime import datetime, timezone

        out: list[Trade] = []
        for t in raw:
            out.append(
                Trade(
                    venue=self.venue,
                    symbol=canonical_symbol,
                    price=Decimal(str(t["price"])),
                    size=Decimal(str(t["amount"])),
                    side=Side.BUY if t.get("side") == "buy" else Side.SELL,
                    ts=datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc),
                    trade_id=str(t.get("id", "")) or None,
                )
            )
        return out

    async def run_ticker_loop(
        self, symbols: list[str], interval_seconds: float = 1.0
    ) -> None:
        """Continuously poll tickers for the given symbols."""
        log.info("ticker_loop_starting", venue=self.venue.value, symbols=symbols)
        while self._running:
            for sym in symbols:
                ticker = await self.fetch_ticker(sym)
                if ticker:
                    await self.cache.set_ticker(ticker)
            await asyncio.sleep(interval_seconds)
