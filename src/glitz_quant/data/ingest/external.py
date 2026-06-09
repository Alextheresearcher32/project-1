"""
External market data sources beyond exchanges:
- CoinGecko for broad coin metadata and reference prices
- Pyth Hermes for oracle prices (low-latency, on-chain)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from glitz_quant.settings import get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class CoinGeckoClient:
    BASE = "https://api.coingecko.com/api/v3"

    def __init__(self) -> None:
        s = get_settings()
        self.api_key = s.coingecko_api_key.get_secret_value() if s.coingecko_api_key else None
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key  # demo key header; pro uses x-cg-pro-api-key
        self.client = httpx.AsyncClient(headers=headers, timeout=20.0)

    async def aclose(self) -> None:
        await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def simple_price(self, coin_ids: list[str], vs: str = "usd") -> dict[str, dict[str, float]]:
        r = await self.client.get(
            f"{self.BASE}/simple/price",
            params={"ids": ",".join(coin_ids), "vs_currencies": vs, "include_24hr_change": "true"},
        )
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def market_data(self, coin_id: str) -> dict[str, Any]:
        r = await self.client.get(f"{self.BASE}/coins/{coin_id}")
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def global_metrics(self) -> dict[str, Any]:
        r = await self.client.get(f"{self.BASE}/global")
        r.raise_for_status()
        return r.json()  # type: ignore[no-any-return]


class PythHermesClient:
    """
    Pyth Network's Hermes web service. Public endpoint, no signup.
    Useful as a venue-independent reference oracle.
    """

    # A handful of common Pyth price feed IDs. Full list:
    # https://www.pyth.network/developers/price-feed-ids
    FEED_IDS = {
        "BTC-USD": "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43",
        "ETH-USD": "0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace",
        "SOL-USD": "0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
    }

    def __init__(self) -> None:
        self.base = get_settings().pyth_hermes_url
        self.client = httpx.AsyncClient(timeout=10.0)

    async def aclose(self) -> None:
        await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=3))
    async def latest_price(self, symbol: str) -> Decimal | None:
        feed_id = self.FEED_IDS.get(symbol)
        if not feed_id:
            return None
        r = await self.client.get(
            f"{self.base}/v2/updates/price/latest",
            params={"ids[]": feed_id, "parsed": "true"},
        )
        r.raise_for_status()
        data = r.json()
        parsed = data.get("parsed", [])
        if not parsed:
            return None
        p = parsed[0].get("price", {})
        # Pyth returns price as integer with separate `expo` (negative exponent)
        try:
            price_int = int(p["price"])
            expo = int(p["expo"])
            return Decimal(price_int) * (Decimal(10) ** Decimal(expo))
        except (KeyError, ValueError, TypeError):
            return None
