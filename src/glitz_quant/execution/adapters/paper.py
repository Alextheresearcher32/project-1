"""
Paper trading adapter. The default venue. Simulates fills using the
live ticker/orderbook from Redis. Strategies and risk engine work
identically to live — only the adapter differs.

Fill model: 'aggressive_limit'. A buy limit at >= ask fills immediately;
a sell limit at <= bid fills immediately. Otherwise the order rests and
we check it again on each poll. Slippage configurable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import cast
from uuid import uuid4

import httpx

from glitz_quant.data.store.redis_cache import RedisCache
from glitz_quant.data.types import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Venue,
)
from glitz_quant.execution.adapters.base import ExchangeAdapter
from glitz_quant.settings import get_exchanges_config
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class PaperAdapter(ExchangeAdapter):
    venue = Venue.PAPER

    def __init__(self, cache: RedisCache, starting_cash_usd: float = 10_000.0) -> None:
        self.cache = cache
        cfg = get_exchanges_config().get("venues", {}).get("paper", {})
        self._fee_bps_taker = Decimal(str(cfg.get("fee_bps_taker", 15)))
        self._fee_bps_maker = Decimal(str(cfg.get("fee_bps_maker", 5)))
        self._slippage_bps = Decimal(str(cfg.get("slippage_bps", 5)))
        self._open: dict[str, Order] = {}     # client_order_id -> Order
        self._reference_venue: Venue = Venue.COINBASE  # source for prices
        self._starting_cash = Decimal(str(starting_cash_usd))
        self._cumulative_pnl = Decimal(0)

    async def start(self) -> None:
        log.info("paper_adapter_started")

    async def stop(self) -> None:
        pass

    def set_reference_venue(self, venue: Venue) -> None:
        self._reference_venue = venue

    async def submit_order(self, order: Order) -> Order:
        ticker = await self.cache.get_ticker(self._reference_venue, order.symbol)
        if ticker is not None:
            bid, ask = ticker.bid, ticker.ask
        else:
            prices = await self._fetch_spot_price(order.symbol)
            if prices is None:
                order.status = OrderStatus.REJECTED
                order.rejected_reason = f"no reference price available for {order.symbol}"
                order.updated_at = datetime.now(timezone.utc)
                return order
            bid, ask = prices

        order.venue_order_id = f"paper-{uuid4().hex[:12]}"
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now(timezone.utc)
        self._open[order.client_order_id] = order

        await self._try_fill(order, bid, ask)
        return order

    async def cancel_order(self, order: Order) -> bool:
        existing = self._open.pop(order.client_order_id, None)
        if existing is None:
            return False
        if existing.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            existing.status = OrderStatus.CANCELLED
            existing.updated_at = datetime.now(timezone.utc)
        return True

    async def cancel_all(self, symbol: str | None = None) -> int:
        count = 0
        for cid, o in list(self._open.items()):
            if symbol and o.symbol != symbol:
                continue
            o.status = OrderStatus.CANCELLED
            o.updated_at = datetime.now(timezone.utc)
            self._open.pop(cid, None)
            count += 1
        return count

    async def poll_order(self, order: Order) -> Order:
        ticker = await self.cache.get_ticker(self._reference_venue, order.symbol)
        if ticker is not None:
            bid, ask = ticker.bid, ticker.ask
        else:
            prices = await self._fetch_spot_price(order.symbol)
            if prices is None:
                return order
            bid, ask = prices
        existing = self._open.get(order.client_order_id)
        if existing is None or existing.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return order
        await self._try_fill(existing, bid, ask)
        return existing

    async def poll_fills(self, order: Order) -> list[Fill]:
        # Paper fills are handled immediately in submit_order()
        return []

    def record_pnl(self, realized_usd: Decimal) -> None:
        self._cumulative_pnl += realized_usd

    async def fetch_total_balance(self) -> Decimal:
        return self._starting_cash + self._cumulative_pnl

    # -------- Internal --------
    async def _fetch_spot_price(self, symbol: str) -> tuple[Decimal, Decimal] | None:
        """Fetch mid price from Coinbase public API and synthesise bid/ask (1 bp spread each side).
        Used as fallback when Redis is unavailable."""
        cb_symbol = symbol.replace("/", "-")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"https://api.coinbase.com/v2/prices/{cb_symbol}/spot")
                r.raise_for_status()
                mid = Decimal(r.json()["data"]["amount"])
            half = mid * Decimal("0.0001")
            log.debug("coinbase_spot_fallback", symbol=symbol, mid=str(mid))
            return mid - half, mid + half
        except Exception as exc:
            log.warning("coinbase_spot_fallback_failed", symbol=symbol, reason=str(exc)[:120])
            return None

    async def _try_fill(self, order: Order, bid: Decimal, ask: Decimal) -> Fill | None:
        if order.status not in (OrderStatus.SUBMITTED, OrderStatus.PARTIAL):
            return None

        fill_price: Decimal | None = None
        is_maker = False

        if order.order_type in (OrderType.MARKET, OrderType.STOP_MARKET):
            fill_price = ask if order.side == Side.BUY else bid
        elif order.order_type in (OrderType.LIMIT, OrderType.LIMIT_MAKER, OrderType.STOP_LIMIT):
            if order.price is None:
                order.status = OrderStatus.REJECTED
                order.rejected_reason = "limit order without price"
                return None
            if order.side == Side.BUY and order.price >= ask:
                fill_price = ask
            elif order.side == Side.SELL and order.price <= bid:
                fill_price = bid
            else:
                is_maker = True
                # rests; no fill yet
                return None

        if fill_price is None:
            return None

        # apply slippage (only against us — buys pay more, sells receive less)
        slip = self._slippage_bps / Decimal(10000)
        if order.side == Side.BUY:
            fill_price *= Decimal(1) + slip
        else:
            fill_price *= Decimal(1) - slip

        fee_bps = self._fee_bps_maker if is_maker else self._fee_bps_taker
        fee = abs(order.size) * fill_price * fee_bps / Decimal(10000)

        order.filled_size = order.size
        order.avg_fill_price = fill_price
        order.fees_paid = fee
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now(timezone.utc)
        self._open.pop(order.client_order_id, None)

        fill = Fill(
            order_id=order.id,
            venue=Venue.PAPER,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            size=order.size,
            fee=fee,
            ts=order.updated_at,
        )
        log.info(
            "paper_filled",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side.value,
            price=str(fill_price),
            size=str(order.size),
            fee=str(fee),
        )
        return cast(Fill, fill)
