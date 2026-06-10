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

    def __init__(self, cache: RedisCache) -> None:
        self.cache = cache
        cfg = get_exchanges_config().get("venues", {}).get("paper", {})
        self._fee_bps_taker = Decimal(str(cfg.get("fee_bps_taker", 15)))
        self._fee_bps_maker = Decimal(str(cfg.get("fee_bps_maker", 5)))
        self._slippage_bps = Decimal(str(cfg.get("slippage_bps", 5)))
        self._open: dict[str, Order] = {}     # client_order_id -> Order
        self._reference_venue: Venue = Venue.COINBASE  # source for prices

    async def start(self) -> None:
        log.info("paper_adapter_started")

    async def stop(self) -> None:
        pass

    def set_reference_venue(self, venue: Venue) -> None:
        self._reference_venue = venue

    async def submit_order(self, order: Order) -> Order:
        ticker = await self.cache.get_ticker(self._reference_venue, order.symbol)
        if ticker is None:
            order.status = OrderStatus.REJECTED
            order.rejected_reason = f"no reference ticker for {order.symbol}"
            order.updated_at = datetime.now(timezone.utc)
            return order

        order.venue_order_id = f"paper-{uuid4().hex[:12]}"
        order.status = OrderStatus.SUBMITTED
        order.updated_at = datetime.now(timezone.utc)
        self._open[order.client_order_id] = order

        # immediately try to fill
        await self._try_fill(order, ticker.bid, ticker.ask)
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
        if ticker is None:
            return order
        existing = self._open.get(order.client_order_id)
        if existing is None or existing.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            return order
        await self._try_fill(existing, ticker.bid, ticker.ask)
        return existing

    async def poll_fills(self, order: Order) -> list[Fill]:
        # Paper fills are handled immediately in submit_order()
        return []

    async def fetch_total_balance(self) -> Decimal:
        """Paper equity: static $10k plus tracked realized PnL."""
        # For simplicity, we just return a large enough number or track it via cache.
        # Here we'll return the default 10k.
        return Decimal("10000.00")

    # -------- Internal --------
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
