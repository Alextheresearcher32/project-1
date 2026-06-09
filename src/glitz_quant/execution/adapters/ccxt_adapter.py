"""
CCXT-based live adapter. Works for Coinbase, Kraken, Binance.US, Gemini —
anything CCXT supports.

CRITICAL: Only used in `live` mode AND only after the LiveTradingGate passes.
The OMS enforces this; this adapter just trusts that gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import ccxt.async_support as ccxt  # type: ignore[import-untyped]

from glitz_quant.data.types import Fill, Order, OrderStatus, OrderType, Side, Venue
from glitz_quant.execution.adapters.base import ExchangeAdapter
from glitz_quant.settings import get_exchanges_config
from glitz_quant.utils.logging import get_logger
from glitz_quant.data.ingest.ccxt_connector import (
    VENUE_TO_CCXT_ID,
    _build_client,
    _canonical_to_venue_symbol,
)

log = get_logger(__name__)


class CCXTAdapter(ExchangeAdapter):
    """Live CEX adapter via ccxt.async_support."""

    def __init__(self, venue: Venue) -> None:
        if venue not in VENUE_TO_CCXT_ID:
            raise ValueError(f"CCXTAdapter doesn't support venue {venue.value}")
        self.venue = venue
        cfg = get_exchanges_config().get("venues", {}).get(venue.value, {})
        self._fee_bps_taker = Decimal(str(cfg.get("fee_bps_taker", 30)))
        self._fee_bps_maker = Decimal(str(cfg.get("fee_bps_maker", 10)))
        self._client: ccxt.Exchange | None = None

    async def start(self) -> None:
        self._client = _build_client(self.venue)
        await self._client.load_markets()
        log.info("ccxt_adapter_started", venue=self.venue.value)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _require_client(self) -> ccxt.Exchange:
        if self._client is None:
            raise RuntimeError(f"{self.venue.value} adapter not started")
        return self._client

    @staticmethod
    def _type_to_ccxt(t: OrderType) -> str:
        return {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.LIMIT_MAKER: "limit",     # via params below
            OrderType.STOP_LIMIT: "limit",      # CCXT varies — venue-specific
            OrderType.STOP_MARKET: "market",
        }[t]

    async def submit_order(self, order: Order) -> Order:
        client = self._require_client()
        venue_sym = _canonical_to_venue_symbol(self.venue, order.symbol)

        params: dict[str, str | bool | float] = {"clientOrderId": order.client_order_id}
        if order.order_type == OrderType.LIMIT_MAKER:
            params["postOnly"] = True

        try:
            raw = await client.create_order(
                symbol=venue_sym,
                type=self._type_to_ccxt(order.order_type),
                side=order.side.value,
                amount=float(order.size),
                price=float(order.price) if order.price else None,
                params=params,
            )
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.rejected_reason = f"{type(e).__name__}: {e}"
            order.updated_at = datetime.now(timezone.utc)
            log.error("ccxt_submit_failed", venue=self.venue.value, err=str(e))
            return order

        order.venue_order_id = str(raw.get("id", ""))
        order.status = OrderStatus.SUBMITTED
        if raw.get("status") == "closed":
            order.status = OrderStatus.FILLED
            order.filled_size = Decimal(str(raw.get("filled", 0)))
            avg = raw.get("average")
            if avg:
                order.avg_fill_price = Decimal(str(avg))
        order.updated_at = datetime.now(timezone.utc)
        return order

    async def cancel_order(self, order: Order) -> bool:
        client = self._require_client()
        venue_sym = _canonical_to_venue_symbol(self.venue, order.symbol)
        try:
            await client.cancel_order(order.venue_order_id, venue_sym)
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now(timezone.utc)
            return True
        except Exception as e:
            log.warning("ccxt_cancel_failed", err=str(e), order_id=order.venue_order_id)
            return False

    async def cancel_all(self, symbol: str | None = None) -> int:
        client = self._require_client()
        venue_sym = _canonical_to_venue_symbol(self.venue, symbol) if symbol else None
        try:
            open_orders = await client.fetch_open_orders(venue_sym)
        except Exception as e:
            log.error("fetch_open_orders_failed", err=str(e))
            return 0
        count = 0
        for o in open_orders:
            try:
                await client.cancel_order(o["id"], o["symbol"])
                count += 1
            except Exception as e:
                log.warning("cancel_one_failed", id=o.get("id"), err=str(e))
        return count

    async def poll_order(self, order: Order) -> Order:
        client = self._require_client()
        venue_sym = _canonical_to_venue_symbol(self.venue, order.symbol)
        if not order.venue_order_id:
            return order
        try:
            raw = await client.fetch_order(order.venue_order_id, venue_sym)
        except Exception as e:
            log.warning("ccxt_poll_failed", err=str(e))
            return order
        status_map = {
            "open": OrderStatus.SUBMITTED,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }
        order.status = status_map.get(raw.get("status", ""), order.status)
        if "filled" in raw and raw["filled"]:
            order.filled_size = Decimal(str(raw["filled"]))
        if raw.get("average"):
            order.avg_fill_price = Decimal(str(raw["average"]))
        order.updated_at = datetime.now(timezone.utc)
        return order

    async def poll_fills(self, order: Order) -> list[Fill]:
        client = self._require_client()
        venue_sym = _canonical_to_venue_symbol(self.venue, order.symbol)
        try:
            trades = await client.fetch_my_trades(venue_sym, limit=50)
        except Exception as e:
            log.warning("fetch_my_trades_failed", err=str(e))
            return []
        out: list[Fill] = []
        for t in trades:
            if t.get("order") != order.venue_order_id:
                continue
            out.append(
                Fill(
                    order_id=order.id,
                    venue=self.venue,
                    symbol=order.symbol,
                    side=Side.BUY if t["side"] == "buy" else Side.SELL,
                    price=Decimal(str(t["price"])),
                    size=Decimal(str(t["amount"])),
                    fee=Decimal(str((t.get("fee") or {}).get("cost", 0))),
                    fee_currency=(t.get("fee") or {}).get("currency", "USD"),
                    ts=datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc),
                    venue_trade_id=str(t.get("id")),
                )
            )
        return out
