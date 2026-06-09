"""
Order Management System. The single chokepoint between strategies and
exchange adapters.

Flow:
  Signal -> sizer -> Order -> RiskEngine.check -> Adapter.submit
                                |
                                +-- rejected -> log incident, do not send

Live-mode safety:
  - Adapter selection routes to PaperAdapter unless LiveTradingGate passes
  - Each adapter is registered with a venue; only enabled venues are reachable
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from glitz_quant.data.store.redis_cache import RedisCache
from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.data.types import (
    Incident,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Signal,
    SignalDirection,
    TimeInForce,
    Venue,
)
from glitz_quant.execution.adapters.base import ExchangeAdapter
from glitz_quant.execution.adapters.paper import PaperAdapter
from glitz_quant.risk.circuit_breakers import CircuitBreakerEngine
from glitz_quant.risk.engine import RiskEngine
from glitz_quant.risk.kill_switch import is_killed
from glitz_quant.settings import LiveTradingGate, Mode, get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


class OMS:
    def __init__(
        self,
        cache: RedisCache,
        store: SupabaseStore | None,
        risk: RiskEngine,
        breakers: CircuitBreakerEngine,
    ) -> None:
        self.cache = cache
        self.store = store
        self.risk = risk
        self.breakers = breakers
        self.adapters: dict[Venue, ExchangeAdapter] = {}
        self.positions: dict[tuple[Venue, str], Position] = {}
        self.daily_pnl_usd = Decimal(0)
        self._account_notional_usd = Decimal(0)  # updated on fills

    def register_adapter(self, adapter: ExchangeAdapter) -> None:
        self.adapters[adapter.venue] = adapter
        log.info("oms_adapter_registered", venue=adapter.venue.value)

    def get_position(self, venue: Venue, symbol: str) -> Position:
        return self.positions.get(
            (venue, symbol),
            Position(symbol=symbol, venue=venue),
        )

    async def process_signal(self, signal: Signal) -> Order | None:
        """Convert a signal to an order, run risk, route to adapter."""
        # 1) Hard halt checks
        if is_killed():
            await self._record_incident("critical", "killed", "kill switch active, signal dropped", signal)
            return None
        if self.breakers.is_halted():
            await self._record_incident("error", "breaker_halt", "circuit breaker halt, signal dropped", signal)
            return None

        # 2) Persist the signal
        if self.store is not None:
            try:
                await self.store.insert_signal(signal)
            except Exception as e:
                log.warning("signal_persist_failed", err=str(e))

        if signal.direction == SignalDirection.FLAT:
            return await self._close_position(signal)

        # 3) Pick venue (for now: paper always; in live, pick first enabled venue)
        venue = self._select_venue(signal)
        if venue is None:
            await self._record_incident("error", "no_venue", "no enabled venue for signal", signal)
            return None

        # 4) Get reference price
        ticker = await self.cache.get_best_ticker(signal.symbol) or await self.cache.get_ticker(
            Venue.COINBASE, signal.symbol
        )
        if ticker is None:
            await self._record_incident("warning", "no_price", "no reference price available", signal)
            return None
        ref_price = ticker.mid

        # 5) Size the order
        size, price = self._size(signal, ref_price)
        if size <= 0:
            await self._record_incident("warning", "zero_size", "computed order size is 0", signal)
            return None

        side = Side.BUY if signal.direction == SignalDirection.LONG else Side.SELL
        order = Order(
            venue=venue,
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            size=size,
            price=price,
            time_in_force=TimeInForce.GTC,
            parent_signal_id=signal.id,
            strategy=signal.strategy,
        )

        # 6) Risk check
        position = self.get_position(venue, signal.symbol)
        decision = self.risk.check_order(
            order=order,
            current_position=position,
            current_account_notional_usd=self._account_notional_usd,
            current_daily_pnl_usd=self.daily_pnl_usd,
            reference_price_usd=ref_price,
        )
        if decision.rejected:
            order.status = OrderStatus.REJECTED
            order.rejected_reason = "; ".join(decision.reasons)
            self.breakers.observe_order_outcome(rejected=True)
            await self._persist_order(order)
            return order

        # 7) Submit
        adapter = self.adapters[venue]
        submitted = await adapter.submit_order(order)
        self.breakers.observe_order_outcome(rejected=submitted.status == OrderStatus.REJECTED)

        await self._persist_order(submitted)

        # 8) If immediately filled (paper often is), update position + PnL
        if submitted.status == OrderStatus.FILLED:
            await self._apply_fill_to_position(submitted)
        return submitted

    async def cancel_all_everywhere(self) -> dict[Venue, int]:
        """Used by kill switch."""
        results: dict[Venue, int] = {}
        for venue, adapter in self.adapters.items():
            try:
                results[venue] = await adapter.cancel_all()
            except Exception as e:
                log.error("cancel_all_failed", venue=venue.value, err=str(e))
                results[venue] = -1
        return results

    # -------- Internal --------
    def _select_venue(self, signal: Signal) -> Venue | None:
        """Pick a venue. Paper unless live gate is open AND a non-paper adapter exists."""
        s = get_settings()
        gate_open, _ = LiveTradingGate.check()
        if s.glitz_mode != Mode.LIVE or not gate_open:
            return Venue.PAPER if Venue.PAPER in self.adapters else None

        # In live mode with gate open: pick first non-paper adapter
        for v in (Venue.COINBASE, Venue.KRAKEN, Venue.BINANCE_US, Venue.GEMINI):
            if v in self.adapters and not self.breakers.is_venue_disabled(v.value):
                return v
        return None

    def _size(self, signal: Signal, ref_price: Decimal) -> tuple[Decimal, Decimal]:
        """
        Convert target_notional_usd to (size, limit_price).
        For LONG: limit price slightly below mid (passive entry).
        For SHORT: limit price slightly above mid.
        """
        notional = signal.target_notional_usd
        size = (notional / ref_price).quantize(Decimal("0.00000001"))
        # passive entry — 5 bps inside mid
        if signal.direction == SignalDirection.LONG:
            limit = (ref_price * Decimal("0.9995")).quantize(Decimal("0.01"))
        else:
            limit = (ref_price * Decimal("1.0005")).quantize(Decimal("0.01"))
        return size, limit

    async def _close_position(self, signal: Signal) -> Order | None:
        """FLAT signal: close any open position in the symbol on the first venue we have one."""
        for (venue, sym), pos in list(self.positions.items()):
            if sym != signal.symbol or pos.is_flat:
                continue
            side = Side.SELL if pos.is_long else Side.BUY
            ticker = await self.cache.get_ticker(venue, sym) or await self.cache.get_best_ticker(sym)
            if ticker is None:
                continue
            limit = (ticker.mid * (Decimal("1.0005") if side == Side.SELL else Decimal("0.9995"))).quantize(Decimal("0.01"))
            order = Order(
                venue=venue,
                symbol=sym,
                side=side,
                order_type=OrderType.LIMIT,
                size=abs(pos.size),
                price=limit,
                parent_signal_id=signal.id,
                strategy=signal.strategy,
            )
            adapter = self.adapters.get(venue)
            if adapter is None:
                continue
            submitted = await adapter.submit_order(order)
            await self._persist_order(submitted)
            if submitted.status == OrderStatus.FILLED:
                await self._apply_fill_to_position(submitted)
            return submitted
        return None

    async def _apply_fill_to_position(self, order: Order) -> None:
        if order.status != OrderStatus.FILLED or order.avg_fill_price is None:
            return
        key = (order.venue, order.symbol)
        pos = self.positions.get(key) or Position(symbol=order.symbol, venue=order.venue)
        signed_size = order.filled_size if order.side == Side.BUY else -order.filled_size

        # PnL accounting: if reducing/flipping, realize the closed portion
        if pos.size != 0 and ((pos.size > 0 and signed_size < 0) or (pos.size < 0 and signed_size > 0)):
            close_size = min(abs(signed_size), abs(pos.size))
            direction = Decimal(1) if pos.size > 0 else Decimal(-1)
            realized_per_unit = (order.avg_fill_price - pos.avg_entry_price) * direction
            realized = realized_per_unit * close_size - order.fees_paid
            pos.realized_pnl += realized
            self.daily_pnl_usd += realized
            self.breakers.observe_realized_pnl(realized)

        new_size = pos.size + signed_size
        if new_size != 0 and (pos.size == 0 or ((pos.size > 0) == (new_size > 0))):
            # opening or growing — update avg entry
            total_cost = pos.size * pos.avg_entry_price + signed_size * order.avg_fill_price
            pos.avg_entry_price = (total_cost / new_size).copy_abs()
        elif new_size == 0:
            pos.avg_entry_price = Decimal(0)

        pos.size = new_size
        pos.last_updated = datetime.now(timezone.utc)
        self.positions[key] = pos

        # Update account-level notional total
        self._account_notional_usd = sum((p.notional for p in self.positions.values()), Decimal(0))

        if self.store is not None:
            try:
                await self.store.upsert_position(pos)
            except Exception as e:
                log.warning("position_persist_failed", err=str(e))

    async def _persist_order(self, order: Order) -> None:
        if self.store is None:
            return
        try:
            await self.store.upsert_order(order)
        except Exception as e:
            log.warning("order_persist_failed", err=str(e))

    async def _record_incident(self, severity: str, kind: str, message: str, signal: Signal | None) -> None:
        inc = Incident(
            severity=severity,
            kind=kind,
            message=message,
            details={"signal_id": str(signal.id) if signal else None, "strategy": signal.strategy if signal else None},
        )
        log.warning("oms_incident", severity=severity, kind=kind, msg=message)
        if self.store is not None:
            try:
                await self.store.insert_incident(inc)
            except Exception as e:
                log.warning("incident_persist_failed", err=str(e))


def build_default_oms(cache: RedisCache, store: SupabaseStore | None) -> OMS:
    """Helper to build an OMS with the paper adapter pre-registered."""
    risk = RiskEngine()
    breakers = CircuitBreakerEngine()
    oms = OMS(cache=cache, store=store, risk=risk, breakers=breakers)
    oms.register_adapter(PaperAdapter(cache=cache))
    return oms
