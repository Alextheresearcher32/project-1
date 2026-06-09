"""
Supabase persistence. Uses asyncpg directly against the Supabase DB URL.
The supabase-py client is also available for auth/realtime/storage.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import asyncpg

from glitz_quant.data.types import (
    Candle,
    Fill,
    Incident,
    Order,
    Position,
    Signal,
    Trade,
)
from glitz_quant.settings import get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


def _to_jsonb(d: dict[str, Any]) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "value"):
            return o.value
        return str(o)

    return json.dumps(d, default=default)


class SupabaseStore:
    def __init__(self) -> None:
        s = get_settings()
        if not s.supabase_db_url:
            raise RuntimeError("SUPABASE_DB_URL is not set in .env")
        self.dsn = s.supabase_db_url.get_secret_value()
        self.pool: asyncpg.Pool | None = None

    async def connect(self, min_size: int = 2, max_size: int = 10) -> None:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=min_size,
                max_size=max_size,
                command_timeout=10.0,
            )
            log.info("supabase_pool_connected", min=min_size, max=max_size)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            raise RuntimeError("SupabaseStore not connected — call .connect() first")
        return self.pool

    # -------- Candles --------
    async def upsert_candles(self, candles: list[Candle]) -> None:
        if not candles:
            return
        rows = [
            (c.venue.value, c.symbol, c.timeframe, c.ts, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
        async with self._require_pool().acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO candles (venue, symbol, timeframe, ts, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (venue, symbol, timeframe, ts) DO UPDATE
                SET open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                rows,
            )

    async def get_candles(
        self, venue: str, symbol: str, timeframe: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        async with self._require_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ts, open, high, low, close, volume
                FROM candles
                WHERE venue = $1 AND symbol = $2 AND timeframe = $3
                ORDER BY ts DESC
                LIMIT $4
                """,
                venue,
                symbol,
                timeframe,
                limit,
            )
        return [dict(r) for r in reversed(rows)]

    # -------- Trades --------
    async def insert_trades(self, trades: list[Trade]) -> None:
        if not trades:
            return
        rows = [
            (t.venue.value, t.symbol, t.price, t.size, t.side.value, t.ts, t.trade_id)
            for t in trades
        ]
        async with self._require_pool().acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO trades (venue, symbol, price, size, side, ts, venue_trade_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                rows,
            )

    # -------- Signals --------
    async def insert_signal(self, sig: Signal) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO signals
                  (id, strategy, symbol, direction, target_notional_usd, confidence,
                   confidence_label, stop_loss_price, take_profit_price, valid_until,
                   reason, metadata, ts)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13)
                """,
                sig.id,
                sig.strategy,
                sig.symbol,
                sig.direction.value,
                sig.target_notional_usd,
                sig.confidence,
                sig.confidence_label.value,
                sig.stop_loss_price,
                sig.take_profit_price,
                sig.valid_until,
                sig.reason,
                _to_jsonb(sig.metadata),
                sig.ts,
            )

    # -------- Orders --------
    async def upsert_order(self, order: Order) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orders
                  (id, client_order_id, venue, symbol, side, order_type, size, price,
                   stop_price, time_in_force, status, filled_size, avg_fill_price,
                   fees_paid, venue_order_id, parent_signal_id, strategy,
                   created_at, updated_at, rejected_reason)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
                ON CONFLICT (id) DO UPDATE SET
                    status = EXCLUDED.status,
                    filled_size = EXCLUDED.filled_size,
                    avg_fill_price = EXCLUDED.avg_fill_price,
                    fees_paid = EXCLUDED.fees_paid,
                    venue_order_id = EXCLUDED.venue_order_id,
                    updated_at = EXCLUDED.updated_at,
                    rejected_reason = EXCLUDED.rejected_reason
                """,
                order.id, order.client_order_id, order.venue.value, order.symbol,
                order.side.value, order.order_type.value, order.size, order.price,
                order.stop_price, order.time_in_force.value, order.status.value,
                order.filled_size, order.avg_fill_price, order.fees_paid,
                order.venue_order_id, order.parent_signal_id, order.strategy,
                order.created_at, order.updated_at, order.rejected_reason,
            )

    async def insert_fill(self, fill: Fill) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO fills
                  (id, order_id, venue, symbol, side, price, size, fee, fee_currency, ts, venue_trade_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                fill.id, fill.order_id, fill.venue.value, fill.symbol, fill.side.value,
                fill.price, fill.size, fill.fee, fill.fee_currency, fill.ts, fill.venue_trade_id,
            )

    # -------- Positions --------
    async def upsert_position(self, pos: Position) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO positions (venue, symbol, size, avg_entry_price, realized_pnl, unrealized_pnl, last_updated)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (venue, symbol) DO UPDATE SET
                    size = EXCLUDED.size,
                    avg_entry_price = EXCLUDED.avg_entry_price,
                    realized_pnl = EXCLUDED.realized_pnl,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    last_updated = EXCLUDED.last_updated
                """,
                pos.venue.value, pos.symbol, pos.size, pos.avg_entry_price,
                pos.realized_pnl, pos.unrealized_pnl, pos.last_updated,
            )

    async def get_position(self, venue: str, symbol: str) -> dict[str, Any] | None:
        async with self._require_pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM positions WHERE venue=$1 AND symbol=$2",
                venue, symbol,
            )
        return dict(row) if row else None

    async def get_all_positions(self) -> list[dict[str, Any]]:
        async with self._require_pool().acquire() as conn:
            rows = await conn.fetch("SELECT * FROM positions WHERE size != 0")
        return [dict(r) for r in rows]

    # -------- Incidents --------
    async def insert_incident(self, inc: Incident) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO incidents (id, severity, kind, message, details, ts)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6)
                """,
                inc.id, inc.severity, inc.kind, inc.message, _to_jsonb(inc.details), inc.ts,
            )

    # -------- Equity snapshots --------
    async def record_equity(
        self, mode: str, total: Decimal, cash: Decimal, positions: Decimal, realized_24h: Decimal
    ) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO equity_snapshots (mode, total_equity_usd, cash_usd, positions_usd, realized_pnl_24h)
                VALUES ($1, $2, $3, $4, $5)
                """,
                mode, total, cash, positions, realized_24h,
            )

    # -------- Agent runs --------
    async def log_agent_run(
        self,
        agent: str,
        provider: str,
        model: str,
        input_: dict[str, Any],
        output: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        async with self._require_pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_runs
                  (agent, provider, model, prompt_tokens, completion_tokens, cost_usd,
                   latency_ms, input, output, error)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,$10)
                """,
                agent, provider, model, prompt_tokens, completion_tokens, cost_usd,
                latency_ms, _to_jsonb(input_), _to_jsonb(output or {}), error,
            )
