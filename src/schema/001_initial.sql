-- ============================================================
-- glitz-quant  —  Initial Supabase schema
-- Run once against your Supabase project (SQL Editor or psql).
-- All tables use gen_random_uuid() for PK defaults.
-- ============================================================

-- Enable pgcrypto for gen_random_uuid() (already available in Supabase)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- candles  —  OHLCV bars from all venues
-- ============================================================
CREATE TABLE IF NOT EXISTS candles (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    venue       text        NOT NULL,
    symbol      text        NOT NULL,
    timeframe   text        NOT NULL,   -- '1m', '15m', '1h', '1d'
    ts          timestamptz NOT NULL,
    open        numeric     NOT NULL,
    high        numeric     NOT NULL,
    low         numeric     NOT NULL,
    close       numeric     NOT NULL,
    volume      numeric     NOT NULL DEFAULT 0,
    UNIQUE (venue, symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS ix_candles_lookup ON candles (venue, symbol, timeframe, ts DESC);

-- ============================================================
-- trades  —  individual market trades (tick data)
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    venue           text        NOT NULL,
    symbol          text        NOT NULL,
    price           numeric     NOT NULL,
    size            numeric     NOT NULL,
    side            text        NOT NULL,   -- 'buy' | 'sell'
    ts              timestamptz NOT NULL,
    venue_trade_id  text
);
CREATE INDEX IF NOT EXISTS ix_trades_lookup ON trades (venue, symbol, ts DESC);

-- ============================================================
-- signals  —  strategy signal log
-- ============================================================
CREATE TABLE IF NOT EXISTS signals (
    id                    uuid        PRIMARY KEY,
    strategy              text        NOT NULL,
    symbol                text        NOT NULL,
    direction             text        NOT NULL,   -- 'long' | 'short' | 'flat'
    target_notional_usd   numeric,
    confidence            numeric,
    confidence_label      text,                   -- 'low' | 'medium' | 'high'
    stop_loss_price       numeric,
    take_profit_price     numeric,
    valid_until           timestamptz,
    reason                text,
    metadata              jsonb       NOT NULL DEFAULT '{}',
    ts                    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_signals_ts     ON signals (ts DESC);
CREATE INDEX IF NOT EXISTS ix_signals_strat  ON signals (strategy, ts DESC);

-- ============================================================
-- orders  —  order lifecycle (one row per order, upserted on fill)
-- ============================================================
CREATE TABLE IF NOT EXISTS orders (
    id                uuid        PRIMARY KEY,
    client_order_id   text        NOT NULL UNIQUE,
    venue             text        NOT NULL,
    symbol            text        NOT NULL,
    side              text        NOT NULL,   -- 'buy' | 'sell'
    order_type        text        NOT NULL,   -- 'limit' | 'limit_maker' | 'stop_limit' | 'market'
    size              numeric     NOT NULL,
    price             numeric,
    stop_price        numeric,
    time_in_force     text        NOT NULL DEFAULT 'gtc',
    status            text        NOT NULL DEFAULT 'pending',
    filled_size       numeric     NOT NULL DEFAULT 0,
    avg_fill_price    numeric,
    fees_paid         numeric     NOT NULL DEFAULT 0,
    venue_order_id    text,
    parent_signal_id  uuid        REFERENCES signals(id) ON DELETE SET NULL,
    strategy          text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    rejected_reason   text
);
CREATE INDEX IF NOT EXISTS ix_orders_signal  ON orders (parent_signal_id);
CREATE INDEX IF NOT EXISTS ix_orders_created ON orders (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_orders_status  ON orders (status, created_at DESC);

-- ============================================================
-- fills  —  individual fill events (one order can have many)
-- ============================================================
CREATE TABLE IF NOT EXISTS fills (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id        uuid        NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    venue           text        NOT NULL,
    symbol          text        NOT NULL,
    side            text        NOT NULL,
    price           numeric     NOT NULL,
    size            numeric     NOT NULL,
    fee             numeric     NOT NULL DEFAULT 0,
    fee_currency    text        NOT NULL DEFAULT 'USD',
    ts              timestamptz NOT NULL DEFAULT now(),
    venue_trade_id  text
);
CREATE INDEX IF NOT EXISTS ix_fills_order ON fills (order_id);
CREATE INDEX IF NOT EXISTS ix_fills_ts    ON fills (ts DESC);

-- ============================================================
-- positions  —  current open positions (upserted on each fill)
-- ============================================================
CREATE TABLE IF NOT EXISTS positions (
    venue               text        NOT NULL,
    symbol              text        NOT NULL,
    size                numeric     NOT NULL DEFAULT 0,
    avg_entry_price     numeric     NOT NULL DEFAULT 0,
    realized_pnl        numeric     NOT NULL DEFAULT 0,
    unrealized_pnl      numeric     NOT NULL DEFAULT 0,
    last_updated        timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, symbol)
);

-- ============================================================
-- equity_snapshots  —  periodic equity curve recording
-- ============================================================
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    mode                text        NOT NULL,   -- 'paper' | 'live'
    total_equity_usd    numeric     NOT NULL,
    cash_usd            numeric     NOT NULL DEFAULT 0,
    positions_usd       numeric     NOT NULL DEFAULT 0,
    realized_pnl_24h    numeric     NOT NULL DEFAULT 0,
    ts                  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_equity_ts ON equity_snapshots (ts DESC);

-- ============================================================
-- incidents  —  circuit breaker trips, kill switch events, OMS warnings
-- ============================================================
CREATE TABLE IF NOT EXISTS incidents (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    severity    text        NOT NULL,   -- 'info' | 'warning' | 'error' | 'critical'
    kind        text        NOT NULL,   -- 'circuit_breaker' | 'kill_switch' | 'order_reject' | etc.
    message     text        NOT NULL,
    details     jsonb       NOT NULL DEFAULT '{}',
    ts          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_incidents_ts   ON incidents (ts DESC);
CREATE INDEX IF NOT EXISTS ix_incidents_kind ON incidents (kind, ts DESC);

-- ============================================================
-- agent_runs  —  LLM call telemetry (latency, cost, tokens, output)
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_runs (
    id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    agent             text        NOT NULL,
    provider          text,
    model             text,
    prompt_tokens     int,
    completion_tokens int,
    cost_usd          numeric,
    latency_ms        int,
    input             jsonb,
    output            jsonb,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_agent_runs_agent ON agent_runs (agent, created_at DESC);
CREATE INDEX IF NOT EXISTS ix_agent_runs_ts    ON agent_runs (created_at DESC);

-- ============================================================
-- research_documents  —  RAG vector store (pgvector)
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS research_documents (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    title       text        NOT NULL,
    content     text        NOT NULL,
    embedding   vector(1536),           -- text-embedding-3-large dimension
    source      text,                   -- URL, file path, or manual
    tags        text[]      NOT NULL DEFAULT '{}',
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_research_embedding
    ON research_documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================
-- Row-level security  — lock down to service role by default
-- ============================================================
ALTER TABLE candles           ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades            ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals           ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders            ENABLE ROW LEVEL SECURITY;
ALTER TABLE fills             ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions         ENABLE ROW LEVEL SECURITY;
ALTER TABLE equity_snapshots  ENABLE ROW LEVEL SECURITY;
ALTER TABLE incidents         ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_runs        ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_documents ENABLE ROW LEVEL SECURITY;

-- Service role bypasses RLS automatically in Supabase.
-- Add anon/authenticated policies here if you expose data via the REST API.
