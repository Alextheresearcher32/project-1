-- ============================================================
-- glitz-quant Supabase schema.
-- Run once via Supabase SQL Editor on your project.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- -------- Market data --------
CREATE TABLE IF NOT EXISTS candles (
    venue        text        NOT NULL,
    symbol       text        NOT NULL,
    timeframe    text        NOT NULL,
    ts           timestamptz NOT NULL,
    open         numeric     NOT NULL,
    high         numeric     NOT NULL,
    low          numeric     NOT NULL,
    close        numeric     NOT NULL,
    volume       numeric     NOT NULL,
    PRIMARY KEY (venue, symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts ON candles (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS trades (
    id          uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    venue       text        NOT NULL,
    symbol      text        NOT NULL,
    price       numeric     NOT NULL,
    size        numeric     NOT NULL,
    side        text        NOT NULL,
    ts          timestamptz NOT NULL,
    venue_trade_id text
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades (symbol, ts DESC);

-- -------- Signals (strategy output) --------
CREATE TABLE IF NOT EXISTS signals (
    id                    uuid PRIMARY KEY,
    strategy              text        NOT NULL,
    symbol                text        NOT NULL,
    direction             text        NOT NULL,
    target_notional_usd   numeric     NOT NULL,
    confidence            real        NOT NULL,
    confidence_label      text        NOT NULL,
    stop_loss_price       numeric,
    take_profit_price     numeric,
    valid_until           timestamptz,
    reason                text,
    metadata              jsonb       DEFAULT '{}'::jsonb,
    ts                    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_signals_strategy_ts ON signals (strategy, ts DESC);

-- -------- Orders --------
CREATE TABLE IF NOT EXISTS orders (
    id                  uuid PRIMARY KEY,
    client_order_id     text        NOT NULL UNIQUE,
    venue               text        NOT NULL,
    symbol              text        NOT NULL,
    side                text        NOT NULL,
    order_type          text        NOT NULL,
    size                numeric     NOT NULL,
    price               numeric,
    stop_price          numeric,
    time_in_force       text        NOT NULL,
    status              text        NOT NULL,
    filled_size         numeric     NOT NULL DEFAULT 0,
    avg_fill_price      numeric,
    fees_paid           numeric     NOT NULL DEFAULT 0,
    venue_order_id      text,
    parent_signal_id    uuid,
    strategy            text,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now(),
    rejected_reason     text
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_created ON orders (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders (strategy);

-- -------- Fills --------
CREATE TABLE IF NOT EXISTS fills (
    id              uuid PRIMARY KEY,
    order_id        uuid NOT NULL REFERENCES orders(id),
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
CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills (order_id);

-- -------- Positions (current state) --------
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

-- -------- Equity curve / account snapshots --------
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts               timestamptz NOT NULL DEFAULT now(),
    mode             text        NOT NULL,        -- paper | live
    total_equity_usd numeric     NOT NULL,
    cash_usd         numeric     NOT NULL,
    positions_usd    numeric     NOT NULL,
    realized_pnl_24h numeric     NOT NULL DEFAULT 0,
    PRIMARY KEY (mode, ts)
);

-- -------- Agent runs (LLM calls) --------
CREATE TABLE IF NOT EXISTS agent_runs (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent           text        NOT NULL,        -- director | quant | risk_advisor | etc.
    provider        text        NOT NULL,        -- anthropic | google | groq | etc.
    model           text        NOT NULL,
    prompt_tokens   int,
    completion_tokens int,
    cost_usd        numeric,
    latency_ms      int,
    input           jsonb,
    output          jsonb,
    error           text,
    ts              timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_ts ON agent_runs (agent, ts DESC);

-- -------- Incidents (circuit breakers, rejects, etc.) --------
CREATE TABLE IF NOT EXISTS incidents (
    id              uuid PRIMARY KEY,
    severity        text        NOT NULL,
    kind            text        NOT NULL,
    message         text        NOT NULL,
    details         jsonb       DEFAULT '{}'::jsonb,
    ts              timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_incidents_severity_ts ON incidents (severity, ts DESC);

-- -------- RAG documents (pgvector) --------
CREATE TABLE IF NOT EXISTS research_documents (
    id              bigserial PRIMARY KEY,
    source          text        NOT NULL,
    chunk_index     int         NOT NULL,
    content         text        NOT NULL,
    embedding       vector(3072),  -- text-embedding-3-large dimension
    metadata        jsonb       DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_source ON research_documents (source);

-- Vector index — HNSW is faster than IVFFlat for our scale
-- Build only after you have data; empty index will error.
-- CREATE INDEX ON research_documents USING hnsw (embedding vector_cosine_ops);
