-- Local dev mirror of Supabase schema. Production uses Supabase managed Postgres.
-- This file is only loaded for the local docker compose Postgres container.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
