"""
Streamlit dashboard. Run:
  uv run streamlit run src/glitz_quant/dashboard/streamlit_app.py

Shows positions, recent signals, recent orders, equity curve,
incidents, and circuit-breaker state. Read-only.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.risk import kill_switch
from glitz_quant.settings import LiveTradingGate, get_settings


st.set_page_config(page_title="glitz-quant", layout="wide", page_icon="📊")


@st.cache_resource
def get_store() -> SupabaseStore | None:
    s = get_settings()
    if not s.supabase_db_url:
        return None
    store = SupabaseStore()
    asyncio.run(store.connect(min_size=1, max_size=4))
    return store


async def _fetch_table(store: SupabaseStore, query: str, *args) -> list[dict]:
    pool = store._require_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *args)
    return [dict(r) for r in rows]


def fetch(store: SupabaseStore, query: str, *args) -> pd.DataFrame:
    rows = asyncio.run(_fetch_table(store, query, *args))
    return pd.DataFrame(rows)


# -------- Header --------
s = get_settings()
gate_open, blockers = LiveTradingGate.check()
killed = kill_switch.is_killed()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Environment", s.glitz_env.value)
c2.metric("Mode", s.glitz_mode.value)
c3.metric("Live Gate", "OPEN ⚠️" if gate_open else "Closed ✅")
c4.metric("Kill Switch", "ARMED 🛑" if killed else "Off")

if killed:
    st.error("KILL SWITCH IS ARMED — orchestrator halted, no new orders.")
elif not gate_open:
    with st.expander("Live gate blockers"):
        for b in blockers:
            st.write(f"- {b}")

store = get_store()
if store is None:
    st.warning("Supabase not configured. Set SUPABASE_DB_URL in .env to see data.")
    st.stop()

# -------- Positions --------
st.header("Open positions")
positions = fetch(store, "SELECT * FROM positions WHERE size != 0 ORDER BY symbol")
if positions.empty:
    st.info("No open positions.")
else:
    st.dataframe(positions, use_container_width=True)

# -------- Recent signals --------
st.header("Recent signals (24h)")
cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
signals = fetch(
    store,
    "SELECT ts, strategy, symbol, direction, target_notional_usd, confidence, reason "
    "FROM signals WHERE ts >= $1 ORDER BY ts DESC LIMIT 200",
    cutoff,
)
st.dataframe(signals, use_container_width=True)

# -------- Recent orders --------
st.header("Recent orders (24h)")
orders = fetch(
    store,
    "SELECT created_at, venue, symbol, side, order_type, size, price, status, strategy, rejected_reason "
    "FROM orders WHERE created_at >= $1 ORDER BY created_at DESC LIMIT 200",
    cutoff,
)
st.dataframe(orders, use_container_width=True)

# -------- Equity curve --------
st.header("Equity curve (7d)")
eq_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
equity = fetch(
    store,
    "SELECT ts, total_equity_usd FROM equity_snapshots WHERE mode=$1 AND ts >= $2 ORDER BY ts",
    s.glitz_mode.value, eq_cutoff,
)
if not equity.empty:
    equity = equity.set_index("ts")
    st.line_chart(equity["total_equity_usd"])
else:
    st.info("No equity snapshots yet.")

# -------- Incidents --------
st.header("Incidents (24h)")
incidents = fetch(
    store,
    "SELECT ts, severity, kind, message FROM incidents WHERE ts >= $1 ORDER BY ts DESC LIMIT 100",
    cutoff,
)
if incidents.empty:
    st.success("No incidents.")
else:
    st.dataframe(incidents, use_container_width=True)

# -------- Agent runs --------
st.header("Agent runs (24h)")
runs = fetch(
    store,
    "SELECT ts, agent, provider, model, latency_ms, prompt_tokens, completion_tokens, error "
    "FROM agent_runs WHERE ts >= $1 ORDER BY ts DESC LIMIT 100",
    cutoff,
)
st.dataframe(runs, use_container_width=True)
