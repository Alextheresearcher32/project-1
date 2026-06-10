"""
FastAPI backend. Read endpoints for dashboard + kill switch.
Mount with: uv run uvicorn glitz_quant.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.risk import kill_switch
from glitz_quant.settings import LiveTradingGate, get_settings
from glitz_quant.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

app = FastAPI(title="glitz-quant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- DI --------
_store: SupabaseStore | None = None


async def get_store() -> SupabaseStore | None:
    return _store


# -------- Auth --------
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    s = get_settings()
    expected = s.api_secret_key.get_secret_value()
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid api key")


# -------- Routes --------
@app.on_event("startup")
async def startup() -> None:
    global _store
    s = get_settings()
    if s.supabase_db_url:
        _store = SupabaseStore()
        try:
            await _store.connect(min_size=1, max_size=4)
        except Exception as e:
            log.warning("supabase_connect_failed_in_api", err=str(e))
            _store = None


@app.on_event("shutdown")
async def shutdown() -> None:
    if _store is not None:
        await _store.close()


@app.get("/health")
async def health() -> dict[str, object]:
    s = get_settings()
    gate_open, reasons = LiveTradingGate.check()
    return {
        "status": "ok",
        "env": s.glitz_env.value,
        "mode": s.glitz_mode.value,
        "live_gate_open": gate_open,
        "live_gate_blockers": reasons,
        "killed": kill_switch.is_killed(),
        "now": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/positions")
async def list_positions(store: SupabaseStore | None = Depends(get_store)) -> list[dict[str, object]]:
    if store is None:
        return []
    rows = await store.get_all_positions()
    return rows


@app.post("/kill", dependencies=[Depends(require_api_key)])
async def kill() -> dict[str, str]:
    kill_switch.arm()
    return {"status": "killed", "ts": datetime.now(timezone.utc).isoformat()}


@app.post("/unkill", dependencies=[Depends(require_api_key)])
async def unkill() -> dict[str, str]:
    kill_switch.disarm()
    return {"status": "disarmed", "ts": datetime.now(timezone.utc).isoformat()}
