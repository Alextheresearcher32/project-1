"""
Global kill routine. Invoked by scripts/kill.sh after the halt file is touched.

Steps:
1. Read kill switch state (must be armed — exit if not).
2. For every venue with credentials, instantiate the live adapter and cancel-all.
3. Write a critical incident to Supabase.
4. Broadcast to alert channels.

This is intentionally simple and synchronous in shape. If anything fails,
it logs and continues — the goal is to cancel as much as possible, then
make noise.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from glitz_quant.data.store.supabase_store import SupabaseStore
from glitz_quant.data.types import Incident, Venue
from glitz_quant.execution.adapters.ccxt_adapter import CCXTAdapter
from glitz_quant.monitoring import alerts
from glitz_quant.risk import kill_switch
from glitz_quant.settings import LiveTradingGate, get_settings
from glitz_quant.utils.logging import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


async def _try_cancel_all_on(venue: Venue) -> int:
    try:
        adapter = CCXTAdapter(venue)
        await adapter.start()
        count = await adapter.cancel_all()
        await adapter.stop()
        log.info("cancelled_all_on_venue", venue=venue.value, count=count)
        return count
    except Exception as e:
        log.error("cancel_all_failed", venue=venue.value, err=str(e))
        return -1


async def main() -> None:
    # Arm the kill switch first (idempotent).
    kill_switch.arm()

    s = get_settings()
    summary: dict[str, int] = {}

    # Cancel on every venue with creds, regardless of mode — when killing we are
    # cleaning up. The CCXT adapter will simply error out if creds are missing.
    candidate_venues = [Venue.COINBASE, Venue.KRAKEN, Venue.BINANCE_US, Venue.GEMINI]
    for v in candidate_venues:
        # Skip venues we have no creds for to avoid spurious failures
        if v == Venue.COINBASE and not s.coinbase_api_key:
            continue
        if v == Venue.KRAKEN and not s.kraken_api_key:
            continue
        if v == Venue.BINANCE_US and not s.binance_us_api_key:
            continue
        if v == Venue.GEMINI and not s.gemini_api_key:
            continue
        summary[v.value] = await _try_cancel_all_on(v)

    # Record incident
    incident = Incident(
        severity="critical",
        kind="kill_switch_armed",
        message=f"Global kill executed. Cancellations: {summary}",
        details={"summary": summary, "live_gate": dict(zip(["open", "blockers"], LiveTradingGate.check()))},
    )

    if s.supabase_db_url:
        try:
            store = SupabaseStore()
            await store.connect(min_size=1, max_size=2)
            await store.insert_incident(incident)
            await store.close()
        except Exception as e:
            log.error("kill_incident_persist_failed", err=str(e))

    msg = (
        f"<b>KILL SWITCH ARMED</b>\n"
        f"time: {datetime.now(timezone.utc).isoformat()}\n"
        f"cancellations: {summary}"
    )
    await alerts.broadcast("critical", msg)
    log.error("kill_complete", summary=summary)


if __name__ == "__main__":
    asyncio.run(main())
