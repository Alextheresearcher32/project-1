"""
Pre-flight health check.

Run:  uv run python -m glitz_quant.scripts.healthcheck

Verifies:
- .env loaded and required keys present for active venues
- Redis reachable
- Supabase DB reachable (if configured)
- LLM provider reachable (cheap ping)
- Live-trading gate state (does not enable — only reports)

Exits non-zero if anything required is broken.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console
from rich.table import Table

from glitz_quant.settings import (
    LiveTradingGate,
    Mode,
    get_exchanges_config,
    get_settings,
)
from glitz_quant.utils.logging import configure_logging, get_logger

console = Console()
log = get_logger(__name__)


async def check_redis() -> tuple[bool, str]:
    try:
        import redis.asyncio as redis  # type: ignore[import-not-found]

        s = get_settings()
        client = redis.from_url(s.redis_url, socket_connect_timeout=3)
        pong = await client.ping()
        await client.aclose()
        return (bool(pong), "ok" if pong else "no pong")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


async def check_supabase() -> tuple[bool, str]:
    s = get_settings()
    if not s.supabase_db_url:
        return (True, "skipped (not configured)")
    try:
        import asyncpg  # type: ignore[import-not-found]

        conn = await asyncpg.connect(
            s.supabase_db_url.get_secret_value(), timeout=5
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return (True, "ok")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


async def check_anthropic() -> tuple[bool, str]:
    s = get_settings()
    if not s.anthropic_api_key:
        return (True, "skipped (no key)")
    try:
        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        client = AsyncAnthropic(api_key=s.anthropic_api_key.get_secret_value())
        # cheap ping — list models has no charge
        await client.models.list(limit=1)
        return (True, "ok")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def check_required_keys_for_enabled_venues() -> tuple[bool, str]:
    s = get_settings()
    venues = get_exchanges_config().get("venues", {})

    venue_to_keys = {
        "coinbase": ("coinbase_api_key", "coinbase_api_secret"),
        "kraken": ("kraken_api_key", "kraken_api_secret"),
        "binance_us": ("binance_us_api_key", "binance_us_api_secret"),
        "gemini": ("gemini_api_key", "gemini_api_secret"),
        "hyperliquid": ("hyperliquid_api_private_key",),
    }

    missing: list[str] = []
    for venue, cfg in venues.items():
        if not cfg.get("enabled"):
            continue
        if venue == "paper":
            continue
        if s.glitz_mode != Mode.LIVE:
            # only enforce key presence in live mode
            continue
        for key in venue_to_keys.get(venue, ()):
            val = getattr(s, key, None)
            if not val:
                missing.append(f"{venue}: missing {key.upper()}")

    if missing:
        return (False, "; ".join(missing))
    return (True, "ok")


async def main() -> int:
    configure_logging()
    s = get_settings()

    console.rule("[bold]glitz-quant healthcheck[/bold]")
    console.print(f"env: [cyan]{s.glitz_env.value}[/cyan]  mode: [cyan]{s.glitz_mode.value}[/cyan]")
    console.print()

    checks = [
        ("Redis", await check_redis()),
        ("Supabase", await check_supabase()),
        ("Anthropic", await check_anthropic()),
        ("Required keys for enabled venues", check_required_keys_for_enabled_venues()),
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    all_ok = True
    for name, (ok, detail) in checks:
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, status, detail)

    console.print(table)
    console.print()

    # Live-trading gate report
    gate_open, reasons = LiveTradingGate.check()
    if gate_open:
        console.print("[bold red]LIVE TRADING GATE: OPEN[/bold red]")
        console.print(
            "[yellow]Real orders will be sent if you switch to live mode.[/yellow]"
        )
    else:
        console.print("[bold green]Live trading gate: closed[/bold green] (safe)")
        for r in reasons:
            console.print(f"  - {r}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
