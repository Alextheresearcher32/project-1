"""Alert routing — Telegram, Discord. Used for circuit breakers, kill events."""

from __future__ import annotations

import httpx

from glitz_quant.settings import get_settings
from glitz_quant.utils.logging import get_logger

log = get_logger(__name__)


async def send_telegram(message: str) -> bool:
    s = get_settings()
    if not s.telegram_bot_token or not s.telegram_chat_id:
        return False
    token = s.telegram_bot_token.get_secret_value()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(url, json={
                "chat_id": s.telegram_chat_id,
                "text": message[:4000],
                "parse_mode": "HTML",
            })
            return r.status_code == 200
        except Exception as e:
            log.warning("telegram_send_failed", err=str(e))
            return False


async def send_discord(message: str) -> bool:
    s = get_settings()
    if not s.discord_webhook_url:
        return False
    url = s.discord_webhook_url.get_secret_value()
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(url, json={"content": message[:2000]})
            return r.status_code in (200, 204)
        except Exception as e:
            log.warning("discord_send_failed", err=str(e))
            return False


async def broadcast(level: str, message: str) -> None:
    """Route by level to configured channels."""
    text = f"[{level.upper()}] {message}"
    await send_telegram(text)
    await send_discord(text)
