"""Alert routing — Telegram, Discord, Email. Used for circuit breakers, kill events."""

from __future__ import annotations

import smtplib
from email.mime.text import MIMEText

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


def send_email(subject: str, body: str) -> bool:
    """Synchronous SMTP send. Called from async context via asyncio.to_thread."""
    s = get_settings()
    if not all([s.smtp_host, s.smtp_user, s.smtp_password, s.alert_from_email, s.alert_to_email]):
        return False
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = s.alert_from_email
        msg["To"] = s.alert_to_email
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(s.smtp_user, s.smtp_password.get_secret_value())
            smtp.sendmail(s.alert_from_email, [s.alert_to_email], msg.as_string())
        return True
    except Exception as e:
        log.warning("email_send_failed", err=str(e))
        return False


async def broadcast(level: str, message: str) -> None:
    """Route by level to configured channels per settings.yaml alert_levels."""
    import asyncio

    from glitz_quant.settings import get_app_config

    alert_levels: dict = get_app_config().get("monitoring", {}).get("alert_levels", {})
    channels = alert_levels.get(level, ["telegram"])
    if isinstance(channels, str):
        channels = [channels]

    text = f"[{level.upper()}] {message}"

    if "telegram" in channels:
        await send_telegram(text)
    if "discord" in channels:
        await send_discord(text)
    if "email" in channels:
        subject = f"glitz-quant [{level.upper()}] alert"
        await asyncio.to_thread(send_email, subject, text)
