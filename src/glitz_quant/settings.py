"""
Centralized settings. Loads .env, settings.yaml, risk.yaml, exchanges.yaml,
and strategies.yaml. Everything in the system reads from here, never
directly from os.environ or yaml files.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# -------- Paths --------
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"


# -------- Enums --------
class Env(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Mode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    XAI = "xai"
    OPENROUTER = "openrouter"


# -------- Env-backed settings --------
class Settings(BaseSettings):
    """Reads from .env and process env vars."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),   # absolute so it works from any CWD
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Runtime
    glitz_env: Env = Env.DEVELOPMENT
    glitz_mode: Mode = Mode.PAPER
    log_level: str = "INFO"
    log_format: str = "json"

    # Storage
    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_db_url: SecretStr | None = None
    redis_url: str = "redis://localhost:6379/0"

    # LLM
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    xai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    llm_primary: LLMProvider = LLMProvider.ANTHROPIC
    llm_fallback: LLMProvider = LLMProvider.OPENROUTER

    # Market data
    coingecko_api_key: SecretStr | None = None
    databento_api_key: SecretStr | None = None
    tardis_api_key: SecretStr | None = None
    cryptocompare_api_key: SecretStr | None = None
    pyth_hermes_url: str = "https://hermes.pyth.network"

    # ThetaData (options/equities market data terminal)
    thetadata_api_key: SecretStr | None = None
    thetadata_host: str = "127.0.0.1"
    thetadata_port: int = 25510

    # Oanda (forex broker — data + execution)
    oanda_api_key: SecretStr | None = None
    oanda_account_id: str | None = None
    oanda_practice: bool = True  # False = live trading account

    # CEX
    coinbase_api_key: SecretStr | None = None
    coinbase_api_secret: SecretStr | None = None
    coinbase_passphrase: SecretStr | None = None
    kraken_api_key: SecretStr | None = None
    kraken_api_secret: SecretStr | None = None
    binance_us_api_key: SecretStr | None = None
    binance_us_api_secret: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    gemini_api_secret: SecretStr | None = None

    # DEX
    solana_rpc_url: str | None = None
    solana_hot_wallet_private_key: SecretStr | None = None
    solana_treasury_pubkey: str | None = None
    jupiter_api_url: str = "https://quote-api.jup.ag/v6"
    evm_rpc_url: str | None = None
    evm_hot_wallet_private_key: SecretStr | None = None
    evm_treasury_address: str | None = None
    hyperliquid_api_private_key: SecretStr | None = None
    hyperliquid_account_address: str | None = None

    # Notifications
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None
    discord_webhook_url: SecretStr | None = None
    sentry_dsn: str | None = None

    # Email
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: SecretStr | None = None
    alert_from_email: str | None = None
    alert_to_email: str | None = None

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: SecretStr = Field(default=SecretStr("dev-only-replace-me"))
    cors_origins: str = "http://localhost:3000"

    # Safety flags — all three required to enable live
    live_trading_enabled: bool = False
    live_trading_confirm_phrase: SecretStr | None = None
    live_trading_max_notional_usd: float = 0.0
    kill_switch_path: str = "/var/run/glitz-quant/HALT"

    @property
    def is_production(self) -> bool:
        return self.glitz_env == Env.PRODUCTION

    @property
    def is_live_mode(self) -> bool:
        return self.glitz_mode == Mode.LIVE


# -------- YAML config loaders --------
def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def get_app_config() -> dict[str, Any]:
    return _load_yaml("settings.yaml")


@lru_cache(maxsize=1)
def get_risk_config() -> dict[str, Any]:
    return _load_yaml("risk.yaml")


@lru_cache(maxsize=1)
def get_exchanges_config() -> dict[str, Any]:
    return _load_yaml("exchanges.yaml")


@lru_cache(maxsize=1)
def get_strategies_config() -> dict[str, Any]:
    return _load_yaml("strategies.yaml")


# -------- Live-trading gate --------
class LiveTradingGate:
    """
    Three independent checks. ALL must pass before the OMS routes
    to a real exchange adapter. If any fails, only paper/sim adapters
    are reachable. This is intentional friction.
    """

    @staticmethod
    def check() -> tuple[bool, list[str]]:
        s = get_settings()
        risk = get_risk_config()
        reasons: list[str] = []

        if not s.live_trading_enabled:
            reasons.append("LIVE_TRADING_ENABLED is false in .env")

        if not risk.get("live_trading_enabled", False):
            reasons.append("live_trading_enabled is false in config/risk.yaml")

        configured = risk.get("live_trading_confirm_phrase", "")
        provided = (
            s.live_trading_confirm_phrase.get_secret_value()
            if s.live_trading_confirm_phrase
            else ""
        )
        if not configured or configured == "REPLACE_ME_WITH_OPENSSL_RAND_HEX":
            reasons.append("live_trading_confirm_phrase not set in risk.yaml")
        elif configured != provided:
            reasons.append("LIVE_TRADING_CONFIRM_PHRASE does not match risk.yaml")

        if s.live_trading_max_notional_usd <= 0:
            reasons.append("LIVE_TRADING_MAX_NOTIONAL_USD is 0 or unset")

        return (len(reasons) == 0, reasons)
