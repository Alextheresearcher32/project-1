"""
Canonical domain types. Everything in the system uses these.
Exchange adapters normalize incoming data into these. Strategies emit
Signals. OMS emits Orders. Risk engine inspects Orders before routing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# -------- Enums --------
class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_MAKER = "limit_maker"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    POST_ONLY = "post_only"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Venue(str, Enum):
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BINANCE_US = "binance_us"
    BINANCE = "binance"
    GEMINI = "gemini"
    HYPERLIQUID = "hyperliquid"
    JUPITER = "jupiter"
    OANDA = "oanda"
    THETADATA = "thetadata"
    PAPER = "paper"


class SignalDirection(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class SignalConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# -------- Base config --------
class _Base(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=False,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


# -------- Market data --------
class Ticker(_Base):
    venue: Venue
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    ts: datetime = Field(default_factory=_now)

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread_bps(self) -> Decimal:
        if self.mid == 0:
            return Decimal(0)
        return (self.ask - self.bid) / self.mid * Decimal(10000)


class Trade(_Base):
    venue: Venue
    symbol: str
    price: Decimal
    size: Decimal
    side: Side
    ts: datetime
    trade_id: str | None = None


class Candle(_Base):
    venue: Venue
    symbol: str
    timeframe: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    ts: datetime


class OrderBookLevel(_Base):
    price: Decimal
    size: Decimal


class OrderBook(_Base):
    venue: Venue
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    ts: datetime = Field(default_factory=_now)

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / Decimal(2)


# -------- Trading objects --------
class Signal(_Base):
    """A strategy's recommendation. Goes into the OMS, not directly to an exchange."""
    id: UUID = Field(default_factory=uuid4)
    strategy: str
    symbol: str
    direction: SignalDirection
    target_notional_usd: Decimal
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: SignalConfidence = SignalConfidence.MEDIUM
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    valid_until: datetime | None = None
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_now)


class Order(BaseModel):
    """Mutable — status changes. Other types are frozen."""
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: UUID = Field(default_factory=uuid4)
    client_order_id: str = Field(default_factory=lambda: f"glitz-{uuid4().hex[:16]}")
    venue: Venue
    symbol: str
    side: Side
    order_type: OrderType
    size: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_size: Decimal = Decimal(0)
    avg_fill_price: Decimal | None = None
    fees_paid: Decimal = Decimal(0)
    venue_order_id: str | None = None
    parent_signal_id: UUID | None = None
    strategy: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    rejected_reason: str | None = None


class Fill(_Base):
    id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    venue: Venue
    symbol: str
    side: Side
    price: Decimal
    size: Decimal
    fee: Decimal = Decimal(0)
    fee_currency: str = "USD"
    ts: datetime = Field(default_factory=_now)
    venue_trade_id: str | None = None


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    symbol: str
    venue: Venue
    size: Decimal = Decimal(0)              # positive = long, negative = short
    avg_entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    unrealized_pnl: Decimal = Decimal(0)
    last_updated: datetime = Field(default_factory=_now)

    @property
    def is_flat(self) -> bool:
        return self.size == 0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def notional(self) -> Decimal:
        return abs(self.size * self.avg_entry_price)


class Incident(_Base):
    id: UUID = Field(default_factory=uuid4)
    severity: str                            # info | warning | error | critical
    kind: str                                # e.g. "circuit_breaker", "kill_switch", "order_reject"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_now)
