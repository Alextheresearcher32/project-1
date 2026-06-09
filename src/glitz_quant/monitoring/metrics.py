"""Prometheus metrics. Exposed on settings.monitoring.prometheus_port."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Trading
orders_submitted_total = Counter(
    "glitz_orders_submitted_total", "Total orders submitted", ["venue", "symbol", "side"]
)
orders_filled_total = Counter(
    "glitz_orders_filled_total", "Total orders filled", ["venue", "symbol", "side"]
)
orders_rejected_total = Counter(
    "glitz_orders_rejected_total", "Total orders rejected", ["reason"]
)
signals_emitted_total = Counter(
    "glitz_signals_emitted_total", "Signals emitted by strategy", ["strategy", "direction"]
)

# Risk
account_equity_usd = Gauge("glitz_account_equity_usd", "Account equity in USD")
account_pnl_daily_usd = Gauge("glitz_account_pnl_daily_usd", "Daily realized PnL in USD")
circuit_breaker_active = Gauge(
    "glitz_circuit_breaker_active", "Whether a circuit breaker is active", ["name"]
)
position_notional_usd = Gauge(
    "glitz_position_notional_usd", "Position notional in USD", ["venue", "symbol"]
)

# Latency
order_submit_latency = Histogram(
    "glitz_order_submit_latency_seconds", "Order submission latency"
)
llm_call_latency = Histogram(
    "glitz_llm_call_latency_seconds", "LLM call latency", ["agent", "provider"]
)


def start_metrics_server(port: int = 9100) -> None:
    start_http_server(port)
