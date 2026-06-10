# glitz-quant

Automated crypto trading platform. Multi-exchange execution, LLM-assisted research, deterministic risk controls.

**Status:** functional skeleton across all layers. Paper trading works end-to-end. Live wiring is gated behind multiple explicit opt-ins.

---

## Hard rules (read before touching anything)

1. **Paper trading is the default.** Live execution requires `LIVE_TRADING_ENABLED=true` + `live_trading_enabled: true` in `risk.yaml` + matching `LIVE_TRADING_CONFIRM_PHRASE` + `--i-understand-the-risks` CLI flag. Any one missing = paper only.
2. **All orders pass through the risk engine.** No strategy module, agent, or script submits an order directly to an exchange adapter. The OMS is the only chokepoint.
3. **LLMs do not size positions or send orders.** They produce signals and theses on a slow cadence (15min–1h). Deterministic code sizes and routes.
4. **Hot wallet caps.** The on-server hot wallet for DEX trading is capped per `config/risk.yaml`. Treasury lives on hardware wallet.
5. **Kill switch is global.** `./scripts/kill.sh` arms the halt file, cancels every open order on every venue with creds, writes an incident, broadcasts to alerts.

---

## What's in this repo

```
glitz-quant/
├── config/                 risk.yaml | exchanges.yaml | strategies.yaml | settings.yaml
├── src/glitz_quant/
│   ├── data/
│   │   ├── types.py        canonical Pydantic types (Ticker, Candle, Order, Fill, …)
│   │   ├── ingest/         CCXT connectors, CoinGecko, Pyth Hermes
│   │   └── store/          Redis cache + Supabase store + schema.sql
│   ├── risk/               engine + circuit breakers + kill switch
│   ├── execution/
│   │   ├── adapters/       base / paper / ccxt (Coinbase, Kraken, Binance.US, Gemini)
│   │   └── oms.py          single execution chokepoint
│   ├── strategies/         indicators + base + BitcoinRangeMomentum
│   ├── research/backtest/  Backtester with Sharpe / DD / win rate / PF
│   ├── agents/             llm_router (litellm) + Director / Quant / RiskAdvisor / ExecAdvisor
│   ├── rag/                LlamaIndex + Supabase pgvector for PDFs
│   ├── ml/train/           FinRL PPO trainer (research only, NOT wired to live)
│   ├── monitoring/         Prometheus metrics + Telegram/Discord alerts
│   ├── api/                FastAPI with /health, /positions, /kill, /unkill
│   ├── dashboard/          Streamlit dashboard
│   ├── orchestrator/       top-level Runner + Typer CLI
│   └── scripts/            healthcheck + kill_all
├── rust/                   Rust workspace: core / feed / book / exec (Phase 2+ for hot path)
├── tests/                  smoke tests
├── scripts/                setup_server.sh | deploy.sh | kill.sh
└── infra/
    ├── systemd/            glitz-quant.service / -api.service / -dashboard.service
    ├── prometheus/         prometheus.yml
    └── postgres/           init.sql (local dev)
```

---

## Setup

### On your server (Ubuntu 22.04 / 24.04):

```bash
# 1. Extract
tar xzf glitz-quant-complete.tar.gz
cd glitz-quant

# 2. Bootstrap system deps (installs uv, rust, docker, ufw rules)
./scripts/setup_server.sh
# log out + back in to pick up docker group

# 3. Start local infra
docker compose up -d            # redis + postgres+pgvector + prometheus + grafana

# 4. Fill in keys
cp .env.example .env
nano .env                       # at minimum: REDIS_URL, ANTHROPIC_API_KEY

# 5. Install Python + Rust deps
uv sync
cargo build --release

# 6. Initialize Supabase schema
# In Supabase SQL Editor, paste contents of:
#   src/glitz_quant/data/store/schema.sql

# 7. Sanity check
uv run python -m glitz_quant.scripts.healthcheck

# 8. Run smoke tests (no network, no DB)
uv run pytest tests/ -v

# 9. Start in paper mode
uv run glitz                    # the orchestrator
# In another terminal:
uv run uvicorn glitz_quant.api.main:app --host 0.0.0.0 --port 8000
uv run streamlit run src/glitz_quant/dashboard/streamlit_app.py
```

### Production (systemd):

```bash
sudo ./scripts/deploy.sh
# follow the printed instructions:
sudo nano /opt/glitz-quant/.env
sudo -u glitz bash -lc 'cd /opt/glitz-quant && uv sync && cargo build --release'
sudo systemctl enable --now glitz-quant.service
sudo systemctl enable --now glitz-quant-api.service
sudo systemctl enable --now glitz-quant-dashboard.service
sudo journalctl -u glitz-quant -f
```

---

## Operating modes

`GLITZ_MODE` in `.env`:

| Mode      | Data         | Orders                  |
|-----------|--------------|-------------------------|
| `backtest`| Historical   | Simulated               |
| `paper`   | Live         | Simulated (fill model)  |
| `live`    | Live         | Real (gated, caps)      |

To open the live gate (don't do this until you've run paper for weeks):

1. In `.env`: `LIVE_TRADING_ENABLED=true`, `LIVE_TRADING_MAX_NOTIONAL_USD=100`, set `LIVE_TRADING_CONFIRM_PHRASE` to your random phrase.
2. In `config/risk.yaml`: `live_trading_enabled: true`, same `live_trading_confirm_phrase`.
3. Run: `uv run glitz --i-understand-the-risks`

If any one is missing, the OMS only routes to the paper adapter.

---

## Kill switch

```bash
./scripts/kill.sh
# or via API:
curl -X POST http://localhost:8000/kill -H "X-API-Key: $API_SECRET_KEY"
```

---

## Adding a strategy

1. Create `src/glitz_quant/strategies/your_strategy.py`, subclass `Strategy`, implement `on_candle(ctx) -> Signal | None`.
2. Add an entry under `strategies:` in `config/strategies.yaml`. Set `enabled: true` only after backtesting.
3. The orchestrator will pick it up on next start.

The `BitcoinRangeMomentum` strategy in `src/glitz_quant/strategies/bitcoin_range.py` is the worked example.

---

## What's still stubbed / not done

- **Hyperliquid + Jupiter (DEX) adapters** — venue enums exist, but adapters aren't built. CEX-only for now.
- **Equity calculation in orchestrator** — uses a placeholder $10k. Wire to real balances before any live mode use.
- **Walk-forward analyzer** — backtester is single-pass.
- **Cloudflare Email Routing setup** — provide your domain and I'll write the DNS records + alias config.
- **Grafana dashboards** — Prometheus is wired up but JSON dashboards aren't authored yet.
- **Multi-agent debate (CAMEL)** — `camel-ai` is in deps; integration code is a separate iteration.
- **OpenHands integration** — not implemented.
- **Vibe-Trading repo integration** — I don't have the link. The agent layer is built independently using your stated providers.
