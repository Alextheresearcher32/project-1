#!/usr/bin/env bash
# ============================================================
# deploy.sh — install glitz-quant on the server.
# ============================================================
# Run AFTER setup_server.sh.
#
# Steps:
#   1. Create /opt/glitz-quant and /var/log/glitz-quant and /var/run/glitz-quant
#   2. Copy repo contents to /opt/glitz-quant
#   3. Install systemd unit files
#   4. (Manual step printed) Edit .env, then enable services
#
# Re-runnable — won't clobber your .env.
# ============================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR=/opt/glitz-quant
LOG_DIR=/var/log/glitz-quant
RUN_DIR=/var/run/glitz-quant

log() { echo "[$(date -Iseconds)] $*"; }

if [[ $EUID -ne 0 ]]; then
  log "this script needs root. re-running with sudo."
  exec sudo bash "$0" "$@"
fi

# 1. user + dirs
if ! id glitz >/dev/null 2>&1; then
  log "creating user 'glitz'..."
  useradd --system --create-home --shell /bin/bash glitz
fi

mkdir -p "$INSTALL_DIR" "$LOG_DIR" "$RUN_DIR"
chown -R glitz:glitz "$INSTALL_DIR" "$LOG_DIR" "$RUN_DIR"

# 2. sync code (preserve .env if it exists)
log "syncing code to $INSTALL_DIR..."
rsync -a --delete \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude 'target' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'data/raw/*' \
  --exclude 'data/processed/*' \
  --exclude 'data/backtests/*' \
  --exclude 'data/models/*' \
  "$REPO_DIR/" "$INSTALL_DIR/"

if [[ ! -f "$INSTALL_DIR/.env" ]]; then
  cp "$REPO_DIR/.env.example" "$INSTALL_DIR/.env"
  log "created $INSTALL_DIR/.env from template — edit it before starting services."
fi

chown -R glitz:glitz "$INSTALL_DIR"
chmod 600 "$INSTALL_DIR/.env"

# 3. install systemd units
log "installing systemd units..."
cp "$REPO_DIR/infra/systemd/"glitz-quant*.service /etc/systemd/system/
systemctl daemon-reload

log ""
log "================================================================"
log "deployment staged at $INSTALL_DIR"
log "next steps:"
log "  1. sudo nano $INSTALL_DIR/.env       # fill in API keys"
log "  2. sudo -u glitz bash -lc 'cd $INSTALL_DIR && uv sync && cargo build --release'"
log "  3. sudo -u glitz bash -lc 'cd $INSTALL_DIR && uv run python -m glitz_quant.scripts.healthcheck'"
log "  4. systemctl enable --now glitz-quant.service"
log "  5. systemctl enable --now glitz-quant-api.service"
log "  6. systemctl enable --now glitz-quant-dashboard.service"
log "  7. journalctl -u glitz-quant -f"
log "================================================================"
