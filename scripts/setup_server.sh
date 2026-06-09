#!/usr/bin/env bash
# ============================================================
# setup_server.sh — one-shot bootstrap for fresh Ubuntu 22.04/24.04.
# ============================================================
# Run as a non-root user with sudo. Idempotent — safe to re-run.
# ============================================================

set -euo pipefail

log() { echo "[$(date -Iseconds)] $*"; }

require_sudo() {
  if ! sudo -n true 2>/dev/null; then
    log "this script needs sudo. you'll be prompted."
    sudo -v
  fi
}

require_sudo

log "updating apt..."
sudo apt-get update -y

log "installing system packages..."
sudo apt-get install -y \
  build-essential pkg-config libssl-dev \
  git curl ca-certificates gnupg lsb-release \
  python3-dev python3-venv \
  postgresql-client redis-tools \
  jq htop tmux ufw fail2ban unattended-upgrades

log "configuring unattended upgrades..."
sudo dpkg-reconfigure -plow unattended-upgrades || true

log "configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 8000/tcp comment "glitz api"
sudo ufw allow 3001/tcp comment "grafana"
sudo ufw --force enable

if ! command -v docker >/dev/null 2>&1; then
  log "installing docker..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  log "added $USER to docker group. log out + back in for it to take effect."
fi

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v cargo >/dev/null 2>&1; then
  log "installing rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

log "creating runtime dirs..."
sudo mkdir -p /var/run/glitz-quant /var/log/glitz-quant
sudo chown "$USER:$USER" /var/run/glitz-quant /var/log/glitz-quant

log "done. next steps:"
echo "  1. log out + back in (for docker group)"
echo "  2. cd into the repo"
echo "  3. cp .env.example .env  &&  edit .env"
echo "  4. docker compose up -d"
echo "  5. uv sync"
echo "  6. cargo build --release"
echo "  7. uv run python -m glitz_quant.scripts.healthcheck"
