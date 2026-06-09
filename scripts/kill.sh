#!/usr/bin/env bash
# ============================================================
# kill.sh — global kill switch.
# ============================================================
# Cancels all open orders, flattens positions, halts orchestrator.
# Anyone with SSH access can hit this.
#
# Usage:
#   ./scripts/kill.sh                  # interactive confirm
#   ./scripts/kill.sh --no-confirm     # skip confirm (for automation)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIRM=true
if [[ "${1:-}" == "--no-confirm" ]]; then
  CONFIRM=false
fi

if [[ "$CONFIRM" == "true" ]]; then
  echo "================================================"
  echo "  GLITZ-QUANT GLOBAL KILL SWITCH"
  echo "================================================"
  echo "This will:"
  echo "  1. Touch the halt file (orchestrator stops accepting new signals)"
  echo "  2. Cancel every open order across every venue"
  echo "  3. Flatten every position (market-out to USDC/USDT)"
  echo "  4. Write incident record to Supabase"
  echo ""
  read -p "Type 'KILL' to proceed: " confirm
  if [[ "$confirm" != "KILL" ]]; then
    echo "aborted."
    exit 1
  fi
fi

HALT_FILE="${KILL_SWITCH_PATH:-/var/run/glitz-quant/HALT}"
mkdir -p "$(dirname "$HALT_FILE")"
touch "$HALT_FILE"
echo "[$(date -Iseconds)] halt file touched: $HALT_FILE"

# Execute the Python kill routine — implemented in Phase 3
if command -v uv >/dev/null 2>&1; then
  uv run python -m glitz_quant.scripts.kill_all
else
  python -m glitz_quant.scripts.kill_all
fi

echo "[$(date -Iseconds)] kill sequence complete."
