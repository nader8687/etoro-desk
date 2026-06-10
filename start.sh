#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# EtoroDesk  —  start.sh
# Stops every running container, then rebuilds and starts the EtoroDesk stack.
#
# Usage:
#   ./start.sh           — full rebuild (default)
#   ./start.sh --no-build — skip rebuild, just restart from existing images
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# ── Parse args ────────────────────────────────────────────────────────────────
BUILD_FLAG="--build"
for arg in "$@"; do
  [[ "$arg" == "--no-build" ]] && BUILD_FLAG=""
done

# ─────────────────────────────────────────────────────────────────────────────
header "━━  EtoroDesk Launcher  ━━"

# ── Step 1: Stop ALL running containers ──────────────────────────────────────
header "1 / 3  ·  Stopping all running containers"

RUNNING=$(docker ps -q)
if [[ -n "$RUNNING" ]]; then
  COUNT=$(echo "$RUNNING" | wc -l | tr -d ' ')
  info "Found ${COUNT} running container(s) — stopping…"
  # List names before stopping so the output is readable
  docker ps --format "   • {{.Names}} ({{.Image}})"
  docker stop $RUNNING
  success "All containers stopped."
else
  info "No running containers."
fi

# ── Step 2: Remove stopped EtoroDesk containers (clean slate) ────────────────
header "2 / 3  ·  Removing old EtoroDesk containers"

cd "$SCRIPT_DIR"
docker compose rm -f 2>/dev/null || true
success "Old containers removed."

# ── Step 3: Build + start EtoroDesk stack ────────────────────────────────────
header "3 / 3  ·  Starting EtoroDesk"

if [[ -n "$BUILD_FLAG" ]]; then
  info "Building images and starting containers…"
else
  info "Starting containers from existing images (--no-build)…"
fi

docker compose up $BUILD_FLAG -d

# ── Health check ──────────────────────────────────────────────────────────────
header "Waiting for services to become healthy…"

MAX_WAIT=60
ELAPSED=0
INTERVAL=3

while true; do
  STATUS_DASH=$(docker inspect --format='{{.State.Health.Status}}' etoro-dashboard 2>/dev/null || echo "missing")
  STATUS_BOT=$(docker inspect --format='{{.State.Health.Status}}' visual-bot 2>/dev/null || echo "missing")

  if [[ "$STATUS_DASH" == "healthy" && "$STATUS_BOT" == "healthy" ]]; then
    break
  fi

  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    warn "Health check timed out after ${MAX_WAIT}s (dashboard=${STATUS_DASH}, bot=${STATUS_BOT})."
    warn "Run  docker compose logs  to debug."
    break
  fi

  echo -e "   ${YELLOW}⏳${RESET}  dashboard=${STATUS_DASH}  visual-bot=${STATUS_BOT}  (${ELAPSED}s / ${MAX_WAIT}s)"
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
done

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
docker compose ps
echo ""
success "EtoroDesk is running:"
echo -e "   📊  Dashboard  →  ${BOLD}http://localhost:8501${RESET}"
echo -e "   🤖  Visual Bot →  ${BOLD}http://localhost:8083${RESET}  (docs: /docs)"
echo ""
