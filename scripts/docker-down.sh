#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Stopping annotation stack"
(cd "$ROOT_DIR/annotate" && docker compose down)

echo "Stopping retrieval stack"
(cd "$ROOT_DIR/retrieval" && docker compose down)

if [[ -f "$ROOT_DIR/synthesis/docker-compose.yml" ]] && \
   [[ -f "$ROOT_DIR/synthesis/.env" ]]; then
    echo "Stopping synthesis stack"
    (cd "$ROOT_DIR/synthesis" && docker compose down)
fi
