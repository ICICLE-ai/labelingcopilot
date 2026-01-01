#!/usr/bin/env bash
set -euo pipefail

# Build and start the annotation pipeline Docker services.
# Requires Docker (with at least 10GB memory allocated).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building base image ==="
docker build \
    --build-arg CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}" \
    --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST-7.0;7.5;8.0;8.6;8.9;9.0+PTX}" \
    --build-arg PYTORCH_VERSION="${PYTORCH_VERSION-2.5.1}" \
    --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION-0.20.1}" \
    --build-arg PYTORCH_PACKAGES="${PYTORCH_PACKAGES-torch==2.5.1 torchvision==0.20.1}" \
    --build-arg PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL-https://download.pytorch.org/whl/cu121}" \
    --build-arg INSTALL_TORCH="${INSTALL_TORCH-1}" \
    -t annotation-base \
    -f models/base/Dockerfile .

echo ""
echo "=== Building and starting all services ==="
docker compose up -d --build

echo ""
echo "=== Waiting for services to become healthy ==="
MAX_WAIT=600
ELAPSED=0
INTERVAL=10

while [ $ELAPSED -lt $MAX_WAIT ]; do
    HEALTHY=$(docker compose ps --format json 2>/dev/null | python3 -c "
import sys, json
lines = sys.stdin.read().strip().split('\n')
healthy = 0
total = 0
for line in lines:
    if not line:
        continue
    svc = json.loads(line)
    name = svc.get('Name', svc.get('name', ''))
    health = svc.get('Health', svc.get('health', ''))
    state = svc.get('State', svc.get('state', ''))
    # Orchestrator has no healthcheck, just needs to be running
    if 'orchestrator' in name:
        if state == 'running':
            healthy += 1
        total += 1
    else:
        total += 1
        if health == 'healthy':
            healthy += 1
print(f'{healthy}/{total}')
" 2>/dev/null || echo "0/0")

    echo "  Services healthy: $HEALTHY (waited ${ELAPSED}s / ${MAX_WAIT}s)"

    if echo "$HEALTHY" | grep -qE "^([0-9]+)/\1$"; then
        echo ""
        echo "=== All services healthy! ==="
        docker compose ps
        echo ""
        echo "Orchestrator is available at http://localhost:8080"
        echo "Run './run_tests.sh' to test the pipeline."
        exit 0
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "=== Timed out waiting for services (${MAX_WAIT}s) ==="
echo "Check logs with: docker compose logs"
docker compose ps
exit 1
