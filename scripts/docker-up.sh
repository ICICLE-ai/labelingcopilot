#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_AGENT=0
WITH_SYNTHESIS=0
SKIP_GPU_CHECK=0
RUNTIME_PROFILE=""
GPU_CHECK_IMAGE=""

usage() {
    cat <<'EOF'
Usage: ./scripts/docker-up.sh [options]

Starts the retrieval and annotation stacks, and optionally runs the top-level
agent demo container and the synthesis API service.

Options:
  --run-agent       Run the top-level agent container after services are healthy
  --with-synthesis  Start the synthesis API service as well
  --skip-gpu-check  Skip the NVIDIA Docker validation check
  -h, --help        Show this help text
EOF
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

is_set() {
    local var_name="$1"
    [[ "${!var_name+x}" == "x" ]]
}

detect_host_arch() {
    uname -m
}

detect_gpu_name() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

detect_runtime_profile() {
    local arch
    local gpu_name

    arch="$(detect_host_arch)"
    gpu_name="$(detect_gpu_name || true)"

    if is_set CUDA_BASE_IMAGE || is_set TORCH_CUDA_ARCH_LIST || is_set PYTORCH_PACKAGES || \
       is_set PYTORCH_INDEX_URL || is_set INSTALL_TORCH; then
        RUNTIME_PROFILE="manual"
        if [[ "${CUDA_BASE_IMAGE:-}" == nvidia/cuda:* ]]; then
            GPU_CHECK_IMAGE="${CUDA_BASE_IMAGE}"
        else
            GPU_CHECK_IMAGE="${GPU_CHECK_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
        fi
        return 0
    fi

    if [[ "$arch" == "aarch64" ]] || [[ "$gpu_name" == *"GB10"* ]] || [[ "$gpu_name" == *"Blackwell"* ]]; then
        RUNTIME_PROFILE="arm-blackwell"
        export CUDA_BASE_IMAGE="nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04"
        export PYTORCH_PACKAGES="torch torchvision"
        export PYTORCH_INDEX_URL=""
        export INSTALL_TORCH="1"
        export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;10.1;10.3;12.0;12.1+PTX"
        GPU_CHECK_IMAGE="nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04"
        return 0
    fi

    RUNTIME_PROFILE="default-x86"
    export CUDA_BASE_IMAGE="nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"
    export PYTORCH_PACKAGES="torch==2.5.1 torchvision==0.20.1"
    export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    export INSTALL_TORCH="1"
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    GPU_CHECK_IMAGE="nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"
}

print_runtime_profile() {
    local arch
    local gpu_name

    arch="$(detect_host_arch)"
    gpu_name="$(detect_gpu_name || true)"

    echo "Runtime profile: ${RUNTIME_PROFILE}"
    echo "  Host arch: ${arch}"
    if [[ -n "$gpu_name" ]]; then
        echo "  GPU: ${gpu_name}"
    fi
    echo "  CUDA_BASE_IMAGE: ${CUDA_BASE_IMAGE:-<unset>}"
    echo "  PYTORCH_PACKAGES: ${PYTORCH_PACKAGES:-<unset>}"
    echo "  PYTORCH_INDEX_URL: ${PYTORCH_INDEX_URL:-<default pip index>}"
    echo "  INSTALL_TORCH: ${INSTALL_TORCH:-<unset>}"
    echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-<unset>}"
}

wait_for_http() {
    local url="$1"
    local name="$2"
    local timeout="${3:-600}"
    local start_ts
    start_ts="$(date +%s)"

    echo "Waiting for $name at $url"
    while true; do
        if python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)" "$url" \
            >/dev/null 2>&1; then
            echo "$name is ready"
            return 0
        fi

        if (( "$(date +%s)" - start_ts >= timeout )); then
            echo "Timed out waiting for $name at $url" >&2
            return 1
        fi

        sleep 5
    done
}

check_required_paths() {
    local missing=0

    for path in \
        "$ROOT_DIR/annotate/docker-compose.yml" \
        "$ROOT_DIR/retrieval/docker-compose.yml" \
        "$ROOT_DIR/agent_demo.py"; do
        if [[ ! -e "$path" ]]; then
            echo "Missing required path: $path" >&2
            missing=1
        fi
    done

    if (( missing )); then
        echo "The repo looks incomplete — re-clone or check missing files above." >&2
        exit 1
    fi
}

check_gpu() {
    echo "Checking NVIDIA Docker access"
    docker run --rm --gpus all "${GPU_CHECK_IMAGE}" nvidia-smi
}

start_annotation() {
    echo "Starting annotation stack"
    (cd "$ROOT_DIR/annotate" && bash ./setup.sh)
}

start_retrieval() {
    echo "Starting retrieval stack"
    (cd "$ROOT_DIR/retrieval" && docker compose up -d --build)
    wait_for_http "http://localhost:8000/health" "retrieval API" 900
}

start_synthesis() {
    local env_file="$ROOT_DIR/synthesis/.env"

    if [[ ! -f "$env_file" ]]; then
        echo "Missing $env_file" >&2
        echo "Create it first with:" >&2
        echo "  cp synthesis/.env.example synthesis/.env" >&2
        exit 1
    fi

    mkdir -p \
        "$ROOT_DIR/synthesis/input-images" \
        "$ROOT_DIR/synthesis/augmented-output"

    echo "Starting synthesis API"
    (cd "$ROOT_DIR/synthesis" && docker compose up -d --build)
    wait_for_http "http://localhost:8090/health" "synthesis API" 900
}

run_agent_container() {
    echo "Running top-level agent container"
    (cd "$ROOT_DIR" && docker compose -f docker-compose.agent.yml run --rm --build agent)
}

while (($#)); do
    case "$1" in
        --run-agent)
            RUN_AGENT=1
            ;;
        --with-synthesis)
            WITH_SYNTHESIS=1
            ;;
        --skip-gpu-check)
            SKIP_GPU_CHECK=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

require_cmd docker
require_cmd python3
check_required_paths
detect_runtime_profile
print_runtime_profile

if (( ! SKIP_GPU_CHECK )); then
    check_gpu
fi

start_annotation
start_retrieval

if (( WITH_SYNTHESIS )); then
    start_synthesis
fi

if (( RUN_AGENT )); then
    run_agent_container
else
    cat <<EOF
Services are up.

Annotation API: http://localhost:8080
Retrieval API:  http://localhost:8000
$(if (( WITH_SYNTHESIS )); then echo "Synthesis API:  http://localhost:8090"; fi)

Next steps:
  Run the top-level agent:
    docker compose -f docker-compose.agent.yml run --rm --build agent

$(if (( ! WITH_SYNTHESIS )); then cat <<'INNER'
  Start synthesis too:
    ./scripts/docker-up.sh --with-synthesis

INNER
fi)
  Stop everything later:
    ./scripts/docker-down.sh
EOF
fi
