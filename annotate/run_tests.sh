#!/usr/bin/env bash
set -euo pipefail

# Set up a Python virtual environment and run the end-to-end detection test.
# Assumes Docker services are already running (see setup.sh).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="test_venv"
REQUIREMENTS="test_requirements.txt"

# Check that the orchestrator is reachable
echo "=== Checking orchestrator health ==="
if ! curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "ERROR: Orchestrator not reachable at http://localhost:8080"
    echo "Run './setup.sh' first to start the services."
    exit 1
fi
echo "  Orchestrator is healthy."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "=== Creating virtual environment ==="
    python3 -m venv "$VENV_DIR"
fi

# Activate and install dependencies
echo ""
echo "=== Installing test dependencies ==="
source "$VENV_DIR/bin/activate"
pip install -q -r "$REQUIREMENTS"

# Run the test
echo ""
echo "=== Running detection pipeline test ==="
python test_pipeline.py

echo ""
echo "=== Test complete ==="
echo "Annotated images and JSON results are in test_outputs/"
echo "Open the .jpg files to see bounding box visualizations."
