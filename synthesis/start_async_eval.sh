#!/bin/bash
# Start async quality evaluation in background

echo "Starting async quality evaluation..."
echo "Log file: ./augmented-output/evaluation.log"
echo ""

# Activate virtualenv and run in background
source env/bin/activate

# Run with watch mode (continuously monitor for new images)
# OOD detection enabled (FORTE_ENABLED=true in .env)
nohup python run_evaluation_async.py \
    --output-dir ./augmented-output \
    --watch \
    --interval 30 \
    > ./augmented-output/evaluation_stdout.log 2>&1 &

EVAL_PID=$!

echo "Evaluation running in background (PID: $EVAL_PID)"
echo ""
echo "To monitor progress:"
echo "  tail -f ./augmented-output/evaluation.log"
echo ""
echo "To stop evaluation:"
echo "  kill $EVAL_PID"
echo ""
echo "Results will be saved to:"
echo "  ./augmented-output/evaluation_results.json"
