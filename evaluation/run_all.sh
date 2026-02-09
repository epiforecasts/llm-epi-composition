#!/bin/bash
# Run all agentic evaluations
# Usage: ./run_all.sh [model]
# If model is not specified, uses claude-sonnet-4-20250514

set -e

MODEL=${1:-claude-sonnet-4-20250514}
RUNS_PER_CONDITION=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENARIOS="scenario_1a scenario_1b scenario_2 scenario_3"
CONDITIONS="r python julia epiaware"

echo "=========================================="
echo "Running all agentic evaluations"
echo "=========================================="
echo "Model: $MODEL"
echo "Runs per condition: $RUNS_PER_CONDITION"
echo ""

for scenario in $SCENARIOS; do
    for condition in $CONDITIONS; do
        for run in $(seq 1 $RUNS_PER_CONDITION); do
            echo ""
            echo ">>> $scenario / $condition / run $run"
            echo ""

            "$SCRIPT_DIR/run_agentic.sh" "$scenario" "$condition" "$MODEL" "$run" || {
                echo "WARNING: Run failed for $scenario/$condition/run_$run"
            }

            echo ""
            echo "---"
        done
    done
done

echo ""
echo "=========================================="
echo "All runs complete"
echo "=========================================="
