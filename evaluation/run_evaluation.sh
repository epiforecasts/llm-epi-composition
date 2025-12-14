#!/bin/bash
# Run evaluation of all experiments
# Usage: bash evaluation/run_evaluation.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "LLM Epi Composition - Code Evaluation"
echo "=========================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Start time: $(date)"
echo ""

# Check prerequisites
echo "Checking prerequisites..."
command -v Rscript >/dev/null 2>&1 || { echo "Rscript not found"; exit 1; }
echo "  Rscript: OK"

# Check for Python venv
if [ -d "venv_pymc" ]; then
    echo "  Python venv: OK"
else
    echo "  Python venv: NOT FOUND (PyMC experiments may fail)"
fi

# Check for Julia
if command -v julia >/dev/null 2>&1; then
    echo "  Julia: OK"
else
    echo "  Julia: NOT FOUND (Turing/EpiAware experiments may fail)"
fi

echo ""

# Run evaluation
echo "Starting evaluation..."
echo ""

Rscript evaluation/run_evaluation.R

echo ""
echo "End time: $(date)"
echo "Results saved to: evaluation/results/"
