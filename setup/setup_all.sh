#!/bin/bash
# Setup all environments for experiment execution
# Run with: bash setup/setup_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "Setting up experiment execution environments"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

check_command() {
    if command -v "$1" &> /dev/null; then
        echo "  $1: $(command -v $1)"
        return 0
    else
        echo "  $1: NOT FOUND"
        return 1
    fi
}

MISSING=0
check_command "R" || MISSING=1
check_command "Rscript" || MISSING=1
check_command "python3" || MISSING=1
check_command "julia" || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Warning: Some prerequisites are missing."
    echo "Install missing tools before running experiments."
    echo ""
fi

echo ""

# R setup
echo "=========================================="
echo "Setting up R environment"
echo "=========================================="
if command -v Rscript &> /dev/null; then
    Rscript setup/setup_r.R
else
    echo "Skipping R setup (Rscript not found)"
fi
echo ""

# Python setup
echo "=========================================="
echo "Setting up Python environment"
echo "=========================================="
if command -v python3 &> /dev/null; then
    bash setup/setup_python.sh
else
    echo "Skipping Python setup (python3 not found)"
fi
echo ""

# Julia setup
echo "=========================================="
echo "Setting up Julia environment"
echo "=========================================="
if command -v julia &> /dev/null; then
    julia setup/setup_julia.jl
else
    echo "Skipping Julia setup (julia not found)"
fi
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
