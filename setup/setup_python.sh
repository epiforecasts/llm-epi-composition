#!/bin/bash
# Setup Python environment for experiment execution
# Run with: bash setup/setup_python.sh

set -e

echo "Setting up Python environment..."

ENV_NAME="pymc"

# Check for conda/mamba first (preferred)
if command -v mamba &> /dev/null; then
    echo "Using mamba..."
    if mamba env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '$ENV_NAME' already exists"
    else
        echo "Creating conda environment..."
        mamba create -n "$ENV_NAME" python=3.11 pymc arviz pandas numpy matplotlib scipy scikit-learn -y
    fi
    echo ""
    echo "Python environment setup complete!"
    echo "To activate: conda activate $ENV_NAME"
    exit 0
fi

if command -v conda &> /dev/null; then
    echo "Using conda..."
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment '$ENV_NAME' already exists"
    else
        echo "Creating conda environment..."
        conda create -n "$ENV_NAME" python=3.11 pymc arviz pandas numpy matplotlib scipy scikit-learn -y
    fi
    echo ""
    echo "Python environment setup complete!"
    echo "To activate: conda activate $ENV_NAME"
    exit 0
fi

# Fall back to venv
echo "No conda found, using venv..."

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.9+ or conda/mamba."
    exit 1
fi

VENV_DIR="venv_pymc"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing Python packages..."
pip install --upgrade pip
pip install pymc arviz numpy pandas matplotlib seaborn scipy scikit-learn

echo ""
echo "Python environment setup complete!"
echo "Virtual environment: $VENV_DIR"
echo "To activate: source $VENV_DIR/bin/activate"
