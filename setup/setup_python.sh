#!/bin/bash
# Setup Python environment for experiment execution
# Run with: bash setup/setup_python.sh

set -e

echo "Setting up Python environment..."

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.9+."
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="venv_pymc"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Installing Python packages..."

# Upgrade pip
pip install --upgrade pip

# Install PyMC and dependencies
pip install pymc arviz numpy pandas matplotlib seaborn

# Additional packages that LLMs might use
pip install scipy scikit-learn

echo ""
echo "Python environment setup complete!"
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate: source $VENV_DIR/bin/activate"
echo ""
echo "Installed packages:"
pip list | grep -E "pymc|arviz|numpy|pandas|scipy"
