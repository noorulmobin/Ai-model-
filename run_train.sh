#!/bin/bash
# Training script that ensures virtual environment is active

cd /home/mubeen/ai

# Activate virtual environment
source venv/bin/activate

# Check if venv is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Virtual environment not activated"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Use python from venv (python3 if python doesn't exist)
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="python"
elif [ -f "venv/bin/python3" ]; then
    PYTHON_CMD="python3"
else
    echo "❌ Error: Python not found in virtual environment"
    exit 1
fi

# Run training with provided arguments or default
if [ $# -eq 0 ]; then
    echo "Running training with default parameters (10,000 samples)..."
    $PYTHON_CMD train.py --generate --samples 10000
else
    echo "Running training with provided arguments..."
    $PYTHON_CMD train.py "$@"
fi
