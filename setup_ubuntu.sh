#!/bin/bash
# Setup script for Ubuntu/Debian systems
# This script installs prerequisites and sets up the environment

echo "=========================================="
echo "FSC Recommendation System - Ubuntu Setup"
echo "=========================================="

# Check if running as root for system package installation
if [ "$EUID" -eq 0 ]; then 
    SUDO=""
else
    SUDO="sudo"
fi

# Step 1: Install system prerequisites
echo ""
echo "Step 1: Installing system prerequisites..."
echo "This may require your password for sudo access."
$SUDO apt update
$SUDO apt install -y python3 python3-pip python3-venv

# Step 2: Verify Python installation
echo ""
echo "Step 2: Verifying Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Error: Python3 is not installed properly"
    exit 1
fi

# Step 3: Create virtual environment
echo ""
echo "Step 3: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    echo "Please make sure python3-venv is installed: sudo apt install python3-venv"
    exit 1
fi

# Step 4: Activate virtual environment
echo ""
echo "Step 4: Activating virtual environment..."
source venv/bin/activate

# Step 5: Upgrade pip
echo ""
echo "Step 5: Upgrading pip..."
pip install --upgrade pip

# Step 6: Install Python dependencies
echo ""
echo "Step 6: Installing Python dependencies..."
pip install -r requirements.txt

# Step 7: Create necessary directories
echo ""
echo "Step 7: Creating directories..."
mkdir -p data/synthetic
mkdir -p models/trained
mkdir -p models/backup

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Virtual environment is now active."
echo ""
echo "Next steps:"
echo "  1. Train the model: python train.py --generate --samples 10000"
echo "  2. Or use interactive menu: python train_simple.py"
echo ""
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
