#!/bin/bash
# Setup virtual environment and train FSC Recommendation Model

echo "=========================================="
echo "FSC Recommendation System - Setup & Train"
echo "=========================================="

# Step 1: Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv

# Step 2: Activate virtual environment
echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo ""
echo "Step 3: Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo ""
echo "Step 4: Installing dependencies..."
pip install -r requirements.txt

# Step 5: Create necessary directories
echo ""
echo "Step 5: Creating directories..."
mkdir -p data/synthetic
mkdir -p models/trained
mkdir -p models/backup

# Step 6: Train the model
echo ""
echo "=========================================="
echo "Step 6: Starting model training..."
echo "=========================================="
python train.py --generate --samples 10000

echo ""
echo "=========================================="
echo "Setup and training complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the trained model, run:"
echo "  python example_usage.py"
echo ""
