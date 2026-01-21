# FSC Subject Recommendation System - AI Model Training

This project implements an AI-powered recommendation system for FSC (Faculty of Science) subject selection based on academic performance, aptitude tests, personality assessment, and interests.

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw input data
│   ├── processed/        # Processed features
│   └── synthetic/        # Synthetic training data
├── models/
│   ├── trained/          # Saved trained models
│   └── checkpoints/      # Training checkpoints
├── src/
│   ├── data_models.py    # Data schemas and structures
│   ├── feature_engineer.py  # Feature engineering
│   ├── scoring.py        # Rule-based scoring system
│   ├── model_trainer.py  # ML model training
│   ├── evaluator.py      # Model evaluation
│   └── predictor.py      # Prediction interface
├── notebooks/
│   └── exploration.ipynb # Data exploration and analysis
├── config/
│   └── model_config.yaml # Model configuration
└── train.py              # Main training script
```

## Features

- **Multi-criteria Decision Making**: Combines academic performance (40%), aptitude (35%), personality (15%), and interests (10%)
- **Multiple ML Approaches**: Classification, Regression, and Ensemble methods
- **Rule-based Baseline**: Implements the documented scoring algorithm
- **Comprehensive Evaluation**: Multiple metrics and validation strategies

## Quick Start

### Setup Virtual Environment & Install Dependencies

**Option 1: Automated Setup (Recommended)**
```bash
chmod +x setup_and_train.sh
./setup_and_train.sh
```

**Option 2: Manual Setup**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data/synthetic models/trained
```

### Training

1. **Activate virtual environment** (if not already active):
```bash
source venv/bin/activate
```

2. **Train the model**:
```bash
# Standard training (10,000 samples)
python train.py --generate --samples 10000

# OR use interactive menu
python train_simple.py

# OR quick test (1,000 samples)
python train.py --generate --samples 1000
```

3. **Evaluate the model**:
```bash
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl --generate-test
```

4. **Test the system** (no training needed):
```bash
python example_usage.py
```

## Model Approaches

1. **Rule-based System**: Implements the exact algorithm from documentation (no training needed)
2. **Multi-class Classification**: Predicts stream directly (requires training)
3. **Regression + Ranking**: Predicts scores for each stream, then ranks (requires training)
4. **Hybrid System**: Combines rule-based and ML for best results (recommended)

## Documentation

- **QUICK_START.md**: Get started in 5 minutes
- **AI_TRAINING_APPROACHES.md**: Detailed comparison of all training approaches
- **TRAINING_GUIDE.md**: Comprehensive training instructions
- **example_usage.py**: Code examples

## Data Requirements

- Matriculation marks (Math, Biology, Physics, Chemistry, Computer)
- Aptitude test scores (5 categories)
- Personality type (MBTI)
- Interest assessments (5 categories)

## Documentation

- **SETUP_COMMANDS.md**: Complete setup instructions with virtual environment
- **QUICK_SETUP.txt**: Quick reference for all commands
- **SIMPLE_TRAINING_GUIDE.md**: Simple training guide
- **TRAINING_SUGGESTIONS.md**: Detailed training options
- **AI_TRAINING_APPROACHES.md**: Comparison of training approaches
