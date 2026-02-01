# FSC Subject Recommendation System

AI-powered recommendation system for FSC (Faculty of Science) subject selection based on academic performance, aptitude tests, personality assessment, and interests.

## Quick Start

### Setup

```bash
# Install system packages (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data/synthetic models/trained
```

### Training

```bash
# Activate virtual environment
source venv/bin/activate

# Train with career guidance features (recommended)
python train.py --generate --samples 10000

# Train without career guidance (original only)
python train.py --generate --samples 10000 --no-career-guidance

# Interactive training menu
python train_simple.py
```

### Usage

```python
from src.predictor import FSCPredictor
from src.data_models import StudentProfile, AcademicPerformance, ...

# Create student profile
profile = StudentProfile(
    academic=AcademicPerformance(...),
    aptitude=AptitudeScores(...),
    personality=PersonalityType.INTP,
    interests=InterestScores(...)
)

# Get recommendation
predictor = FSCPredictor(
    model_path='models/trained/best_model.pkl',
    scaler_path='models/trained/scaler.pkl',
    use_career_guidance=True  # Includes career guidance Q&A
)

result = predictor.recommend(profile)
print(f"Recommended: {result['ml_prediction']}")
```

## Features

- **Multi-criteria Decision**: Academic (40%), Aptitude (35%), Personality (15%), Interests (10%)
- **Enhanced Training**: 59 features (42 original + 17 career guidance features)
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM
- **Career Guidance**: Integrated Q&A dataset for career advice
- **Rule-based Baseline**: No training needed, fully interpretable

## Project Structure

```
.
├── src/                    # Core modules
│   ├── data_models.py     # Data structures
│   ├── scoring.py         # Rule-based algorithm
│   ├── feature_engineer.py # Feature extraction (59 features)
│   ├── career_guidance_features.py # Career guidance features
│   ├── model_trainer.py   # ML training
│   └── predictor.py       # Prediction interface
├── scripts/
│   └── generate_data.py   # Synthetic data generator
├── train.py               # Main training script
├── evaluate.py            # Model evaluation
└── example_usage.py       # Usage examples
```

## Training Options

| Option | Features | Command |
|--------|----------|---------|
| **With Career Guidance** (Recommended) | 59 features | `python train.py --generate --samples 10000` |
| **Without Career Guidance** | 42 features | `python train.py --generate --samples 10000 --no-career-guidance` |
| **Quick Test** | 59 features | `python train.py --generate --samples 1000` |

## Data Requirements

- Matriculation marks: Math, Biology, Physics, Chemistry, Computer
- Aptitude scores: Mathematical, Scientific, Verbal, Logical, Spatial
- Personality: MBTI type (INTP, INTJ, etc.)
- Interests: Medical, Engineering, Computer, Research, Arts

## Model Performance

- **Expected Accuracy**: 91-93% (with career guidance features)
- **Best Models**: XGBoost, LightGBM
- **Training Time**: 15-30 minutes (10,000 samples)

## Troubleshooting

**Virtual environment not active?**
```bash
source venv/bin/activate
```

**Career guidance not loading?**
```bash
pip install datasets transformers
```

**Large files in Git?**
```bash
git rm --cached models/trained/*.pkl
git commit -m "Remove model files"
```

## Examples

```bash
# Test rule-based system (no training needed)
python example_usage.py

# Train model
python train.py --generate --samples 10000

# Evaluate model
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl
```

---

**For detailed information, see code comments and docstrings.**
