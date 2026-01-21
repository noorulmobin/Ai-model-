# Quick Start Guide - FSC Recommendation System

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Training (5 minutes)

### Step 1: Generate Training Data
```bash
python scripts/generate_data.py --samples 10000 --output data/synthetic/training_data.csv
```

### Step 2: Train the Model
```bash
python train.py --generate --samples 10000
```

This will:
- Generate 10,000 synthetic student profiles
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost, etc.)
- Select the best model
- Save to `models/trained/`

### Step 3: Test the Model
```bash
python example_usage.py
```

## Using the System

### Option 1: Rule-Based (No Training Required)

```python
from src.data_models import StudentProfile, AcademicPerformance, AptitudeScores, InterestScores, PersonalityType
from src.scoring import RuleBasedScorer

# Create student profile
profile = StudentProfile(
    academic=AcademicPerformance(
        mathematics=82, biology=76, physics=71, chemistry=74, computer=79, aggregate=76.4
    ),
    aptitude=AptitudeScores(
        mathematical_ability=85, scientific_aptitude=78, verbal_ability=72,
        logical_reasoning=81, spatial_ability=68
    ),
    personality=PersonalityType.INTP,
    interests=InterestScores(
        medicine_healthcare=30, engineering_technology=75,
        computers_programming=90, research_science=60, creative_arts=40
    )
)

# Get recommendation
scorer = RuleBasedScorer()
result = scorer.recommend(profile)

print(f"Recommended: {result.top_recommendation.stream.value}")
print(f"Match: {result.top_recommendation.match_percentage:.1f}%")
```

### Option 2: ML Model (After Training)

```python
from src.predictor import FSCPredictor

# Load trained model
predictor = FSCPredictor(
    model_path='models/trained/best_model.pkl',
    scaler_path='models/trained/scaler.pkl'
)

# Get recommendation
recommendation = predictor.recommend(profile, use_ml=True)
print(f"ML Prediction: {recommendation['ml_prediction']}")
```

## Model Approaches Summary

| Approach | Training Time | Accuracy | Use Case |
|----------|--------------|----------|----------|
| **Rule-Based** | None | Baseline | Production, Interpretable |
| **Logistic Regression** | Fast | Good | Quick baseline |
| **Random Forest** | Medium | Very Good | Balanced performance |
| **XGBoost** | Medium | Excellent | Best accuracy |
| **LightGBM** | Fast | Excellent | Fast + Accurate |

## File Structure

```
.
├── src/                    # Core modules
│   ├── data_models.py     # Data structures
│   ├── scoring.py         # Rule-based algorithm
│   ├── feature_engineer.py # Feature extraction
│   ├── model_trainer.py   # ML training
│   └── predictor.py       # Prediction interface
├── scripts/
│   └── generate_data.py   # Synthetic data generator
├── config/
│   └── model_config.yaml  # Configuration
├── train.py               # Main training script
├── evaluate.py            # Evaluation script
└── example_usage.py       # Usage examples
```

## Next Steps

1. **Read TRAINING_GUIDE.md** for detailed training instructions
2. **Collect real data** to replace synthetic data
3. **Customize scoring** in `src/scoring.py` if needed
4. **Deploy** using the predictor interface

## Troubleshooting

**Import errors?**
- Make sure you're in the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`

**Model not found?**
- Train the model first: `python train.py --generate`

**Low accuracy?**
- Increase training samples: `--samples 20000`
- Check data quality
- Try different models (see TRAINING_GUIDE.md)
