# FSC Recommendation System - AI Model Training Guide

## Overview

This guide explains how to train an AI model for the FSC Subject Recommendation System. The system uses machine learning to recommend FSC streams (Pre-Medical, Pre-Engineering, ICS, General Science) based on student profiles.

## Training Approaches

### 1. **Rule-Based System (Baseline)**
- **No training required** - implements the exact algorithm from documentation
- Uses weighted scoring: Academic (40%), Aptitude (35%), Personality (15%), Interest (10%)
- Provides interpretable recommendations with reasoning
- **Best for**: Understanding the logic, baseline comparisons, production use without ML

### 2. **Multi-Class Classification**
- Predicts the recommended stream directly (4 classes)
- Models: Logistic Regression, Random Forest, SVM, XGBoost, LightGBM
- **Best for**: Direct stream prediction, when you have labeled data

### 3. **Regression + Ranking**
- Predicts match scores for each stream, then ranks them
- Models: Ridge, Random Forest, SVR, XGBoost, LightGBM
- **Best for**: Getting confidence scores for all streams

### 4. **Ensemble Methods**
- Combines multiple models for better accuracy
- Uses voting or stacking approaches
- **Best for**: Maximum accuracy, production systems

## Step-by-Step Training Process

### Step 1: Data Preparation

#### Option A: Generate Synthetic Data (Recommended for Testing)
```bash
python scripts/generate_data.py --samples 10000 --output data/synthetic/training_data.csv
```

#### Option B: Use Real Data
1. Prepare CSV file with columns:
   - `math`, `bio`, `physics`, `chem`, `computer`, `aggregate`
   - `apt_math`, `apt_science`, `apt_verbal`, `apt_logical`, `apt_spatial`
   - `int_medical`, `int_engineering`, `int_computer`, `int_research`, `int_arts`
   - `personality` (MBTI type: INTP, INTJ, etc.)

2. Save as `data/synthetic/training_data.csv`

### Step 2: Train the Model

```bash
# Generate data and train
python train.py --generate --samples 10000

# Or use existing data
python train.py --data data/synthetic/training_data.csv
```

**What happens during training:**
1. Data is loaded and converted to student profiles
2. Features are extracted (academic, aptitude, interests, personality)
3. Data is split into train/validation/test sets
4. Multiple models are trained and compared
5. Best model is selected based on validation accuracy
6. Models are saved to `models/trained/`

### Step 3: Evaluate the Model

```bash
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl --generate-test
```

This will:
- Generate test data
- Evaluate model accuracy
- Show classification report
- Generate confusion matrix visualization

### Step 4: Use the Model

```python
from src.predictor import FSCPredictor
from src.data_models import StudentProfile, AcademicPerformance, ...

# Load model
predictor = FSCPredictor(
    model_path='models/trained/best_model.pkl',
    scaler_path='models/trained/scaler.pkl'
)

# Make prediction
recommendation = predictor.recommend(student_profile, use_ml=True)
print(f"Recommended: {recommendation['ml_prediction']}")
```

## Model Selection Strategy

### For Production Use:

1. **Start with Rule-Based System**
   - No training needed
   - Fully interpretable
   - Matches documented algorithm exactly

2. **Train ML Model for Comparison**
   - Use synthetic data initially
   - Compare ML predictions vs rule-based
   - If ML performs better, use it

3. **Collect Real Data**
   - Gather actual student profiles and outcomes
   - Retrain with real data
   - Continuously improve

### Model Comparison:

The training script automatically compares:
- **Logistic Regression**: Fast, interpretable
- **Random Forest**: Good baseline, handles non-linearity
- **SVM**: Good for complex boundaries
- **XGBoost**: Often best accuracy, handles missing data
- **LightGBM**: Fast training, good accuracy

**Selection Criteria:**
- Accuracy on validation set
- Training time
- Inference speed
- Interpretability needs

## Feature Engineering

The system automatically creates features:

### Base Features:
- Academic marks (6): math, bio, physics, chem, computer, aggregate
- Aptitude scores (5): math, science, verbal, logical, spatial
- Interest scores (5): medical, engineering, computer, research, arts
- Personality (16): One-hot encoded MBTI types

### Derived Features:
- Science average (physics + chemistry) / 2
- Math-Physics combination
- Bio-Chemistry combination
- Math-Logic aptitude combo
- Science-Verbal aptitude combo
- Medical interest combo
- Tech interest combo
- Interaction features (e.g., math × math_aptitude)

**Total: ~50 features**

## Training with Real Data

### Data Requirements:

1. **Minimum Dataset Size:**
   - 1000+ samples for basic training
   - 5000+ samples for reliable models
   - 10000+ samples for production quality

2. **Data Quality:**
   - All fields must be filled
   - Marks/scores in 0-100 range
   - Valid personality types
   - Representative distribution across streams

3. **Labeling:**
   - Use rule-based system to generate initial labels
   - Or use actual student outcomes (which stream they chose/succeeded in)
   - Validate labels with domain experts

### Training Tips:

1. **Start Small**: Begin with 1000 samples, test, then scale up
2. **Monitor Overfitting**: Check train vs validation accuracy gap
3. **Feature Importance**: Use Random Forest feature importance to understand what matters
4. **Cross-Validation**: Use 5-fold CV for robust evaluation
5. **Hyperparameter Tuning**: Use GridSearchCV for best parameters

## Advanced Training Options

### Custom Model Training:

```python
from src.model_trainer import ModelTrainer
from src.data_models import StudentProfile

# Load your data
profiles = load_your_profiles()

# Create trainer
trainer = ModelTrainer(random_state=42)

# Train specific model type
X, y_class, y_reg = trainer.prepare_data(profiles)
X_train, X_val, y_train, y_val = train_test_split(X, y_class, test_size=0.2)

# Train only classification
results = trainer.train_classification_models(X_train, y_train, X_val, y_val)

# Or train only regression
results = trainer.train_regression_models(X_train, y_train, X_val, y_val)
```

### Hyperparameter Tuning:

Modify `config/model_config.yaml` or use GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
```

## Evaluation Metrics

### Classification Metrics:
- **Accuracy**: Overall correctness
- **Precision**: Per-class precision
- **Recall**: Per-class recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

### Regression Metrics (for score prediction):
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

### Business Metrics:
- **Recommendation Acceptance Rate**: How often students follow recommendations
- **Success Rate**: How well students perform in recommended streams
- **Satisfaction Score**: Student/parent satisfaction with recommendations

## Troubleshooting

### Low Accuracy:
1. **Check data quality**: Missing values, outliers, incorrect ranges
2. **Increase training data**: More samples usually help
3. **Feature engineering**: Add more relevant features
4. **Try different models**: Some models work better for certain data

### Overfitting:
1. **Reduce model complexity**: Lower max_depth, fewer trees
2. **Add regularization**: L1/L2 penalties
3. **More training data**: Reduces overfitting
4. **Cross-validation**: Use proper train/val/test splits

### Slow Training:
1. **Reduce dataset size**: Start with subset
2. **Use faster models**: LightGBM > XGBoost > Random Forest
3. **Parallel processing**: Set n_jobs=-1
4. **Feature selection**: Remove irrelevant features

## Production Deployment

### Model Serving:

1. **Save Model**:
```python
trainer.save_model('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
```

2. **Load and Use**:
```python
predictor = FSCPredictor('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
recommendation = predictor.recommend(student_profile)
```

3. **API Integration** (Flask example):
```python
from flask import Flask, request, jsonify
from src.predictor import FSCPredictor

app = Flask(__name__)
predictor = FSCPredictor('models/trained/best_model.pkl', 'models/trained/scaler.pkl')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    profile = create_profile_from_json(data)
    result = predictor.recommend(profile)
    return jsonify(result)
```

### Model Updates:

1. **Retrain Periodically**: Monthly/quarterly with new data
2. **A/B Testing**: Compare old vs new model performance
3. **Monitor Performance**: Track accuracy, latency, user feedback
4. **Version Control**: Save model versions with metadata

## Next Steps

1. **Collect Real Data**: Gather actual student profiles and outcomes
2. **Validate with Experts**: Have counselors review recommendations
3. **Deploy Gradually**: Start with rule-based, add ML as backup
4. **Continuous Improvement**: Monitor, collect feedback, retrain

## Resources

- **Documentation**: See README.md for system overview
- **Example Usage**: Run `python example_usage.py`
- **Configuration**: Edit `config/model_config.yaml`
- **Code Structure**: See individual module docstrings

## Support

For questions or issues:
1. Check this guide and README.md
2. Review code comments and docstrings
3. Test with example_usage.py
4. Validate data format matches expected schema
