# AI Model Training Approaches for FSC Recommendation System

## Executive Summary

This document outlines **4 different approaches** to train an AI model for the FSC Subject Recommendation System, each with different trade-offs in accuracy, interpretability, and implementation complexity.

---

## Approach 1: Rule-Based System (Baseline - No ML Training)

### Overview
Implements the exact algorithm from your documentation using if-else logic and weighted scoring.

### Implementation
- **File**: `src/scoring.py`
- **Training Required**: ❌ None
- **Data Required**: None (works immediately)

### How It Works
1. Calculates scores for each stream:
   - Academic Score (40% weight)
   - Aptitude Score (35% weight)
   - Personality Match (15% weight)
   - Interest Score (10% weight)
2. Combines scores: `Final = (Academic × 0.40) + (Aptitude × 0.35) + (Personality × 0.15) + (Interest × 0.10)`
3. Ranks streams by final score

### Pros
✅ **No training needed** - works immediately  
✅ **Fully interpretable** - exact reasoning for each recommendation  
✅ **Matches documentation** - implements your exact algorithm  
✅ **No data dependency** - doesn't need historical data  
✅ **Fast inference** - instant predictions  
✅ **Production-ready** - can deploy immediately

### Cons
❌ **Rigid rules** - can't learn from data  
❌ **No adaptation** - doesn't improve with experience  
❌ **Limited patterns** - may miss complex relationships

### Best For
- **Immediate deployment** without training
- **Interpretable recommendations** for counselors
- **Baseline comparison** for ML models
- **Production use** when explainability is critical

### Usage
```python
from src.scoring import RuleBasedScorer
scorer = RuleBasedScorer()
result = scorer.recommend(student_profile)
```

---

## Approach 2: Supervised Learning - Multi-Class Classification

### Overview
Train a classifier to predict the recommended stream directly (4 classes: Pre-Medical, Pre-Engineering, ICS, General Science).

### Implementation
- **File**: `src/model_trainer.py` → `train_classification_models()`
- **Training Required**: ✅ Yes (30 minutes - 2 hours)
- **Data Required**: 5,000-10,000+ labeled student profiles

### How It Works
1. **Feature Extraction**: Convert student profile to ~50 features
   - Academic marks (6 features)
   - Aptitude scores (5 features)
   - Interest scores (5 features)
   - Personality encoding (16 features)
   - Derived features (20+ features)
2. **Label Generation**: Use rule-based system to create labels
   - For each profile, get top recommendation → that's the label
3. **Model Training**: Train multiple classifiers
   - Logistic Regression
   - Random Forest
   - SVM
   - XGBoost
   - LightGBM
4. **Selection**: Choose best model based on validation accuracy

### Models Available
| Model | Training Time | Accuracy | Interpretability |
|-------|--------------|----------|------------------|
| Logistic Regression | Fast (1 min) | Good | High |
| Random Forest | Medium (5 min) | Very Good | Medium |
| SVM | Medium (10 min) | Good | Low |
| XGBoost | Medium (8 min) | Excellent | Medium |
| LightGBM | Fast (3 min) | Excellent | Medium |

### Pros
✅ **Learns from data** - can discover patterns  
✅ **High accuracy** - often outperforms rule-based  
✅ **Handles complexity** - captures non-linear relationships  
✅ **Multiple models** - can ensemble for better results

### Cons
❌ **Requires training data** - need labeled examples  
❌ **Training time** - takes time to train  
❌ **Less interpretable** - harder to explain predictions  
❌ **Data dependency** - quality depends on training data

### Best For
- **Large datasets** with real student outcomes
- **Maximum accuracy** requirements
- **Complex patterns** in data
- **Production systems** with sufficient data

### Usage
```bash
# Train
python train.py --generate --samples 10000

# Use
from src.predictor import FSCPredictor
predictor = FSCPredictor('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
recommendation = predictor.recommend(profile, use_ml=True)
```

---

## Approach 3: Regression + Ranking

### Overview
Train regression models to predict match scores for each stream, then rank by score.

### Implementation
- **File**: `src/model_trainer.py` → `train_regression_models()`
- **Training Required**: ✅ Yes (30 minutes - 2 hours)
- **Data Required**: 5,000-10,000+ profiles with scores

### How It Works
1. **Multi-output Regression**: Predict 4 scores simultaneously
   - One score per stream (Pre-Medical, Pre-Engineering, ICS, General Science)
2. **Target Generation**: Use rule-based system to generate scores
   - For each profile, calculate match percentage for each stream
3. **Model Training**: Train regression models
   - Ridge Regression
   - Random Forest Regressor
   - SVR
   - XGBoost Regressor
   - LightGBM Regressor
4. **Ranking**: Sort streams by predicted scores

### Pros
✅ **Score predictions** - get confidence for all streams  
✅ **Ranking flexibility** - can show top 3 recommendations  
✅ **Continuous values** - more nuanced than classification  
✅ **Better for comparison** - can compare stream scores

### Cons
❌ **More complex** - requires multi-output models  
❌ **Needs scores** - requires score labels, not just classes  
❌ **Similar accuracy** - may not outperform classification

### Best For
- **Score-based recommendations** - when you need match percentages
- **Multiple recommendations** - showing top 3 streams
- **Comparison needs** - comparing stream suitability

### Usage
```python
# Training automatically includes regression models
# Access via model_trainer.train_regression_models()
```

---

## Approach 4: Hybrid System (Recommended for Production)

### Overview
Combine rule-based system with ML model, using both for robust recommendations.

### Implementation
- **Files**: `src/predictor.py` → `recommend()` method
- **Training Required**: ✅ Partial (ML component needs training)
- **Data Required**: 5,000-10,000+ profiles

### How It Works
1. **Dual Prediction**: Get recommendations from both systems
   - Rule-based recommendation
   - ML model prediction
2. **Consensus Logic**:
   - If both agree → High confidence recommendation
   - If they disagree → Show both, explain difference
   - Use ML probabilities for confidence scores
3. **Fallback**: If ML model unavailable, use rule-based

### Pros
✅ **Best of both worlds** - accuracy + interpretability  
✅ **Robust** - doesn't fail if ML model unavailable  
✅ **Confidence scores** - ML provides probability estimates  
✅ **Explainable** - rule-based provides reasoning  
✅ **Production-ready** - handles edge cases

### Cons
❌ **More complex** - requires both systems  
❌ **Slightly slower** - runs both models  
❌ **Maintenance** - need to maintain both

### Best For
- **Production deployment** - maximum reliability
- **High-stakes decisions** - when accuracy is critical
- **Explainable AI** - need both accuracy and reasoning
- **Gradual rollout** - can start with rule-based, add ML later

### Usage
```python
predictor = FSCPredictor('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
result = predictor.recommend(profile, use_ml=True)

# Result contains:
# - ML prediction
# - ML probabilities
# - Rule-based recommendation
# - Comparison and warnings
```

---

## Recommended Training Strategy

### Phase 1: Start with Rule-Based (Week 1)
1. ✅ Deploy rule-based system immediately
2. ✅ Start collecting real student data
3. ✅ Validate recommendations with counselors

### Phase 2: Train ML Model (Week 2-3)
1. ✅ Generate synthetic data for initial training
2. ✅ Train classification models
3. ✅ Compare ML vs rule-based on test set
4. ✅ If ML performs better, integrate it

### Phase 3: Hybrid System (Week 4+)
1. ✅ Deploy hybrid system (both models)
2. ✅ Collect real student outcomes
3. ✅ Retrain ML model with real data quarterly
4. ✅ Continuously improve

---

## Data Requirements by Approach

| Approach | Minimum Data | Optimal Data | Data Type |
|----------|-------------|--------------|-----------|
| Rule-Based | 0 | 0 | None needed |
| Classification | 1,000 | 10,000+ | Labeled profiles |
| Regression | 1,000 | 10,000+ | Profiles with scores |
| Hybrid | 1,000 | 10,000+ | Labeled profiles |

### Data Generation Options

1. **Synthetic Data** (Quick Start):
   ```bash
   python scripts/generate_data.py --samples 10000
   ```
   - Generates realistic student profiles
   - Uses correlations between features
   - Good for initial training

2. **Real Data** (Production):
   - Collect actual student profiles
   - Include outcomes (which stream they chose/succeeded in)
   - Validate with domain experts

---

## Performance Comparison

### Expected Accuracy (on synthetic data)

| Approach | Accuracy | Training Time | Inference Time |
|----------|----------|---------------|----------------|
| Rule-Based | Baseline | 0 | <1ms |
| Logistic Regression | 75-80% | 1 min | <1ms |
| Random Forest | 80-85% | 5 min | <5ms |
| XGBoost | 85-90% | 8 min | <5ms |
| LightGBM | 85-90% | 3 min | <3ms |
| Hybrid | 85-92% | 8 min | <10ms |

*Note: Actual accuracy depends on data quality and distribution*

---

## Implementation Checklist

### For Rule-Based System:
- [x] ✅ Implemented in `src/scoring.py`
- [x] ✅ Ready to use immediately
- [ ] Deploy to production
- [ ] Collect user feedback

### For ML Training:
- [x] ✅ Feature engineering implemented
- [x] ✅ Model training pipeline ready
- [x] ✅ Data generator available
- [ ] Generate/collect training data
- [ ] Train models
- [ ] Evaluate and select best model
- [ ] Deploy ML model

### For Hybrid System:
- [x] ✅ Predictor interface supports both
- [ ] Train ML component
- [ ] Implement consensus logic
- [ ] Deploy hybrid system
- [ ] Monitor both systems

---

## Quick Decision Guide

**Choose Rule-Based if:**
- You need to deploy immediately
- Interpretability is critical
- You don't have training data yet
- You want exact algorithm implementation

**Choose ML Classification if:**
- You have 5,000+ labeled examples
- You want maximum accuracy
- You can train models
- You're okay with less interpretability

**Choose Regression if:**
- You need score predictions for all streams
- You want to show top 3 recommendations
- You need confidence scores

**Choose Hybrid if:**
- You want best of both worlds
- Production deployment
- High reliability requirements
- You can maintain both systems

---

## Next Steps

1. **Review this document** - understand all approaches
2. **Start with rule-based** - deploy immediately
3. **Generate synthetic data** - `python scripts/generate_data.py --samples 10000`
4. **Train ML model** - `python train.py --generate`
5. **Compare results** - evaluate ML vs rule-based
6. **Deploy hybrid** - use both systems together

For detailed training instructions, see **TRAINING_GUIDE.md**
