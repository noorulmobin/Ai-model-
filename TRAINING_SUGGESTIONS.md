# Practical Training Suggestions for FSC Recommendation Model

## 🚀 Quick Training Options (Choose Based on Your Situation)

### Option 1: Start with Synthetic Data (Recommended First Step)

**Best for**: Testing, learning, initial model development

```bash
# Step 1: Generate 10,000 synthetic student profiles
python scripts/generate_data.py --samples 10000 --output data/synthetic/training_data.csv

# Step 2: Train all models and find the best one
python train.py --data data/synthetic/training_data.csv

# Step 3: Evaluate the trained model
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl --generate-test
```

**Time Required**: 15-30 minutes  
**Data Needed**: None (generated automatically)  
**Result**: Trained model ready to use

---

### Option 2: One-Command Training (Easiest)

**Best for**: Quick start, testing the system

```bash
# This does everything: generates data AND trains model
python train.py --generate --samples 10000
```

**Time Required**: 15-30 minutes  
**What it does**:
1. Generates synthetic data if needed
2. Loads student profiles
3. Trains 5 different ML models
4. Selects the best one
5. Saves to `models/trained/`

---

### Option 3: Train with Your Own Real Data

**Best for**: Production use, when you have actual student data

#### Step 1: Prepare Your Data

Create a CSV file with these columns:
```csv
student_id,math,bio,physics,chem,computer,aggregate,apt_math,apt_science,apt_verbal,apt_logical,apt_spatial,int_medical,int_engineering,int_computer,int_research,int_arts,personality
STU_001,85,78,82,80,88,82.6,88,85,75,90,70,40,85,90,75,50,INTP
STU_002,75,90,78,85,70,79.6,80,88,85,75,65,95,30,40,60,90,45,ISFJ
...
```

#### Step 2: Train with Your Data

```bash
python train.py --data path/to/your/data.csv
```

**Minimum Data**: 1,000 samples (basic)  
**Recommended**: 5,000-10,000 samples (good accuracy)  
**Optimal**: 10,000+ samples (production quality)

---

## 📊 Training Strategy Recommendations

### Strategy A: Incremental Training (Recommended)

**Phase 1: Start Small**
```bash
# Train with 1,000 samples first (quick test)
python train.py --generate --samples 1000
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl --generate-test
```

**Phase 2: Scale Up**
```bash
# If results look good, train with more data
python train.py --generate --samples 5000
```

**Phase 3: Production Training**
```bash
# Final training with full dataset
python train.py --generate --samples 10000
```

**Why this approach?**
- ✅ Catch errors early
- ✅ Test quickly
- ✅ Iterate faster
- ✅ Save time on large datasets

---

### Strategy B: Model Comparison Training

**Compare all models to find the best:**

```python
# Custom training script to compare models
from src.model_trainer import ModelTrainer
from scripts.generate_data import generate_dataset, load_profiles_from_dataframe
import pandas as pd

# Generate data
df, profiles = generate_dataset(10000, 'data/synthetic/training_data.csv')

# Train
trainer = ModelTrainer(random_state=42)
results = trainer.train_all_models(profiles, test_size=0.2)

# Compare results
print("\n=== Model Comparison ===")
for model_name, metrics in results['models']['classification'].items():
    print(f"{model_name}:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")

print(f"\nBest Model: {results['best_model_name']}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

**This shows you:**
- Which model performs best
- Training vs validation accuracy (detect overfitting)
- Model comparison table

---

### Strategy C: Real Data Collection + Training

**Step 1: Collect Real Student Data**

Create a data collection form/script:
```python
# Example: data_collection.py
from src.data_models import StudentProfile, AcademicPerformance, ...

def collect_student_data():
    print("Enter student information:")
    
    # Academic
    math = float(input("Mathematics marks: "))
    bio = float(input("Biology marks: "))
    physics = float(input("Physics marks: "))
    chem = float(input("Chemistry marks: "))
    computer = float(input("Computer marks: "))
    
    # Aptitude (if available)
    apt_math = float(input("Math aptitude score: "))
    # ... etc
    
    # Create profile
    profile = StudentProfile(...)
    return profile
```

**Step 2: Save to CSV**
```python
import pandas as pd

# Collect multiple profiles
profiles = []
for i in range(100):  # Collect 100 profiles
    profile = collect_student_data()
    profiles.append(profile)

# Convert to DataFrame and save
df = convert_profiles_to_dataframe(profiles)
df.to_csv('data/real/student_data.csv', index=False)
```

**Step 3: Train with Real Data**
```bash
python train.py --data data/real/student_data.csv
```

---

## 🎯 Specific Training Scenarios

### Scenario 1: "I want to test the system quickly"

```bash
# 5-minute test
python train.py --generate --samples 1000
python example_usage.py
```

### Scenario 2: "I want the best possible model"

```bash
# Train with large dataset
python train.py --generate --samples 20000

# Evaluate thoroughly
python evaluate.py --model models/trained/best_model.pkl \
                    --scaler models/trained/scaler.pkl \
                    --generate-test --test-samples 5000
```

### Scenario 3: "I have real data but it's small (<1000 samples)"

```bash
# Option A: Use synthetic data to augment
python scripts/generate_data.py --samples 9000 --output data/synthetic/augment.csv

# Combine with real data
# (manually merge CSVs or use pandas)

# Option B: Train with what you have (may have lower accuracy)
python train.py --data data/real/small_dataset.csv
```

### Scenario 4: "I want to retrain periodically"

```bash
# Create a retraining script: retrain.sh
#!/bin/bash
# Retrain model monthly

# Backup old model
cp models/trained/best_model.pkl models/backup/best_model_$(date +%Y%m%d).pkl

# Retrain with new data
python train.py --data data/real/updated_data.csv

# Evaluate new model
python evaluate.py --model models/trained/best_model.pkl \
                   --scaler models/trained/scaler.pkl \
                   --test-data data/real/test_data.csv

echo "Retraining complete!"
```

---

## 🔧 Advanced Training Options

### Option 1: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.model_trainer import ModelTrainer

# Load data
trainer = ModelTrainer()
X, y_class, _ = trainer.prepare_data(profiles)
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

### Option 2: Cross-Validation Training

```python
from sklearn.model_selection import cross_val_score
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
X, y_class, _ = trainer.prepare_data(profiles)

# 5-fold cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y_class, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### Option 3: Ensemble Training

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer()
X, y_class, _ = trainer.prepare_data(profiles)
X_train, X_val, y_train, y_val = train_test_split(X, y_class, test_size=0.2)

# Train individual models
class_results = trainer.train_classification_models(X_train, y_train, X_val, y_val)

# Create ensemble from top 3 models
top_models = sorted(class_results.items(), 
                   key=lambda x: x[1]['val_accuracy'], 
                   reverse=True)[:3]

# Train ensemble
ensemble_result = trainer.train_ensemble(
    X_train, y_train, X_val, y_val,
    {name: result for name, result in top_models}
)
```

---

## 📈 Training Monitoring

### Monitor Training Progress

```python
# Add this to train.py or create monitoring script
import time
from tqdm import tqdm

def train_with_monitoring(profiles):
    trainer = ModelTrainer()
    
    print("Preparing data...")
    start_time = time.time()
    X, y_class, y_reg = trainer.prepare_data(profiles)
    print(f"Data preparation: {time.time() - start_time:.2f}s")
    
    print("Training models...")
    results = trainer.train_all_models(profiles)
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Best model: {results['best_model_name']}")
    print(f"Accuracy: {results['test_accuracy']:.4f}")
    
    return results
```

### Save Training Logs

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename=f'training_logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use in training
logger.info(f"Starting training with {len(profiles)} profiles")
logger.info(f"Best model: {results['best_model_name']}")
logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
```

---

## ⚡ Performance Tips

### 1. Speed Up Training

```python
# Use fewer samples for quick iteration
python train.py --generate --samples 5000  # Faster than 10000

# Use faster models
# LightGBM is fastest, then XGBoost, then Random Forest
```

### 2. Improve Accuracy

```python
# Use more training data
python train.py --generate --samples 20000

# Train for longer (more trees)
# Modify model_config.yaml to increase n_estimators
```

### 3. Handle Memory Issues

```python
# Train in batches if dataset is too large
def train_in_batches(profiles, batch_size=5000):
    results_list = []
    for i in range(0, len(profiles), batch_size):
        batch = profiles[i:i+batch_size]
        trainer = ModelTrainer()
        results = trainer.train_all_models(batch)
        results_list.append(results)
    return results_list
```

---

## ✅ Training Checklist

Before training:
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Data ready (synthetic or real)
- [ ] Enough disk space (models can be 10-50MB)
- [ ] Python 3.8+ installed

During training:
- [ ] Monitor training progress
- [ ] Check for errors
- [ ] Note training time
- [ ] Save training logs

After training:
- [ ] Evaluate model: `python evaluate.py`
- [ ] Test predictions: `python example_usage.py`
- [ ] Compare with rule-based system
- [ ] Save model and metadata
- [ ] Document training parameters

---

## 🎓 Learning Path

### Beginner:
1. Start with rule-based: `python example_usage.py`
2. Generate small dataset: `python scripts/generate_data.py --samples 1000`
3. Train one model: Modify `train.py` to train only one model
4. Evaluate: `python evaluate.py`

### Intermediate:
1. Train all models: `python train.py --generate --samples 10000`
2. Compare results
3. Tune hyperparameters
4. Use cross-validation

### Advanced:
1. Collect real data
2. Feature engineering improvements
3. Ensemble methods
4. Production deployment
5. Continuous retraining pipeline

---

## 💡 Pro Tips

1. **Start Simple**: Use rule-based first, then add ML
2. **Validate Early**: Test with small dataset before full training
3. **Save Everything**: Models, scalers, metadata, logs
4. **Compare Always**: ML vs rule-based, different models
5. **Monitor Overfitting**: Watch train vs validation accuracy gap
6. **Version Control**: Save model versions with dates
7. **Document**: Note what worked, what didn't

---

## 🆘 Troubleshooting

**Problem**: Training takes too long
- **Solution**: Reduce samples or use faster models (LightGBM)

**Problem**: Low accuracy
- **Solution**: More training data, check data quality, try different models

**Problem**: Out of memory
- **Solution**: Train in batches, reduce features, use smaller models

**Problem**: Model not saving
- **Solution**: Check directory permissions, ensure `models/trained/` exists

**Problem**: Import errors
- **Solution**: `pip install -r requirements.txt`, check Python path

---

## 📞 Next Steps

1. **Choose your approach** from the options above
2. **Run the training command** for your scenario
3. **Evaluate the results** with `evaluate.py`
4. **Compare with rule-based** system
5. **Iterate and improve** based on results

Good luck with your training! 🚀
