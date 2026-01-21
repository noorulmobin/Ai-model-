# Simple Training Guide - Quick Start

## 🎯 Three Easy Ways to Train Your Model

### Method 1: Interactive Menu (Easiest)

Just run this and choose an option:

```bash
python train_simple.py
```

You'll see a menu:
```
1. Quick Test (1000 samples, ~3 minutes)
2. Standard Training (10,000 samples, ~20 minutes)
3. Large Dataset (20,000 samples, ~45 minutes)
4. Train with Your Own Data
5. Compare All Models
```

**Recommended for beginners!**

---

### Method 2: One Command (Fastest)

For standard training, just run:

```bash
python train.py --generate --samples 10000
```

This will:
- ✅ Generate 10,000 synthetic student profiles
- ✅ Train 5 different ML models
- ✅ Pick the best one
- ✅ Save it to `models/trained/`

**Takes about 15-30 minutes**

---

### Method 3: Step by Step (Most Control)

```bash
# Step 1: Generate data
python scripts/generate_data.py --samples 10000 --output data/synthetic/training_data.csv

# Step 2: Train
python train.py --data data/synthetic/training_data.csv

# Step 3: Evaluate
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl --generate-test
```

---

## 📊 Which Method Should You Use?

| Your Situation | Recommended Method | Command |
|---------------|-------------------|---------|
| **First time, want to test** | Method 1 (Interactive) | `python train_simple.py` → Choose 1 |
| **Want to train quickly** | Method 2 (One command) | `python train.py --generate --samples 10000` |
| **Have your own data** | Method 1 → Option 4 | `python train_simple.py` → Choose 4 |
| **Want to compare models** | Method 1 → Option 5 | `python train_simple.py` → Choose 5 |
| **Need best accuracy** | Method 2 with more samples | `python train.py --generate --samples 20000` |

---

## ⏱️ Training Time Estimates

| Samples | Time | Best For |
|---------|------|----------|
| 1,000 | 2-3 min | Quick test |
| 5,000 | 8-12 min | Development |
| 10,000 | 15-30 min | Standard training |
| 20,000 | 30-60 min | Production quality |

---

## ✅ After Training

Once training completes, you'll see:
```
✅ Training completed!
Best Model: xgboost
Test Accuracy: 0.8542
Models saved to models/trained/
```

Then test it:
```bash
python example_usage.py
```

---

## 🆘 Common Issues

**"Module not found" error?**
```bash
pip install -r requirements.txt
```

**"File not found" error?**
- Make sure you're in the project root directory
- Run: `cd /home/mubeen/ai`

**Training too slow?**
- Use fewer samples: `--samples 5000`
- Or use quick test: `python train_simple.py` → Choose 1

**Want to use your own data?**
1. Prepare CSV with required columns (see TRAINING_SUGGESTIONS.md)
2. Run: `python train_simple.py` → Choose 4
3. Enter path to your CSV file

---

## 🎓 Learning Path

**Day 1: Quick Test**
```bash
python train_simple.py  # Choose option 1
```

**Day 2: Standard Training**
```bash
python train.py --generate --samples 10000
```

**Day 3: Evaluate & Compare**
```bash
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl
```

**Day 4: Use in Your Code**
```python
from src.predictor import FSCPredictor
predictor = FSCPredictor('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
result = predictor.recommend(student_profile)
```

---

## 💡 Pro Tips

1. **Start small**: Test with 1000 samples first
2. **Save time**: Use `train_simple.py` for interactive training
3. **Compare results**: Always check which model performs best
4. **Test immediately**: Run `example_usage.py` after training

---

That's it! You're ready to train. 🚀

For more details, see:
- `TRAINING_SUGGESTIONS.md` - Detailed training options
- `AI_TRAINING_APPROACHES.md` - Understanding different approaches
- `TRAINING_GUIDE.md` - Comprehensive guide
