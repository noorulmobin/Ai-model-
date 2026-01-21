# Setup Commands - Virtual Environment & Training

## 🚀 Complete Setup & Training Commands

### Option 1: Automated Setup Script (Easiest)

```bash
# Make the script executable
chmod +x setup_and_train.sh

# Run the setup and training script
./setup_and_train.sh
```

This will:
1. ✅ Create virtual environment
2. ✅ Install all dependencies
3. ✅ Create necessary directories
4. ✅ Train the model automatically

---

### Option 2: Manual Step-by-Step Commands

#### Step 1: Navigate to Project Directory
```bash
cd /home/mubeen/ai
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

#### Step 3: Activate Virtual Environment
```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your command prompt.

#### Step 4: Upgrade pip
```bash
pip install --upgrade pip
```

#### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 6: Create Directories
```bash
mkdir -p data/synthetic
mkdir -p models/trained
mkdir -p models/backup
```

#### Step 7: Train the Model
```bash
# Standard training (10,000 samples)
python train.py --generate --samples 10000

# OR use interactive menu
python train_simple.py

# OR quick test (1,000 samples)
python train.py --generate --samples 1000
```

---

## 📋 Complete Command Sequence (Copy & Paste)

```bash
# Navigate to project
cd /home/mubeen/ai

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/synthetic models/trained models/backup

# Train model (choose one):
# Option A: Standard training
python train.py --generate --samples 10000

# Option B: Interactive menu
python train_simple.py

# Option C: Quick test
python train.py --generate --samples 1000
```

---

## 🔄 Using Virtual Environment in Future Sessions

### Activate Virtual Environment
```bash
cd /home/mubeen/ai
source venv/bin/activate
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Check if Virtual Environment is Active
Look for `(venv)` in your terminal prompt, or run:
```bash
which python
# Should show: /home/mubeen/ai/venv/bin/python
```

---

## 🎯 Training Options After Setup

### Option 1: Standard Training (Recommended)
```bash
source venv/bin/activate
python train.py --generate --samples 10000
```

### Option 2: Interactive Menu
```bash
source venv/bin/activate
python train_simple.py
```

### Option 3: Quick Test
```bash
source venv/bin/activate
python train.py --generate --samples 1000
```

### Option 4: Large Dataset
```bash
source venv/bin/activate
python train.py --generate --samples 20000
```

---

## ✅ Verify Installation

After setup, verify everything works:

```bash
# Activate virtual environment
source venv/bin/activate

# Check Python version (should be 3.8+)
python --version

# Check installed packages
pip list

# Test imports
python -c "import numpy, pandas, sklearn, xgboost, lightgbm; print('All packages installed!')"

# Run example (no training needed)
python example_usage.py
```

---

## 🐛 Troubleshooting

### Problem: "python3: command not found"
**Solution**: Install Python 3.8 or higher
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Check version
python3 --version
```

### Problem: "Permission denied" when running script
**Solution**: Make script executable
```bash
chmod +x setup_and_train.sh
```

### Problem: "Module not found" after activation
**Solution**: Make sure virtual environment is activated
```bash
# Check if activated (should see venv in prompt)
# If not, activate again:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Problem: Virtual environment not found
**Solution**: Recreate it
```bash
# Remove old venv
rm -rf venv

# Create new one
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📝 Quick Reference Card

```bash
# SETUP (One-time)
cd /home/mubeen/ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data/synthetic models/trained

# TRAINING (Every time you want to train)
source venv/bin/activate
python train.py --generate --samples 10000

# TESTING (After training)
python example_usage.py

# EVALUATION
python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl
```

---

## 🎓 Next Steps After Setup

1. **Test the system** (no training needed):
   ```bash
   python example_usage.py
   ```

2. **Train your first model**:
   ```bash
   python train.py --generate --samples 10000
   ```

3. **Evaluate the model**:
   ```bash
   python evaluate.py --model models/trained/best_model.pkl --scaler models/trained/scaler.pkl --generate-test
   ```

4. **Use in your code**:
   ```python
   from src.predictor import FSCPredictor
   predictor = FSCPredictor('models/trained/best_model.pkl', 'models/trained/scaler.pkl')
   ```

---

That's it! You're all set! 🚀
