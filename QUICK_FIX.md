# Quick Fix - Virtual Environment Not Active

## The Problem
When you run `python train.py`, it says "Command 'python' not found" because the virtual environment isn't active in your current shell.

## Solution: Activate Virtual Environment First

**In your terminal, run these commands:**

```bash
# 1. Navigate to project
cd /home/mubeen/ai

# 2. Activate virtual environment (IMPORTANT!)
source venv/bin/activate

# 3. Verify it's active (you should see "venv" in your prompt)
which python  # Should show: /home/mubeen/ai/venv/bin/python

# 4. Now run training
python train.py --generate --samples 10000
```

---

## OR Use the Helper Script (Easiest)

I've created a script that handles everything:

```bash
cd /home/mubeen/ai
./run_train.sh
```

Or with custom parameters:
```bash
./run_train.sh --generate --samples 10000
./run_train.sh --generate --samples 5000
```

---

## Alternative: Use Python3 Explicitly

If `python` command doesn't work, use `python3` but make sure venv is active:

```bash
cd /home/mubeen/ai
source venv/bin/activate
python3 train.py --generate --samples 10000
```

---

## How to Know Virtual Environment is Active

You should see `(venv)` at the beginning of your terminal prompt:
```
(venv) mubeen@mubeen-HP-EliteBook-840-G7-Notebook-PC:~/ai$
```

If you don't see `(venv)`, the environment is NOT active!

---

## Complete Working Commands

```bash
# Make sure you're in the project directory
cd /home/mubeen/ai

# Activate virtual environment (do this every time you open a new terminal)
source venv/bin/activate

# Verify activation
which python
# Should output: /home/mubeen/ai/venv/bin/python

# Now training will work
python train.py --generate --samples 10000
```

---

## Quick Test

Run this to verify everything works:

```bash
cd /home/mubeen/ai
source venv/bin/activate
python -c "import numpy, pandas, sklearn; print('✅ All packages installed!')"
python example_usage.py
```

If this works, then training will work too!
