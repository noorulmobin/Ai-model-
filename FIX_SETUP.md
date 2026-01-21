# Fix Setup Issues - Ubuntu/Debian

## Problem: Missing System Packages

If you see errors like:
- `ensurepip is not available`
- `python3-venv package needed`
- `pip not found`

## Solution: Install Prerequisites First

### Step 1: Install Required System Packages

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

### Step 2: Verify Installation

```bash
python3 --version
pip3 --version
```

### Step 3: Now Create Virtual Environment

```bash
cd /home/mubeen/ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Create Directories and Train

```bash
mkdir -p data/synthetic models/trained models/backup
python train.py --generate --samples 10000
```

---

## Complete Fixed Commands (Copy & Paste)

```bash
# Install system packages (requires sudo password)
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

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

# Train the model
python train.py --generate --samples 10000
```

---

## OR Use the Automated Script

I've created a script that does everything automatically:

```bash
cd /home/mubeen/ai
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

This script will:
1. ✅ Install system prerequisites (python3, pip, venv)
2. ✅ Create virtual environment
3. ✅ Install Python packages
4. ✅ Create directories
5. ✅ Set everything up

---

## Alternative: Use System Python (No Virtual Environment)

If you prefer not to use virtual environment:

```bash
# Install system packages
sudo apt update
sudo apt install -y python3 python3-pip

# Install dependencies globally (not recommended but works)
pip3 install --user -r requirements.txt

# Run training
python3 train.py --generate --samples 10000
```

**Note**: Using virtual environment is recommended to avoid conflicts with system packages.

---

## Troubleshooting

### Issue: "sudo: command not found"
**Solution**: You need administrator access. Contact your system administrator or use a user account with sudo privileges.

### Issue: "E: Unable to locate package python3-venv"
**Solution**: Try:
```bash
sudo apt update
sudo apt install python3.10-venv
# Or for your specific Python version:
python3 --version  # Check version first
sudo apt install python3.X-venv  # Replace X with your version
```

### Issue: "Permission denied"
**Solution**: Make sure you have write permissions in the project directory:
```bash
cd /home/mubeen/ai
ls -la  # Check permissions
```

### Issue: Virtual environment created but activation fails
**Solution**: Recreate it:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

---

## Quick Fix Commands

**If you just need to install prerequisites:**
```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv
```

**Then continue with setup:**
```bash
cd /home/mubeen/ai
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p data/synthetic models/trained
python train.py --generate --samples 10000
```

---

## Verify Everything Works

After setup, test:

```bash
# Check virtual environment is active (should see venv in prompt)
which python  # Should show: /home/mubeen/ai/venv/bin/python

# Test imports
python -c "import numpy, pandas, sklearn; print('✅ All packages installed!')"

# Run example
python example_usage.py
```

---

That should fix all the issues! 🚀
