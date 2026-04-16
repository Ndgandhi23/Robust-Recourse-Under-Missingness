#!/bin/bash
# Setup script for Amarel HPC (Rutgers)
# Run this once after cloning the repo:
#   bash setup_amarel.sh

set -e

echo "=== Loading modules ==="
module purge
module load python/3.8.2

echo "=== Creating virtual environment ==="
python -m venv ~/.venvs/recourse
source ~/.venvs/recourse/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Downloading data ==="
python download_data.py

echo "=== Verifying installation ==="
python -c "from pipeline import load_diabetes; load_diabetes()"

echo ""
echo "=== Setup complete ==="
echo "To activate later:  module load python/3.8.2 && source ~/.venvs/recourse/bin/activate"
