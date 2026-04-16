#!/bin/bash
# Setup script for Amarel HPC (Rutgers)
# Run this once after cloning the repo:
#   bash setup_amarel.sh

set -e

echo "=== Installing miniconda (if needed) ==="
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p ~/miniconda3
    rm ~/miniconda.sh
fi

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate base

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Downloading data ==="
python download_data.py

echo "=== Verifying installation ==="
python -c "from pipeline import load_diabetes; load_diabetes()"

echo ""
echo "=== Setup complete ==="
echo "To activate later:  eval \"\$(~/miniconda3/bin/conda shell.bash hook)\" && conda activate base"
