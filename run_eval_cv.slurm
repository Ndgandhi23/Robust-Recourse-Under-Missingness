#!/bin/bash
#SBATCH --job-name=recourse-cv
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_cv_%j.out
#SBATCH --error=logs/eval_cv_%j.err

module purge
module load python/3.10
source ~/.venvs/recourse/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo ""

python run_eval_cv.py

echo ""
echo "Job finished: $(date)"
