#!/bin/bash
#SBATCH --job-name=recourse-eval
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate base

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Working dir: $(pwd)"
echo ""

python run_eval_all.py --workers $SLURM_CPUS_PER_TASK

echo ""
echo "Job finished: $(date)"
