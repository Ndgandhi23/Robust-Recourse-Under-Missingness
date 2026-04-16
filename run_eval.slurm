#!/bin/bash
#SBATCH --job-name=recourse-eval
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

module purge
module load python/3.8.2
source ~/.venvs/recourse/bin/activate

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo ""

python run_eval_all.py

echo ""
echo "Job finished: $(date)"
