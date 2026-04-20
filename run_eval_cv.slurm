#!/bin/bash
# SLURM array job: evaluates one registered dataset per array index.
# Submit with:
#   sbatch --array=0-$(python -c "from pipeline.datasets import list_datasets; print(len(list_datasets()) - 1)") run_eval_cv.slurm
# Or manually (e.g. just index 0):
#   sbatch --array=0 run_eval_cv.slurm
#SBATCH --job-name=recourse-cv
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_cv_%A_%a.out
#SBATCH --error=logs/eval_cv_%A_%a.err

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate base

cd $SLURM_SUBMIT_DIR
mkdir -p logs results

# Pull dataset name by array index from the registry
DATASET=$(python -c "from pipeline.datasets import list_datasets; print(list_datasets()[$SLURM_ARRAY_TASK_ID])")

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Array index: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo ""

python run_eval_cv.py --dataset $DATASET --workers $SLURM_CPUS_PER_TASK

echo ""
echo "Job finished: $(date)"
