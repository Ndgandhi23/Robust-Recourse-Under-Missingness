#!/usr/bin/env python3
"""
Run the full pipeline for every registered dataset:
  1. Download data
  2. 4-fold stratified CV evaluation (run_eval_cv.py)
  3. Generate figures across all datasets

Usage:
    python run_all.py                     # all datasets, 1 worker
    python run_all.py --workers 8         # all datasets, 8 workers
    python run_all.py --datasets fico     # just fico
    python run_all.py --skip-download     # skip download step
    python run_all.py --skip-eval         # only regenerate figures from existing results
"""

import argparse
import subprocess
import sys
import time

from pipeline.datasets import list_datasets


def run(cmd, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFAILED: {label}  (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  {label} done ({elapsed:.0f}s)")


def main():
    available = list_datasets()

    parser = argparse.ArgumentParser(description="Run full pipeline for all datasets")
    parser.add_argument("--datasets", nargs="+", default=available,
                        choices=available, help=f"Datasets to run (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for per-person evaluation")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the data download step")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only regenerate figures")
    args = parser.parse_args()

    t_total = time.time()

    # 1. Download
    if not args.skip_download and not args.skip_eval:
        run([sys.executable, "download_data.py"] + args.datasets,
            "Download datasets")

    # 2. CV evaluation per dataset
    if not args.skip_eval:
        for name in args.datasets:
            run([sys.executable, "run_eval_cv.py",
                 "--dataset", name,
                 "--workers", str(args.workers)],
                f"CV evaluation: {name}")

    # 3. Figures
    run([sys.executable, "figures.py"], "Generate figures")

    elapsed = time.time() - t_total
    print(f"\n{'#' * 70}")
    print(f"#  ALL DONE  ({elapsed:.0f}s total)")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
