"""
download_data.py
----------------
Downloads datasets used by the pipeline.

Usage:
    python download_data.py              # download all datasets
    python download_data.py diabetes     # download one dataset
    python download_data.py german australian  # download specific datasets
"""

import sys
from pipeline.datasets import get_dataset, list_datasets


def main():
    available = list_datasets()

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = available

    for name in names:
        if name not in available:
            print(f"Unknown dataset: {name}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)

    for name in names:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        cfg = get_dataset(name)
        if cfg.download is not None:
            cfg.download()
        else:
            print(f"  no download function defined for {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
