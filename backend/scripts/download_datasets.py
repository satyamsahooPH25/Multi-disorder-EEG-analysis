"""Download and prepare open-source EEG datasets for training.

Targets the most relevant CC0 open datasets:

  - OpenNeuro ds004504 — Alzheimer's, FTD, Healthy (88 subjects)
  - OpenNeuro ds002778 — Parkinson's UC San Diego (31 subjects)
  - OpenNeuro ds003478 — Parkinson's Iowa (28 subjects)

Requires `awscli` or `openneuro-py` to be installed for ds004504/ds002778
(they are hosted on AWS S3). For convenience this script wraps both:

    pip install openneuro-py
    python -m scripts.download_datasets --datasets ds004504 ds002778
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASETS = {
    "ds004504": "Alzheimer's & FTD",
    "ds002778": "Parkinson's UC San Diego",
    "ds003478": "Parkinson's Iowa",
}


def download_openneuro(dataset_id: str, target: Path):
    try:
        import openneuro
    except ImportError as e:
        print("openneuro-py is required. Install with: pip install openneuro-py")
        raise SystemExit(1) from e
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_id} -> {target}")
    openneuro.download(dataset=dataset_id, target_dir=str(target))
    print(f"Done {dataset_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ds004504"])
    p.add_argument("--out", type=str, default="./data/datasets")
    args = p.parse_args()

    out = Path(args.out)
    for ds in args.datasets:
        if ds not in DATASETS:
            print(f"unknown dataset: {ds} (known: {list(DATASETS)})")
            continue
        download_openneuro(ds, out / ds)


if __name__ == "__main__":
    sys.exit(main())
