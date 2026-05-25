"""Standalone CLI for training HAMD-Net.

Usage:
    python -m scripts.train --epochs 12 --n_per_class 64
    python -m scripts.train --data ./data/openneuro_ds004504  # real data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.trainer import train_synthetic  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_per_class", type=int, default=32)
    p.add_argument("--duration_s", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ckpt = train_synthetic(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_per_class=args.n_per_class,
        duration_s=args.duration_s,
        seed=args.seed,
        device=args.device,
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
