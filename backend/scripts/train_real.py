"""Train HAMD-Net on the combined real corpus (ds004504 + ds002778 + Olejarczyk SCZ)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.trainer import train  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--per_subject_seconds", type=float, default=120.0)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--source", type=str, default="real",
                   choices=["real", "synthetic"])
    p.add_argument("--patience", type=int, default=8,
                   help="early stop after N epochs without val improvement")
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--min_epochs", type=int, default=8)
    args = p.parse_args()

    ckpt = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        source=args.source,
        max_subjects_per_class=args.max_per_class,
        per_subject_seconds=args.per_subject_seconds,
        patience=args.patience,
        min_delta=args.min_delta,
        min_epochs=args.min_epochs,
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
