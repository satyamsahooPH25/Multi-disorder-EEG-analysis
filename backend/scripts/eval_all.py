"""Evaluate the loaded HAMD-Net checkpoint on every available subject."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.ml.hamd_net import HAMDConfig, CLASS_NAMES  # noqa: E402
from app.ml.inference_engine import InferenceEngine  # noqa: E402
from app.ml.real_dataset import list_subjects, load_subject  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = InferenceEngine(checkpoint="data/checkpoints/hamdnet_latest.pt",
                           device=device, cfg=HAMDConfig())
    subjects = list_subjects()
    print(f"checkpoint loaded · device {device} · {len(subjects)} subjects\n")

    correct_per: dict[str, list[int]] = {}
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    eval_seconds = 60.0
    for i, s in enumerate(subjects):
        sig, sf, ch = load_subject(s)
        n = int(eval_seconds * sf)
        if sig.shape[1] > n:
            sig = sig[:, :n]
        res = eng.infer_array(sig, sfreq=sf, channels=ch)
        pred_idx = CLASS_NAMES.index(res.predicted_class)
        confusion[s.label, pred_idx] += 1
        ok = res.predicted_class == s.label_name
        correct_per.setdefault(s.label_name, []).append(int(ok))
        marker = "ok" if ok else "MISS"
        print(f"  [{i+1:3d}/{len(subjects)}] {marker:4s} "
              f"{s.subject_id:14s} {s.source:15s} "
              f"truth={s.label_name:14s} -> {res.predicted_class:14s} "
              f"({res.confidence:.2f})", flush=True)

    print(f"{'class':<14} {'correct/total':<14} accuracy")
    print("-" * 44)
    overall_c = overall_t = 0
    for cls in CLASS_NAMES:
        items = correct_per.get(cls, [])
        if not items:
            continue
        correct = sum(items)
        total = len(items)
        overall_c += correct
        overall_t += total
        print(f"{cls:<14} {correct}/{total:<12} {correct / total * 100:.1f}%")
    print("-" * 44)
    print(f"{'overall':<14} {overall_c}/{overall_t:<12} {overall_c / overall_t * 100:.1f}%")

    print("\nconfusion matrix (rows = truth, cols = predicted)")
    header = "    " + " ".join(f"{c[:5]:>6}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(confusion):
        cells = " ".join(f"{int(v):>6d}" for v in row)
        print(f"{CLASS_NAMES[i][:5]:<5}{cells}")


if __name__ == "__main__":
    main()
