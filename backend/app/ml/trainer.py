"""HAMD-Net training pipeline.

Supports training on:
  * the synthetic dataset (no downloads required)
  * the real OpenNeuro ds004504 corpus (Alzheimer's / FTD / Healthy)

Emits epoch-by-epoch metrics to the shared `TrainingState` so the frontend
training dashboard updates in real time over Server-Sent Events.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..core.state import DEFAULT_CHECKPOINT, TrainingState, state as app_state
from .hamd_net import CLASS_NAMES, HAMDConfig, build_model
from .preprocessing import PreprocConfig, preprocess_array
from .real_dataset import list_subjects, load_subject
from .synthetic import generate_dataset


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(windows.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _fit_channels(windows: np.ndarray, n_channels: int) -> np.ndarray:
    if windows.shape[1] >= n_channels:
        return windows[:, :n_channels, :]
    pad = np.zeros((windows.shape[0], n_channels - windows.shape[1],
                    windows.shape[2]), dtype=windows.dtype)
    return np.concatenate([windows, pad], axis=1)


def windows_from_subjects(X, y, sfreq: float, cfg: HAMDConfig,
                           per_subject_seconds: float | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    pre = PreprocConfig(target_sfreq=cfg.sfreq, n_channels=cfg.n_channels,
                        window_seconds=cfg.n_samples / cfg.sfreq,
                        overlap=0.5)
    all_windows, all_labels = [], []
    for sig, label in zip(X, y):
        if per_subject_seconds is not None:
            n = int(per_subject_seconds * sfreq)
            if sig.shape[1] > n:
                sig = sig[:, :n]
        result = preprocess_array(sig, sfreq, cfg=pre)
        windows = _fit_channels(result.windows, cfg.n_channels)
        all_windows.append(windows)
        all_labels.append(np.full(windows.shape[0], label, dtype=np.int64))
    return np.concatenate(all_windows), np.concatenate(all_labels)


def _publish(ts: TrainingState, payload: dict):
    for q in list(ts.listeners):
        try:
            q.put_nowait(payload)
        except Exception:
            pass
    msg = payload.get("message")
    if msg:
        print(f"[trainer] {msg}", flush=True)


def _snapshot(ts: TrainingState) -> dict:
    return {
        "running": ts.running,
        "epoch": ts.epoch,
        "total_epochs": ts.total_epochs,
        "train_loss": list(ts.train_loss),
        "val_loss": list(ts.val_loss),
        "train_acc": list(ts.train_acc),
        "val_acc": list(ts.val_acc),
        "best_val_acc": ts.best_val_acc,
        "started_at": ts.started_at,
        "finished_at": ts.finished_at,
        "message": ts.message,
    }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_real_data(cfg: HAMDConfig, max_subjects_per_class: int | None,
                     per_subject_seconds: float | None,
                     ts: TrainingState) -> tuple[np.ndarray, np.ndarray]:
    subjects = list_subjects()
    if not subjects:
        raise FileNotFoundError(
            "No real dataset found. Run "
            "`python -m scripts.download_datasets --datasets ds004504` first."
        )
    by_label: dict[int, list] = {}
    for s in subjects:
        by_label.setdefault(s.label, []).append(s)
    selected = []
    for label, items in by_label.items():
        if max_subjects_per_class:
            items = items[:max_subjects_per_class]
        selected.extend(items)
    sources = sorted({s.source for s in selected})
    ts.message = (f"Loading {len(selected)} subjects from "
                  f"{', '.join(sources)}")
    _publish(ts, _snapshot(ts))

    X, y = [], []
    sfreq = None
    for i, s in enumerate(selected):
        try:
            sig, sf, _ = load_subject(s)
        except Exception as exc:
            print(f"[trainer] skipped {s.subject_id} ({s.source}): {exc}",
                  flush=True)
            continue
        # Resample on the fly to the canonical model sampling rate so all
        # corpora share one window length.
        if sf != cfg.sfreq:
            ratio = cfg.sfreq / sf
            new_n = int(round(sig.shape[1] * ratio))
            idx = np.linspace(0, sig.shape[1] - 1, new_n).astype(np.int64)
            sig = sig[:, idx]
            sf = cfg.sfreq
        sfreq = sf
        X.append(sig)
        y.append(s.label)
        if (i + 1) % 5 == 0:
            ts.message = f"Loaded {i+1}/{len(selected)} subjects"
            _publish(ts, _snapshot(ts))

    Xw, yw = windows_from_subjects(X, np.array(y), sfreq or cfg.sfreq, cfg,
                                    per_subject_seconds=per_subject_seconds)
    return Xw, yw


def train(
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_per_class: int = 32,
    duration_s: float = 30.0,
    seed: int = 42,
    device: str = "auto",
    source: str = "synthetic",
    max_subjects_per_class: int | None = None,
    per_subject_seconds: float | None = 60.0,
    patience: int = 8,
    min_delta: float = 1e-3,
    min_epochs: int = 8,
) -> Path:
    """Train HAMD-Net.

    Stops early when validation accuracy hasn't improved by `min_delta`
    for `patience` consecutive epochs (after a `min_epochs` warm-up).
    """
    ts = app_state.training
    cfg = app_state.cfg
    ts.reset()
    ts.running = True
    ts.total_epochs = epochs

    dev = _resolve_device(device)
    ts.message = f"device {dev.type}{':' + str(dev.index) if dev.index is not None else ''}"
    _publish(ts, _snapshot(ts))

    if source == "real":
        Xw, yw = _load_real_data(cfg, max_subjects_per_class,
                                 per_subject_seconds, ts)
    else:
        ts.message = f"Generating synthetic dataset ({n_per_class} per class)"
        _publish(ts, _snapshot(ts))
        X, y, _ = generate_dataset(n_per_class=n_per_class,
                                    duration_s=duration_s,
                                    sfreq=cfg.sfreq, seed=seed)
        ts.message = "Preprocessing into windows"
        _publish(ts, _snapshot(ts))
        Xw, yw = windows_from_subjects(X, y, cfg.sfreq, cfg)

    ts.message = (f"dataset windows={Xw.shape[0]} "
                  f"shape={tuple(Xw.shape[1:])}")
    _publish(ts, _snapshot(ts))

    dataset = WindowDataset(Xw, yw)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))
    pin = dev.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=pin, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size,
                         pin_memory=pin, num_workers=0)

    counts = np.bincount(yw, minlength=len(CLASS_NAMES))
    inv = 1.0 / np.maximum(counts, 1)
    weights = inv / inv.sum() * len(CLASS_NAMES)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=dev)

    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best = 0.0
    epochs_since_best = 0
    last_epoch = 0
    DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        if ts.cancel_event.is_set():
            ts.message = "Cancelled"
            break
        ts.epoch = epoch + 1
        last_epoch = epoch + 1
        model.train()
        t_loss = t_correct = t_total = 0
        for xb, yb in train_dl:
            xb = xb.to(dev, non_blocking=pin)
            yb = yb.to(dev, non_blocking=pin)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item() * xb.size(0)
            t_correct += (logits.argmax(-1) == yb).sum().item()
            t_total += xb.size(0)
        sched.step()
        train_loss = t_loss / max(t_total, 1)
        train_acc = t_correct / max(t_total, 1)

        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(dev, non_blocking=pin)
                yb = yb.to(dev, non_blocking=pin)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                v_loss += loss.item() * xb.size(0)
                v_correct += (logits.argmax(-1) == yb).sum().item()
                v_total += xb.size(0)
        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)

        ts.train_loss.append(train_loss)
        ts.val_loss.append(val_loss)
        ts.train_acc.append(train_acc)
        ts.val_acc.append(val_acc)

        improved = val_acc > best + min_delta
        if improved:
            best = val_acc
            ts.best_val_acc = best
            epochs_since_best = 0
            torch.save({"model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "classes": CLASS_NAMES,
                        "source": source,
                        "device": dev.type},
                       DEFAULT_CHECKPOINT)
        else:
            epochs_since_best += 1

        flag = "*" if improved else f"+{epochs_since_best}"
        ts.message = (f"epoch {epoch+1}/{epochs} "
                       f"train={train_acc:.3f} val={val_acc:.3f} "
                       f"best={best:.3f} {flag} ({dev.type})")
        _publish(ts, _snapshot(ts))

        if (epoch + 1) >= min_epochs and epochs_since_best >= patience:
            ts.message = (f"early stop at epoch {epoch+1}/{epochs} — "
                          f"no val gain ≥ {min_delta} for {patience} epochs "
                          f"(best {best:.3f})")
            _publish(ts, _snapshot(ts))
            break

    ts.running = False
    ts.finished_at = time.time()
    if not ts.message.startswith("Cancelled") and not ts.message.startswith("early stop"):
        ts.message = (f"done after {last_epoch}/{epochs} epochs — "
                      f"best val acc {best:.3f} on {dev.type}")
    _publish(ts, _snapshot(ts))
    app_state.reload_engine()
    return DEFAULT_CHECKPOINT


def train_synthetic(*args, **kwargs) -> Path:
    """Backwards-compatible alias for synthetic training."""
    kwargs.setdefault("source", "synthetic")
    return train(*args, **kwargs)
