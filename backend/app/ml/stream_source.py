"""Source providers for the live EEG stream.

Two modes are supported:

  * `real` — loops a real recording from OpenNeuro ds004504 (eyes-closed
             resting EEG, 19 channels, 500 Hz, ~10 minutes per subject).
             The signal is the same array a clinician would see on the
             scanner — no synthesis.
  * `synthetic` — biologically-plausible generator (kept as a fallback so
             the platform remains demoable on machines that do not have
             the dataset downloaded yet).

The streamer also pre-computes per-band filtered copies of the signal so
the frontend can render each frequency band as its own coloured trace
without needing scipy in the browser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from .real_dataset import Subject, list_subjects, load_subject
from .synthetic import generate_subject

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _bandpass(sig: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * sfreq
    lo_n = max(lo / nyq, 1e-3)
    hi_n = min(hi / nyq, 0.999)
    b, a = sp_signal.butter(4, [lo_n, hi_n], btype="bandpass")
    return sp_signal.filtfilt(b, a, sig, axis=-1).astype(np.float32)


@dataclass
class StreamPayload:
    signal: np.ndarray              # (channels, samples) full series
    bands: dict[str, np.ndarray]     # band -> (channels, samples)
    channels: list[str]
    sfreq: float
    source: str
    source_path: str
    label: Optional[int]
    label_name: Optional[str]
    subject_id: Optional[str]


def _build_bands(signal: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in BANDS.items():
        out[name] = _bandpass(signal, sfreq, lo, hi)
    return out


def real_stream(subject: Subject) -> StreamPayload:
    sig, sfreq, channels = load_subject(subject)
    sig = sig.astype(np.float32) * 1e6  # scale to µV-range floats
    bands = _build_bands(sig, sfreq)
    return StreamPayload(
        signal=sig,
        bands=bands,
        channels=channels,
        sfreq=sfreq,
        source=subject.source,
        source_path=subject.relative_path(),
        label=subject.label,
        label_name=subject.label_name,
        subject_id=subject.subject_id,
    )


def synthetic_stream(label: int = 0, sfreq: float = 250.0,
                      duration_s: float = 120.0,
                      seed: Optional[int] = None) -> StreamPayload:
    sig, meta = generate_subject(label=label, duration_s=duration_s,
                                  sfreq=sfreq, seed=seed)
    sig = sig.astype(np.float32) * 1e6
    bands = _build_bands(sig, sfreq)
    return StreamPayload(
        signal=sig,
        bands=bands,
        channels=meta["channels"],
        sfreq=sfreq,
        source="synthetic",
        source_path=f"in-memory:synthetic/label={label}",
        label=meta["label"],
        label_name=meta["label_name"],
        subject_id=None,
    )


def list_real_streams() -> list[dict]:
    """Return display metadata for every real subject available locally."""
    out = []
    for s in list_subjects():
        out.append({
            "subject_id": s.subject_id,
            "label": s.label,
            "label_name": s.label_name,
            "age": s.age,
            "gender": s.gender,
            "mmse": s.mmse,
            "path": s.relative_path(),
            "source": s.source,
        })
    return out
