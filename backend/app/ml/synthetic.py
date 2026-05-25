"""Biologically plausible synthetic EEG generator.

Produces multi-channel EEG mimicking known signatures of each disorder, so
the platform is fully demoable without downloading multi-GB clinical
datasets. The generator is also used to bootstrap training data and to
unit-test the preprocessing + inference pipeline end-to-end.

Disorder signatures encoded (consistent with literature):
  * Healthy        : balanced spectrum, dominant alpha (8-13 Hz) at occipital
  * Alzheimer's    : slowing — increased delta/theta, reduced alpha/beta
  * Parkinson's    : reduced beta-band activity, mild theta increase
  * FTD            : frontal slowing, reduced fast activity (frontal-anterior)
  * Schizophrenia  : decreased alpha, increased gamma noise, frontal theta
"""

from __future__ import annotations

import numpy as np

from .hamd_net import CLASS_NAMES
from .preprocessing import STANDARD_19


def _band_signal(t: np.ndarray, freq_lo: float, freq_hi: float,
                 amplitude: float, rng: np.random.Generator) -> np.ndarray:
    n = len(t)
    sig = np.zeros(n)
    n_components = rng.integers(3, 7)
    for _ in range(n_components):
        f = rng.uniform(freq_lo, freq_hi)
        phase = rng.uniform(0, 2 * np.pi)
        amp = amplitude * rng.uniform(0.6, 1.4)
        sig += amp * np.sin(2 * np.pi * f * t + phase)
    return sig


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise approximation via filtering of white noise in frequency domain."""
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    return np.fft.irfft(spectrum, n=n)


def _channel_weight(channel: str, region: str) -> float:
    """Return amplitude weighting for a brain region."""
    ch = channel.lower()
    if region == "frontal":
        return 1.6 if ch.startswith(("fp", "f")) else 0.7
    if region == "central":
        return 1.5 if ch.startswith("c") or ch == "cz" else 0.7
    if region == "occipital":
        return 1.7 if ch.startswith("o") else 0.6
    if region == "temporal":
        return 1.4 if ch.startswith("t") else 0.7
    if region == "parietal":
        return 1.4 if ch.startswith("p") or ch == "pz" else 0.7
    return 1.0


def generate_subject(label: int, duration_s: float = 30.0,
                     sfreq: float = 250.0, channels: list[str] | None = None,
                     seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Generate synthetic EEG for a single subject.

    Returns (signal, metadata) where signal is (n_channels, n_samples).
    """
    rng = np.random.default_rng(seed)
    channels = channels or STANDARD_19
    n_samples = int(duration_s * sfreq)
    t = np.arange(n_samples) / sfreq
    signal = np.zeros((len(channels), n_samples))

    profile = {
        0: {  # Healthy
            "delta": (1.0, "frontal"),
            "theta": (1.2, "central"),
            "alpha": (3.0, "occipital"),
            "beta":  (1.4, "central"),
            "gamma": (0.6, "frontal"),
        },
        1: {  # Alzheimer's — slowing, alpha loss
            "delta": (2.5, "frontal"),
            "theta": (2.6, "frontal"),
            "alpha": (1.0, "occipital"),
            "beta":  (0.6, "central"),
            "gamma": (0.4, "frontal"),
        },
        2: {  # Parkinson's — reduced beta, mild theta
            "delta": (1.2, "frontal"),
            "theta": (1.8, "central"),
            "alpha": (2.4, "occipital"),
            "beta":  (0.6, "central"),
            "gamma": (0.7, "frontal"),
        },
        3: {  # FTD — frontal slowing
            "delta": (2.2, "frontal"),
            "theta": (2.4, "frontal"),
            "alpha": (1.5, "occipital"),
            "beta":  (0.7, "frontal"),
            "gamma": (0.5, "frontal"),
        },
        4: {  # Schizophrenia — alpha loss, gamma noise
            "delta": (1.3, "frontal"),
            "theta": (2.0, "frontal"),
            "alpha": (1.2, "occipital"),
            "beta":  (1.5, "central"),
            "gamma": (1.8, "frontal"),
        },
    }[label]

    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    for ci, ch in enumerate(channels):
        sig = np.zeros(n_samples)
        for band, (lo, hi) in bands.items():
            amp, region = profile[band]
            w = _channel_weight(ch, region)
            sig += w * _band_signal(t, lo, hi, amp, rng)
        sig += 0.6 * _pink_noise(n_samples, rng)
        if rng.random() < 0.05:
            ix = rng.integers(0, n_samples - 50)
            sig[ix : ix + 50] += rng.standard_normal(50) * 8.0
        signal[ci] = sig

    signal *= 1e-5  # convert into volt-like scale (tens of microvolts)
    meta = {
        "label": int(label),
        "label_name": CLASS_NAMES[label],
        "duration_s": duration_s,
        "sfreq": sfreq,
        "channels": channels,
    }
    return signal.astype(np.float32), meta


def generate_dataset(n_per_class: int = 20, duration_s: float = 30.0,
                     sfreq: float = 250.0, seed: int = 0):
    """Build a balanced synthetic dataset across all classes."""
    X, y, meta = [], [], []
    rng = np.random.default_rng(seed)
    for label in range(len(CLASS_NAMES)):
        for i in range(n_per_class):
            sig, m = generate_subject(label, duration_s=duration_s,
                                      sfreq=sfreq,
                                      seed=int(rng.integers(0, 2**31 - 1)))
            X.append(sig)
            y.append(label)
            meta.append(m)
    return np.stack(X), np.array(y, dtype=np.int64), meta
