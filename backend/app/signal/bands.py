"""Frequency-band power and spectrogram computations for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class BandPower:
    band: str
    power_per_channel: list[float]
    relative_power_per_channel: list[float]


def compute_psd(data: np.ndarray, sfreq: float, fmax: float = 45.0):
    """Welch PSD per channel.

    Returns (freqs, psd) where psd has shape (n_channels, n_freqs).
    """
    nperseg = min(int(sfreq * 2), data.shape[-1])
    freqs, psd = sp_signal.welch(data, fs=sfreq, nperseg=nperseg,
                                 noverlap=nperseg // 2, axis=-1)
    mask = freqs <= fmax
    return freqs[mask], psd[..., mask]


def band_power(data: np.ndarray, sfreq: float) -> dict[str, BandPower]:
    freqs, psd = compute_psd(data, sfreq)
    total = np.trapz(psd, freqs, axis=-1) + 1e-12
    out: dict[str, BandPower] = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        bp = np.trapz(psd[..., mask], freqs[mask], axis=-1)
        rel = bp / total
        out[name] = BandPower(
            band=name,
            power_per_channel=bp.tolist(),
            relative_power_per_channel=rel.tolist(),
        )
    return out


def spectrogram(data: np.ndarray, sfreq: float, channel_idx: int = 0,
                fmax: float = 45.0):
    """Compute spectrogram for a single channel for visualization."""
    nperseg = min(int(sfreq), data.shape[-1])
    f, t, Sxx = sp_signal.spectrogram(data[channel_idx], fs=sfreq,
                                      nperseg=nperseg,
                                      noverlap=nperseg // 2)
    mask = f <= fmax
    Sxx_db = 10 * np.log10(Sxx[mask] + 1e-12)
    return f[mask], t, Sxx_db


def topomap_data(data: np.ndarray, sfreq: float,
                 channels: Optional[list[str]] = None) -> dict:
    """Aggregate band power per channel into topomap-friendly payload."""
    powers = band_power(data, sfreq)
    return {
        "channels": channels or [],
        "bands": {
            name: {
                "power": bp.power_per_channel,
                "relative": bp.relative_power_per_channel,
            }
            for name, bp in powers.items()
        },
    }
