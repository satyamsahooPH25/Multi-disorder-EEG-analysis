"""EEG preprocessing pipeline using MNE-Python.

Each step is exposed individually so the frontend can render the full
pipeline visually — raw signal -> filtered -> artifact-removed ->
re-referenced -> normalized -> windowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mne
import numpy as np

mne.set_log_level("WARNING")

STANDARD_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]

# Many modern caps replace the old "Tx" labels with new "Tx/Px" forms.
# Allow either spelling when matching channels.
CHANNEL_ALIASES = {
    "T3": ["T3", "T7"],
    "T4": ["T4", "T8"],
    "T5": ["T5", "P7"],
    "T6": ["T6", "P8"],
}


@dataclass
class PreprocConfig:
    target_sfreq: float = 250.0
    l_freq: float = 0.5
    h_freq: float = 45.0
    notch_freq: Optional[float] = 50.0
    window_seconds: float = 4.0
    overlap: float = 0.5
    n_channels: int = 19
    channels: list[str] = field(default_factory=lambda: STANDARD_19.copy())
    use_ica: bool = False
    reference: str = "average"


@dataclass
class PreprocResult:
    raw_signal: np.ndarray            # (C, T) before processing
    filtered: np.ndarray              # after band-pass + notch
    referenced: np.ndarray            # after re-reference
    normalized: np.ndarray            # z-scored per channel
    windows: np.ndarray               # (N_windows, C, T_win)
    sfreq: float
    channels: list[str]
    duration_s: float
    pipeline_log: list[dict]


def load_edf(path: str, cfg: PreprocConfig) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return raw


def _alias_lookup(target: str, available_lower: dict[str, str]) -> str | None:
    candidates = CHANNEL_ALIASES.get(target, [target])
    for cand in candidates:
        hit = available_lower.get(cand.lower())
        if hit:
            return hit
    return None


def select_channels(raw: mne.io.Raw, cfg: PreprocConfig) -> mne.io.Raw:
    """Intersect requested channels with what's available, accepting modern
    aliases (T3↔T7, T4↔T8, T5↔P7, T6↔P8). Channels are also renamed back
    to the canonical Tx form so downstream layers see a consistent montage."""
    available_lower = {ch.lower(): ch for ch in raw.ch_names}
    picks: list[str] = []
    rename: dict[str, str] = {}
    for target in cfg.channels:
        match = _alias_lookup(target, available_lower)
        if match is not None:
            picks.append(match)
            if match != target:
                rename[match] = target
    if not picks:
        eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
        picks = [raw.ch_names[i] for i in eeg_picks[: cfg.n_channels]]
    raw.pick_channels(picks)
    if rename:
        raw.rename_channels(rename)
    return raw


def preprocess_raw(raw: mne.io.Raw, cfg: PreprocConfig) -> PreprocResult:
    log: list[dict] = []
    raw_data = raw.get_data().copy()
    sfreq_in = raw.info["sfreq"]
    log.append({
        "step": "load",
        "channels": raw.ch_names,
        "sfreq": sfreq_in,
        "duration_s": raw.n_times / sfreq_in,
    })

    if abs(sfreq_in - cfg.target_sfreq) > 0.5:
        raw.resample(cfg.target_sfreq, npad="auto")
        log.append({"step": "resample", "from": sfreq_in,
                    "to": cfg.target_sfreq})

    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq,
               method="iir", verbose="ERROR")
    log.append({"step": "bandpass", "l_freq": cfg.l_freq,
                "h_freq": cfg.h_freq})

    if cfg.notch_freq is not None:
        raw.notch_filter(cfg.notch_freq, verbose="ERROR")
        log.append({"step": "notch", "freq": cfg.notch_freq})

    filtered = raw.get_data().copy()

    if cfg.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        log.append({"step": "reference", "type": "average"})
    referenced = raw.get_data().copy()

    if cfg.use_ica:
        try:
            ica = mne.preprocessing.ICA(n_components=min(15, len(raw.ch_names)),
                                        random_state=42, max_iter="auto")
            ica.fit(raw, verbose="ERROR")
            ica.apply(raw, verbose="ERROR")
            log.append({"step": "ica", "n_components": ica.n_components_})
            referenced = raw.get_data().copy()
        except Exception as e:
            log.append({"step": "ica", "skipped": True, "reason": str(e)})

    data = raw.get_data()
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-7
    normalized = (data - mean) / std
    log.append({"step": "normalize", "method": "per-channel z-score"})

    win_samples = int(cfg.window_seconds * cfg.target_sfreq)
    step = int(win_samples * (1 - cfg.overlap))
    n_total = normalized.shape[1]
    starts = list(range(0, n_total - win_samples + 1, step))
    if not starts:
        starts = [0]
    windows = np.stack([normalized[:, s : s + win_samples] for s in starts])
    log.append({"step": "window",
                "window_samples": win_samples,
                "n_windows": len(starts),
                "overlap": cfg.overlap})

    return PreprocResult(
        raw_signal=raw_data,
        filtered=filtered,
        referenced=referenced,
        normalized=normalized,
        windows=windows,
        sfreq=cfg.target_sfreq,
        channels=raw.ch_names,
        duration_s=raw.n_times / cfg.target_sfreq,
        pipeline_log=log,
    )


def preprocess_array(arr: np.ndarray, sfreq: float,
                     channel_names: Optional[list[str]] = None,
                     cfg: Optional[PreprocConfig] = None) -> PreprocResult:
    """Run preprocessing on a raw NumPy array (C, T)."""
    cfg = cfg or PreprocConfig()
    if channel_names is None:
        channel_names = STANDARD_19[: arr.shape[0]]
    info = mne.create_info(ch_names=channel_names,
                           sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(arr, info, verbose="ERROR")
    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    except Exception:
        pass
    return preprocess_raw(raw, cfg)


def preprocess_edf(path: str, cfg: Optional[PreprocConfig] = None) -> PreprocResult:
    cfg = cfg or PreprocConfig()
    raw = load_edf(path, cfg)
    raw = select_channels(raw, cfg)
    try:
        raw.set_montage("standard_1020", on_missing="ignore", verbose="ERROR")
    except Exception:
        pass
    return preprocess_raw(raw, cfg)
