"""End-to-end inference: EDF/array -> preprocessed windows -> HAMD-Net ->
multi-disorder probability report with attention-based interpretability."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .hamd_net import CLASS_NAMES, HAMDNet, HAMDConfig, build_model
from .preprocessing import PreprocConfig, PreprocResult, preprocess_array, preprocess_edf
from ..signal.bands import band_power, compute_psd
from ..signal.topomap import positions_for


@dataclass
class InferenceResult:
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float
    per_window_probs: list[list[float]]
    channel_importance: list[float]
    temporal_attention: list[list[float]]
    band_powers: dict
    psd: dict
    pipeline_log: list[dict]
    channels: list[str]
    sfreq: float
    duration_s: float
    n_windows: int
    inference_ms: float
    electrode_positions: list[dict]
    raw_signal_preview: list[list[float]] = field(default_factory=list)
    filtered_preview: list[list[float]] = field(default_factory=list)
    referenced_preview: list[list[float]] = field(default_factory=list)
    normalized_preview: list[list[float]] = field(default_factory=list)
    window_preview: list[list[float]] = field(default_factory=list)
    preview_sfreq: float = 0.0
    preview_seconds: float = 0.0


class InferenceEngine:
    def __init__(self, checkpoint: Optional[str] = None,
                 device: str = "cpu",
                 cfg: Optional[HAMDConfig] = None):
        self.device = torch.device(device)
        self.cfg = cfg or HAMDConfig()
        self.model: HAMDNet = build_model(self.cfg).to(self.device)
        self.checkpoint_path = checkpoint
        self.is_trained = False
        if checkpoint and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location=self.device)
            self.model.load_state_dict(state["model"]
                                       if "model" in state else state)
            self.is_trained = True
        self.model.eval()

    @torch.no_grad()
    def infer_array(self, arr: np.ndarray, sfreq: float,
                    channels: Optional[list[str]] = None,
                    preview_seconds: float = 4.0) -> InferenceResult:
        cfg = PreprocConfig(target_sfreq=self.cfg.sfreq,
                            n_channels=self.cfg.n_channels)
        preproc = preprocess_array(arr, sfreq, channels, cfg)
        return self._infer_from_preproc(preproc, preview_seconds)

    @torch.no_grad()
    def infer_edf(self, path: str,
                  preview_seconds: float = 4.0) -> InferenceResult:
        cfg = PreprocConfig(target_sfreq=self.cfg.sfreq,
                            n_channels=self.cfg.n_channels)
        preproc = preprocess_edf(path, cfg)
        return self._infer_from_preproc(preproc, preview_seconds)

    def _infer_from_preproc(self, preproc: PreprocResult,
                            preview_seconds: float) -> InferenceResult:
        t0 = time.time()
        windows = preproc.windows
        if windows.shape[1] != self.cfg.n_channels:
            windows = self._fit_channels(windows)
        x = torch.from_numpy(windows.astype(np.float32)).to(self.device)
        logits, info = self.model(x, return_attn=True)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        per_window_probs = probs.tolist()
        agg = probs.mean(axis=0)
        if not self.is_trained:
            agg = self._heuristic_calibration(preproc, agg)
        predicted = int(np.argmax(agg))

        # Reduce attention from (B, n_heads, T, T) → (B, T): average heads,
        # then average query positions to get per-key attention received.
        attn = info["attention"].mean(dim=1).mean(dim=1).cpu().numpy()
        temporal_attention = attn.tolist()
        spatial_act = info["spatial"].abs().mean(dim=(0, 2)).cpu().numpy()
        ch_importance = self._project_spatial_to_channels(spatial_act,
                                                          preproc.channels)

        band = band_power(preproc.normalized, preproc.sfreq)
        bands_payload = {
            name: {
                "absolute": bp.power_per_channel,
                "relative": bp.relative_power_per_channel,
            }
            for name, bp in band.items()
        }
        freqs, psd = compute_psd(preproc.normalized, preproc.sfreq)
        psd_payload = {
            "freqs": freqs.tolist(),
            "psd": psd.mean(axis=0).tolist(),
            "psd_per_channel": psd.tolist(),
        }

        # The raw_signal / filtered arrays may live at a different rate
        # (e.g. 500 Hz native) than `preproc.sfreq` (always 250 Hz post-
        # resample). Slice each one to `preview_seconds` of its own rate,
        # then downsample to a uniform 200 points so all five stages share
        # a single time axis on the wire (and the JSON payload stays well
        # under the 1 MB WS frame limit).
        n_raw = int(min(preview_seconds * preproc.sfreq,
                         preproc.raw_signal.shape[1]))
        n_preview = int(min(preview_seconds * preproc.sfreq,
                            preproc.normalized.shape[1]))

        def _down(arr: np.ndarray, n_keep: int, points: int = 200) -> list[list[float]]:
            sub = arr[:, :n_keep]
            if sub.shape[1] == 0:
                return []
            if sub.shape[1] > points:
                idx = np.linspace(0, sub.shape[1] - 1, points).astype(np.int64)
                sub = sub[:, idx]
            return np.round(sub.astype(np.float32), 4).tolist()

        raw_preview = _down(preproc.raw_signal, n_raw)
        filt_preview = _down(preproc.filtered, n_preview)
        ref_preview = _down(preproc.referenced, n_preview)
        norm_preview = _down(preproc.normalized, n_preview)
        # Last (most recent) 4-second window — what actually went into
        # the network for this prediction.
        window_preview: list[list[float]] = []
        if preproc.windows.size > 0:
            window_preview = _down(preproc.windows[-1], preproc.windows.shape[2])

        elapsed_ms = (time.time() - t0) * 1000.0
        return InferenceResult(
            probabilities={CLASS_NAMES[i]: float(agg[i])
                           for i in range(len(CLASS_NAMES))},
            predicted_class=CLASS_NAMES[predicted],
            confidence=float(agg[predicted]),
            per_window_probs=per_window_probs,
            channel_importance=ch_importance,
            temporal_attention=temporal_attention[: min(8, len(temporal_attention))],
            band_powers=bands_payload,
            psd=psd_payload,
            pipeline_log=preproc.pipeline_log,
            channels=preproc.channels,
            sfreq=preproc.sfreq,
            duration_s=preproc.duration_s,
            n_windows=int(windows.shape[0]),
            inference_ms=elapsed_ms,
            electrode_positions=positions_for(preproc.channels),
            raw_signal_preview=raw_preview,
            filtered_preview=filt_preview,
            referenced_preview=ref_preview,
            normalized_preview=norm_preview,
            window_preview=window_preview,
            preview_sfreq=preproc.sfreq,
            preview_seconds=n_preview / preproc.sfreq if preproc.sfreq else 0.0,
        )

    def _fit_channels(self, windows: np.ndarray) -> np.ndarray:
        """Pad or crop channel dimension to match model expectation."""
        n = self.cfg.n_channels
        if windows.shape[1] >= n:
            return windows[:, :n, :]
        pad = np.zeros((windows.shape[0], n - windows.shape[1],
                        windows.shape[2]), dtype=windows.dtype)
        return np.concatenate([windows, pad], axis=1)

    def _project_spatial_to_channels(self, spatial_act: np.ndarray,
                                      channels: list[str]) -> list[float]:
        """Project spatial-filter activations back to electrode space.

        The spatial conv collapses the channel dimension via grouped 19x1
        kernels, so a true projection requires the kernel weights. This
        approximation uses the magnitude of the input data filtered through
        the first conv block to give a per-electrode importance estimate.
        """
        weights = self.model.spatial.spatial.weight.detach().cpu().numpy()
        # weights shape: (n_filters, n_filters_in/groups, C, 1)
        per_ch = np.abs(weights).sum(axis=(0, 1, 3))
        per_ch = per_ch[: len(channels)]
        if per_ch.size == 0:
            return [1.0 / max(len(channels), 1)] * len(channels)
        per_ch = per_ch / (per_ch.sum() + 1e-9)
        return per_ch.tolist()

    def _heuristic_calibration(self, preproc: PreprocResult,
                               probs: np.ndarray) -> np.ndarray:
        """When no checkpoint is loaded, mix model output with a rule-based
        prior so the demo produces meaningful predictions on synthetic /
        unseen data. Once a real checkpoint is loaded this is bypassed."""
        bands = band_power(preproc.normalized, preproc.sfreq)
        rel = {name: float(np.mean(bp.relative_power_per_channel))
               for name, bp in bands.items()}
        prior = np.zeros(len(CLASS_NAMES))
        prior[0] = 0.4 * rel.get("alpha", 0.0) + 0.2 * rel.get("beta", 0.0)
        prior[1] = 0.5 * rel.get("delta", 0.0) + 0.4 * rel.get("theta", 0.0) - 0.4 * rel.get("alpha", 0.0)
        prior[2] = 0.3 * rel.get("theta", 0.0) - 0.4 * rel.get("beta", 0.0) + 0.2 * rel.get("alpha", 0.0)
        prior[3] = 0.4 * rel.get("delta", 0.0) + 0.4 * rel.get("theta", 0.0) - 0.2 * rel.get("beta", 0.0)
        prior[4] = 0.5 * rel.get("gamma", 0.0) + 0.2 * rel.get("theta", 0.0) - 0.3 * rel.get("alpha", 0.0)
        prior = prior - prior.min()
        prior = np.exp(prior * 4) / np.exp(prior * 4).sum()
        mixed = 0.4 * probs + 0.6 * prior
        return mixed / mixed.sum()
