"""Pydantic schemas for inference / dataset / training endpoints."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PipelineStep(BaseModel):
    step: str
    detail: dict


class BandPayload(BaseModel):
    absolute: list[float]
    relative: list[float]


class PSDPayload(BaseModel):
    freqs: list[float]
    psd: list[float]
    psd_per_channel: list[list[float]]


class ElectrodePosition(BaseModel):
    channel: str
    x: float
    y: float


class InferenceResponse(BaseModel):
    id: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    per_window_probs: list[list[float]]
    channels: list[str]
    sfreq: float
    duration_s: float
    n_windows: int
    inference_ms: float
    band_powers: dict[str, BandPayload]
    psd: PSDPayload
    pipeline_log: list[dict]
    channel_importance: list[float]
    temporal_attention: list[list[float]]
    electrode_positions: list[ElectrodePosition]
    raw_signal_preview: list[list[float]]
    filtered_preview: list[list[float]]
    referenced_preview: list[list[float]] = []
    normalized_preview: list[list[float]] = []
    window_preview: list[list[float]] = []
    preview_sfreq: float = 0.0
    preview_seconds: float = 0.0
    is_synthetic: bool = False
    model_trained: bool = False


class SyntheticInferenceRequest(BaseModel):
    label: int = Field(0, ge=0, le=4)
    duration_s: float = Field(30.0, gt=0, le=300)
    sfreq: float = 250.0
    seed: Optional[int] = None


class RawArrayRequest(BaseModel):
    values: list[list[float]]
    sfreq: float
    channels: Optional[list[str]] = None


class TrainStartRequest(BaseModel):
    epochs: int = 60
    batch_size: int = 32
    lr: float = 5e-4
    n_per_class: int = 32
    duration_s: float = 30.0
    seed: int = 42
    device: str = "auto"
    source: str = "real"
    max_subjects_per_class: Optional[int] = None
    per_subject_seconds: Optional[float] = 120.0
    patience: int = 8
    min_delta: float = 1e-3
    min_epochs: int = 8


class TrainStatus(BaseModel):
    running: bool
    epoch: int
    total_epochs: int
    train_loss: list[float]
    val_loss: list[float]
    train_acc: list[float]
    val_acc: list[float]
    best_val_acc: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    message: str = ""


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: str
    url: str
    subjects: int
    classes: list[str]
    license: str


class ModelInfo(BaseModel):
    name: str
    parameters: int
    input_channels: int
    input_samples: int
    sfreq: float
    classes: list[str]
    is_trained: bool
    checkpoint_path: Optional[str] = None
    architecture: list[dict]
