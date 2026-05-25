"""In-memory application state — inference engine, training jobs, results."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..ml.hamd_net import HAMDConfig
from ..ml.inference_engine import InferenceEngine, InferenceResult


BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BACKEND_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "hamdnet_latest.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingState:
    running: bool = False
    epoch: int = 0
    total_epochs: int = 0
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    message: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    listeners: list[asyncio.Queue] = field(default_factory=list)

    def reset(self) -> None:
        self.epoch = 0
        self.total_epochs = 0
        self.train_loss.clear()
        self.val_loss.clear()
        self.train_acc.clear()
        self.val_acc.clear()
        self.best_val_acc = 0.0
        self.started_at = time.time()
        self.finished_at = None
        self.message = ""
        self.cancel_event.clear()


class AppState:
    def __init__(self):
        self.cfg = HAMDConfig()
        ckpt = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else None
        self.engine = InferenceEngine(checkpoint=ckpt, cfg=self.cfg)
        self.results: dict[str, InferenceResult] = {}
        self.training = TrainingState()
        self.lock = threading.Lock()

    def reload_engine(self):
        ckpt = str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else None
        self.engine = InferenceEngine(checkpoint=ckpt, cfg=self.cfg)


state = AppState()
