"""All HTTP/WebSocket routes for the CognitiveScreen backend."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import (APIRouter, BackgroundTasks, HTTPException,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.state import CHECKPOINT_DIR, DEFAULT_CHECKPOINT, state
from ..ml.hamd_net import CLASS_NAMES
from ..ml.real_dataset import list_subjects
from ..ml.stream_source import (BANDS, list_real_streams, real_stream,
                                synthetic_stream)
from ..ml.synthetic import generate_subject
from ..ml.trainer import _publish, _snapshot, train
from ..schemas.inference import (DatasetInfo, InferenceResponse, ModelInfo,
                                 RawArrayRequest, SyntheticInferenceRequest,
                                 TrainStartRequest, TrainStatus)
from .fhir import build_diagnostic_report


router = APIRouter()


def _to_response(rid: str, result, is_synthetic: bool) -> InferenceResponse:
    return InferenceResponse(
        id=rid,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        probabilities=result.probabilities,
        per_window_probs=result.per_window_probs,
        channels=result.channels,
        sfreq=result.sfreq,
        duration_s=result.duration_s,
        n_windows=result.n_windows,
        inference_ms=result.inference_ms,
        band_powers={k: {"absolute": v["absolute"], "relative": v["relative"]}
                     for k, v in result.band_powers.items()},
        psd={"freqs": result.psd["freqs"],
             "psd": result.psd["psd"],
             "psd_per_channel": result.psd["psd_per_channel"]},
        pipeline_log=[{"step": s.get("step", "step"), **s}
                      for s in result.pipeline_log],
        channel_importance=result.channel_importance,
        temporal_attention=result.temporal_attention,
        electrode_positions=[{"channel": p["channel"], "x": p["x"], "y": p["y"]}
                             for p in result.electrode_positions],
        raw_signal_preview=result.raw_signal_preview,
        filtered_preview=result.filtered_preview,
        referenced_preview=result.referenced_preview,
        normalized_preview=result.normalized_preview,
        window_preview=result.window_preview,
        preview_sfreq=result.preview_sfreq,
        preview_seconds=result.preview_seconds,
        is_synthetic=is_synthetic,
        model_trained=state.engine.is_trained,
    )


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "model_trained": state.engine.is_trained,
        "checkpoint": str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else None,
    }


@router.get("/model/info", response_model=ModelInfo)
async def model_info():
    cfg = state.cfg
    architecture = [
        {"name": "Spatial Conv (EEGNet-style)",
         "detail": f"depthwise spatial conv across {cfg.n_channels} electrodes",
         "out_shape": [cfg.spatial_filters, "T/4"]},
        {"name": "TCN",
         "detail": f"{cfg.tcn_layers} dilated causal blocks, kernel={cfg.tcn_kernel}",
         "out_shape": [cfg.tcn_channels, "T/4"]},
        {"name": "Bi-LSTM",
         "detail": f"2 layers, hidden={cfg.lstm_hidden}, bidirectional",
         "out_shape": ["T/4", cfg.lstm_hidden * 2]},
        {"name": "Multi-head Self-Attention",
         "detail": f"2 layers, {cfg.n_heads} heads, dim={cfg.attn_dim}",
         "out_shape": ["T/4", cfg.attn_dim]},
        {"name": "Classification Head",
         "detail": f"global avg pool -> MLP -> {cfg.n_classes}-way softmax",
         "out_shape": [cfg.n_classes]},
    ]
    return ModelInfo(
        name="HAMD-Net",
        parameters=state.engine.model.num_params(),
        input_channels=cfg.n_channels,
        input_samples=cfg.n_samples,
        sfreq=cfg.sfreq,
        classes=CLASS_NAMES,
        is_trained=state.engine.is_trained,
        checkpoint_path=str(DEFAULT_CHECKPOINT)
        if DEFAULT_CHECKPOINT.exists() else None,
        architecture=architecture,
    )


@router.post("/inference/synthetic", response_model=InferenceResponse)
async def inference_synthetic(req: SyntheticInferenceRequest):
    sig, _meta = generate_subject(label=req.label, duration_s=req.duration_s,
                                  sfreq=req.sfreq, seed=req.seed)
    rid = str(uuid.uuid4())
    result = state.engine.infer_array(sig, sfreq=req.sfreq)
    state.results[rid] = result
    return _to_response(rid, result, is_synthetic=True)


@router.post("/inference/subject/{subject_id}", response_model=InferenceResponse)
async def inference_subject(subject_id: str):
    """Run a full inference pass on a real ds004504 subject (whole 10-min recording)."""
    from ..ml.real_dataset import list_subjects, load_subject
    subj = next((s for s in list_subjects() if s.subject_id == subject_id), None)
    if subj is None:
        raise HTTPException(404, f"subject {subject_id} not found")
    sig, sfreq, channels = load_subject(subj)
    rid = str(uuid.uuid4())
    result = state.engine.infer_array(sig, sfreq=sfreq, channels=channels)
    state.results[rid] = result
    return _to_response(rid, result, is_synthetic=False)


@router.post("/inference/raw", response_model=InferenceResponse)
async def inference_raw(req: RawArrayRequest):
    """Inference on a raw (channels × samples) array — used by the live page
    to snapshot the current rolling window into a persisted, FHIR-exportable
    result."""
    arr = np.asarray(req.values, dtype=np.float32)
    if arr.ndim != 2:
        raise HTTPException(400, "values must be a 2-D array (channels x samples)")
    rid = str(uuid.uuid4())
    result = state.engine.infer_array(arr, sfreq=req.sfreq, channels=req.channels)
    state.results[rid] = result
    return _to_response(rid, result, is_synthetic=False)


@router.get("/inference/{rid}", response_model=InferenceResponse)
async def get_result(rid: str):
    result = state.results.get(rid)
    if result is None:
        raise HTTPException(404, "result not found")
    return _to_response(rid, result, is_synthetic=False)


@router.get("/inference/{rid}/fhir")
async def get_fhir(rid: str, patient_id: str = "anonymous"):
    result = state.results.get(rid)
    if result is None:
        raise HTTPException(404, "result not found")
    bundle = build_diagnostic_report(result, patient_id=patient_id,
                                     report_id=rid)
    return JSONResponse(bundle)


@router.get("/datasets", response_model=list[DatasetInfo])
async def datasets():
    return [
        DatasetInfo(
            id="ds004504",
            name="OpenNeuro ds004504 — Alzheimer's & FTD EEG",
            description=(
                "Resting-state EEG from 88 subjects: Alzheimer's, "
                "Frontotemporal Dementia, and healthy controls. "
                "19-channel 10-20 montage."
            ),
            url="https://openneuro.org/datasets/ds004504",
            subjects=88,
            classes=["AD", "FTD", "Healthy"],
            license="CC0",
        ),
        DatasetInfo(
            id="ds002778",
            name="OpenNeuro ds002778 — Parkinson's UC San Diego",
            description=(
                "Resting EEG, on/off medication, 32 channels, 31 subjects: "
                "Parkinson's disease and matched controls."
            ),
            url="https://openneuro.org/datasets/ds002778",
            subjects=31,
            classes=["PD", "Healthy"],
            license="CC0",
        ),
        DatasetInfo(
            id="ds003478",
            name="OpenNeuro ds003478 — Parkinson's Iowa",
            description=(
                "Resting EEG from 28 Parkinson's patients and matched controls."
            ),
            url="https://openneuro.org/datasets/ds003478",
            subjects=28,
            classes=["PD", "Healthy"],
            license="CC0",
        ),
        DatasetInfo(
            id="kaggle-schiz",
            name="Schizophrenia EEG (Kaggle / Olejarczyk & Jernajczyk)",
            description=(
                "14 schizophrenia patients vs 14 healthy controls, "
                "19-channel resting-state EEG."
            ),
            url="https://www.kaggle.com/datasets/broach/button-tone-sz",
            subjects=28,
            classes=["SCZ", "Healthy"],
            license="CC-BY",
        ),
        DatasetInfo(
            id="tuh-eeg",
            name="TUH EEG Corpus",
            description=(
                "Largest publicly available clinical EEG corpus, "
                "30,000+ recordings used by foundation models like CLEF."
            ),
            url="https://isip.piconepress.com/projects/tuh_eeg/",
            subjects=15000,
            classes=["mixed clinical"],
            license="open access (registration required)",
        ),
    ]


@router.post("/train/start", response_model=TrainStatus)
async def train_start(req: TrainStartRequest, bg: BackgroundTasks):
    if state.training.running:
        raise HTTPException(409, "training already running")

    def _run():
        try:
            train(epochs=req.epochs,
                  batch_size=req.batch_size,
                  lr=req.lr,
                  n_per_class=req.n_per_class,
                  duration_s=req.duration_s,
                  seed=req.seed,
                  device=req.device,
                  source=req.source,
                  max_subjects_per_class=req.max_subjects_per_class,
                  per_subject_seconds=req.per_subject_seconds,
                  patience=req.patience,
                  min_delta=req.min_delta,
                  min_epochs=req.min_epochs)
        except Exception as e:
            ts = state.training
            ts.message = f"failed: {e}"
            ts.running = False
            ts.finished_at = time.time()
            _publish(ts, _snapshot(ts))

    bg.add_task(_run)
    state.training.running = True
    state.training.started_at = time.time()
    return TrainStatus(**_snapshot(state.training))


@router.post("/train/cancel")
async def train_cancel():
    state.training.cancel_event.set()
    return {"ok": True}


@router.get("/train/status", response_model=TrainStatus)
async def train_status():
    return TrainStatus(**_snapshot(state.training))


@router.get("/train/stream")
async def train_stream():
    async def gen():
        q: asyncio.Queue = asyncio.Queue()
        state.training.listeners.append(q)
        try:
            yield f"data: {json.dumps(_snapshot(state.training))}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            try:
                state.training.listeners.remove(q)
            except ValueError:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/streams/real")
async def streams_real():
    """List every real subject available for the live stream."""
    return list_real_streams()


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Stream live EEG (real recording or synthetic) with per-band channels.

    Initial JSON params:
        { "source": "real"|"synthetic",
          "subject_id": "sub-001" (optional, real),
          "label": 0..4 (optional, synthetic),
          "chunk_s": 0.2,
          "speed": 1.0 }
    """
    await ws.accept()
    try:
        params = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
    except Exception:
        params = {}
    source = params.get("source", "real")
    chunk_s = float(params.get("chunk_s", 0.2))
    speed = float(params.get("speed", 1.0))

    payload = None
    if source == "real":
        subject_id = params.get("subject_id")
        all_subjects = list_subjects()
        if not all_subjects:
            await ws.send_json({"type": "error",
                                 "message": "No real dataset available — falling back to synthetic"})
            payload = synthetic_stream(label=int(params.get("label", 0)))
        else:
            subj = next((s for s in all_subjects if s.subject_id == subject_id),
                        all_subjects[0])
            try:
                payload = real_stream(subj)
            except Exception as e:
                await ws.send_json({"type": "error",
                                    "message": f"Failed to load {subj.subject_id}: {e}"})
                payload = synthetic_stream(label=int(params.get("label", 0)))
    else:
        payload = synthetic_stream(label=int(params.get("label", 0)))

    sfreq = payload.sfreq
    chunk = max(int(chunk_s * sfreq), 1)
    n_total = payload.signal.shape[1]
    rolling_samples = int(sfreq * 4)

    await ws.send_json({
        "type": "meta",
        "source": payload.source,
        "source_path": payload.source_path,
        "subject_id": payload.subject_id,
        "label": payload.label,
        "label_name": payload.label_name,
        "channels": payload.channels,
        "sfreq": sfreq,
        "duration_s": n_total / sfreq,
        "bands": list(BANDS.keys()),
        "rolling_seconds": rolling_samples / sfreq,
    })

    rolling = np.zeros((payload.signal.shape[0], rolling_samples),
                       dtype=np.float32)
    pos = 0
    sleep_s = chunk_s / max(speed, 0.01)
    pred_every = max(int(1.0 / max(chunk_s, 0.01)), 1)
    tick = 0
    try:
        while True:
            if pos + chunk >= n_total:
                pos = 0
            piece = payload.signal[:, pos : pos + chunk]
            band_pieces = {b: payload.bands[b][:, pos : pos + chunk]
                            for b in payload.bands}
            pos += chunk
            rolling = np.concatenate([rolling[:, chunk:], piece], axis=1)

            await ws.send_json({
                "type": "chunk",
                "values": piece.tolist(),
                "bands": {b: bp.tolist() for b, bp in band_pieces.items()},
                "position_s": pos / sfreq,
            })

            tick += 1
            if tick % pred_every == 0:
                result = state.engine.infer_array(rolling, sfreq=sfreq,
                                                  channels=payload.channels)
                await ws.send_json({
                    "type": "prediction",
                    "probabilities": result.probabilities,
                    "predicted_class": result.predicted_class,
                    "confidence": result.confidence,
                    "band_powers": result.band_powers,
                    "channel_importance": result.channel_importance,
                    "temporal_attention": result.temporal_attention,
                    "psd": result.psd,
                    "electrode_positions": [
                        {"channel": p["channel"], "x": p["x"], "y": p["y"]}
                        for p in result.electrode_positions
                    ],
                    "preproc": {
                        "raw": result.raw_signal_preview,
                        "filtered": result.filtered_preview,
                        "referenced": result.referenced_preview,
                        "normalized": result.normalized_preview,
                        "window": result.window_preview,
                        "input_sfreq": float(sfreq),
                        "preview_sfreq": result.preview_sfreq,
                        "preview_seconds": result.preview_seconds,
                        "log": result.pipeline_log,
                        "channels": result.channels,
                        "n_windows": result.n_windows,
                        "inference_ms": result.inference_ms,
                    },
                })
            await asyncio.sleep(sleep_s)
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


@router.get("/checkpoints")
async def checkpoints():
    out = []
    for p in CHECKPOINT_DIR.glob("*.pt"):
        st = p.stat()
        out.append({"name": p.name, "size_bytes": st.st_size,
                    "modified": st.st_mtime})
    return out
