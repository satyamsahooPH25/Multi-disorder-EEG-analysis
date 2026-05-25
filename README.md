# Temple · CognitiveScreen

**Multi-disorder EEG analysis platform.** A research-grade end-to-end system
that classifies five neurological / psychiatric conditions —
**Healthy controls · Alzheimer's · Parkinson's · Frontotemporal dementia ·
Schizophrenia** — directly from raw scalp EEG, with full live visualisation
of every preprocessing step, every model attention head, and every
prediction.

<img width="1919" height="1079" alt="Screenshot 2026-05-25 210803" src="https://github.com/user-attachments/assets/dc4a1fff-404a-449d-b467-95c8e36f8d06" />
<img width="1919" height="1079" alt="Screenshot 2026-05-25 211015" src="https://github.com/user-attachments/assets/4711373e-1c23-4ed2-89bb-80b8ce9e4d80" />


---

## Table of contents

1. [Run · quickstart](#run--quickstart)
2. [What you get](#what-you-get)
3. [The model — HAMD-Net](#the-model--hamd-net)
4. [Datasets](#datasets)
5. [Training results](#training-results)
6. [End-to-end pipeline](#end-to-end-pipeline)
7. [Pages in the UI](#pages-in-the-ui)
8. [API surface](#api-surface)
9. [FHIR R4 export](#fhir-r4-export)
10. [Project layout](#project-layout)
11. [Troubleshooting](#troubleshooting)
12. [Disclaimer](#disclaimer)

---

## Run · quickstart

Two terminals, in this exact order. Both must stay running while you use the
platform. The trained checkpoint at
`backend/data/checkpoints/hamdnet_latest.pt` is auto-loaded on startup, so
you do **not** need to re-train to use the app.

> **Datasets are not committed to the repository** (4.58 GB of raw EEG
> exceeds GitHub's per-push budget). All three corpora are freely
> re-downloadable in one command — see step **0** below. Without the
> datasets the backend boots fine, but the live stream subject list
> comes back empty, so run the dataset setup first.

### 0 · Download the datasets  (one-time, ≈5 GB · ~10 min on a fast link)

```powershell
cd C:\Users\hp\Desktop\Karma\Temple\backend
# first run only:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install openneuro-py requests

# fetch all three EEG corpora in one go (re-runnable; cached files are skipped)
python -m scripts.setup_datasets
```

The script ([`backend/scripts/setup_datasets.py`](backend/scripts/setup_datasets.py))
pulls **OpenNeuro ds004504** (Alzheimer's + FTD + Healthy, 2.7 GB), **OpenNeuro
ds002778** (Parkinson's UC San Diego, 545 MB), and the **Olejarczyk-Jernajczyk
RepOD** schizophrenia EEG (250 MB) into `backend/data/datasets/`. Re-running
is safe — already-downloaded subjects are detected and skipped. Use
`--only ds004504` or `--skip ds002778` to fetch a subset.

After this completes you'll have 147 subjects available for live playback
under `/live`.

### 1 · Backend  (terminal A)

```powershell
cd C:\Users\hp\Desktop\Karma\Temple\backend

# first run only:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# every time:
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
# wait for:  INFO: Application startup complete.
```

### 2 · Frontend  (terminal B)

```powershell
cd C:\Users\hp\Desktop\Karma\Temple\frontend

# first run only:
npm install

# every time:
npm run dev
# wait for:  ✓ Ready in <n>s
```

### 3 · Open the UI

<http://localhost:3000>  →  pick any subject under **Live stream** →
**Start stream**.

### Optional · re-train from scratch  (after step 0, before step 1)

```powershell
cd C:\Users\hp\Desktop\Karma\Temple\backend

# (datasets already downloaded by step 0)

# train on CUDA, with early stopping (≈25 min on RTX 4050)
python -m scripts.train_real --epochs 80 --batch_size 32 --lr 5e-4 `
  --per_subject_seconds 120 --patience 10 --device cuda --source real

# evaluate the new checkpoint on every subject
python -m scripts.eval_all
```

---

## What you get

A single integrated workflow that goes from a raw EEG recording to a
clinically interpretable, FHIR-encoded report:

- **Five-class HAMD-Net classifier** (`Healthy / AD / PD / FTD / SCZ`)
  trained on a combined 147-subject corpus.
- **Live WebSocket EEG streaming** — replays real `.set` / `.bdf` / `.edf`
  recordings at native rate with per-band colour overlay.
- **Live preprocessing scope** — every prediction snapshots the rolling
  4 s window at every transformation stage (raw → bandpass → notch →
  re-reference → z-score → window), so you can see exactly what the
  network sees.

<img width="1919" height="1079" alt="Screenshot 2026-05-25 211003" src="https://github.com/user-attachments/assets/d81ab908-7956-4d96-8b80-4480bfca170e" />


- **Interpretability** — temporal attention heatmap, per-electrode
  importance, topographic band-power maps, per-window probability strip.
- **Training dashboard** — start a run from the browser, watch loss /
  accuracy stream in real-time over Server-Sent Events, with
  patience-based early stopping.


- **FHIR R4 DiagnosticReport export** with SNOMED CT codes for all five
  classes — every prediction is one HTTP call away from being ingestible
  by an HL7-compliant EHR.

---

## The model — HAMD-Net

**HAMD-Net** (Hybrid Attention Multi-Disorder Network) fuses four
biologically-aware sub-modules in sequence:

| # | Block | Role |
| --- | --- | --- |
| 1 | Spatial Conv (EEGNet-style) | Learns inter-electrode spatial filters analogous to ICA / CSP |
| 2 | Temporal Conv Net (TCN) | 4 dilated causal blocks → ~1.5 s receptive field |
| 3 | Bi-LSTM | Long-range bidirectional integration across the 4 s window |
| 4 | Multi-head Self-Attention | 2 layers · 8 heads · attention exposed for interpretability |
| 5 | MLP head | Global average pool → 5-way softmax |

Implementation: [`backend/app/ml/hamd_net.py`](backend/app/ml/hamd_net.py).

The `/architecture` page in the UI renders the full data-flow as an
interactive Mermaid diagram, with an **info button below it** that expands
into the per-block math (KaTeX-rendered equations and intuition):

<img width="1537" height="567" alt="Screenshot 2026-05-26 001830" src="https://github.com/user-attachments/assets/1ccf4a6c-0724-45a7-9953-730624220695" />

---

## Datasets

The combined real-EEG corpus used for training and live playback:

| Dataset | Subjects | Classes covered | Format / rate |
| --- | --- | --- | --- |
| [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504) | 88 | Alzheimer's · FTD · Healthy | EEGLAB `.set` · 19 ch · 500 Hz |
| [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778) | 31 | Parkinson's · Healthy | BioSemi `.bdf` · 41 ch · 512 Hz |
| [Olejarczyk-Jernajczyk RepOD](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441) | 28 | Schizophrenia · Healthy | EDF · 19 ch · 250 Hz |
| **Combined** | **147** | All 5 classes | Unified to 19-ch 10-20 montage @ 250 Hz |

All three loaders return data in the same canonical 19-channel 10-20
montage (with automatic alias resolution between T3↔T7, T4↔T8, T5↔P7,
T6↔P8) and the preprocessing pipeline resamples everything to 250 Hz.
Subject discovery and on-disk indexing live in
[`backend/app/ml/real_dataset.py`](backend/app/ml/real_dataset.py).

For a fully runnable demo without downloading anything, the platform ships
a biologically-plausible **synthetic generator**
([`backend/app/ml/synthetic.py`](backend/app/ml/synthetic.py)) that
simulates the canonical signature of each disorder
(Alzheimer's slowing, schizophrenia gamma noise, Parkinson's beta
suppression, etc.).

---

## Training results

The shipped checkpoint was trained with class-weighted cross-entropy and
early stopping (patience 10, min ε = 1 × 10⁻³). Training stopped at
**epoch 60 / 80** with peak validation accuracy **94.5 %**. Subject-level
evaluation across the full 147-subject corpus:

| Class | Correct / Total | Accuracy |
| --- | --- | --- |
| Healthy | 57 / 59 | 96.6 % |
| Alzheimer's | 36 / 36 | 100 % |
| Parkinson's | 15 / 15 | 100 % |
| FTD | 23 / 23 | 100 % |
| Schizophrenia | 14 / 14 | 100 % |
| **Overall** | **145 / 147** | **98.6 %** |

Confusion matrix (rows = ground truth, columns = predicted):

```text
          Healthy  AD   PD   FTD   SCZ
Healthy     57    1    0    0    1
AD           0   36    0    0    0
PD           0    0   15    0    0
FTD          0    0    0   23    0
SCZ          0    0    0    0   14
```

Reproduce with `python -m scripts.eval_all` after
`python -m scripts.train_real ...`.

---

## End-to-end pipeline

```text
Combined real corpus  (147 subjects · .set/.bdf/.edf · 19ch unified 10-20)
  ├─ ds004504 — Alzheimer's, FTD, Healthy   (88 · 500 Hz)
  ├─ ds002778 — Parkinson's, Healthy        (31 · 512 Hz)
  └─ Olejarczyk-Jernajczyk — Schizophrenia  (28 · 250 Hz)
   │
   ▼
[ MNE preprocessing ]
   ├─ resample → 250 Hz (all corpora normalised to one rate)
   ├─ band-pass 0.5-45 Hz IIR
   ├─ notch 50 Hz
   ├─ average reference
   ├─ z-score per channel
   └─ 4 s sliding windows (50 % overlap)
   │
   ├─────► [ Band decomposition ] ─► Butterworth band-pass for δ/θ/α/β/γ
   │       (pre-computed once per subject; streamed to the browser
   │        alongside the raw signal so the live trace can colour each
   │        band by its frequency range)
   ▼
[ HAMD-Net inference (CUDA) ]
   ├─ probabilities per window → aggregated
   ├─ attention weights extracted
   └─ spatial filter activations → channel importance
   │
   ▼
[ Outputs ]
   ├─ Live WebSocket: meta + raw chunk + 5 band chunks + 1 Hz prediction
   │   (each prediction includes a 5-stage preprocessing snapshot)
   ├─ FHIR R4 DiagnosticReport bundle
   └─ Live dashboard (band-coloured EEG · topomaps · attention · prob strip)
```

---

## Pages in the UI

| Route | What it shows |
| --- | --- |
| `/` | Overview · pipeline summary · capability cards |
| `/live` | **Unified live analysis surface** — pick any of 147 subjects across all 5 classes; stream raw + δ/θ/α/β/γ band-decomposed channels with per-band colour overlay; get HAMD-Net predictions every second; preprocessing-pipeline scope; topomaps; channel importance; rolling 2-min probability strip |
| `/architecture` | Mermaid flow diagram of HAMD-Net + an **i** button that expands the full mathematical formulation (KaTeX equations for every block, plus the training objective and early-stop criterion) |
| `/train` | Hyperparameter form (CUDA / CPU, real / synthetic source, epochs, patience, min-epochs, …); live training metrics over Server-Sent Events |
| `/research` | Dataset metadata, download instructions, sample provenance |

---

## API surface

The backend mounts every endpoint under `/api`. Auto-generated OpenAPI
docs live at <http://localhost:8765/docs>.

| Method · Path | Description |
| --- | --- |
| `GET  /api/health` | Liveness probe + checkpoint state |
| `GET  /api/model/info` | HAMD-Net architecture summary, parameter count, classes |
| `GET  /api/datasets` | Dataset metadata for the `/research` page |
| `GET  /api/streams/real` | List of all 147 subjects available for live playback |
| `WS   /api/ws/live` | Bidirectional EEG stream + 1 Hz HAMD-Net prediction |
| `POST /api/inference/synthetic` | One-shot inference on a synthetic recording |
| `POST /api/inference/raw` | Inference on a raw NumPy array supplied as JSON |
| `POST /api/inference/subject/{id}` | Inference on a full real subject |
| `GET  /api/inference/{id}` | Retrieve a prior result |
| `GET  /api/inference/{id}/fhir` | FHIR R4 `Bundle` for the prediction |
| `POST /api/train/start` | Start a training run with given hyperparameters |
| `POST /api/train/cancel` | Stop the current run |
| `GET  /api/train/status` | One-shot status snapshot |
| `GET  /api/train/stream` | Server-Sent Events stream of live training metrics |
| `GET  /api/checkpoints` | List all checkpoints on disk |

---

## FHIR R4 export

Every inference result is exposed as a FHIR `Bundle` of one
`DiagnosticReport` plus per-class `Observation` resources at
`/api/inference/<id>/fhir`. SNOMED CT codes are attached to each disorder
so the bundle can be ingested by any HL7-compliant EHR:

| Class | SNOMED CT |
| --- | --- |
| Healthy | `17621005` |
| Alzheimer's | `26929004` |
| Parkinson's | `49049000` |
| FTD | `230270009` |
| Schizophrenia | `58214004` |

Implementation: [`backend/app/api/fhir.py`](backend/app/api/fhir.py).

---

## Project layout

```text
Temple/
├── backend/                          FastAPI · PyTorch · MNE-Python
│   ├── app/
│   │   ├── ml/
│   │   │   ├── hamd_net.py           HAMD-Net architecture
│   │   │   ├── preprocessing.py      MNE pipeline + channel aliasing
│   │   │   ├── synthetic.py          Synthetic EEG generator
│   │   │   ├── real_dataset.py       3-corpus loader (.set/.bdf/.edf)
│   │   │   ├── stream_source.py      Real / synthetic WS data sources
│   │   │   ├── inference_engine.py   End-to-end inference + interpretability
│   │   │   └── trainer.py            Class-weighted CE + early stopping
│   │   ├── signal/                   PSD, band power, topomap helpers
│   │   ├── api/                      HTTP + WS routes, FHIR R4 builder
│   │   ├── core/                     Application state
│   │   └── schemas/                  Pydantic request/response models
│   ├── scripts/
│   │   ├── setup_datasets.py         ONE-SHOT downloader (run me first)
│   │   ├── train.py                  CLI training (synthetic)
│   │   ├── train_real.py             CLI training (real corpora, CUDA)
│   │   ├── eval_all.py               Subject-level evaluation
│   │   ├── download_datasets.py      OpenNeuro fetcher (used by setup_datasets)
│   │   └── download_schizophrenia.py RepOD fetcher  (used by setup_datasets)
│   └── data/
│       ├── checkpoints/              Saved model state dicts
│       ├── datasets/                 Local copies of ds004504/ds002778/SCZ
│       └── synthetic/                Cached synthetic samples
└── frontend/                          Next.js 14 · React · Tailwind
    └── src/
        ├── app/
        │   ├── page.tsx               Overview
        │   ├── live/page.tsx          Live stream + preprocessing scope
        │   ├── train/page.tsx         Training dashboard
        │   ├── architecture/page.tsx  Mermaid diagram + math panel
        │   └── research/page.tsx      Dataset list
        ├── components/
        │   ├── BandedEEGTrace.tsx     Multi-band live EEG plot
        │   ├── PreprocessingScope.tsx 5-stage transformation viewer
        │   ├── MermaidDiagram.tsx     Theme-aware Mermaid wrapper
        │   ├── HAMDNetMath.tsx        Per-block equations + Mermaid spec
        │   ├── Equation.tsx           KaTeX wrapper
        │   ├── TopoMap.tsx            10-20 topographic plot
        │   ├── ChannelImportance.tsx  Per-electrode bar chart
        │   ├── PredictionCard.tsx     Top-1 + probability bars
        │   └── …                      PSD, spectrogram, attention heatmap
        └── lib/                       Typed API client, formatters
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'app'` when starting uvicorn | Wrong working directory | Always `cd backend` before launching; the FastAPI app lives in `backend/app/`, not at the repo root |
| `[WinError 10048] only one usage of each socket address …` | Port 8765 already taken by an old backend | `Get-NetTCPConnection -LocalPort 8765 -State Listen \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` |
| `[WinError 10013] An attempt was made to access a socket in a way forbidden …` | Hyper-V / WSL2 reserved the port range | `netsh interface ipv4 show excludedportrange protocol=tcp` — if 8765 falls inside any range, pick a different port (e.g. `--port 9011`) and set `BACKEND_URL=http://127.0.0.1:9011` + `NEXT_PUBLIC_BACKEND_WS=ws://127.0.0.1:9011` for the frontend |
| `EADDRINUSE :::3000` on `npm run dev` | Old Next.js dev server still running | `Get-NetTCPConnection -LocalPort 3000 -State Listen \| ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }` (or use `npx next dev -p 3001`) |
| `keepalive ping failed … AssertionError` in backend log | Known race in uvicorn's legacy websockets handler | Cosmetic; safe to ignore. To silence: `pip install wsproto` then start uvicorn with `--ws wsproto` |
| `ModuleNotFoundError: No module named 'mne'` | Skipped `pip install -r requirements.txt` | Activate the venv and run `pip install -r requirements.txt` from inside `backend/` |

---

## Disclaimer

CognitiveScreen is **research / decision-support software**. It is **not a
clinical diagnostic device**. All predictions must be reviewed by a
qualified clinician before any action is taken. Datasets are used under
their respective open-access licences (see the `/research` page or the
upstream dataset pages for full terms).
