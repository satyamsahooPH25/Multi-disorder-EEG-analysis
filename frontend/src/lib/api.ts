export type InferenceResponse = {
  id: string;
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  per_window_probs: number[][];
  channels: string[];
  sfreq: number;
  duration_s: number;
  n_windows: number;
  inference_ms: number;
  band_powers: Record<string, { absolute: number[]; relative: number[] }>;
  psd: { freqs: number[]; psd: number[]; psd_per_channel: number[][] };
  pipeline_log: Array<Record<string, unknown> & { step: string }>;
  channel_importance: number[];
  temporal_attention: number[][];
  electrode_positions: { channel: string; x: number; y: number }[];
  raw_signal_preview: number[][];
  filtered_preview: number[][];
  is_synthetic: boolean;
  model_trained: boolean;
};

export type ModelInfo = {
  name: string;
  parameters: number;
  input_channels: number;
  input_samples: number;
  sfreq: number;
  classes: string[];
  is_trained: boolean;
  checkpoint_path: string | null;
  architecture: { name: string; detail: string; out_shape: unknown[] }[];
};

export type DatasetInfo = {
  id: string;
  name: string;
  description: string;
  url: string;
  subjects: number;
  classes: string[];
  license: string;
};

export type TrainStatus = {
  running: boolean;
  epoch: number;
  total_epochs: number;
  train_loss: number[];
  val_loss: number[];
  train_acc: number[];
  val_acc: number[];
  best_val_acc: number;
  started_at: number | null;
  finished_at: number | null;
  message: string;
};

export type RealStreamSubject = {
  subject_id: string;
  label: number;
  label_name: string;
  age: number | null;
  gender: string | null;
  mmse: number | null;
  path: string;
  source: string;
};

const BASE = "/api";

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText}: ${text}`);
  }
  return (await r.json()) as T;
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(jsonOrThrow<{ status: string; model_trained: boolean }>),
  modelInfo: () => fetch(`${BASE}/model/info`).then(jsonOrThrow<ModelInfo>),
  datasets: () => fetch(`${BASE}/datasets`).then(jsonOrThrow<DatasetInfo[]>),
  uploadEdf: (f: File) => {
    const fd = new FormData();
    fd.append("file", f);
    return fetch(`${BASE}/inference/upload`, { method: "POST", body: fd })
      .then(jsonOrThrow<InferenceResponse>);
  },
  syntheticInference: (label: number, duration_s = 30) =>
    fetch(`${BASE}/inference/synthetic`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ label, duration_s }),
    }).then(jsonOrThrow<InferenceResponse>),
  getResult: (id: string) =>
    fetch(`${BASE}/inference/${id}`).then(jsonOrThrow<InferenceResponse>),
  fhirUrl: (id: string) => `${BASE}/inference/${id}/fhir`,
  trainStart: (body: {
    epochs: number; batch_size: number; lr: number;
    n_per_class: number; duration_s: number; seed: number;
    device?: string; source?: string;
    max_subjects_per_class?: number | null;
    per_subject_seconds?: number | null;
    patience?: number;
    min_delta?: number;
    min_epochs?: number;
  }) =>
    fetch(`${BASE}/train/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(jsonOrThrow<TrainStatus>),
  realStreams: () => fetch(`${BASE}/streams/real`).then(jsonOrThrow<RealStreamSubject[]>),
  trainStatus: () => fetch(`${BASE}/train/status`).then(jsonOrThrow<TrainStatus>),
  trainCancel: () => fetch(`${BASE}/train/cancel`, { method: "POST" }).then(jsonOrThrow),
  trainStreamUrl: () => `${BASE}/train/stream`,
  liveWebSocketUrl: () => {
    const override = process.env.NEXT_PUBLIC_BACKEND_WS;
    if (override) return `${override}/api/ws/live`;
    if (typeof window !== "undefined") {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const host = window.location.hostname;
      return `${proto}://${host}:8765/api/ws/live`;
    }
    return "ws://127.0.0.1:8765/api/ws/live";
  },
};
