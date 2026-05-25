"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Square, Database } from "lucide-react";
import { api, RealStreamSubject } from "@/lib/api";
import { CLASS_NAMES, CLASS_COLORS, BAND_COLORS, fmtPct } from "@/lib/utils";
import { BandedEEGTrace } from "@/components/BandedEEGTrace";
import { TopoMap } from "@/components/TopoMap";
import { ChannelImportance } from "@/components/ChannelImportance";
import { PreprocessingScope, PreprocStage } from "@/components/PreprocessingScope";

const ALL_BANDS = ["delta", "theta", "alpha", "beta", "gamma"];

type Meta = {
  source: string;
  source_path: string;
  subject_id: string | null;
  label: number | null;
  label_name: string | null;
  channels: string[];
  sfreq: number;
  duration_s: number;
  bands: string[];
  rolling_seconds: number;
};

type Prediction = {
  probabilities: Record<string, number>;
  predicted_class: string;
  confidence: number;
  band_powers: Record<string, { absolute: number[]; relative: number[] }>;
  channel_importance: number[];
  temporal_attention: number[][];
  psd: { freqs: number[]; psd: number[]; psd_per_channel: number[][] };
  electrode_positions: { channel: string; x: number; y: number }[];
  preproc?: PreprocPayload;
};

type PreprocPayload = {
  raw: number[][];
  filtered: number[][];
  referenced: number[][];
  normalized: number[][];
  window: number[][];
  input_sfreq: number;
  preview_sfreq: number;
  preview_seconds: number;
  log: Array<Record<string, unknown> & { step: string }>;
  channels: string[];
  n_windows: number;
  inference_ms: number;
};

export default function LivePage() {
  const [subjects, setSubjects] = useState<RealStreamSubject[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [rolling, setRolling] = useState<number[][]>([]);
  const [bandRolling, setBandRolling] = useState<Record<string, number[][]>>({});
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<{ t: number; probs: Record<string, number> }[]>([]);
  const [position, setPosition] = useState(0);
  const [activeBands, setActiveBands] = useState<string[]>([...ALL_BANDS]);
  const [activeBandTopo, setActiveBandTopo] = useState("alpha");
  const [showRaw, setShowRaw] = useState(true);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [scopeChannel, setScopeChannel] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const rollingRef = useRef<number[][]>([]);
  const bandRollingRef = useRef<Record<string, number[][]>>({});

  useEffect(() => {
    api.realStreams().then((rows) => {
      setSubjects(rows);
      if (rows.length) setSelectedId(rows[0].subject_id);
    }).catch(() => {});
  }, []);

  function start() {
    if (running) return;
    setHistory([]);
    setPrediction(null);
    setMeta(null);
    setPosition(0);
    rollingRef.current = [];
    bandRollingRef.current = {};
    const ws = new WebSocket(api.liveWebSocketUrl());
    wsRef.current = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({
        source: "real",
        subject_id: selectedId,
        chunk_s: 0.2,
        speed: 1.0,
      }));
      setRunning(true);
    };
    ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data));
    ws.onclose = () => setRunning(false);
    ws.onerror = () => setRunning(false);
  }

  function stop() {
    wsRef.current?.close();
    wsRef.current = null;
    setRunning(false);
  }

  function handleMessage(m: any) {
    if (m.type === "meta") {
      setMeta(m as Meta);
    } else if (m.type === "chunk") {
      setPosition(m.position_s);
      const piece: number[][] = m.values;
      const targetLen = (meta?.rolling_seconds ?? 4) * (meta?.sfreq ?? 500);
      let cur = rollingRef.current;
      if (cur.length === 0) {
        cur = piece.map((row) => [...row]);
      } else {
        cur = cur.map((row, i) => row.concat(piece[i] ?? []));
      }
      cur = cur.map((row) => row.slice(-targetLen));
      rollingRef.current = cur;
      setRolling([...cur]);

      const bands: Record<string, number[][]> = m.bands ?? {};
      const nextBand: Record<string, number[][]> = {};
      for (const b of Object.keys(bands)) {
        const prev = bandRollingRef.current[b];
        const incoming = bands[b];
        let merged: number[][];
        if (!prev) {
          merged = incoming.map((row) => [...row]);
        } else {
          merged = prev.map((row, i) => row.concat(incoming[i] ?? []));
        }
        merged = merged.map((row) => row.slice(-targetLen));
        nextBand[b] = merged;
      }
      bandRollingRef.current = nextBand;
      setBandRolling({ ...nextBand });
    } else if (m.type === "prediction") {
      setPrediction(m);
      setHistory((h) => [...h.slice(-119), { t: Date.now(), probs: m.probabilities }]);
      setAnalysisCount((c) => c + 1);
    } else if (m.type === "error") {
      console.error("ws error", m.message);
    }
  }

  useEffect(() => () => wsRef.current?.close(), []);

  const subjectsByLabel = useMemo(() => {
    const out: Record<string, RealStreamSubject[]> = {};
    for (const s of subjects) {
      const k = s.label_name;
      out[k] = out[k] || [];
      out[k].push(s);
    }
    return out;
  }, [subjects]);

  const selected = useMemo(
    () => subjects.find((s) => s.subject_id === selectedId) ?? null,
    [subjects, selectedId]
  );

  const bandTopoValues = prediction?.band_powers[activeBandTopo]?.relative ?? [];

  return (
    <div className="space-y-6">
      <header>
        <div className="label-mono mb-2">Live analysis · real EEG</div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Real-time CognitiveScreen
        </h1>
        <p className="text-zinc-400 mt-2 max-w-2xl">
          Streams a real eyes-closed resting EEG recording from one of three
          public corpora — <code className="text-xs font-mono text-zinc-300">ds004504</code>{" "}
          (Alzheimer's / FTD / healthy),{" "}
          <code className="text-xs font-mono text-zinc-300">ds002778</code>{" "}
          (Parkinson's UC&nbsp;San&nbsp;Diego) and{" "}
          <code className="text-xs font-mono text-zinc-300">Olejarczyk-Jernajczyk</code>{" "}
          (resting schizophrenia EEG) — over WebSocket, runs HAMD-Net inference
          once per second on the rolling 4-second window, and decomposes the
          signal into the canonical delta/theta/alpha/beta/gamma bands.
        </p>
      </header>

      <section className="surface p-4">
        <div className="flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-accent-400" />
          <div className="text-sm font-medium">Subject pool</div>
          <span className="tag ml-auto">{subjects.length} subjects · 3 corpora</span>
        </div>
        <div className="grid grid-cols-5 gap-3">
          {CLASS_NAMES.map((label) => {
            const items = subjectsByLabel[label] ?? [];
            return (
              <div key={label}>
                <div className="text-xs text-zinc-500 mb-2">
                  <span className="inline-block w-2 h-2 rounded-sm mr-2"
                        style={{ background: CLASS_COLORS[label] ?? "#5b8df5" }} />
                  {label} · {items.length}
                </div>
                <div className="flex flex-wrap gap-1">
                  {items.map((s) => {
                    const active = selectedId === s.subject_id;
                    return (
                      <button
                        key={s.subject_id}
                        onClick={() => setSelectedId(s.subject_id)}
                        disabled={running}
                        title={`${s.subject_id} · ${s.source}`}
                        className={`px-2 py-1 rounded text-[11px] font-mono border transition ${
                          active
                            ? "border-accent-500 bg-accent-500/15 text-white"
                            : "border-ink-600 text-zinc-400 hover:text-white hover:border-ink-500"
                        } disabled:opacity-50`}
                      >
                        {s.subject_id.replace("sub-", "").replace("olj-", "")}
                      </button>
                    );
                  })}
                  {items.length === 0 && (
                    <span className="text-[11px] text-zinc-600 italic">no subjects</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="surface p-4 flex items-center gap-3 flex-wrap">
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          <span className="label-mono">selected</span>
          <span className="font-mono text-zinc-200">
            {selected?.subject_id ?? "—"}
          </span>
          {selected && (
            <span className="font-mono text-zinc-500">
              · {selected.label_name} · {selected.source}
              {selected.gender ? ` · ${selected.gender}` : ""}
              {selected.age ? ` · age ${selected.age}` : ""}
              {typeof selected.mmse === "number" ? ` · MMSE ${selected.mmse}` : ""}
            </span>
          )}
        </div>
        {selected && (
          <span className="tag font-mono normal-case tracking-normal text-[10px] truncate max-w-md"
                title={selected.path}>
            {selected.path}
          </span>
        )}
        <div className="flex-1" />
        {meta && running && (
          <span className="label-mono">
            position {position.toFixed(1)} / {meta.duration_s.toFixed(0)} s
          </span>
        )}
        {!running ? (
          <button
            onClick={start}
            disabled={!selectedId}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-500 hover:bg-accent-600 disabled:opacity-50 rounded-md text-sm font-medium"
          >
            <Play className="w-3.5 h-3.5" /> Start stream
          </button>
        ) : (
          <button
            onClick={stop}
            className="flex items-center gap-1.5 px-3 py-1.5 surface surface-hover text-sm font-medium"
          >
            <Square className="w-3.5 h-3.5" /> Stop
          </button>
        )}
        {running && <span className="flex items-center gap-2 text-xs text-zinc-300"><span className="live-dot" /> live</span>}
      </section>

      <section className="surface p-3 flex items-center gap-2 flex-wrap">
        <span className="label-mono mr-2">show bands</span>
        <button
          onClick={() => setShowRaw((v) => !v)}
          className={`px-2.5 py-1 rounded text-xs border transition ${
            showRaw
              ? "border-zinc-300 text-white bg-zinc-300/10"
              : "border-ink-600 text-zinc-400"
          }`}
        >
          <span className="inline-block w-2 h-2 rounded-sm mr-2" style={{ background: "#cbd1e0" }} />
          raw
        </button>
        {ALL_BANDS.map((b) => {
          const active = activeBands.includes(b);
          return (
            <button
              key={b}
              onClick={() =>
                setActiveBands((cur) =>
                  cur.includes(b) ? cur.filter((x) => x !== b) : [...cur, b]
                )
              }
              className={`px-2.5 py-1 rounded text-xs border transition ${
                active
                  ? "border-accent-500 bg-accent-500/15 text-white"
                  : "border-ink-600 text-zinc-400 hover:text-white"
              }`}
            >
              <span className="inline-block w-2 h-2 rounded-sm mr-2"
                    style={{ background: BAND_COLORS[b] }} />
              {b}
            </button>
          );
        })}
      </section>

      <section className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          {meta ? (
            <BandedEEGTrace
              channels={meta.channels}
              raw={rolling}
              bands={bandRolling}
              sfreq={meta.sfreq}
              visibleBands={activeBands}
              showRaw={showRaw}
            />
          ) : (
            <div className="surface p-10 text-center text-zinc-500 text-sm">
              {running ? "buffering…" : "select a subject and press Start"}
            </div>
          )}
        </div>
        <div className="space-y-4">
          <div className="surface p-5">
            <div className="text-sm font-medium mb-2">Live prediction</div>
            {prediction ? (
              <>
                <div
                  className="text-2xl font-semibold tracking-tight mb-1"
                  style={{ color: CLASS_COLORS[prediction.predicted_class] }}
                >
                  {prediction.predicted_class}
                </div>
                <div className="label-mono mb-3">
                  confidence {fmtPct(prediction.confidence)} · analyses {analysisCount}
                </div>
                <div className="space-y-2">
                  {Object.entries(prediction.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([cls, p]) => (
                      <div key={cls}>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span>{cls}</span>
                          <span className="font-mono text-zinc-500">{fmtPct(p, 1)}</span>
                        </div>
                        <div className="h-1 bg-ink-700 rounded">
                          <div
                            className="h-full rounded"
                            style={{ width: `${p * 100}%`, background: CLASS_COLORS[cls] }}
                          />
                        </div>
                      </div>
                    ))}
                </div>
                <div className="text-[11px] text-zinc-500 mt-3">
                  {selected && (
                    <>ground truth label · <span className="font-mono">{selected.label_name}</span></>
                  )}
                </div>
              </>
            ) : (
              <div className="text-zinc-500 text-sm">first prediction arrives after ~1 s</div>
            )}
          </div>
          <div className="surface p-4">
            <div className="text-sm font-medium mb-2">Band power (rolling)</div>
            {prediction ? (
              <div className="space-y-1.5">
                {ALL_BANDS.map((b) => {
                  const v = prediction.band_powers[b];
                  if (!v) return null;
                  const mean = v.relative.reduce((a, c) => a + c, 0) / Math.max(v.relative.length, 1);
                  return (
                    <div key={b} className="flex items-center gap-2 text-xs">
                      <span className="w-3 h-2 rounded-sm shrink-0"
                            style={{ background: BAND_COLORS[b] }} />
                      <span className="w-12 text-zinc-400 capitalize">{b}</span>
                      <div className="flex-1 h-1.5 bg-ink-700 rounded">
                        <div className="h-full rounded"
                             style={{ width: `${mean * 200}%`, background: BAND_COLORS[b] }} />
                      </div>
                      <span className="font-mono text-zinc-500 w-12 text-right">
                        {(mean * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-zinc-500 text-xs">awaiting first window…</div>
            )}
          </div>
        </div>
      </section>

      {prediction?.preproc && (
        <section>
          <div className="flex items-baseline justify-between mb-3">
            <div>
              <h2 className="text-lg font-semibold">Preprocessing pipeline</h2>
              <div className="text-xs text-zinc-500 mt-0.5">
                Each prediction snapshots the rolling 4&nbsp;s window at every
                preprocessing stage so you can see exactly what the network sees.
              </div>
            </div>
            <div className="flex gap-1 flex-wrap max-w-md justify-end">
              {(prediction.preproc.channels.length
                ? prediction.preproc.channels
                : meta?.channels ?? []
              ).map((ch, i) => {
                const active = scopeChannel === i;
                return (
                  <button
                    key={ch}
                    onClick={() => setScopeChannel(i)}
                    className={`px-1.5 py-0.5 rounded text-[10px] font-mono border transition ${
                      active
                        ? "border-accent-500 bg-accent-500/15 text-white"
                        : "border-ink-700 text-zinc-500 hover:text-white"
                    }`}
                  >
                    {ch}
                  </button>
                );
              })}
            </div>
          </div>
          <PreprocessingScope
            stages={buildPreprocStages(prediction.preproc)}
            activeChannel={scopeChannel}
            channels={prediction.preproc.channels}
            inferenceMs={prediction.preproc.inference_ms}
            nWindows={prediction.preproc.n_windows}
            pipelineLog={prediction.preproc.log}
          />
        </section>
      )}

      {prediction && (
        <section>
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-lg font-semibold">Topographic distribution</h2>
            <div className="flex gap-2">
              {ALL_BANDS.map((b) => (
                <button
                  key={b}
                  onClick={() => setActiveBandTopo(b)}
                  className={`px-3 py-1.5 rounded-md text-xs border transition ${
                    activeBandTopo === b
                      ? "border-accent-500 text-white bg-accent-500/15"
                      : "border-ink-600 text-zinc-400 hover:text-white hover:border-ink-500"
                  }`}
                >
                  <span className="inline-block w-2 h-2 rounded-sm mr-2"
                        style={{ background: BAND_COLORS[b] }} />
                  {b}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <TopoMap
              positions={prediction.electrode_positions}
              values={bandTopoValues}
              title={`${activeBandTopo} band — relative power`}
              legendLabel={activeBandTopo}
            />
            <TopoMap
              positions={prediction.electrode_positions}
              values={prediction.channel_importance}
              title="Model channel importance"
              legendLabel="attention"
            />
            <ChannelImportance
              channels={meta?.channels ?? []}
              importance={prediction.channel_importance}
            />
          </div>
        </section>
      )}

      <section>
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-lg font-semibold">Probability strip · last ~2 minutes</h2>
          <div className="label-mono">prediction every 1 s</div>
        </div>
        <ProbabilityStrip history={history} />
      </section>
    </div>
  );
}

function buildPreprocStages(p: PreprocPayload): PreprocStage[] {
  return [
    {
      key: "raw",
      label: "1 · Raw",
      detail: "as captured from the recording",
      color: "#cbd1e0",
      data: p.raw,
      sfreq: p.input_sfreq,
    },
    {
      key: "filtered",
      label: "2 · Band-pass + notch",
      detail: "0.5–45 Hz IIR · 50 Hz notch",
      color: "#5b8df5",
      data: p.filtered,
      sfreq: p.preview_sfreq,
    },
    {
      key: "referenced",
      label: "3 · Average reference",
      detail: "common-average re-referencing",
      color: "#22c55e",
      data: p.referenced,
      sfreq: p.preview_sfreq,
    },
    {
      key: "normalized",
      label: "4 · Z-score",
      detail: "per-channel mean/std normalisation",
      color: "#f59e0b",
      data: p.normalized,
      sfreq: p.preview_sfreq,
    },
    {
      key: "window",
      label: "5 · Window → HAMD-Net",
      detail: "4 s × 19 ch tensor fed to the model",
      color: "#39c0ed",
      data: p.window,
      sfreq: p.preview_sfreq,
    },
  ];
}

function ProbabilityStrip({
  history,
}: { history: { t: number; probs: Record<string, number> }[] }) {
  const W = 1000;
  const H = 140;
  const margin = { left: 50, right: 16, top: 16, bottom: 24 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const N = Math.max(history.length, 1);
  const cellW = innerW / N;

  return (
    <div className="surface p-4">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {[0, 0.5, 1].map((p) => {
          const y = margin.top + innerH * (1 - p);
          return (
            <g key={p}>
              <line x1={margin.left} x2={W - margin.right}
                    y1={y} y2={y}
                    stroke="#1a1f2e" strokeWidth="0.5" />
              <text x={margin.left - 6} y={y + 3}
                    textAnchor="end" fontSize="10" fill="#8d94a8"
                    fontFamily="JetBrains Mono, monospace">
                {(p * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {history.map((step, wi) => {
          let acc = 0;
          return (
            <g key={wi}>
              {CLASS_NAMES.map((cls) => {
                const v = step.probs[cls] ?? 0;
                const y0 = margin.top + innerH * (1 - acc - v);
                const h = innerH * v;
                acc += v;
                return (
                  <rect
                    key={cls}
                    x={margin.left + wi * cellW}
                    y={y0}
                    width={Math.max(cellW + 0.5, 0.5)}
                    height={h}
                    fill={CLASS_COLORS[cls]}
                    opacity={0.85}
                  />
                );
              })}
            </g>
          );
        })}
        <text x={margin.left} y={H - 6} fontSize="10" fill="#8d94a8">past</text>
        <text x={W - margin.right} y={H - 6} fontSize="10" fill="#8d94a8" textAnchor="end">now</text>
      </svg>
      <div className="flex flex-wrap gap-3 mt-2 px-2 label-mono">
        {CLASS_NAMES.map((c) => (
          <span key={c} className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-sm" style={{ background: CLASS_COLORS[c] }} />
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}
