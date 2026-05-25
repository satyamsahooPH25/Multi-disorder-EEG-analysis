"use client";

import { useEffect, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { Cpu, Play, Square } from "lucide-react";
import { api, TrainStatus } from "@/lib/api";
import { fmtPct } from "@/lib/utils";

export default function TrainPage() {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [epochs, setEpochs] = useState(80);
  const [batch, setBatch] = useState(32);
  const [lr, setLr] = useState(5e-4);
  const [n, setN] = useState(32);
  const [device, setDevice] = useState<"auto" | "cuda" | "cpu">("auto");
  const [source, setSource] = useState<"real" | "synthetic">("real");
  const [perSubject, setPerSubject] = useState(120);
  const [patience, setPatience] = useState(10);
  const [minEpochs, setMinEpochs] = useState(12);
  const [busy, setBusy] = useState(false);
  const sseRef = useRef<EventSource | null>(null);

  useEffect(() => {
    api.trainStatus().then(setStatus).catch(() => {});
    const es = new EventSource(api.trainStreamUrl());
    sseRef.current = es;
    es.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        setStatus(m);
      } catch {}
    };
    return () => es.close();
  }, []);

  async function start() {
    setBusy(true);
    try {
      await api.trainStart({
        epochs, batch_size: batch, lr,
        n_per_class: n, duration_s: 30, seed: 42,
        device, source,
        per_subject_seconds: source === "real" ? perSubject : null,
        patience, min_epochs: minEpochs,
      });
    } finally {
      setBusy(false);
    }
  }

  async function cancel() {
    await api.trainCancel();
  }

  const data = (status?.train_loss ?? []).map((tl, i) => ({
    epoch: i + 1,
    train_loss: tl,
    val_loss: status?.val_loss[i],
    train_acc: status?.train_acc[i],
    val_acc: status?.val_acc[i],
  }));

  return (
    <div className="space-y-6">
      <header>
        <div className="label-mono mb-2">Training</div>
        <h1 className="text-2xl font-semibold tracking-tight">
          HAMD-Net training dashboard
        </h1>
        <p className="text-zinc-400 mt-2 max-w-2xl">
          Trains on synthetic data or the combined real corpus —{" "}
          <code className="text-xs font-mono text-zinc-300">ds004504</code>{" "}
          (Alzheimer&apos;s · FTD · Healthy) +{" "}
          <code className="text-xs font-mono text-zinc-300">ds002778</code>{" "}
          (Parkinson&apos;s) +{" "}
          <code className="text-xs font-mono text-zinc-300">Olejarczyk-Jernajczyk</code>{" "}
          (Schizophrenia). Training stops early when validation accuracy plateaus.
          The best checkpoint is hot-loaded into the live inference engine.
        </p>
      </header>

      <section className="grid grid-cols-2 gap-4">
        <div className="surface p-4 space-y-3">
          <div className="text-sm font-medium flex items-center gap-2">
            <Cpu className="w-4 h-4 text-accent-400" /> Hyperparameters
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Select label="Source" value={source}
                    options={[["real", "ds004504 (real)"], ["synthetic", "synthetic"]]}
                    onChange={(v) => setSource(v as any)} />
            <Select label="Device" value={device}
                    options={[["auto", "auto"], ["cuda", "cuda"], ["cpu", "cpu"]]}
                    onChange={(v) => setDevice(v as any)} />
            <Field label="Epochs (max)" value={epochs} onChange={(v) => setEpochs(+v)} />
            <Field label="Batch size" value={batch} onChange={(v) => setBatch(+v)} />
            <Field label="Learning rate" value={lr} step={0.0001} onChange={(v) => setLr(+v)} />
            {source === "real" ? (
              <Field label="Seconds per subject" value={perSubject}
                     onChange={(v) => setPerSubject(+v)} />
            ) : (
              <Field label="Synthetic per class" value={n} onChange={(v) => setN(+v)} />
            )}
            <Field label="Early-stop patience" value={patience}
                   onChange={(v) => setPatience(+v)} />
            <Field label="Min epochs" value={minEpochs}
                   onChange={(v) => setMinEpochs(+v)} />
          </div>
          <div className="flex gap-2 pt-2">
            {!status?.running ? (
              <button
                onClick={start}
                disabled={busy}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-500 hover:bg-accent-600 rounded-md text-sm font-medium"
              >
                <Play className="w-3.5 h-3.5" /> Start training
              </button>
            ) : (
              <button
                onClick={cancel}
                className="flex items-center gap-1.5 px-3 py-1.5 surface surface-hover text-sm font-medium"
              >
                <Square className="w-3.5 h-3.5" /> Cancel
              </button>
            )}
          </div>
        </div>

        <div className="surface p-4">
          <div className="text-sm font-medium mb-3">Status</div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <Stat label="State" value={status?.running ? "running" : "idle"} />
            <Stat label="Epoch" value={`${status?.epoch ?? 0} / ${status?.total_epochs ?? 0}`} />
            <Stat
              label="Best val accuracy"
              value={status?.best_val_acc ? fmtPct(status.best_val_acc) : "—"}
            />
            <Stat
              label="Latest train loss"
              value={(status?.train_loss?.at(-1) ?? 0).toFixed(4)}
            />
          </div>
          <div className="mt-4 text-xs text-zinc-500 font-mono break-words">
            {status?.message || "—"}
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-4">
        <Chart title="Loss" data={data} keys={["train_loss", "val_loss"]} colors={["#5b8df5", "#f59e0b"]} />
        <Chart title="Accuracy" data={data} keys={["train_acc", "val_acc"]} colors={["#22c55e", "#39c0ed"]} />
      </section>
    </div>
  );
}

function Field({
  label, value, step = 1, onChange,
}: { label: string; value: number; step?: number; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <div className="label-mono mb-1">{label}</div>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-ink-700 border border-ink-500 rounded px-2 py-1.5 text-sm font-mono"
      />
    </label>
  );
}

function Select({
  label, value, options, onChange,
}: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <div className="label-mono mb-1">{label}</div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-ink-700 border border-ink-500 rounded px-2 py-1.5 text-sm font-mono"
      >
        {options.map(([k, v]) => <option key={k} value={k}>{v}</option>)}
      </select>
    </label>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="label-mono">{label}</div>
      <div className="font-mono text-base">{value}</div>
    </div>
  );
}

function Chart({
  title, data, keys, colors,
}: { title: string; data: any[]; keys: string[]; colors: string[] }) {
  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="label-mono">epoch</div>
      </div>
      <div style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <XAxis dataKey="epoch" tick={{ fill: "#8d94a8", fontSize: 11 }} stroke="#3a4055" />
            <YAxis tick={{ fill: "#8d94a8", fontSize: 11 }} stroke="#3a4055" />
            <Tooltip
              contentStyle={{ background: "#0a0e15", border: "1px solid #1a1f2e", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#e7eaf2" }}
            />
            <Legend wrapperStyle={{ color: "#8d94a8", fontSize: 11 }} />
            {keys.map((k, i) => (
              <Line
                key={k}
                type="monotone"
                dataKey={k}
                stroke={colors[i]}
                dot={false}
                strokeWidth={1.6}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
