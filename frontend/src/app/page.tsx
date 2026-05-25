"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Activity, Brain, Cpu, Database, Radio, Zap } from "lucide-react";
import { api, ModelInfo } from "@/lib/api";
import { fmtNum } from "@/lib/utils";

export default function HomePage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="space-y-8">
      <header>
        <div className="label-mono mb-2">Temple · CognitiveScreen</div>
        <h1 className="text-3xl font-semibold tracking-tight">
          Multi-disorder EEG analysis platform
        </h1>
        <p className="text-zinc-400 mt-2 max-w-2xl">
          A live HAMD-Net hybrid attention model that streams real clinical
          EEG from <code className="text-xs font-mono text-zinc-300">OpenNeuro
          ds004504</code> over WebSocket and continuously infers probability
          scores across Alzheimer&apos;s, Parkinson&apos;s, FTD, schizophrenia
          and healthy controls — with band-decomposed signals, attention
          heatmaps, and a FHIR-compatible diagnostic report.
        </p>
      </header>

      {err && (
        <div className="surface p-4 border-risk-high text-risk-high text-sm">
          Backend not reachable: {err}
        </div>
      )}

      <section className="grid grid-cols-4 gap-4">
        <Stat label="Disorders detected" value={info ? String(info.classes.length) : "5"} />
        <Stat label="Model parameters" value={info ? fmtNum(info.parameters) : "—"} />
        <Stat label="Input channels" value={info ? String(info.input_channels) : "19"} />
        <Stat label="Sampling rate" value={info ? `${info.sfreq} Hz` : "250 Hz"} />
      </section>

      <section className="grid grid-cols-2 gap-4">
        <ActionCard
          href="/live" icon={Radio}
          title="Live analysis"
          desc="Stream a real ds004504 subject in real time and watch HAMD-Net produce predictions, topomaps, and attention every second."
        />
        <ActionCard
          href="/train" icon={Cpu}
          title="Train HAMD-Net"
          desc="Run a CUDA training session against ds004504 or the synthetic dataset and watch metrics update over Server-Sent Events."
        />
        <ActionCard
          href="/architecture" icon={Brain}
          title="Model architecture"
          desc="Inspect every block of the network — spatial conv, TCN, Bi-LSTM, multi-head attention — with parameter counts and tensor shapes."
        />
        <ActionCard
          href="/research" icon={Database}
          title="Open-source datasets"
          desc="OpenNeuro AD/FTD, Parkinson's UCSD/Iowa, Schizophrenia Kaggle, TUH EEG corpus."
        />
      </section>

      <section className="surface p-5">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-accent-400" />
          <div className="text-sm font-medium">Pipeline</div>
        </div>
        <ol className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs text-zinc-400">
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> Real EEG ingest from OpenNeuro ds004504 (.set)</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> Resample → band-pass 0.5-45 Hz → notch 50 Hz</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> Average reference, optional ICA artifact removal</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> Per-channel z-score · 4 s sliding windows</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> Backend pre-computes 5 band-pass filtered streams</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> WebSocket pushes raw + per-band chunks to UI</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> HAMD-Net inference + attention extraction · 1 Hz</li>
          <li className="flex gap-2"><Zap className="w-3 h-3 mt-0.5 text-accent-400" /> Aggregated report → FHIR R4 bundle export</li>
        </ol>
      </section>

      <section className="surface p-5">
        <div className="text-sm font-medium mb-3">What this platform demonstrates</div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm text-zinc-400">
          <Bullet>Hybrid deep architecture: spatial CNN + dilated TCN + Bi-LSTM + multi-head self-attention.</Bullet>
          <Bullet>Five-class neurological disorder discrimination from raw scalp EEG.</Bullet>
          <Bullet>Open-source dataset integration · CUDA-accelerated training.</Bullet>
          <Bullet>End-to-end interpretability — channel importance, temporal attention, band power.</Bullet>
          <Bullet>Standards compliance — FHIR R4 DiagnosticReport bundle for EHR ingestion.</Bullet>
          <Bullet>Live bedside-style streaming with band-decomposed colour overlay.</Bullet>
        </div>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="surface p-4">
      <div className="label-mono mb-1">{label}</div>
      <div className="text-2xl font-semibold tracking-tight">{value}</div>
    </div>
  );
}

function ActionCard({
  href, icon: Icon, title, desc,
}: { href: string; icon: React.ComponentType<{ className?: string }>; title: string; desc: string }) {
  return (
    <Link href={href} className="surface surface-hover p-5 transition-colors">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-accent-400" />
        <div className="font-medium">{title}</div>
      </div>
      <p className="text-sm text-zinc-400 leading-relaxed">{desc}</p>
    </Link>
  );
}

function Bullet({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="mt-1.5 w-1 h-1 rounded-full bg-accent-400 shrink-0" />
      <span>{children}</span>
    </div>
  );
}
