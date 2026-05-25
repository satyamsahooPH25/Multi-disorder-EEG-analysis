"use client";

import { useEffect, useState } from "react";
import { Brain, Info, ChevronUp } from "lucide-react";
import { api, ModelInfo } from "@/lib/api";
import { fmtNum } from "@/lib/utils";
import { MermaidDiagram } from "@/components/MermaidDiagram";
import { HAMDNetMath, HAMD_NET_MERMAID } from "@/components/HAMDNetMath";

export default function ArchitecturePage() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showMath, setShowMath] = useState(false);

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="surface p-4 text-risk-high text-sm">{err}</div>;
  if (!info) return <div className="text-zinc-500 text-sm">Loading…</div>;

  return (
    <div className="space-y-6">
      <header>
        <div className="label-mono mb-2">Model</div>
        <h1 className="text-2xl font-semibold tracking-tight flex items-center gap-2">
          <Brain className="w-6 h-6 text-accent-400" />
          {info.name}
        </h1>
        <p className="text-zinc-400 mt-2 max-w-2xl">
          A biologically-informed hybrid network that fuses spatial filtering
          (EEGNet-style), dilated temporal convolutions, bi-directional LSTM,
          and stacked multi-head self-attention. Trained end-to-end with
          class-weighted cross-entropy on aggregated 4-second windows.
        </p>
      </header>

      <section className="grid grid-cols-4 gap-4">
        <Stat label="Parameters" value={fmtNum(info.parameters)} />
        <Stat label="Input channels" value={String(info.input_channels)} />
        <Stat label="Window samples" value={String(info.input_samples)} />
        <Stat label="Sample rate" value={`${info.sfreq} Hz`} />
      </section>

      <section className="surface p-5">
        <div className="flex items-baseline justify-between mb-3">
          <div>
            <div className="text-sm font-medium">Architecture &amp; data flow</div>
            <div className="label-mono mt-0.5">
              preprocessing &rarr; spatial &rarr; TCN &rarr; bi-LSTM &rarr; attention &rarr; head
            </div>
          </div>
        </div>

        <div className="bg-ink-900/40 rounded-md p-4 border border-ink-700">
          <MermaidDiagram chart={HAMD_NET_MERMAID} />
        </div>

        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={() => setShowMath((v) => !v)}
            aria-expanded={showMath}
            title={showMath ? "Hide math" : "Show the math behind each block"}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium border transition ${
              showMath
                ? "border-accent-500 bg-accent-500/15 text-white"
                : "border-ink-500 surface-hover text-zinc-300"
            }`}
          >
            {showMath ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <Info className="w-4 h-4" />
            )}
            {showMath ? "Hide math" : "Show the math"}
          </button>
          <span className="text-xs text-zinc-500">
            Click to expand the equations and intuition behind every block.
          </span>
        </div>

        {showMath && (
          <div className="mt-5 pt-5 border-t border-ink-700">
            <div className="text-sm font-medium mb-3">
              Mathematical formulation
            </div>
            <HAMDNetMath />
          </div>
        )}
      </section>

      <section>
        <div className="text-sm font-medium mb-3">Forward pass · per-block summary</div>
        <div className="space-y-2">
          {info.architecture.map((b, i) => (
            <div key={b.name} className="surface p-4 flex items-start gap-4">
              <div className="w-10 h-10 rounded-md bg-accent-500/15 text-accent-400 flex items-center justify-center font-mono text-sm">
                {i + 1}
              </div>
              <div className="flex-1">
                <div className="font-medium">{b.name}</div>
                <div className="text-sm text-zinc-400 mt-1">{b.detail}</div>
              </div>
              <div className="text-xs font-mono text-zinc-500 mt-2">
                out = [{b.out_shape.map(String).join(", ")}]
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="surface p-5">
        <div className="text-sm font-medium mb-3">Output classes</div>
        <div className="grid grid-cols-5 gap-3">
          {info.classes.map((c) => (
            <div key={c} className="surface p-3 border-ink-500">
              <div className="font-medium">{c}</div>
              <div className="label-mono mt-1">multi-class softmax</div>
            </div>
          ))}
        </div>
      </section>

      <section className="surface p-5">
        <div className="text-sm font-medium mb-2">Checkpoint</div>
        {info.is_trained ? (
          <div className="text-sm">
            Loaded:{" "}
            <code className="text-xs font-mono text-zinc-400">
              {info.checkpoint_path}
            </code>
          </div>
        ) : (
          <div className="text-sm text-zinc-400">
            No checkpoint loaded. Run a training session under{" "}
            <a className="underline" href="/train">/train</a>.
          </div>
        )}
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
