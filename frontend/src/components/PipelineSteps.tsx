"use client";

import { Check } from "lucide-react";

type Props = {
  log: Array<Record<string, unknown> & { step: string }>;
};

const ORDER = [
  "load",
  "resample",
  "bandpass",
  "notch",
  "reference",
  "ica",
  "normalize",
  "window",
];

export function PipelineSteps({ log }: Props) {
  const byStep = new Map<string, Record<string, unknown>>();
  for (const entry of log) byStep.set(entry.step, entry);

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-sm font-medium">Preprocessing pipeline</div>
        <div className="label-mono">MNE-Python</div>
      </div>
      <ol className="space-y-2.5">
        {ORDER.map((step) => {
          const e = byStep.get(step);
          const enabled = e && !("skipped" in e && (e as { skipped: boolean }).skipped);
          return (
            <li key={step} className="flex items-start gap-3">
              <div
                className={`mt-0.5 w-5 h-5 rounded-full border flex items-center justify-center shrink-0 ${
                  enabled
                    ? "border-accent-500 bg-accent-500/15 text-accent-400"
                    : "border-ink-500 text-zinc-600"
                }`}
              >
                {enabled ? <Check className="w-3 h-3" /> : <span className="w-1 h-1 rounded-full bg-zinc-600" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm capitalize">
                  {labelFor(step)}
                </div>
                <div className="text-xs text-zinc-500 font-mono break-all">
                  {detailFor(step, e)}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function labelFor(step: string) {
  return ({
    load: "1 · Load EDF",
    resample: "2 · Resample",
    bandpass: "3 · Band-pass filter",
    notch: "4 · Notch filter",
    reference: "5 · Re-reference",
    ica: "6 · ICA artifact removal",
    normalize: "7 · Per-channel z-score",
    window: "8 · Sliding-window split",
  } as Record<string, string>)[step] ?? step;
}

function detailFor(step: string, e?: Record<string, unknown>) {
  if (!e) return "skipped";
  if (step === "load") {
    const ch = (e.channels as string[] | undefined)?.length ?? 0;
    const f = e.sfreq as number | undefined;
    const d = e.duration_s as number | undefined;
    return `${ch} channels · ${f ?? "?"} Hz · ${(d ?? 0).toFixed(1)} s`;
  }
  if (step === "resample") return `${e.from} Hz → ${e.to} Hz`;
  if (step === "bandpass") return `${e.l_freq}–${e.h_freq} Hz IIR`;
  if (step === "notch") return `${e.freq} Hz`;
  if (step === "reference") return `${e.type}`;
  if (step === "ica") {
    if ("skipped" in e) return `skipped: ${(e as { reason: string }).reason}`;
    return `${e.n_components} components`;
  }
  if (step === "normalize") return `method: ${e.method}`;
  if (step === "window") {
    return `window=${e.window_samples} samples · n=${e.n_windows} · overlap ${(e.overlap as number) * 100}%`;
  }
  return JSON.stringify(e);
}
