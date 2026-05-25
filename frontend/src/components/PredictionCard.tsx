"use client";

import { CLASS_COLORS, fmtPct } from "@/lib/utils";

type Props = {
  predictedClass: string;
  confidence: number;
  probabilities: Record<string, number>;
};

export function PredictionCard({ predictedClass, confidence, probabilities }: Props) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="surface p-5">
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-sm font-medium">Diagnostic prediction</div>
        <div className="label-mono">multi-disorder probabilities</div>
      </div>
      <div className="flex items-baseline gap-3 mt-3 mb-4">
        <div className="text-3xl font-semibold tracking-tight"
             style={{ color: CLASS_COLORS[predictedClass] ?? "#e7eaf2" }}>
          {predictedClass}
        </div>
        <div className="label-mono">confidence {fmtPct(confidence)}</div>
      </div>
      <div className="space-y-2">
        {sorted.map(([cls, p]) => (
          <div key={cls}>
            <div className="flex items-center justify-between text-xs mb-1">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm"
                      style={{ background: CLASS_COLORS[cls] ?? "#8d94a8" }} />
                <span className="text-zinc-300">{cls}</span>
              </div>
              <span className="font-mono text-zinc-400">{fmtPct(p, 2)}</span>
            </div>
            <div className="h-1.5 bg-ink-700 rounded overflow-hidden">
              <div
                className="h-full rounded"
                style={{
                  width: `${Math.max(0, Math.min(1, p)) * 100}%`,
                  background: CLASS_COLORS[cls] ?? "#3b6ee0",
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
