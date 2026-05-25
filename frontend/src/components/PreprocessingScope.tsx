"use client";

import { useMemo } from "react";

export type PreprocStage = {
  key: string;
  label: string;
  detail: string;
  color: string;
  data: number[][] | undefined;
  sfreq: number;
};

const W = 720;
const H_PER_STAGE = 56;
const PAD_X = 56;
const PAD_TOP = 8;

export function PreprocessingScope({
  stages,
  activeChannel,
  channels,
  inferenceMs,
  nWindows,
  pipelineLog,
}: {
  stages: PreprocStage[];
  activeChannel: number;
  channels: string[];
  inferenceMs?: number;
  nWindows?: number;
  pipelineLog?: Array<Record<string, unknown> & { step: string }>;
}) {
  const totalH = PAD_TOP + stages.length * H_PER_STAGE + 16;

  const tracks = useMemo(() => {
    return stages.map((s) => {
      const ch = s.data?.[activeChannel] ?? [];
      if (!ch.length) return { ...s, path: "", min: 0, max: 0, points: 0 };
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < ch.length; i++) {
        const v = ch[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const span = Math.max(max - min, 1e-9);
      const innerW = W - PAD_X - 12;
      const dx = innerW / Math.max(ch.length - 1, 1);
      let path = `M ${PAD_X} ${(H_PER_STAGE - 12) / 2 + 6}`;
      for (let i = 0; i < ch.length; i++) {
        const x = PAD_X + i * dx;
        const norm = (ch[i] - min) / span;
        const y = 6 + (H_PER_STAGE - 16) * (1 - norm);
        path += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      return { ...s, path, min, max, points: ch.length };
    });
  }, [stages, activeChannel]);

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-3">
        <div>
          <div className="text-sm font-medium">Preprocessing scope</div>
          <div className="label-mono mt-0.5">
            channel{" "}
            <span className="font-mono text-zinc-300">
              {channels[activeChannel] ?? "—"}
            </span>{" "}
            · pipeline runs once per second on the rolling 4 s window
          </div>
        </div>
        <div className="flex gap-3 text-[11px] text-zinc-500 font-mono">
          {typeof inferenceMs === "number" && (
            <span>inference {inferenceMs.toFixed(0)} ms</span>
          )}
          {typeof nWindows === "number" && <span>{nWindows} windows</span>}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${totalH}`} width="100%">
        {tracks.map((t, idx) => {
          const y0 = PAD_TOP + idx * H_PER_STAGE;
          return (
            <g key={t.key} transform={`translate(0 ${y0})`}>
              <rect x={0} y={0} width={W}
                    height={H_PER_STAGE - 4}
                    fill={idx % 2 === 0 ? "#0c1018" : "#0a0e15"}
                    rx={4} />
              <text x={6} y={16} fontSize={11}
                    fill="#e7eaf2"
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight={600}>
                {t.label}
              </text>
              <text x={6} y={30} fontSize={9}
                    fill="#8d94a8"
                    fontFamily="JetBrains Mono, monospace">
                {t.detail}
              </text>
              <text x={6} y={H_PER_STAGE - 10} fontSize={9}
                    fill="#5b647a"
                    fontFamily="JetBrains Mono, monospace">
                {t.points
                  ? `${t.points} pts @ ${t.sfreq.toFixed(0)} Hz`
                  : "no data"}
              </text>
              <line x1={PAD_X} x2={W - 12}
                    y1={(H_PER_STAGE - 4) / 2}
                    y2={(H_PER_STAGE - 4) / 2}
                    stroke="#1a1f2e" strokeWidth={0.5}
                    strokeDasharray="2 4" />
              {t.path && (
                <path d={t.path} fill="none"
                      stroke={t.color} strokeWidth={1.1}
                      vectorEffect="non-scaling-stroke" />
              )}
              {t.points > 0 && (
                <text x={W - 12} y={H_PER_STAGE - 10} fontSize={9}
                      fill="#5b647a" textAnchor="end"
                      fontFamily="JetBrains Mono, monospace">
                  range {fmt(t.min)} → {fmt(t.max)}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {pipelineLog && pipelineLog.length > 0 && (
        <div className="mt-3 grid grid-cols-3 gap-2">
          {pipelineLog.map((step, i) => (
            <div key={i}
                 className="text-[10px] font-mono text-zinc-500
                            border border-ink-700 rounded px-2 py-1">
              <span className="text-zinc-300 mr-2">{i + 1}.</span>
              <span className="text-accent-400 mr-2">{step.step}</span>
              <span className="truncate inline-block align-bottom max-w-[180px]"
                    title={JSON.stringify(step)}>
                {summarise(step)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  if (a >= 1000) return v.toExponential(1);
  return v.toFixed(2);
}

function summarise(step: Record<string, unknown> & { step: string }): string {
  switch (step.step) {
    case "load":
      return `${(step.sfreq as number)?.toFixed?.(0) ?? "?"} Hz · ${(step.duration_s as number)?.toFixed?.(1) ?? "?"} s`;
    case "resample":
      return `${(step.from as number)?.toFixed?.(0)} → ${(step.to as number)?.toFixed?.(0)} Hz`;
    case "bandpass":
      return `${step.l_freq}-${step.h_freq} Hz IIR`;
    case "notch":
      return `${step.freq} Hz`;
    case "reference":
      return `${step.type}`;
    case "normalize":
      return `${step.method}`;
    case "window":
      return `${step.n_windows}× win, ${(step as any).window_samples} smp, overlap ${(((step as any).overlap as number) * 100).toFixed(0)}%`;
    default:
      return JSON.stringify(step).slice(0, 40);
  }
}
