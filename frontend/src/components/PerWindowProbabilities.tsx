"use client";

import { CLASS_COLORS } from "@/lib/utils";

type Props = {
  perWindow: number[][];
  classes: string[];
  title?: string;
  height?: number;
};

export function PerWindowProbabilities({
  perWindow,
  classes,
  title = "Per-window class probability over time",
  height = 220,
}: Props) {
  const N = perWindow.length;
  const W = 1000;
  const H = height;
  const margin = { left: 50, right: 16, top: 28, bottom: 28 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const cellW = innerW / Math.max(N, 1);

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex gap-3 label-mono">
          {classes.map((c) => (
            <span key={c} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm" style={{ background: CLASS_COLORS[c] }} />
              {c}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {[0, 0.25, 0.5, 0.75, 1].map((p) => {
          const y = margin.top + innerH * (1 - p);
          return (
            <g key={p}>
              <line x1={margin.left} x2={W - margin.right}
                    y1={y} y2={y}
                    stroke="#1a1f2e" strokeWidth="0.5" />
              <text x={margin.left - 6} y={y + 3} textAnchor="end"
                    fontSize="10" fill="#8d94a8"
                    fontFamily="JetBrains Mono, monospace">
                {(p * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {perWindow.map((probs, wi) => {
          let acc = 0;
          return (
            <g key={wi}>
              {classes.map((cls, ci) => {
                const v = probs[ci] ?? 0;
                const y0 = margin.top + innerH * (1 - acc - v);
                const h = innerH * v;
                acc += v;
                return (
                  <rect
                    key={cls}
                    x={margin.left + wi * cellW}
                    y={y0}
                    width={Math.max(cellW - 0.5, 0.5)}
                    height={h}
                    fill={CLASS_COLORS[cls] ?? "#3b6ee0"}
                    opacity={0.85}
                  />
                );
              })}
            </g>
          );
        })}
        <text x={margin.left} y={18} fontSize="10" fill="#8d94a8">probability</text>
        <text x={W - margin.right} y={18} fontSize="10" fill="#8d94a8" textAnchor="end">
          window index
        </text>
      </svg>
    </div>
  );
}
