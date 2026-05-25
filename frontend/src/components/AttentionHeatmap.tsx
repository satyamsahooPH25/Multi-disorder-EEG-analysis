"use client";

import { interpolateRgb } from "d3-interpolate";

type Props = {
  attention: number[][];
  title?: string;
  height?: number;
};

export function AttentionHeatmap({
  attention,
  title = "Self-attention heatmap (window × time-step)",
  height = 240,
}: Props) {
  const N = attention.length;
  const T = attention[0]?.length ?? 0;
  const W = 1000;
  const H = height;
  const margin = { left: 50, right: 16, top: 28, bottom: 28 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const cellW = innerW / Math.max(T, 1);
  const cellH = innerH / Math.max(N, 1);

  const flat = attention.flat();
  const mn = Math.min(...flat);
  const mx = Math.max(...flat);
  const lo = interpolateRgb("#06090e", "#3b6ee0");
  const hi = interpolateRgb("#3b6ee0", "#22c55e");
  const color = (v: number) => {
    const t = (v - mn) / Math.max(mx - mn, 1e-9);
    return t < 0.5 ? lo(t * 2) : hi((t - 0.5) * 2);
  };

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="label-mono">last attention layer · head-averaged</div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {attention.map((row, r) =>
          row.map((v, c) => (
            <rect
              key={`${r}-${c}`}
              x={margin.left + c * cellW}
              y={margin.top + r * cellH}
              width={cellW + 0.5}
              height={cellH + 0.5}
              fill={color(v)}
            />
          ))
        )}
        {attention.map((_, r) => (
          <text
            key={r}
            x={margin.left - 6}
            y={margin.top + r * cellH + cellH / 2 + 3}
            textAnchor="end"
            fontSize="9"
            fill="#8d94a8"
            fontFamily="JetBrains Mono, monospace"
          >
            w{r}
          </text>
        ))}
        <text x={margin.left} y={18} fontSize="10" fill="#8d94a8">window</text>
        <text x={W - margin.right} y={18} fontSize="10" fill="#8d94a8" textAnchor="end">
          time-step
        </text>
      </svg>
    </div>
  );
}
