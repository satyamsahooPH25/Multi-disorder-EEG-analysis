"use client";

import { BAND_COLORS } from "@/lib/utils";

type Props = {
  bandPowers: Record<string, { absolute: number[]; relative: number[] }>;
  channels: string[];
  title?: string;
  mode?: "relative" | "absolute";
};

export function BandPowerChart({
  bandPowers,
  channels,
  title = "Band power per channel",
  mode = "relative",
}: Props) {
  const bands = Object.keys(bandPowers);
  const N = channels.length;
  const W = 1000;
  const H = 280;
  const margin = { left: 60, right: 16, top: 28, bottom: 32 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const groupW = innerW / Math.max(N, 1);
  const barW = (groupW / bands.length) * 0.9;

  const max = Math.max(
    1e-9,
    ...bands.flatMap((b) => bandPowers[b][mode] as number[])
  );

  const yTicks = 4;

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="flex gap-3 label-mono">
          {bands.map((b) => (
            <span key={b} className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm" style={{ background: BAND_COLORS[b] }} />
              {b}
            </span>
          ))}
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const y = margin.top + (innerH * i) / yTicks;
          const v = max - (max * i) / yTicks;
          return (
            <g key={i}>
              <line x1={margin.left} x2={W - margin.right}
                    y1={y} y2={y}
                    stroke="#1a1f2e" strokeWidth="0.5" />
              <text x={margin.left - 6} y={y + 3} textAnchor="end"
                    fontSize="10" fill="#8d94a8"
                    fontFamily="JetBrains Mono, monospace">
                {mode === "relative" ? `${(v * 100).toFixed(0)}%` : v.toExponential(0)}
              </text>
            </g>
          );
        })}
        {channels.map((ch, ci) => (
          <g key={ch}>
            {bands.map((b, bi) => {
              const v = bandPowers[b][mode][ci] ?? 0;
              const h = (v / max) * innerH;
              const x = margin.left + ci * groupW + bi * (barW / bands.length * bands.length / bands.length);
              const xx = margin.left + ci * groupW + bi * (groupW / bands.length);
              const w = (groupW / bands.length) * 0.85;
              return (
                <rect
                  key={b}
                  x={xx}
                  y={margin.top + innerH - h}
                  width={w}
                  height={h}
                  fill={BAND_COLORS[b]}
                  opacity={0.9}
                />
              );
            })}
            <text
              x={margin.left + ci * groupW + groupW / 2}
              y={H - 12}
              fontSize="9"
              fill="#8d94a8"
              textAnchor="middle"
              fontFamily="JetBrains Mono, monospace"
            >
              {ch}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
