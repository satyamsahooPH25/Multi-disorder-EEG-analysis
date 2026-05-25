"use client";

import { useMemo } from "react";
import { interpolateRgb } from "d3-interpolate";

type Props = {
  /** PSD per channel (rows = channels, cols = frequencies) — used as a
   *  channel-x-frequency heatmap when no time axis is available. */
  psdPerChannel: number[][];
  freqs: number[];
  channels: string[];
  title?: string;
  height?: number;
};

export function Spectrogram({
  psdPerChannel,
  freqs,
  channels,
  title = "Power spectral density (per channel)",
  height = 320,
}: Props) {
  const W = 1000;
  const H = height;
  const margin = { left: 50, right: 16, top: 28, bottom: 28 };
  const innerW = W - margin.left - margin.right;
  const innerH = H - margin.top - margin.bottom;
  const C = channels.length;
  const F = freqs.length;
  const cellW = innerW / Math.max(F, 1);
  const cellH = innerH / Math.max(C, 1);

  const log = useMemo(
    () => psdPerChannel.map((row) => row.map((v) => Math.log10(v + 1e-12))),
    [psdPerChannel]
  );

  const flat = useMemo(() => log.flat(), [log]);
  const mn = useMemo(() => Math.min(...flat), [flat]);
  const mx = useMemo(() => Math.max(...flat), [flat]);
  const lo = interpolateRgb("#06090e", "#3b6ee0");
  const hi = interpolateRgb("#3b6ee0", "#ef4444");
  const color = (v: number) => {
    const t = (v - mn) / Math.max(mx - mn, 1e-9);
    return t < 0.5 ? lo(t * 2) : hi((t - 0.5) * 2);
  };

  const xTicks = [4, 8, 13, 30, 45].filter((f) => f <= (freqs[F - 1] ?? 0));

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="label-mono">log power · dB scale</div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {log.map((row, r) =>
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
        {channels.map((ch, i) => (
          <text
            key={ch}
            x={margin.left - 6}
            y={margin.top + i * cellH + cellH / 2 + 3}
            textAnchor="end"
            fontSize="9"
            fill="#8d94a8"
            fontFamily="JetBrains Mono, monospace"
          >
            {ch}
          </text>
        ))}
        {xTicks.map((f) => {
          const idx = freqs.findIndex((x) => x >= f);
          if (idx < 0) return null;
          const x = margin.left + idx * cellW;
          return (
            <g key={f}>
              <line x1={x} x2={x}
                    y1={margin.top}
                    y2={margin.top + innerH}
                    stroke="#1a1f2e" strokeWidth="0.5" />
              <text
                x={x}
                y={H - 8}
                textAnchor="middle"
                fontSize="10"
                fill="#8d94a8"
                fontFamily="JetBrains Mono, monospace"
              >
                {f} Hz
              </text>
            </g>
          );
        })}
        <text x={margin.left} y={18} fontSize="10" fill="#8d94a8">
          channel
        </text>
        <text x={W - margin.right} y={18} fontSize="10" fill="#8d94a8" textAnchor="end">
          frequency (Hz)
        </text>
      </svg>
    </div>
  );
}
