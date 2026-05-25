"use client";

import { useMemo } from "react";

type Props = {
  channels: string[];
  values: number[][];        // (channels, samples)
  sfreq: number;
  height?: number;
  channelHeight?: number;
  title?: string;
  showAxis?: boolean;
  highlightChannelIdx?: number;
};

export function EEGTimeSeries({
  channels,
  values,
  sfreq,
  height,
  channelHeight = 26,
  title,
  showAxis = true,
  highlightChannelIdx,
}: Props) {
  const W = 1000;
  const C = channels.length;
  const H = height ?? Math.max(C * channelHeight + 32, 200);
  const samples = values[0]?.length ?? 0;
  const duration = samples / sfreq;

  const paths = useMemo(() => {
    if (!samples || !C) return [];
    const out: string[] = [];
    for (let c = 0; c < C; c++) {
      const sig = values[c] ?? [];
      const max = Math.max(1e-9, ...sig.map((v) => Math.abs(v)));
      const yCenter = 16 + c * channelHeight + channelHeight / 2;
      const amp = (channelHeight - 4) / 2;
      const step = W / Math.max(samples - 1, 1);
      let d = "";
      for (let i = 0; i < samples; i++) {
        const x = i * step;
        const y = yCenter - (sig[i] / max) * amp;
        d += i === 0 ? `M${x.toFixed(1)} ${y.toFixed(1)}` : ` L${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      out.push(d);
    }
    return out;
  }, [values, samples, C, channelHeight]);

  const ticks = 5;
  return (
    <div className="surface p-4">
      {title && (
        <div className="flex items-baseline justify-between mb-2">
          <div className="text-sm font-medium">{title}</div>
          <div className="label-mono">
            {C} channels · {sfreq} Hz · {duration.toFixed(1)} s
          </div>
        </div>
      )}
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W + 60} ${H}`} width="100%" preserveAspectRatio="none">
          <rect x="0" y="0" width={W + 60} height={H} fill="transparent" />
          {channels.map((ch, c) => {
            const yCenter = 16 + c * channelHeight + channelHeight / 2;
            const isHighlighted = highlightChannelIdx === c;
            return (
              <g key={ch}>
                <line
                  x1={50} x2={50 + W}
                  y1={yCenter} y2={yCenter}
                  stroke="#1a1f2e" strokeDasharray="2,4" strokeWidth="0.5"
                />
                <text
                  x="44" y={yCenter + 3}
                  textAnchor="end"
                  fontSize="10"
                  fill={isHighlighted ? "#5b8df5" : "#8d94a8"}
                  fontFamily="JetBrains Mono, monospace"
                >
                  {ch}
                </text>
                <path
                  d={paths[c]}
                  transform="translate(50,0)"
                  fill="none"
                  stroke={isHighlighted ? "#5b8df5" : "#9ca3b8"}
                  strokeWidth={isHighlighted ? 1 : 0.7}
                  opacity={isHighlighted ? 1 : 0.85}
                />
              </g>
            );
          })}
          {showAxis && Array.from({ length: ticks + 1 }).map((_, i) => {
            const t = (duration * i) / ticks;
            const x = 50 + (W * i) / ticks;
            return (
              <g key={i}>
                <line x1={x} x2={x} y1={H - 18} y2={H - 14} stroke="#3a4055" />
                <text
                  x={x} y={H - 4}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#8d94a8"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {t.toFixed(1)}s
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
