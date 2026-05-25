"use client";

import { useMemo } from "react";
import { BAND_COLORS } from "@/lib/utils";

type Props = {
  channels: string[];
  raw: number[][];                          // (channels, samples)
  bands: Record<string, number[][]>;        // band -> (channels, samples)
  sfreq: number;
  channelHeight?: number;
  visibleBands?: string[];
  showRaw?: boolean;
  highlightChannelIdx?: number;
};

const ALL_BANDS = ["delta", "theta", "alpha", "beta", "gamma"];

export function BandedEEGTrace({
  channels,
  raw,
  bands,
  sfreq,
  channelHeight = 30,
  visibleBands = ALL_BANDS,
  showRaw = true,
  highlightChannelIdx,
}: Props) {
  const W = 1000;
  const C = channels.length;
  const H = Math.max(C * channelHeight + 36, 220);
  const samples = raw[0]?.length ?? 0;
  const duration = samples / sfreq;

  const traces = useMemo(() => {
    const out: { idx: number; band: string | "raw"; d: string; color: string; opacity: number; width: number }[] = [];
    if (!samples || !C) return out;

    const buildPath = (sig: number[], yCenter: number, max: number, amp: number) => {
      const step = W / Math.max(samples - 1, 1);
      let d = "";
      for (let i = 0; i < samples; i++) {
        const x = i * step;
        const y = yCenter - (sig[i] / max) * amp;
        d += i === 0 ? `M${x.toFixed(1)} ${y.toFixed(1)}` : ` L${x.toFixed(1)} ${y.toFixed(1)}`;
      }
      return d;
    };

    for (let c = 0; c < C; c++) {
      const yCenter = 18 + c * channelHeight + channelHeight / 2;
      const ampRaw = (channelHeight - 6) / 2;
      const ampBand = (channelHeight - 6) / 2.5;
      const rawSig = raw[c] ?? [];
      const rawMax = Math.max(1e-9, ...rawSig.map(Math.abs));
      if (showRaw) {
        out.push({
          idx: c,
          band: "raw",
          d: buildPath(rawSig, yCenter, rawMax, ampRaw),
          color: highlightChannelIdx === c ? "#5b8df5" : "#3a4055",
          opacity: 0.55,
          width: 0.9,
        });
      }
      for (const band of visibleBands) {
        const sig = bands[band]?.[c] ?? [];
        if (!sig.length) continue;
        const max = Math.max(1e-9, ...sig.map(Math.abs));
        out.push({
          idx: c,
          band,
          d: buildPath(sig, yCenter, max, ampBand),
          color: BAND_COLORS[band] ?? "#5b8df5",
          opacity: 0.95,
          width: 1.0,
        });
      }
    }
    return out;
  }, [raw, bands, samples, C, channelHeight, visibleBands, showRaw, highlightChannelIdx]);

  const ticks = 5;

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">Live EEG · band-decomposed</div>
        <div className="label-mono">{C} channels · {sfreq.toFixed(0)} Hz · 4 s rolling</div>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W + 60} ${H}`} width="100%" preserveAspectRatio="none">
          {channels.map((ch, c) => {
            const yCenter = 18 + c * channelHeight + channelHeight / 2;
            return (
              <g key={ch}>
                <line
                  x1={50} x2={50 + W}
                  y1={yCenter} y2={yCenter}
                  stroke="#1a1f2e"
                  strokeDasharray="2,4"
                  strokeWidth="0.5"
                />
                <text
                  x="44"
                  y={yCenter + 3}
                  textAnchor="end"
                  fontSize="10"
                  fill={highlightChannelIdx === c ? "#5b8df5" : "#8d94a8"}
                  fontFamily="JetBrains Mono, monospace"
                >
                  {ch}
                </text>
              </g>
            );
          })}
          <g transform="translate(50,0)">
            {traces.map((tr, i) => (
              <path
                key={i}
                d={tr.d}
                fill="none"
                stroke={tr.color}
                strokeWidth={tr.width}
                opacity={tr.opacity}
              />
            ))}
          </g>
          {Array.from({ length: ticks + 1 }).map((_, i) => {
            const t = (duration * i) / ticks;
            const x = 50 + (W * i) / ticks;
            return (
              <g key={i}>
                <line x1={x} x2={x} y1={H - 18} y2={H - 14} stroke="#3a4055" />
                <text
                  x={x}
                  y={H - 4}
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
      <div className="flex flex-wrap items-center gap-3 mt-2 px-2 label-mono">
        {showRaw && (
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5" style={{ background: "#3a4055" }} />
            raw
          </span>
        )}
        {visibleBands.map((b) => (
          <span key={b} className="flex items-center gap-1.5">
            <span className="w-3 h-0.5" style={{ background: BAND_COLORS[b] }} />
            {b}
          </span>
        ))}
      </div>
    </div>
  );
}
