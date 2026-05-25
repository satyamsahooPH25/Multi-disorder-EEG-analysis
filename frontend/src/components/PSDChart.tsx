"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceArea } from "recharts";
import { BAND_COLORS } from "@/lib/utils";

type Props = {
  freqs: number[];
  psd: number[];
  title?: string;
  height?: number;
};

const BANDS: { name: string; lo: number; hi: number }[] = [
  { name: "delta", lo: 1, hi: 4 },
  { name: "theta", lo: 4, hi: 8 },
  { name: "alpha", lo: 8, hi: 13 },
  { name: "beta", lo: 13, hi: 30 },
  { name: "gamma", lo: 30, hi: 45 },
];

export function PSDChart({
  freqs,
  psd,
  title = "Average PSD across channels",
  height = 260,
}: Props) {
  const data = freqs.map((f, i) => ({ f, p: Math.log10((psd[i] ?? 0) + 1e-12) }));

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="label-mono">log power · Hz</div>
      </div>
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
            {BANDS.map((b) => (
              <ReferenceArea
                key={b.name}
                x1={b.lo}
                x2={b.hi}
                fill={BAND_COLORS[b.name]}
                fillOpacity={0.06}
                stroke="none"
              />
            ))}
            <XAxis
              dataKey="f"
              type="number"
              domain={[0, 45]}
              tick={{ fill: "#8d94a8", fontSize: 11 }}
              stroke="#3a4055"
              label={{ value: "frequency (Hz)", position: "insideBottom", offset: -10, fill: "#8d94a8", fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: "#8d94a8", fontSize: 11 }}
              stroke="#3a4055"
              tickFormatter={(v) => v.toFixed(1)}
            />
            <Tooltip
              contentStyle={{ background: "#0a0e15", border: "1px solid #1a1f2e", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#e7eaf2" }}
              labelFormatter={(v) => `${(v as number).toFixed(2)} Hz`}
              formatter={(v: number) => v.toFixed(3)}
            />
            <Line dataKey="p" stroke="#5b8df5" dot={false} strokeWidth={1.6} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
