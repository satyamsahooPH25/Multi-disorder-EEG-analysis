"use client";

import { useMemo } from "react";
import { interpolateRgb } from "d3-interpolate";

type Position = { channel: string; x: number; y: number };

type Props = {
  positions: Position[];
  values: number[];
  title?: string;
  size?: number;
  legendLabel?: string;
  colorRamp?: [string, string, string];
};

const DEFAULT_RAMP: [string, string, string] = ["#0f1420", "#3b6ee0", "#ef4444"];

export function TopoMap({
  positions,
  values,
  title,
  size = 280,
  legendLabel = "intensity",
  colorRamp = DEFAULT_RAMP,
}: Props) {
  const margin = 18;
  const r = (size - margin * 2) / 2;
  const cx = size / 2;
  const cy = size / 2;

  const max = Math.max(1e-9, ...values.map(Math.abs));
  const min = Math.min(...values);
  const interp = useMemo(() => {
    const lo = interpolateRgb(colorRamp[0], colorRamp[1]);
    const hi = interpolateRgb(colorRamp[1], colorRamp[2]);
    return (v: number) => {
      const t = (v - min) / Math.max(max - min, 1e-9);
      return t < 0.5 ? lo(t * 2) : hi((t - 0.5) * 2);
    };
  }, [min, max, colorRamp]);

  const grid = useMemo(() => {
    const N = 60;
    const cells: { gx: number; gy: number; v: number }[] = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const gx = -1 + (2 * i) / (N - 1);
        const gy = -1 + (2 * j) / (N - 1);
        if (gx * gx + gy * gy > 1) continue;
        let num = 0;
        let den = 0;
        for (let p = 0; p < positions.length; p++) {
          const dx = gx - positions[p].x;
          const dy = gy - positions[p].y;
          const d2 = dx * dx + dy * dy + 0.0008;
          const w = 1 / (d2 * d2);
          num += w * values[p];
          den += w;
        }
        cells.push({ gx, gy, v: num / den });
      }
    }
    return cells;
  }, [positions, values]);

  const cellSize = (2 * r) / 60;

  return (
    <div className="surface p-4">
      {title && (
        <div className="flex items-baseline justify-between mb-2">
          <div className="text-sm font-medium">{title}</div>
          <div className="label-mono">{legendLabel}</div>
        </div>
      )}
      <div className="flex items-center justify-center">
        <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
          <defs>
            <clipPath id="head-clip">
              <circle cx={cx} cy={cy} r={r} />
            </clipPath>
          </defs>
          <g clipPath="url(#head-clip)">
            {grid.map((c, i) => (
              <rect
                key={i}
                x={cx + c.gx * r - cellSize / 2}
                y={cy - c.gy * r - cellSize / 2}
                width={cellSize + 0.6}
                height={cellSize + 0.6}
                fill={interp(c.v)}
              />
            ))}
          </g>
          <circle
            cx={cx} cy={cy} r={r}
            fill="none" stroke="#3a4055" strokeWidth="1.5"
          />
          <path
            d={`M${cx - 8},${cy - r} q8,-10 16,0`}
            fill="none" stroke="#3a4055" strokeWidth="1.5"
          />
          <ellipse cx={cx - r} cy={cy} rx={5} ry={10}
                   fill="none" stroke="#3a4055" strokeWidth="1.2" />
          <ellipse cx={cx + r} cy={cy} rx={5} ry={10}
                   fill="none" stroke="#3a4055" strokeWidth="1.2" />
          {positions.map((p, i) => (
            <g key={p.channel}>
              <circle
                cx={cx + p.x * r}
                cy={cy - p.y * r}
                r={3.5}
                fill="#0a0e15"
                stroke="#e7eaf2"
                strokeWidth="1"
              />
              <text
                x={cx + p.x * r}
                y={cy - p.y * r - 6}
                fontSize="8.5"
                fill="#cbd1e0"
                textAnchor="middle"
                fontFamily="JetBrains Mono, monospace"
              >
                {p.channel}
              </text>
            </g>
          ))}
        </svg>
      </div>
      <div className="flex items-center justify-between mt-2 px-2 label-mono">
        <span>min</span>
        <div className="flex-1 mx-3 h-1.5 rounded"
             style={{ background: `linear-gradient(to right, ${colorRamp[0]}, ${colorRamp[1]}, ${colorRamp[2]})` }} />
        <span>max</span>
      </div>
    </div>
  );
}
