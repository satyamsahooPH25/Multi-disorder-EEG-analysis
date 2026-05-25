"use client";

type Props = {
  channels: string[];
  importance: number[];
  title?: string;
};

export function ChannelImportance({
  channels,
  importance,
  title = "Channel importance (spatial-filter activation)",
}: Props) {
  const max = Math.max(1e-9, ...importance);
  const sorted = channels
    .map((c, i) => ({ ch: c, v: importance[i] ?? 0 }))
    .sort((a, b) => b.v - a.v);

  return (
    <div className="surface p-4">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-sm font-medium">{title}</div>
        <div className="label-mono">normalized</div>
      </div>
      <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
        {sorted.map(({ ch, v }) => (
          <div key={ch} className="flex items-center gap-2">
            <div className="w-8 text-xs font-mono text-zinc-400">{ch}</div>
            <div className="flex-1 h-2 bg-ink-700 rounded">
              <div className="h-full rounded bg-accent-500"
                   style={{ width: `${(v / max) * 100}%` }} />
            </div>
            <div className="w-12 text-right text-[11px] font-mono text-zinc-500">
              {(v * 100).toFixed(2)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
