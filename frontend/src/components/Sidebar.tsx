"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  Brain,
  Cpu,
  Database,
  Gauge,
  LineChart,
  Radio,
} from "lucide-react";
import { cn } from "@/lib/utils";

const items = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/live", label: "Live Stream", icon: Radio },
  { href: "/train", label: "Training", icon: Cpu },
  { href: "/architecture", label: "Architecture", icon: Brain },
  { href: "/research", label: "Datasets", icon: Database },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="fixed left-0 top-0 h-screen w-60 border-r border-ink-600 bg-ink-900 flex flex-col">
      <div className="px-5 py-6 border-b border-ink-600">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-accent-400" />
          <div className="font-semibold tracking-tight">CognitiveScreen</div>
        </div>
        <div className="label-mono mt-1">Temple · v0.1</div>
      </div>
      <nav className="flex-1 px-3 py-4 space-y-1">
        {items.map((it) => {
          const active = pathname === it.href ||
            (it.href !== "/" && pathname?.startsWith(it.href));
          const Icon = it.icon;
          return (
            <Link
              key={it.href}
              href={it.href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                active
                  ? "bg-ink-700 text-white"
                  : "text-zinc-400 hover:text-white hover:bg-ink-800"
              )}
            >
              <Icon className="w-4 h-4" />
              {it.label}
            </Link>
          );
        })}
      </nav>
      <div className="px-5 py-4 border-t border-ink-600 text-[11px] text-zinc-500 leading-relaxed">
        <div className="flex items-center gap-2 mb-2">
          <LineChart className="w-3.5 h-3.5 text-accent-400" />
          <span className="text-zinc-300 font-medium">HAMD-Net</span>
        </div>
        Hybrid Attention Multi-Disorder Network<br />
        TCN · Bi-LSTM · Multi-head Attention
      </div>
    </aside>
  );
}
