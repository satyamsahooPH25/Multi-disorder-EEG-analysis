"use client";

import { useEffect, useState } from "react";
import { ExternalLink, Database } from "lucide-react";
import { api, DatasetInfo } from "@/lib/api";

export default function ResearchPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.datasets().then(setDatasets).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="surface p-4 text-risk-high text-sm">{err}</div>;

  return (
    <div className="space-y-6">
      <header>
        <div className="label-mono mb-2">Open-source datasets</div>
        <h1 className="text-2xl font-semibold tracking-tight">
          Training & evaluation corpora
        </h1>
        <p className="text-zinc-400 mt-2 max-w-2xl">
          The HAMD-Net architecture is designed to be trained on the
          combination of these public EEG datasets covering Alzheimer&apos;s,
          frontotemporal dementia, Parkinson&apos;s, schizophrenia, and large
          general clinical corpora used by foundation models like CLEF.
        </p>
      </header>

      <section className="grid grid-cols-1 gap-3">
        {datasets.map((d) => (
          <div key={d.id} className="surface p-5 flex gap-4 items-start">
            <div className="w-10 h-10 rounded-md bg-accent-500/15 text-accent-400 flex items-center justify-center shrink-0">
              <Database className="w-4 h-4" />
            </div>
            <div className="flex-1">
              <div className="flex items-baseline justify-between gap-3">
                <div className="font-medium">{d.name}</div>
                <a
                  href={d.url}
                  target="_blank"
                  className="flex items-center gap-1 text-xs text-accent-400 hover:underline"
                >
                  source <ExternalLink className="w-3 h-3" />
                </a>
              </div>
              <div className="text-sm text-zinc-400 mt-1 leading-relaxed">{d.description}</div>
              <div className="flex flex-wrap gap-2 mt-3">
                <span className="tag">{d.subjects} subjects</span>
                <span className="tag">{d.license}</span>
                {d.classes.map((c) => (
                  <span key={c} className="tag">{c}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </section>

      <section className="surface p-5">
        <div className="text-sm font-medium mb-3">Acquiring the data</div>
        <pre className="bg-ink-900 border border-ink-600 rounded p-3 text-xs font-mono overflow-x-auto leading-relaxed">
{`# from backend/
pip install openneuro-py
python -m scripts.download_datasets --datasets ds004504 ds002778 ds003478

# preprocessed npz files end up under backend/data/datasets/<id>/
# train HAMD-Net on real data:
python -m scripts.train --epochs 30 --device cuda`}
        </pre>
      </section>
    </div>
  );
}
