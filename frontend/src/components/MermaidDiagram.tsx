"use client";

import { useEffect, useId, useRef, useState } from "react";
import mermaid from "mermaid";

type Props = {
  chart: string;
};

let initialized = false;

export function MermaidDiagram({ chart }: Props) {
  const id = useId().replace(/[:]/g, "_");
  const ref = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!initialized) {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "base",
        themeVariables: {
          background: "#0a0e15",
          primaryColor: "#0f1421",
          primaryTextColor: "#e7eaf2",
          primaryBorderColor: "#1a1f2e",
          lineColor: "#3a4055",
          secondaryColor: "#0c1018",
          tertiaryColor: "#0f1421",
          fontFamily: "JetBrains Mono, ui-monospace, monospace",
          fontSize: "13px",
          mainBkg: "#0f1421",
          edgeLabelBackground: "#0a0e15",
          clusterBkg: "#0c1018",
          clusterBorder: "#1a1f2e",
          nodeTextColor: "#e7eaf2",
        },
        flowchart: {
          curve: "basis",
          padding: 12,
          htmlLabels: true,
        },
      });
      initialized = true;
    }

    let cancelled = false;
    (async () => {
      try {
        const out = await mermaid.render(`m_${id}`, chart);
        if (!cancelled && ref.current) {
          ref.current.innerHTML = out.svg;
          if (out.bindFunctions) out.bindFunctions(ref.current);
        }
      } catch (e: any) {
        if (!cancelled) setErr(e?.message ?? String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [chart, id]);

  return (
    <div className="overflow-x-auto">
      <div ref={ref}
           className="mermaid-host flex justify-center min-h-[400px]"
           style={{ color: "#e7eaf2" }} />
      {err && (
        <pre className="text-xs text-risk-high font-mono whitespace-pre-wrap">
          {err}
        </pre>
      )}
    </div>
  );
}
