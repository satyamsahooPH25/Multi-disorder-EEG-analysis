"use client";

import { useEffect, useRef } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

export function Equation({
  tex,
  display = true,
}: {
  tex: string;
  display?: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(tex, ref.current, {
        displayMode: display,
        throwOnError: false,
        output: "html",
        strict: "ignore",
      });
    } catch (e) {
      if (ref.current) ref.current.textContent = tex;
    }
  }, [tex, display]);
  return (
    <span ref={ref}
          className={display ? "block my-2 text-zinc-200" : "inline text-zinc-200"} />
  );
}
