import fs from "node:fs";
import { JSDOM } from "jsdom";

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  pretendToBeVisual: true,
});
Object.defineProperty(globalThis, "window", { value: dom.window, configurable: true });
Object.defineProperty(globalThis, "document", { value: dom.window.document, configurable: true });
for (const k of ["HTMLElement", "Element", "SVGElement", "DOMParser", "Node"]) {
  Object.defineProperty(globalThis, k, { value: dom.window[k], configurable: true });
}

const { default: mermaid } = await import("mermaid");

const src = fs.readFileSync("src/components/HAMDNetMath.tsx", "utf8");
const m = src.match(/HAMD_NET_MERMAID\s*=\s*`([\s\S]*?)`;/);
if (!m) {
  console.error("no chart found");
  process.exit(2);
}
const chart = m[1];

try {
  const ok = await mermaid.parse(chart, { suppressErrors: false });
  if (ok === false) throw new Error("mermaid.parse returned false");
  console.log("OK · chart parses cleanly · " + chart.length + " chars");
} catch (e) {
  console.error("PARSE ERROR:", e?.message ?? e);
  process.exit(1);
}
