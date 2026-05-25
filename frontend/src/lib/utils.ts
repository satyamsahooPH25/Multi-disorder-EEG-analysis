import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtMs(ms: number) {
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtNum(n: number) {
  return Intl.NumberFormat("en", { notation: "compact" }).format(n);
}

export function fmtPct(p: number, digits = 1) {
  return `${(p * 100).toFixed(digits)}%`;
}

export const CLASS_NAMES = ["Healthy", "Alzheimer's", "Parkinson's", "FTD", "Schizophrenia"];

export const CLASS_COLORS: Record<string, string> = {
  Healthy: "#22c55e",
  "Alzheimer's": "#7c5cff",
  "Parkinson's": "#39c0ed",
  FTD: "#f59e0b",
  Schizophrenia: "#ef4444",
};

export const BAND_COLORS: Record<string, string> = {
  delta: "#7c5cff",
  theta: "#39c0ed",
  alpha: "#22c55e",
  beta: "#f59e0b",
  gamma: "#ef4444",
};
