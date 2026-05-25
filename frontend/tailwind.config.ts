import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx,js,jsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "ui-monospace"],
      },
      colors: {
        ink: {
          900: "#06090e",
          800: "#0a0e15",
          700: "#0f1420",
          600: "#1a1f2e",
          500: "#252a3a",
          400: "#3a4055",
        },
        accent: {
          400: "#5b8df5",
          500: "#3b6ee0",
          600: "#2e58b8",
        },
        signal: {
          delta: "#7c5cff",
          theta: "#39c0ed",
          alpha: "#22c55e",
          beta:  "#f59e0b",
          gamma: "#ef4444",
        },
        risk: {
          low:  "#22c55e",
          mid:  "#f59e0b",
          high: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};

export default config;
