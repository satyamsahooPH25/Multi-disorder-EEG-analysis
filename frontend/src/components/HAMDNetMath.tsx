"use client";

import { Equation } from "./Equation";

type Block = {
  title: string;
  intuition: string;
  equations: { label?: string; tex: string }[];
  notes?: string[];
};

const BLOCKS: Block[] = [
  {
    title: "0 · Input window",
    intuition:
      "Each prediction operates on a 4-second slice of the 19-channel EEG, sampled at 250 Hz after preprocessing. " +
      "The tensor X has shape (C, T) where C = 19 electrodes and T = 1000 time points.",
    equations: [
      { label: "shape", tex: "X \\in \\mathbb{R}^{C \\times T},\\quad C=19,\\ T=1000" },
      {
        label: "z-score per channel",
        tex: "\\tilde X_{c,t} = \\frac{X_{c,t} - \\mu_c}{\\sigma_c + \\varepsilon}",
      },
    ],
    notes: [
      "μc, σc are computed per channel over T (so each electrode has zero mean, unit variance over the window).",
    ],
  },
  {
    title: "1 · Spatial Conv (EEGNet-style)",
    intuition:
      "First a temporal 1×64 conv mixes nearby time steps within each electrode, then a depthwise spatial conv " +
      "across all 19 electrodes learns electrode-mixing patterns analogous to ICA / CSP.",
    equations: [
      { label: "temporal", tex: "Y^{(1)}_{f,c,t} = \\mathrm{ELU}\\!\\Big(\\mathrm{BN}\\big(\\sum_{\\tau} W^{(t)}_{f,\\tau}\\,\\tilde X_{c,\\,t-\\tau}\\big)\\Big)" },
      { label: "spatial (depthwise)", tex: "Y^{(2)}_{f,t} = \\mathrm{ELU}\\!\\Big(\\mathrm{BN}\\big(\\sum_{c=1}^{C} W^{(s)}_{f,c}\\,Y^{(1)}_{f,c,t}\\big)\\Big)" },
      { label: "average pool ↓4", tex: "Z^{(\\text{sp})}_{f,t'} = \\tfrac{1}{4}\\sum_{i=0}^{3} Y^{(2)}_{f,\\,4t'+i}" },
    ],
    notes: [
      "Output shape: (F, T/4) with F = 32 spatial filters.",
      "The depthwise (groups = F/2) spatial kernel is what gives the per-electrode importance map shown on the live topomap.",
    ],
  },
  {
    title: "2 · TCN (Dilated Causal Conv)",
    intuition:
      "Four stacked dilated causal blocks expand the receptive field exponentially without growing parameter count, " +
      "capturing short-to-mid range temporal patterns (≈100 ms to several seconds).",
    equations: [
      { label: "dilated causal conv", tex: "h^{(\\ell)}_t = \\sum_{k=0}^{K-1} W^{(\\ell)}_{k}\\,h^{(\\ell-1)}_{\\,t - d_\\ell\\,k}" },
      { label: "block (residual)", tex: "h^{(\\ell)} = h^{(\\ell-1)} + \\mathrm{Drop}\\big(\\mathrm{ELU}(\\mathrm{BN}(h^{(\\ell)}))\\big)" },
      { label: "dilation schedule", tex: "d_\\ell = 2^{\\ell-1},\\quad \\ell \\in \\{1,2,3,4\\}" },
    ],
    notes: [
      "Kernel K = 7, channels C_TCN = 96.",
      "Total receptive field after 4 layers: 1 + (K−1)·(1+2+4+8) = 91 samples ≈ 1.46 s of EEG context per output token.",
    ],
  },
  {
    title: "3 · Bi-LSTM",
    intuition:
      "A 2-layer bidirectional LSTM integrates information across the full 4-second sequence in both temporal directions. " +
      "Useful for slow oscillations and resting-state rhythms.",
    equations: [
      {
        label: "forward LSTM cell",
        tex:
          "\\begin{aligned}\n" +
          "i_t &= \\sigma(W_i x_t + U_i h_{t-1} + b_i)\\\\\n" +
          "f_t &= \\sigma(W_f x_t + U_f h_{t-1} + b_f)\\\\\n" +
          "o_t &= \\sigma(W_o x_t + U_o h_{t-1} + b_o)\\\\\n" +
          "g_t &= \\tanh(W_g x_t + U_g h_{t-1} + b_g)\\\\\n" +
          "c_t &= f_t \\odot c_{t-1} + i_t \\odot g_t\\\\\n" +
          "h_t &= o_t \\odot \\tanh(c_t)\n" +
          "\\end{aligned}",
      },
      {
        label: "bi-directional concat",
        tex: "\\overrightarrow{h}_t,\\; \\overleftarrow{h}_t \\;\\to\\; H_t = [\\overrightarrow{h}_t \\,\\Vert\\, \\overleftarrow{h}_t] \\in \\mathbb{R}^{2H}",
      },
    ],
    notes: [
      "Hidden size H = 128 → output dim 2H = 256 per time step.",
    ],
  },
  {
    title: "4 · Multi-Head Self-Attention",
    intuition:
      "Two stacked Transformer-style attention layers integrate information globally across the sequence. " +
      "Heads can specialise on different time scales / frequency bands. The attention matrix is exposed " +
      "in the prediction payload and drives the temporal-attention heatmap in the live UI.",
    equations: [
      { label: "Q / K / V projections", tex: "Q = H W_Q,\\quad K = H W_K,\\quad V = H W_V" },
      {
        label: "scaled dot-product attention",
        tex: "\\mathrm{Attn}(Q,K,V) = \\mathrm{softmax}\\!\\Big(\\tfrac{Q K^{\\top}}{\\sqrt{d_k}}\\Big)V",
      },
      {
        label: "multi-head",
        tex: "\\mathrm{MHA}(H) = \\big[\\mathrm{Attn}_1\\,\\Vert\\,\\dots\\,\\Vert\\,\\mathrm{Attn}_h\\big]\\,W_O",
      },
      {
        label: "block",
        tex:
          "\\begin{aligned}\n" +
          "Z' &= \\mathrm{LN}(H + \\mathrm{MHA}(H))\\\\\n" +
          "Z  &= \\mathrm{LN}(Z' + \\mathrm{FFN}(Z'))\n" +
          "\\end{aligned}",
      },
    ],
    notes: [
      "h = 8 heads, d_model = d_attn = 128, d_k = 16.",
      "The temporal attention exported to the live UI is the per-key average across heads and queries: aₜ = (1/h·T) Σ_{h,q} A^h_{q,t}.",
    ],
  },
  {
    title: "5 · Classification head",
    intuition:
      "Global average pool over time collapses the sequence into a single 128-d embedding, " +
      "then a 2-layer MLP produces logits over the 5 disorder classes.",
    equations: [
      { label: "global avg pool", tex: "\\bar Z = \\tfrac{1}{T}\\sum_{t=1}^{T} Z_t \\in \\mathbb{R}^{d}" },
      { label: "logits", tex: "\\ell = W_2\\,\\mathrm{ELU}(W_1 \\bar Z + b_1) + b_2 \\in \\mathbb{R}^{5}" },
      {
        label: "softmax over classes",
        tex: "p_k = \\frac{\\exp(\\ell_k)}{\\sum_{j=1}^{5}\\exp(\\ell_j)},\\quad k \\in \\{\\text{Healthy, AD, PD, FTD, SCZ}\\}",
      },
      {
        label: "window aggregation",
        tex: "p^{\\text{subj}}_k = \\frac{1}{N_w}\\sum_{w=1}^{N_w} p^{(w)}_k",
      },
    ],
    notes: [
      "Softmax + arg-max gives the predicted class; the max probability is reported as `confidence`.",
      "For multi-window subjects, per-window probabilities are averaged for the final report.",
    ],
  },
  {
    title: "6 · Training objective",
    intuition:
      "Class-weighted cross-entropy with early stopping. Weights compensate for the imbalance between classes " +
      "(Healthy: 59, AD: 36, PD: 15, FTD: 23, SCZ: 14 subjects).",
    equations: [
      { label: "class weight", tex: "w_k = \\frac{N}{K\\,n_k}\\quad\\text{(inverse frequency)}" },
      {
        label: "weighted cross-entropy",
        tex: "\\mathcal{L} = -\\frac{1}{B}\\sum_{i=1}^{B} w_{y_i}\\,\\log p^{(i)}_{y_i}",
      },
      {
        label: "early-stop criterion",
        tex:
          "\\text{stop if } \\#\\{e \\le E\\,:\\, v_e \\le v^{*} + \\delta\\} \\ge P\\;\\text{ and }\\; E \\ge E_{\\min}",
      },
    ],
    notes: [
      "v_e = validation accuracy at epoch e, v* = best so far, δ = min_delta (1e-3), P = patience (10), E_min = min_epochs (12).",
      "Optimizer: Adam, lr = 5e-4, batch = 32. Best-by-val checkpoint is hot-loaded into the inference engine on save.",
    ],
  },
];

export function HAMDNetMath() {
  return (
    <div className="space-y-5">
      {BLOCKS.map((b) => (
        <div key={b.title}
             className="border border-ink-700 rounded-lg p-4 bg-ink-800/40">
          <div className="text-sm font-semibold text-accent-400 mb-1">
            {b.title}
          </div>
          <p className="text-sm text-zinc-400 mb-3 leading-relaxed">
            {b.intuition}
          </p>
          <div className="space-y-2">
            {b.equations.map((e, i) => (
              <div key={i} className="bg-ink-900/60 rounded px-3 py-2">
                {e.label && (
                  <div className="text-[10px] uppercase tracking-wider
                                  text-zinc-500 mb-1 font-mono">
                    {e.label}
                  </div>
                )}
                <Equation tex={e.tex} />
              </div>
            ))}
          </div>
          {b.notes && b.notes.length > 0 && (
            <ul className="mt-3 space-y-1">
              {b.notes.map((n, i) => (
                <li key={i}
                    className="text-xs text-zinc-500 flex gap-2">
                  <span className="text-accent-500/60">›</span>
                  <span>{n}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
}

export const HAMD_NET_MERMAID = `flowchart TD
  X["Input EEG window<br/>X in R^{19 x 1000}<br/>4 s at 250 Hz · 19 channels"]:::input

  subgraph PRE ["Preprocessing — per 1 s tick on rolling buffer"]
    direction TB
    P1["Resample to 250 Hz"]:::pre
    P2["Band-pass 0.5–45 Hz IIR"]:::pre
    P3["Notch 50 Hz"]:::pre
    P4["Average reference"]:::pre
    P5["Z-score per channel"]:::pre
    P6["Slide 4 s window<br/>50% overlap"]:::pre
    P1 --> P2 --> P3 --> P4 --> P5 --> P6
  end

  X --> PRE

  subgraph SP ["1 · Spatial Conv — EEGNet-style"]
    direction TB
    S1["Temporal Conv 1×64<br/>+ BN + ELU"]:::block
    S2["Depthwise Spatial Conv 19×1<br/>+ BN + ELU"]:::block
    S3["Avg-pool ↓4 + Dropout"]:::block
    S1 --> S2 --> S3
  end
  PRE -- "shape 19 × 1000" --> SP

  subgraph TCN ["2 · TCN — 4 dilated causal blocks"]
    direction TB
    T1["Block 1 · dilation 1"]:::block
    T2["Block 2 · dilation 2"]:::block
    T3["Block 3 · dilation 4"]:::block
    T4["Block 4 · dilation 8"]:::block
    T1 --> T2 --> T3 --> T4
  end
  SP -- "shape 32 × 250" --> TCN

  subgraph LSTM ["3 · Bi-LSTM — 2 layers · hidden 128"]
    direction LR
    L1["Forward LSTM"]:::block
    L2["Backward LSTM"]:::block
    LC["Concat → 256-d"]:::block
    L1 --> LC
    L2 --> LC
  end
  TCN -- "shape 96 × 250" --> LSTM

  subgraph ATTN ["4 · Multi-Head Self-Attention — 2 layers · 8 heads · d=128"]
    direction TB
    A1["Q · K · V projections"]:::block
    A2["Scaled dot-product attention"]:::block
    A3["FFN + residual + LayerNorm"]:::block
    A1 --> A2 --> A3
  end
  LSTM -- "shape 250 × 256" --> ATTN

  subgraph HEAD ["5 · Classification Head"]
    direction TB
    H1["Global avg pool over T"]:::block
    H2["MLP 128 → 64 → 5"]:::block
    H3["Softmax"]:::block
    H1 --> H2 --> H3
  end
  ATTN -- "shape 250 × 128" --> HEAD

  HEAD --> OUT["Probabilities p in Δ^5<br/>Healthy · AD · PD · FTD · SCZ"]:::output

  ATTN -. "attention weights" .-> ATTNOUT["Temporal attention a_t<br/>avg over heads + queries"]:::side
  SP   -. "filter weights"    .-> CHIMP["Per-electrode<br/>channel importance"]:::side

  classDef input  fill:#1f2937,stroke:#5b8df5,color:#e7eaf2,stroke-width:1.5px
  classDef pre    fill:#0f1421,stroke:#39c0ed,color:#cbd1e0
  classDef block  fill:#0f1421,stroke:#5b8df5,color:#cbd1e0
  classDef output fill:#1a2233,stroke:#22c55e,color:#e7eaf2,stroke-width:1.5px
  classDef side   fill:#0f1421,stroke:#f59e0b,color:#f59e0b,stroke-dasharray:4 3
`;
