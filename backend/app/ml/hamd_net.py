"""HAMD-Net: Hybrid Attention Multi-Disorder Network for EEG classification.

Biologically informed hybrid architecture combining:
  - Spatial convolutional feature extractor (channel mixing across electrodes)
  - Temporal Convolutional Network (TCN) for short-range dependencies
  - Bi-directional LSTM for long-range temporal context
  - Multi-head self-attention for global temporal integration
  - Per-class output head for multi-disorder classification
    (Alzheimer's, Parkinson's, FTD, Schizophrenia, Healthy Control)

The attention weights are exposed for interpretability so the frontend can
render electrode-level + temporal heatmaps explaining each prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_NAMES = ["Healthy", "Alzheimer's", "Parkinson's", "FTD", "Schizophrenia"]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class HAMDConfig:
    n_channels: int = 19          # standard 10-20 montage subset
    n_samples: int = 1000         # 4 s @ 250 Hz
    sfreq: float = 250.0
    spatial_filters: int = 32
    temporal_filters: int = 64
    tcn_channels: int = 96
    tcn_kernel: int = 7
    tcn_layers: int = 4
    lstm_hidden: int = 128
    n_heads: int = 8
    attn_dim: int = 128
    dropout: float = 0.3
    n_classes: int = NUM_CLASSES


class SpatialConv(nn.Module):
    """Learnable spatial filter — mixes electrodes into virtual channels.

    Mirrors EEGNet's depthwise spatial convolution: a 1xN_channels kernel
    that learns inter-electrode spatial patterns analogous to ICA / CSP.
    """

    def __init__(self, n_channels: int, n_filters: int, dropout: float):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters // 2, kernel_size=(1, 64),
                                  padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(n_filters // 2)
        self.spatial = nn.Conv2d(n_filters // 2, n_filters,
                                 kernel_size=(n_channels, 1),
                                 groups=n_filters // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, 4))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.spatial(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.drop(x)
        # Squeeze channel dim of conv2d -> (B, F, T')
        return x.squeeze(2)


class TCNBlock(nn.Module):
    """Dilated causal convolution block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.padding = padding
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : x.shape[-1] - self.padding] if self.padding > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        out = self._crop(self.conv1(x))
        out = F.elu(self.bn1(out))
        out = self.drop(out)
        out = self._crop(self.conv2(out))
        out = F.elu(self.bn2(out))
        out = self.drop(out)
        return F.elu(out + res)


class TCN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int,
                 kernel: int, dropout: float):
        super().__init__()
        blocks = []
        ch_in = in_ch
        for i in range(layers):
            blocks.append(TCNBlock(ch_in, out_ch, kernel, 2 ** i, dropout))
            ch_in = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Returns attention weights alongside output for interpretability."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        attn_out, attn_w = self.attn(x, x, x, need_weights=True,
                                     average_attn_weights=False)
        x = self.norm(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, attn_w


class HAMDNet(nn.Module):
    """Hybrid Attention Multi-Disorder Network."""

    def __init__(self, cfg: Optional[HAMDConfig] = None):
        super().__init__()
        self.cfg = cfg or HAMDConfig()
        c = self.cfg

        self.spatial = SpatialConv(c.n_channels, c.spatial_filters, c.dropout)
        self.proj_in = nn.Conv1d(c.spatial_filters, c.tcn_channels, 1)
        self.tcn = TCN(c.tcn_channels, c.tcn_channels, c.tcn_layers,
                       c.tcn_kernel, c.dropout)
        self.lstm = nn.LSTM(c.tcn_channels, c.lstm_hidden,
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=c.dropout)
        self.proj_attn = nn.Linear(c.lstm_hidden * 2, c.attn_dim)
        self.pos_enc = PositionalEncoding(c.attn_dim)
        self.attn1 = MultiHeadSelfAttention(c.attn_dim, c.n_heads, c.dropout)
        self.attn2 = MultiHeadSelfAttention(c.attn_dim, c.n_heads, c.dropout)
        self.head = nn.Sequential(
            nn.Linear(c.attn_dim, c.attn_dim // 2),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.attn_dim // 2, c.n_classes),
        )
        self._last_attention = None
        self._last_spatial = None

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        # Spatial mix -> (B, F, T')
        spatial = self.spatial(x)
        self._last_spatial = spatial.detach()
        h = self.proj_in(spatial)
        h = self.tcn(h)
        # Move time to dim 1 for LSTM/attention: (B, T', F)
        h = h.transpose(1, 2)
        h, _ = self.lstm(h)
        h = self.proj_attn(h)
        h = self.pos_enc(h)
        h, a1 = self.attn1(h)
        h, a2 = self.attn2(h)
        # Average attention across heads, last layer for interpretability
        self._last_attention = a2.detach().mean(dim=1)
        # Global average pool over time
        pooled = h.mean(dim=1)
        logits = self.head(pooled)
        if return_attn:
            return logits, {"attention": a2, "spatial": spatial}
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return F.softmax(self.forward(x), dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: Optional[HAMDConfig] = None) -> HAMDNet:
    return HAMDNet(cfg)


if __name__ == "__main__":
    model = build_model()
    x = torch.randn(2, 19, 1000)
    y, info = model(x, return_attn=True)
    print(f"Output shape: {y.shape}")
    print(f"Attention shape: {info['attention'].shape}")
    print(f"Params: {model.num_params():,}")
