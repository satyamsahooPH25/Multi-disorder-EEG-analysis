"""Approximate 2D coordinates for the standard 10-20 montage so the frontend
can render topographic maps without depending on MNE in the browser."""

from __future__ import annotations

import math


# Approximate 2D positions on a unit circle (head outline). x = left/right,
# y = posterior/anterior. Values normalized to roughly [-1, 1].
ELECTRODE_POSITIONS_2D: dict[str, tuple[float, float]] = {
    "Fp1": (-0.30,  0.95), "Fp2": (0.30,  0.95),
    "F7":  (-0.80,  0.55), "F3":  (-0.45, 0.55),
    "Fz":  ( 0.00,  0.55), "F4":  (0.45, 0.55), "F8": (0.80, 0.55),
    "T3":  (-0.95,  0.00), "C3":  (-0.50, 0.00), "Cz": (0.00, 0.00),
    "C4":  ( 0.50,  0.00), "T4":  (0.95, 0.00),
    "T5":  (-0.80, -0.55), "P3":  (-0.45, -0.55),
    "Pz":  ( 0.00, -0.55), "P4":  (0.45, -0.55), "T6": (0.80, -0.55),
    "O1":  (-0.30, -0.95), "O2":  (0.30, -0.95),
    "A1":  (-1.05,  0.05), "A2":  (1.05, 0.05),
    "FC5": (-0.65,  0.30), "FC1": (-0.25, 0.30),
    "FC2": ( 0.25,  0.30), "FC6": (0.65, 0.30),
    "CP5": (-0.65, -0.30), "CP1": (-0.25, -0.30),
    "CP2": ( 0.25, -0.30), "CP6": (0.65, -0.30),
    "PO3": (-0.25, -0.80), "PO4": (0.25, -0.80),
    "Oz":  ( 0.00, -1.00),
}


def get_position(channel: str) -> tuple[float, float] | None:
    key = channel.replace(" ", "").upper()
    for k, v in ELECTRODE_POSITIONS_2D.items():
        if k.upper() == key:
            return v
    return None


def positions_for(channels: list[str]) -> list[dict]:
    out = []
    for i, ch in enumerate(channels):
        pos = get_position(ch)
        if pos is None:
            angle = (i / max(len(channels), 1)) * 2 * math.pi
            pos = (math.cos(angle) * 0.7, math.sin(angle) * 0.7)
        out.append({"channel": ch, "x": pos[0], "y": pos[1]})
    return out
