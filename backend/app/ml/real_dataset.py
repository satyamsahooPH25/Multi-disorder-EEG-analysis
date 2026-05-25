"""Real EEG dataset loaders.

Three corpora are wired into the pipeline:

  * OpenNeuro ds004504  — Alzheimer's, FTD, Healthy   (88 subjects, .set, 19ch, 500 Hz)
  * OpenNeuro ds002778  — Parkinson's UC San Diego    (31 subjects, .bdf, 41ch, 512 Hz)
  * RepOD Olejarczyk-Jernajczyk — Schizophrenia      (28 subjects, .edf, 19ch, 250 Hz)

The HAMD-Net 5-class label space is:
  0 Healthy · 1 Alzheimer's · 2 Parkinson's · 3 FTD · 4 Schizophrenia
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mne
import numpy as np

from .preprocessing import select_channels, PreprocConfig

mne.set_log_level("WARNING")


LABEL_NAMES = ["Healthy", "Alzheimer's", "Parkinson's", "FTD", "Schizophrenia"]
DS004504_GROUP_TO_LABEL = {"C": 0, "A": 1, "F": 3}


@dataclass
class Subject:
    subject_id: str
    label: int
    label_name: str
    file_path: Path
    source: str
    age: Optional[int] = None
    gender: Optional[str] = None
    mmse: Optional[int] = None
    extra: Optional[dict] = None

    def relative_path(self) -> str:
        rel = self.file_path.as_posix()
        if rel.startswith("./"):
            return rel[2:]
        return rel


# --------------------------------------------------------------------------
# ds004504  — Alzheimer's / FTD / Healthy
# --------------------------------------------------------------------------

def index_ds004504(root: Path) -> list[Subject]:
    root = Path(root)
    participants = root / "participants.tsv"
    if not participants.exists():
        return []
    out: list[Subject] = []
    with participants.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["participant_id"]
            grp = row.get("Group", "").strip()
            if grp not in DS004504_GROUP_TO_LABEL:
                continue
            path = root / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
            if not path.exists():
                continue
            label = DS004504_GROUP_TO_LABEL[grp]
            try:
                age = int(row.get("Age", "0") or 0)
            except ValueError:
                age = None
            try:
                mmse = int(row.get("MMSE", "0") or 0)
            except ValueError:
                mmse = None
            out.append(Subject(
                subject_id=sid,
                label=label,
                label_name=LABEL_NAMES[label],
                file_path=path,
                source="ds004504",
                age=age,
                gender=row.get("Gender"),
                mmse=mmse,
            ))
    return out


# --------------------------------------------------------------------------
# ds002778  — Parkinson's UC San Diego
# --------------------------------------------------------------------------

def index_ds002778(root: Path) -> list[Subject]:
    root = Path(root)
    participants = root / "participants.tsv"
    if not participants.exists():
        return []
    rows: dict[str, dict] = {}
    with participants.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows[row["participant_id"]] = row

    out: list[Subject] = []
    for sid, row in rows.items():
        is_pd = sid.startswith("sub-pd")
        is_hc = sid.startswith("sub-hc")
        if not (is_pd or is_hc):
            continue
        # Choose the off-meds session for PD subjects (closer to natural state),
        # the only session for healthy controls.
        if is_pd:
            session_dir = root / sid / "ses-off" / "eeg"
            ses = "ses-off"
        else:
            session_dir = root / sid / "ses-hc" / "eeg"
            ses = "ses-hc"
        if not session_dir.exists():
            continue
        bdf = next(session_dir.glob("*_task-rest_eeg.bdf"), None)
        if not bdf:
            continue
        label = 2 if is_pd else 0
        try:
            age = int(row.get("age", "0") or 0)
        except ValueError:
            age = None
        out.append(Subject(
            subject_id=sid,
            label=label,
            label_name=LABEL_NAMES[label],
            file_path=bdf,
            source="ds002778",
            age=age,
            gender=row.get("gender"),
            extra={"session": ses},
        ))
    return out


# --------------------------------------------------------------------------
# Olejarczyk-Jernajczyk Schizophrenia EEG (RepOD)
# --------------------------------------------------------------------------

def index_schizophrenia(root: Path) -> list[Subject]:
    root = Path(root)
    if not root.exists():
        return []
    out: list[Subject] = []
    for path in sorted(root.glob("*.edf")):
        name = path.stem  # e.g. "h01" or "s07"
        if not name:
            continue
        prefix = name[0].lower()
        try:
            num = int(name[1:])
        except ValueError:
            continue
        if prefix == "s":
            label = 4
        elif prefix == "h":
            label = 0
        else:
            continue
        out.append(Subject(
            subject_id=f"olj-{name}",
            label=label,
            label_name=LABEL_NAMES[label],
            file_path=path,
            source="olejarczyk_scz",
            extra={"index": num},
        ))
    return out


# --------------------------------------------------------------------------
# Combined view
# --------------------------------------------------------------------------

DEFAULT_ROOTS = {
    "ds004504": Path("data/datasets/ds004504"),
    "ds002778": Path("data/datasets/ds002778"),
    "olejarczyk_scz": Path("data/datasets/schizophrenia_olejarczyk"),
}


def list_subjects(roots: Optional[dict[str, Path]] = None,
                  sources: Optional[Iterable[str]] = None) -> list[Subject]:
    roots = roots or DEFAULT_ROOTS
    out: list[Subject] = []
    if sources is None or "ds004504" in sources:
        out.extend(index_ds004504(roots["ds004504"]))
    if sources is None or "ds002778" in sources:
        out.extend(index_ds002778(roots["ds002778"]))
    if sources is None or "olejarczyk_scz" in sources:
        out.extend(index_schizophrenia(roots["olejarczyk_scz"]))
    return out


def load_subject(subj: Subject,
                  cfg: Optional[PreprocConfig] = None
                  ) -> tuple[np.ndarray, float, list[str]]:
    """Load a subject into a (channels, samples) array plus its sample rate
    and the canonical channel names. Channels are restricted to the standard
    19-channel 10-20 montage so all three corpora share a single layout."""
    cfg = cfg or PreprocConfig()
    suffix = subj.file_path.suffix.lower()
    if suffix == ".set":
        raw = mne.io.read_raw_eeglab(str(subj.file_path),
                                       preload=True, verbose="ERROR")
    elif suffix == ".bdf":
        raw = mne.io.read_raw_bdf(str(subj.file_path),
                                    preload=True, verbose="ERROR")
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(str(subj.file_path),
                                    preload=True, verbose="ERROR")
    else:
        raise ValueError(f"unsupported format: {suffix}")
    raw = select_channels(raw, cfg)
    return (raw.get_data().astype(np.float32),
            float(raw.info["sfreq"]),
            list(raw.ch_names))
