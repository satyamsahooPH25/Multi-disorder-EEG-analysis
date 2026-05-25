"""Download the Olejarczyk-Jernajczyk schizophrenia EEG dataset from RepOD.

DOI: 10.18150/repod.0107441
14 schizophrenia patients + 14 matched healthy controls.
19-channel resting-state EEG, 250 Hz, EDF format, ~9 MB per file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

DATASET_DOI = "doi:10.18150/repod.0107441"
API_BASE = "https://repod.icm.edu.pl/api"
OUT_DIR = Path("./data/datasets/schizophrenia_olejarczyk")


def fetch_file_list() -> list[dict]:
    url = f"{API_BASE}/datasets/:persistentId/?persistentId={DATASET_DOI}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    j = r.json()
    return [f["dataFile"] for f in j["data"]["latestVersion"]["files"]]


def download_file(file_id: int, name: str, target: Path) -> None:
    url = f"{API_BASE}/access/datafile/{file_id}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1000:
        print(f"  - {name} (cached)")
        return
    print(f"  > {name}")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with target.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 18):
                if chunk:
                    f.write(chunk)


def main():
    files = fetch_file_list()
    edfs = [f for f in files if f["filename"].lower().endswith(".edf")]
    print(f"Found {len(edfs)} EDFs (Olejarczyk-Jernajczyk schizophrenia EEG)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(edfs):
        target = OUT_DIR / f["filename"]
        download_file(f["id"], f["filename"], target)
        print(f"  {i+1}/{len(edfs)} ok")
    print(f"\nSaved to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
