"""One-shot setup: download every EEG corpus the live stream replays.

Run this **before** starting the backend if you want the /live page to
play back real recordings. The shipped checkpoint will load and predict
either way, but with no datasets on disk the subject list under
/api/streams/real comes back empty.

Pulls the three corpora HAMD-Net was trained on (~5 GB total, one-time
download):

  - OpenNeuro ds004504  — Alzheimer's + FTD + Healthy (88 subjects, .set)
  - OpenNeuro ds002778  — Parkinson's UC San Diego    (31 subjects, .bdf)
  - RepOD Olejarczyk    — Schizophrenia + Healthy     (28 subjects, .edf)

Usage (from backend/):

    pip install -r requirements.txt          # one-time
    pip install openneuro-py requests        # one-time
    python -m scripts.setup_datasets         # the actual download
    python -m uvicorn app.main:app --port 8765   # then start the backend

Pass --skip <id> to skip a corpus you already have, or --only <id> to
download just one. Existing subjects are detected and skipped automatically
so re-running this script is safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"

CORPORA = {
    "ds004504": {
        "label": "Alzheimer's + FTD + Healthy (OpenNeuro)",
        "size_mb": 2700,
        "expect_marker": "participants.tsv",
    },
    "ds002778": {
        "label": "Parkinson's UC San Diego (OpenNeuro)",
        "size_mb": 545,
        "expect_marker": "participants.tsv",
    },
    "schizophrenia_olejarczyk": {
        "label": "Schizophrenia + Healthy (Olejarczyk-Jernajczyk RepOD)",
        "size_mb": 250,
        "expect_marker": "h01.edf",
    },
}


def is_present(corpus_id: str) -> bool:
    target = ROOT / corpus_id
    marker = CORPORA[corpus_id]["expect_marker"]
    return (target / marker).exists()


def fetch_openneuro(dataset_id: str) -> None:
    try:
        import openneuro
    except ImportError:
        print("  ! openneuro-py not installed.\n"
              "    Run: pip install openneuro-py")
        raise SystemExit(1)
    target = ROOT / dataset_id
    target.mkdir(parents=True, exist_ok=True)
    openneuro.download(dataset=dataset_id, target_dir=str(target))


def fetch_olejarczyk_scz() -> None:
    """Download every EDF in the Olejarczyk-Jernajczyk RepOD dataset."""
    try:
        import requests
    except ImportError:
        print("  ! requests not installed.\n"
              "    Run: pip install requests")
        raise SystemExit(1)
    api_base = "https://repod.icm.edu.pl/api"
    doi = "doi:10.18150/repod.0107441"
    target = ROOT / "schizophrenia_olejarczyk"
    target.mkdir(parents=True, exist_ok=True)

    list_url = f"{api_base}/datasets/:persistentId/?persistentId={doi}"
    r = requests.get(list_url, timeout=60)
    r.raise_for_status()
    files = [f["dataFile"]
             for f in r.json()["data"]["latestVersion"]["files"]
             if f["dataFile"]["filename"].lower().endswith(".edf")]
    print(f"    {len(files)} EDFs to fetch")
    for i, f in enumerate(files, 1):
        out = target / f["filename"]
        if out.exists() and out.stat().st_size > 1000:
            print(f"    [{i}/{len(files)}] {f['filename']}  (cached)")
            continue
        print(f"    [{i}/{len(files)}] {f['filename']}")
        url = f"{api_base}/access/datafile/{f['id']}"
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with out.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 18):
                    if chunk:
                        fh.write(chunk)


FETCHERS = {
    "ds004504": lambda: fetch_openneuro("ds004504"),
    "ds002778": lambda: fetch_openneuro("ds002778"),
    "schizophrenia_olejarczyk": fetch_olejarczyk_scz,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--only", nargs="+", choices=list(CORPORA),
                   help="download just these corpora")
    p.add_argument("--skip", nargs="+", choices=list(CORPORA), default=[],
                   help="skip these corpora")
    p.add_argument("--force", action="store_true",
                   help="re-download even if already present")
    args = p.parse_args()

    selected = list(args.only) if args.only else list(CORPORA)
    selected = [c for c in selected if c not in args.skip]

    total_mb = sum(CORPORA[c]["size_mb"] for c in selected)
    print(f"\nTemple/CognitiveScreen · dataset setup")
    print(f"  target: {ROOT}")
    print(f"  selected ({len(selected)}, ~{total_mb / 1000:.1f} GB total):")
    for c in selected:
        present = "[OK]" if is_present(c) else " -  "
        print(f"    {present} {c:30s}  {CORPORA[c]['label']}")
    print()

    failed: list[str] = []
    for c in selected:
        if is_present(c) and not args.force:
            print(f"~ {c}: already on disk, skipping (use --force to redo)")
            continue
        print(f"> fetching {c} ({CORPORA[c]['label']})")
        try:
            FETCHERS[c]()
        except SystemExit:
            raise
        except Exception as exc:
            print(f"  ! {c} failed: {exc}")
            failed.append(c)
            continue
        if is_present(c):
            print(f"  OK {c}")
        else:
            print(f"  ! {c}: download finished but marker file is missing")
            failed.append(c)

    print()
    if failed:
        print(f"finished with {len(failed)} failure(s): {', '.join(failed)}")
        print("re-run the script to retry only the missing corpora.")
        return 1
    print("All datasets ready. You can now start the backend:")
    print("  python -m uvicorn app.main:app --host 127.0.0.1 --port 8765")
    return 0


if __name__ == "__main__":
    sys.exit(main())
