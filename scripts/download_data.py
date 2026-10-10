#!/usr/bin/env python
"""Download the UCI Heart Disease data (all four hospitals) into data/raw/.

Usage:
    python scripts/download_data.py               # download from UCI
    python scripts/download_data.py --zip heart+disease.zip   # use a local copy

The script tries the current UCI archive zip first, then the legacy
per-file URLs. If the machine has no internet (e.g. a locked-down lab),
download https://archive.ics.uci.edu/dataset/45/heart+disease manually
and pass the zip with --zip.

After downloading it writes data/raw/CHECKSUMS.sha256 and builds the
cleaned dataset data/processed/heart_clean.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))

FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
    "heart-disease.names",
]
ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
LEGACY_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"


def _extract(zip_bytes: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for f in FILES:
            if f not in names:
                raise RuntimeError(f"{f} missing from zip")
            (RAW / f).write_bytes(zf.read(names[f]))


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cardiorisk-mlops/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def download() -> None:
    try:
        print(f"Downloading {ZIP_URL}")
        _extract(_fetch(ZIP_URL))
        return
    except Exception as exc:  # noqa: BLE001 - fall back to legacy URLs
        print(f"  zip download failed ({exc}); trying legacy per-file URLs")
    for f in FILES:
        print(f"Downloading {LEGACY_BASE + f}")
        (RAW / f).write_bytes(_fetch(LEGACY_BASE + f))


def write_checksums() -> None:
    lines = []
    for f in FILES:
        digest = hashlib.sha256((RAW / f).read_bytes()).hexdigest()
        lines.append(f"{digest}  {f}")
    (RAW / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, help="use a locally downloaded UCI zip instead of the internet")
    ap.add_argument("--skip-if-present", action="store_true", help="do nothing if raw files exist")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.skip_if_present and all((RAW / f).exists() for f in FILES):
        print("Raw files already present - skipping download")
    elif args.zip:
        _extract(args.zip.read_bytes())
    else:
        download()

    write_checksums()

    from cardiorisk.data import build_clean_dataset

    df = build_clean_dataset()
    print(f"Clean dataset: {len(df)} rows, positive rate {df['target'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
