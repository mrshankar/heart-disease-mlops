"""Loading and cleaning of the UCI Heart Disease data.

Design choice: instead of using only the 303-row Cleveland subset (the usual
choice), all four hospitals are combined (920 rows). This triples the data but
brings heavy missingness in `ca`, `thal` and `slope`, plus zeros that really
mean "not measured" (`chol == 0` in the Swiss file). Cleaning therefore turns
those sentinels into NaN and leaves imputation to the model pipeline, so the
exact same logic runs at training time and at serving time.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def load_raw(raw_dir: Path | str = config.RAW_DIR) -> pd.DataFrame:
    """Read the four processed UCI files and stack them with a `source` column."""
    raw_dir = Path(raw_dir)
    frames = []
    for source, fname in config.SOURCES.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run `python scripts/download_data.py` first"
            )
        df = pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")
        df[config.SOURCE_COL] = source
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d raw rows from %d sources", len(combined), len(frames))
    return combined


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy: binary target, sentinel zeros -> NaN, sane dtypes.

    Rows are never dropped for missing feature values (the pipeline imputes),
    only for a missing target or exact duplicates.
    """
    out = df.copy()

    # Binary target: 0 = no disease, 1..4 = disease of increasing severity.
    out = out.dropna(subset=["num"])
    out[config.TARGET] = (out["num"].astype(float) > 0).astype(int)
    out = out.drop(columns=["num"])

    # Zero cholesterol / resting BP are physiologically impossible -> missing.
    for col in ("chol", "trestbps"):
        out.loc[out[col] == 0, col] = np.nan

    # Out-of-range numeric values are treated as data-entry errors.
    for col, (lo, hi) in config.VALID_RANGES.items():
        bad = (out[col] < lo) | (out[col] > hi)
        if bad.any():
            log.info("%s: %d out-of-range values set to NaN", col, int(bad.sum()))
        out.loc[bad, col] = np.nan

    # Categorical codes outside the documented levels are also set to NaN.
    for col, levels in config.CATEGORY_LEVELS.items():
        bad = out[col].notna() & ~out[col].isin(levels)
        out.loc[bad, col] = np.nan

    numeric = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    out[numeric] = out[numeric].astype(float)

    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.info("Dropped %d duplicate rows", before - len(out))

    return out[config.INPUT_FEATURES + [config.SOURCE_COL, config.TARGET]]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values per feature, overall and per source."""
    per_source = df.groupby(config.SOURCE_COL)[config.INPUT_FEATURES].apply(
        lambda g: g.isna().mean() * 100
    )
    per_source.loc["ALL"] = df[config.INPUT_FEATURES].isna().mean() * 100
    return per_source.round(1)


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_clean_dataset(
    raw_dir: Path | str = config.RAW_DIR, out_csv: Path | str = config.CLEAN_CSV
) -> pd.DataFrame:
    """Raw files -> cleaned CSV on disk. Returns the cleaned frame."""
    df = clean(load_raw(raw_dir))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows, sha256=%s)", out_csv, len(df), file_sha256(out_csv)[:12])
    return df


def load_clean(path: Path | str = config.CLEAN_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return build_clean_dataset(out_csv=path)
    return pd.read_csv(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    frame = build_clean_dataset()
    print(frame[config.TARGET].value_counts().rename("count"))
    print(missing_report(frame))
