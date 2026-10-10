"""Unit tests for data loading and cleaning."""
import numpy as np
import pandas as pd
import pytest

from cardiorisk import config
from cardiorisk.data import build_clean_dataset, clean, load_raw, missing_report


def test_load_raw_has_all_sources(raw_df):
    assert set(raw_df[config.SOURCE_COL]) == set(config.SOURCES)
    assert len(raw_df) == 920
    assert list(raw_df.columns[:14]) == config.RAW_COLUMNS


def test_load_raw_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path)


def test_target_is_binary(clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert "num" not in clean_df.columns


def test_target_mapping(tiny_raw):
    out = clean(tiny_raw)
    # num 0 -> 0, num >= 1 -> 1, missing num dropped
    assert out[config.TARGET].tolist() == [0, 1, 1, 0]


def test_sentinel_zeros_become_nan(tiny_raw):
    out = clean(tiny_raw)
    assert np.isnan(out.loc[1, "chol"])        # chol == 0
    assert np.isnan(out.loc[2, "trestbps"])    # trestbps == 0
    assert (clean(load_raw())["chol"] == 0).sum() == 0


def test_invalid_category_codes_become_nan(tiny_raw):
    out = clean(tiny_raw)
    row = out.loc[3]
    assert np.isnan(row["ca"]) and np.isnan(row["thal"])


def test_duplicates_dropped(tiny_raw):
    assert len(clean(tiny_raw)) == 4


def test_clean_output_schema(clean_df):
    assert list(clean_df.columns) == config.INPUT_FEATURES + [config.SOURCE_COL, config.TARGET]
    for col in config.INPUT_FEATURES:
        assert pd.api.types.is_float_dtype(clean_df[col])


def test_numeric_values_within_valid_ranges(clean_df):
    for col, (lo, hi) in config.VALID_RANGES.items():
        s = clean_df[col].dropna()
        assert s.between(lo, hi).all(), col


def test_clean_does_not_mutate_input(tiny_raw):
    before = tiny_raw.copy()
    clean(tiny_raw)
    pd.testing.assert_frame_equal(before, tiny_raw)


def test_missing_report_shape(clean_df):
    rep = missing_report(clean_df)
    assert "ALL" in rep.index
    assert rep.loc["switzerland", "chol"] == 100.0
    assert ((rep >= 0) & (rep <= 100)).all().all()


def test_build_clean_dataset_writes_csv(tmp_path):
    out = tmp_path / "clean.csv"
    df = build_clean_dataset(out_csv=out)
    assert out.exists()
    assert len(pd.read_csv(out)) == len(df) == 918
