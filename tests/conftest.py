"""Shared fixtures. Tests never depend on a previously trained model on disk:
a small model is trained into a temp dir so the suite runs from a clean clone."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cardiorisk import config
from cardiorisk.data import clean, load_raw
from cardiorisk.features import build_pipeline, split_xy


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture()
def tiny_raw() -> pd.DataFrame:
    """Hand-made rows covering the cleaning edge cases."""
    rows = [
        # age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal num source
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0, "cleveland"],
        [67, 1, 4, 160, 0, 0, 2, 108, 1, 1.5, 2, 3, 3, 2, "switzerland"],   # chol 0 -> NaN
        [54, 0, 3, 0, 250, 0, 0, 187, 0, 0.0, np.nan, np.nan, np.nan, 1, "va_long_beach"],  # bp 0
        [45, 1, 2, 130, 204, 0, 0, 172, 0, 1.4, 1, 9, 5, 0, "hungarian"],   # ca/thal invalid
        [45, 1, 2, 130, 204, 0, 0, 172, 0, 1.4, 1, 9, 5, 0, "hungarian"],   # duplicate
        [50, 1, 4, 120, 900, 0, 0, 300, 1, 1.0, 2, 0, 7, np.nan, "cleveland"],  # no target
    ]
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS + [config.SOURCE_COL])


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory, clean_df):
    """Fit a quick LR pipeline and write model.joblib + metadata like train.py does."""
    import joblib

    out = tmp_path_factory.mktemp("model")
    X, y = split_xy(clean_df)
    pipe = build_pipeline(LogisticRegression(C=0.3, penalty="l1", solver="liblinear", max_iter=2000))
    pipe.fit(X, y)
    joblib.dump(pipe, out / "model.joblib")
    (out / "model_metadata.json").write_text(json.dumps({
        "model_version": "test+0000", "decision_threshold": 0.45, "model_family": "logistic_regression",
        "input_features": config.INPUT_FEATURES}))
    return out


@pytest.fixture()
def sample_patient() -> dict:
    return {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0, "restecg": 2,
            "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7}


@pytest.fixture()
def healthy_patient() -> dict:
    return {"age": 37, "sex": 0, "cp": 3, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0,
            "thalach": 185, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3}
