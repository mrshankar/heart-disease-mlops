"""Feature engineering and the preprocessing pipeline.

The whole transformation lives inside one sklearn Pipeline that is saved
with the model, so the API gets raw clinical values and applies exactly the
same steps as training (no train/serve skew).

Pipeline layout
    raw 13 features
      -> ClinicalFeatures   adds hr_reserve_pct and age_x_oldpeak
      -> ColumnTransformer
           numeric:     median impute -> RobustScaler (outlier-tolerant)
           categorical: missing -> -1 level -> one-hot (unknown ignored)
      -> classifier
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from . import config


class ClinicalFeatures(BaseEstimator, TransformerMixin):
    """Add domain-motivated features.

    hr_reserve_pct  thalach / (220 - age): how close the patient got to the
                    age-predicted maximum heart rate. Low values (chronotropic
                    incompetence) are a known cardiac risk marker and remove the
                    age confounding from raw thalach.
    age_x_oldpeak   ST depression weighted by age - ST changes in older
                    patients are more often ischaemic.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=config.INPUT_FEATURES).copy()
        predicted_max = (220.0 - X["age"]).clip(lower=100.0)
        X["hr_reserve_pct"] = X["thalach"] / predicted_max
        X["age_x_oldpeak"] = X["age"] * X["oldpeak"] / 50.0
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(config.INPUT_FEATURES + config.ENGINEERED_FEATURES, dtype=object)


def build_preprocessor(add_missing_indicators: bool = False) -> Pipeline:
    """Return the feature pipeline (no estimator).

    add_missing_indicators adds a binary "was missing" column per numeric
    feature. It is OFF by default: in this dataset missingness is mostly a
    proxy for which hospital the patient came from (e.g. every Swiss row lacks
    cholesterol and 93% of Swiss patients are positive), so indicators let the
    model learn the site rather than the physiology. The trade-off is measured
    as an ablation in train.py and logged to MLflow.
    """
    numeric = config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
        ("scale", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, config.CATEGORICAL_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([("clinical", ClinicalFeatures()), ("columns", columns)])


def build_pipeline(estimator, add_missing_indicators: bool = False) -> Pipeline:
    """Full model pipeline: preprocessing + estimator under the step name 'model'."""
    return Pipeline([("prep", build_preprocessor(add_missing_indicators)), ("model", estimator)])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.INPUT_FEATURES].copy(), df[config.TARGET].astype(int).copy()
