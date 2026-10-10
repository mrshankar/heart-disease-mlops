"""Unit tests for feature engineering, the model pipeline and evaluation helpers."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from cardiorisk import config
from cardiorisk.evaluate import choose_threshold, classification_metrics
from cardiorisk.features import ClinicalFeatures, build_pipeline, build_preprocessor, split_xy
from cardiorisk.train import load_params, make_estimator


def test_clinical_features_values():
    X = pd.DataFrame([{**{f: 1.0 for f in config.INPUT_FEATURES}, "age": 60, "thalach": 128, "oldpeak": 2.5}])
    out = ClinicalFeatures().transform(X)
    assert out.loc[0, "hr_reserve_pct"] == pytest.approx(128 / 160)
    assert out.loc[0, "age_x_oldpeak"] == pytest.approx(60 * 2.5 / 50)
    assert list(ClinicalFeatures().get_feature_names_out()) == config.INPUT_FEATURES + config.ENGINEERED_FEATURES


def test_preprocessor_handles_missing_values(clean_df):
    X, _ = split_xy(clean_df)
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()
    assert Xt.shape[0] == len(X)


def test_missing_indicator_option_adds_columns(clean_df):
    X, _ = split_xy(clean_df)
    base = build_preprocessor(False).fit_transform(X).shape[1]
    with_ind = build_preprocessor(True).fit_transform(X).shape[1]
    assert with_ind > base


def test_preprocessor_ignores_unknown_category(clean_df):
    X, _ = split_xy(clean_df)
    prep = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["cp"] = 99  # unseen level must not crash serving
    assert prep.transform(row).shape[1] == prep.transform(X.iloc[[0]]).shape[1]


@pytest.mark.parametrize("estimator", [
    LogisticRegression(max_iter=2000, solver="liblinear"),
    RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0),
])
def test_pipeline_learns_signal(clean_df, estimator):
    X, y = split_xy(clean_df)
    auc = cross_val_score(build_pipeline(estimator), X, y, cv=3, scoring="roc_auc").mean()
    assert auc > 0.80  # well above chance - guards against broken preprocessing


def test_predict_proba_shape_and_range(trained_model_dir, clean_df):
    import joblib

    model = joblib.load(trained_model_dir / "model.joblib")
    X, _ = split_xy(clean_df)
    proba = model.predict_proba(X.head(20))
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_training_is_reproducible(clean_df):
    X, y = split_xy(clean_df)
    p1 = build_pipeline(RandomForestClassifier(n_estimators=30, random_state=7)).fit(X, y).predict_proba(X)
    p2 = build_pipeline(RandomForestClassifier(n_estimators=30, random_state=7)).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_model_handles_all_optional_missing(trained_model_dir):
    import joblib

    model = joblib.load(trained_model_dir / "model.joblib")
    row = pd.DataFrame([{f: np.nan for f in config.INPUT_FEATURES} | {"age": 50, "sex": 1, "cp": 4}])
    assert 0 <= model.predict_proba(row)[0, 1] <= 1


def test_classification_metrics_perfect():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["accuracy"] == 1 and m["roc_auc"] == 1 and m["recall"] == 1 and m["specificity"] == 1


def test_classification_metrics_threshold_changes_predictions():
    y, p = [0, 1, 1], [0.3, 0.4, 0.9]
    assert classification_metrics(y, p, 0.5)["recall"] == 0.5
    assert classification_metrics(y, p, 0.35)["recall"] == 1.0


def test_choose_threshold_respects_bounds_and_specificity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.4 + rng.random(400) * 0.6, 0, 1)
    t = choose_threshold(y, p, beta=2, lo=0.25, hi=0.65)
    assert 0.25 <= t <= 0.65
    t_strict = choose_threshold(y, p, beta=2, lo=0.25, hi=0.65, min_specificity=0.9)
    assert t_strict >= t
    spec = ((p[y == 0] >= t_strict) == 0).mean()
    assert spec >= 0.9 or t_strict == 0.65


def test_params_file_valid():
    params = load_params()
    assert params["selection_metric"] == "roc_auc"
    assert len(params["models"]) >= 2
    for spec in params["models"].values():
        est = make_estimator(spec, 42)
        assert hasattr(est, "predict_proba")
