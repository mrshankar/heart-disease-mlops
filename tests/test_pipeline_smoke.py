"""End-to-end smoke tests: EDA + a miniature training run into temp dirs."""
import json

import joblib
import pytest
import yaml

from cardiorisk import config


@pytest.fixture()
def isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "models" / "model_metadata.json")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "artifacts").as_uri())
    return tmp_path


def test_eda_produces_figures(isolated_outputs):
    from cardiorisk.eda import run_eda

    (isolated_outputs / "reports").mkdir()
    summary = run_eda(isolated_outputs / "reports" / "figures")
    assert summary["rows"] == 918
    assert len(list((isolated_outputs / "reports" / "figures").glob("*.png"))) == 7


def test_mini_training_run(isolated_outputs, monkeypatch):
    import errno
    import pathlib

    from cardiorisk.train import run_training

    # Regression: on the Prayogshala VM /tmp is a different filesystem from $HOME, so
    # renames out of the temp dir fail with EXDEV. Simulate that for every rename.
    def no_cross_device_rename(self, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link (simulated)")

    monkeypatch.setattr(pathlib.Path, "replace", no_cross_device_rename)
    monkeypatch.setattr(pathlib.Path, "rename", no_cross_device_rename)

    params = yaml.safe_load(config.PARAMS_FILE.read_text())
    params["experiment_name"] = "smoke-test"
    params["cv_folds"] = 3
    params["models"] = {
        "logistic_regression": {"estimator": "LogisticRegression", "fixed": {"solver": "liblinear"},
                                "grid": {"C": [0.1, 1.0]}},
        "random_forest": {"estimator": "RandomForestClassifier", "fixed": {"n_estimators": 30},
                          "grid": {"max_depth": [4]}},
    }
    pfile = isolated_outputs / "params.yaml"
    pfile.write_text(yaml.safe_dump(params))
    (isolated_outputs / "reports" / "figures").mkdir(parents=True)

    report = run_training(pfile)

    assert report["champion"] in params["models"]
    assert report["champion_test"]["test_roc_auc"] > 0.8
    meta = json.loads(config.METADATA_PATH.read_text())
    assert params["threshold"]["min"] <= meta["decision_threshold"] <= params["threshold"]["max"]
    assert meta["input_features"] == config.INPUT_FEATURES
    model = joblib.load(config.MODEL_PATH)
    assert hasattr(model, "predict_proba")

    import mlflow

    runs = mlflow.search_runs(experiment_names=["smoke-test"])
    kinds = set(runs["tags.run_type"])
    assert {"pipeline", "model", "grid_candidate", "ablation", "champion"} <= kinds
    assert (config.FIGURES_DIR / "model_champion_roc_pr.png").exists()
