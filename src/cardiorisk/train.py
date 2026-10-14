"""Model selection, tuning, evaluation and packaging with MLflow tracking.

Run:  python -m cardiorisk.train            (uses params.yaml)

MLflow run hierarchy (experiment "heart-disease-risk"):

    training-pipeline-<timestamp>                 parent run: config, data hash, EDA plots
      |- model:logistic_regression                best config + CV/test metrics + plots + model
      |    |- grid:logistic_regression:000 ...    one run per hyper-parameter candidate
      |- model:random_forest
      |- model:gradient_boosting
      |- ablation:missing_indicators              champion family re-run with indicators
      `- champion                                 tuned threshold, per-site metrics, registry

Outputs: models/model.joblib, models/model_metadata.json, reports/metrics.json
"""
from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from . import __version__, config
from .data import file_sha256, load_clean
from .evaluate import (
    CV_SCORING,
    choose_threshold,
    classification_metrics,
    feature_importance,
    plot_calibration,
    plot_confusion,
    plot_importance,
    plot_model_comparison,
    plot_roc_pr,
)
from .features import build_pipeline, split_xy

log = logging.getLogger("cardiorisk.train")

ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}


# ------------------------------------------------------------------ helpers
def load_params(path: Path | str = config.PARAMS_FILE) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def git_commit() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def setup_mlflow(experiment: str) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{config.PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", (config.PROJECT_ROOT / "mlartifacts").as_uri())
        mlflow.create_experiment(experiment, artifact_location=artifact_root)
    mlflow.set_experiment(experiment)
    log.info("MLflow tracking URI: %s", uri)


def make_estimator(spec: dict, seed: int):
    cls = ESTIMATORS[spec["estimator"]]
    kwargs = dict(spec.get("fixed") or {})
    if "random_state" in cls().get_params():
        kwargs["random_state"] = seed
    return cls(**kwargs)


def log_sklearn_model(model, name: str, X_example: pd.DataFrame, registered: str | None = None):
    """Log a model with signature + input example; works on MLflow 2.x and 3.x."""
    signature = infer_signature(X_example, model.predict_proba(X_example)[:, 1])
    # code_paths ships the cardiorisk package (custom transformer) inside the
    # MLflow model so `mlflow.pyfunc.load_model` works in a fresh environment.
    kwargs = dict(signature=signature, input_example=X_example.head(3),
                  registered_model_name=registered, serialization_format="cloudpickle",
                  code_paths=[str(Path(__file__).resolve().parent)],
                  pip_requirements=_serving_requirements())
    if int(mlflow.__version__.split(".")[0]) >= 3:
        return mlflow.sklearn.log_model(model, name=name, **kwargs)
    return mlflow.sklearn.log_model(model, artifact_path=name, **kwargs)


def _serving_requirements() -> list[str] | None:
    """Pinned runtime deps for the MLflow model (stops MLflow guessing, e.g. 'cardiorisk' from PyPI)."""
    req = config.PROJECT_ROOT / "requirements-serve.txt"
    if not req.exists():
        return None
    lines = [ln.split("#")[0].strip() for ln in req.read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("-")]


def _clean_params(d: dict) -> dict:
    return {k.replace("model__", ""): ("None" if v is None else v) for k, v in d.items()}


# --------------------------------------------------------------- main flow
def tune_model(name: str, spec: dict, X, y, cv, params: dict, tmp: Path):
    """Grid search one model family; logs every candidate and the best config."""
    seed = params["seed"]
    pipe = build_pipeline(make_estimator(spec, seed), params["add_missing_indicators"])
    grid = {f"model__{k}": v for k, v in spec["grid"].items()}
    search = GridSearchCV(pipe, grid, scoring=CV_SCORING, refit=params["selection_metric"],
                          cv=cv, n_jobs=-1, return_train_score=True)
    t0 = time.time()
    search.fit(X, y)
    fit_seconds = time.time() - t0
    res = pd.DataFrame(search.cv_results_)

    with mlflow.start_run(run_name=f"model:{name}", nested=True) as run:
        mlflow.set_tags({"run_type": "model", "model_family": name, "estimator": spec["estimator"]})
        # every hyper-parameter candidate gets its own child run
        for i, row in res.iterrows():
            with mlflow.start_run(run_name=f"grid:{name}:{i:03d}", nested=True):
                mlflow.set_tags({"run_type": "grid_candidate", "model_family": name})
                mlflow.log_params(_clean_params(row["params"]))
                mlflow.log_metrics({f"cv_{m}_mean": row[f"mean_test_{m}"] for m in CV_SCORING})
                mlflow.log_metrics({f"cv_{m}_std": row[f"std_test_{m}"] for m in CV_SCORING})
                mlflow.log_metric("cv_train_roc_auc_mean", row["mean_train_roc_auc"])
                mlflow.log_metric("rank_roc_auc", int(row["rank_test_roc_auc"]))

        best = res.loc[search.best_index_]
        cv_metrics = {}
        for m in CV_SCORING:
            cv_metrics[f"cv_{m}_mean"] = float(best[f"mean_test_{m}"])
            cv_metrics[f"cv_{m}_std"] = float(best[f"std_test_{m}"])
        cv_metrics["cv_train_roc_auc_mean"] = float(best["mean_train_roc_auc"])
        cv_metrics["overfit_gap_roc_auc"] = cv_metrics["cv_train_roc_auc_mean"] - cv_metrics["cv_roc_auc_mean"]

        mlflow.log_params({**spec.get("fixed", {}), **_clean_params(search.best_params_),
                           "n_candidates": len(res), "add_missing_indicators": params["add_missing_indicators"]})
        mlflow.log_metrics({**cv_metrics, "grid_fit_seconds": fit_seconds})

        csv = tmp / f"cv_results_{name}.csv"
        res.drop(columns=["params"]).to_csv(csv, index=False)
        mlflow.log_artifact(str(csv), "cv")
        log_sklearn_model(search.best_estimator_, "model", X)
        run_id = run.info.run_id

    log.info("%-20s best CV ROC-AUC %.4f +/- %.4f  (%d candidates, %.1fs)", name,
             cv_metrics["cv_roc_auc_mean"], cv_metrics["cv_roc_auc_std"], len(res), fit_seconds)
    return search, cv_metrics, run_id


def evaluate_on_test(name, model, X_test, y_test, threshold, tmp: Path, prefix="test") -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, proba, threshold)
    figs = [
        plot_roc_pr(y_test, proba, f"{name} - hold-out test", tmp / f"{name}_roc_pr.png"),
        plot_confusion(y_test, proba, threshold, name, tmp / f"{name}_confusion.png"),
        plot_calibration(y_test, proba, name, tmp / f"{name}_calibration.png"),
    ]
    return {f"{prefix}_{k}": v for k, v in metrics.items()}, figs, proba


def run_training(params_path: Path | str = config.PARAMS_FILE) -> dict:
    params = load_params(params_path)
    seed = params["seed"]
    np.random.seed(seed)

    df = load_clean()
    X, y = split_xy(df)
    strata = df[config.SOURCE_COL] + "_" + df[config.TARGET].astype(str)
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, df[config.SOURCE_COL], test_size=params["test_size"], stratify=strata, random_state=seed)
    cv = StratifiedKFold(n_splits=params["cv_folds"], shuffle=True, random_state=seed)

    setup_mlflow(params["experiment_name"])
    tmp = Path(tempfile.mkdtemp(prefix="cardiorisk_"))
    data_hash = file_sha256(config.CLEAN_CSV)
    started = datetime.now(timezone.utc)

    with mlflow.start_run(run_name=f"training-pipeline-{started:%Y%m%d-%H%M%S}") as parent:
        mlflow.set_tags({"run_type": "pipeline", "git_commit": git_commit(),
                         "package_version": __version__, "python": platform.python_version(),
                         "sklearn": sklearn.__version__})
        mlflow.log_params({"seed": seed, "test_size": params["test_size"], "cv_folds": params["cv_folds"],
                           "selection_metric": params["selection_metric"], "data_sha256": data_hash[:16],
                           "n_rows": len(df), "n_train": len(X_train), "n_test": len(X_test),
                           "features": ",".join(config.INPUT_FEATURES)})
        mlflow.log_artifact(str(config.CLEAN_CSV), "data")
        mlflow.log_artifact(str(params_path), "config")
        if config.FIGURES_DIR.exists():
            for p in sorted(config.FIGURES_DIR.glob("0*.png")):
                mlflow.log_artifact(str(p), "eda")

        # ---------------- 1. tune every model family on CV
        results, searches = {}, {}
        for name, spec in params["models"].items():
            search, cv_metrics, run_id = tune_model(name, spec, X_train, y_train, cv, params, tmp)
            searches[name] = search
            test_metrics, figs, _ = evaluate_on_test(name, search.best_estimator_, X_test, y_test, 0.5, tmp)
            with mlflow.start_run(run_id=run_id, nested=True):
                mlflow.log_metrics(test_metrics)
                for f in figs:
                    mlflow.log_artifact(str(f), "plots")
            results[name] = {**cv_metrics, **test_metrics, "best_params": _clean_params(search.best_params_),
                             "run_id": run_id}

        summary = pd.DataFrame(results).T
        comp = plot_model_comparison(summary.astype({c: float for c in summary.columns if c.startswith("cv_")}),
                                     tmp / "model_comparison.png")
        mlflow.log_artifact(str(comp), "comparison")
        summary.drop(columns=["best_params"]).to_csv(tmp / "model_comparison.csv")
        mlflow.log_artifact(str(tmp / "model_comparison.csv"), "comparison")

        # ---------------- 2. champion = best CV score (test set NOT used for selection)
        metric = f"cv_{params['selection_metric']}_mean"
        champ_name = max(results, key=lambda k: results[k][metric])
        champion = searches[champ_name].best_estimator_
        log.info("Champion: %s (%s = %.4f)", champ_name, metric, results[champ_name][metric])

        # ---------------- 3. ablation: same family with missing-value indicators
        abl_spec = params["models"][champ_name]
        abl_pipe = build_pipeline(make_estimator(abl_spec, seed), add_missing_indicators=True)
        abl_pipe.set_params(**searches[champ_name].best_params_)
        abl_oof = cross_val_predict(abl_pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        abl_pipe.fit(X_train, y_train)
        abl_test = abl_pipe.predict_proba(X_test)[:, 1]
        base_oof = cross_val_predict(champion, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        with mlflow.start_run(run_name="ablation:missing_indicators", nested=True):
            mlflow.set_tags({"run_type": "ablation", "model_family": champ_name})
            mlflow.log_param("add_missing_indicators", True)
            abl = {"oof_roc_auc_with_indicators": classification_metrics(y_train, abl_oof)["roc_auc"],
                   "oof_roc_auc_without_indicators": classification_metrics(y_train, base_oof)["roc_auc"],
                   "test_roc_auc_with_indicators": classification_metrics(y_test, abl_test)["roc_auc"]}
            mlflow.log_metrics(abl)
        log.info("Ablation (missing indicators): %s", {k: round(v, 4) for k, v in abl.items()})

        # ---------------- 4. decision threshold from out-of-fold predictions
        tcfg = params["threshold"]
        threshold = choose_threshold(y_train.values, base_oof, tcfg["beta"], tcfg["min"], tcfg["max"],
                                     tcfg.get("min_specificity", 0.0))

        # ---------------- 5. champion evaluation at tuned threshold + per site
        test_metrics, figs, test_proba = evaluate_on_test("champion", champion, X_test, y_test, threshold, tmp)
        per_site = {}
        for site in sorted(src_test.unique()):
            m = (src_test == site).values
            per_site[site] = {k: round(float(v), 4) for k, v in
                              classification_metrics(y_test[m], test_proba[m], threshold).items()
                              if k in ("accuracy", "recall", "precision", "roc_auc")}
            per_site[site]["n"] = int(m.sum())
        imp = feature_importance(champion, X_test, y_test, seed=seed)
        figs.append(plot_importance(imp, champ_name, tmp / "champion_permutation_importance.png"))

        with mlflow.start_run(run_name="champion", nested=True) as champ_run:
            mlflow.set_tags({"run_type": "champion", "model_family": champ_name})
            mlflow.log_params({"model_family": champ_name, "decision_threshold": threshold,
                               **results[champ_name]["best_params"]})
            mlflow.log_metrics({k: v for k, v in results[champ_name].items() if k.startswith("cv_")})
            mlflow.log_metrics(test_metrics)
            mlflow.log_metric("test_accuracy_at_0_5",
                              classification_metrics(y_test, test_proba, 0.5)["accuracy"])
            for site, m in per_site.items():
                mlflow.log_metrics({f"site_{site}_{k}": v for k, v in m.items() if k != "n"})
            for f in figs:
                mlflow.log_artifact(str(f), "plots")
            imp.to_csv(tmp / "permutation_importance.csv")
            mlflow.log_artifact(str(tmp / "permutation_importance.csv"), "plots")
            info = log_sklearn_model(champion, "model", X_train, registered=params["registered_model_name"])
            champ_run_id = champ_run.info.run_id

        registry_version = None
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{params['registered_model_name']}'")
            registry_version = max(int(v.version) for v in versions if v.run_id == champ_run_id)
            client.set_registered_model_alias(params["registered_model_name"], "champion", registry_version)
            client.set_model_version_tag(params["registered_model_name"], str(registry_version),
                                         "decision_threshold", str(threshold))
        except Exception as exc:  # noqa: BLE001 - registry is optional (e.g. file store)
            log.warning("Model registry step skipped: %s", exc)

        mlflow.log_metrics({f"champion_{k}": v for k, v in test_metrics.items()
                            if isinstance(v, float)})
        mlflow.set_tag("champion_family", champ_name)
        parent_id = parent.info.run_id

    # ---------------- 6. package the model for serving
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, config.MODEL_PATH)
    metadata = {
        "model_name": params["registered_model_name"],
        "model_version": f"{__version__}+{data_hash[:8]}",
        "registry_version": registry_version,
        "model_family": champ_name,
        "estimator": params["models"][champ_name]["estimator"],
        "best_params": results[champ_name]["best_params"],
        "decision_threshold": threshold,
        "trained_at": started.isoformat(),
        "git_commit": git_commit(),
        "data_sha256": data_hash,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "input_features": config.INPUT_FEATURES,
        "engineered_features": config.ENGINEERED_FEATURES,
        "cv_metrics": {k: round(v, 4) for k, v in results[champ_name].items() if k.startswith("cv_")},
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "per_site_test_metrics": per_site,
        "mlflow": {"parent_run_id": parent_id, "champion_run_id": champ_run_id,
                   "model_uri": getattr(info, "model_uri", None)},
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))

    report = {
        "champion": champ_name,
        "decision_threshold": threshold,
        "models": {k: {m: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                       for m, v in r.items()} for k, r in results.items()},
        "ablation_missing_indicators": {k: round(v, 4) for k, v in abl.items()},
        "champion_test": metadata["test_metrics"],
        "per_site_test": per_site,
        "permutation_importance": imp.sort_values(ascending=False).round(4).to_dict(),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for f in list(tmp.glob("*.png")):
        # shutil.move, not Path.replace: the temp dir can be on another filesystem
        # (e.g. /tmp on a separate disk), where a rename fails with EXDEV.
        shutil.move(str(f), str(config.FIGURES_DIR / f"model_{f.name}"))
    shutil.rmtree(tmp, ignore_errors=True)
    log.info("Saved %s and %s", config.MODEL_PATH, config.METADATA_PATH)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("mlflow", "alembic", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    report = run_training()
    print(json.dumps({k: report[k] for k in ("champion", "decision_threshold", "champion_test",
                                             "ablation_missing_indicators", "per_site_test")}, indent=2))


if __name__ == "__main__":
    main()
