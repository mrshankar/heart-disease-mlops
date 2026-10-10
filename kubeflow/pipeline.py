"""Kubeflow Pipelines (KFP v2) version of the training workflow - for BITS Prayogshala.

The same steps as the GitHub Actions `test` + `train` jobs, run as pipeline steps on
the Kubeflow cluster:

    fetch-and-test  ->  train-and-track  ->  quality-gate  ->  (passes) publish-model

Each step downloads the repository as a zip (no git needed in the base image),
installs the pinned requirements and calls the project's own modules, so the
pipeline cannot drift from the code tested in CI.

    pip install "kfp>=2.7,<3"
    python kubeflow/pipeline.py --compile-only          # -> kubeflow/heart_disease_pipeline.yaml
    # upload the YAML in the Kubeflow Pipelines UI, or submit from a notebook:
    python kubeflow/pipeline.py --repo-zip-url https://github.com/<you>/heart-disease-mlops/archive/refs/heads/main.zip
"""

import argparse
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Input, Metrics, Model, Output

BASE_IMAGE = "python:3.11-slim"


@dsl.component(base_image=BASE_IMAGE)
def fetch_and_test(repo_zip_url: str) -> str:
    """Download the repo, install pinned deps, run flake8 + pytest. Returns 'ok' or raises."""
    import io
    import pathlib
    import subprocess
    import sys
    import urllib.request
    import zipfile

    data = urllib.request.urlopen(repo_zip_url, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall("/tmp/src")
    root = next(p for p in pathlib.Path("/tmp/src").iterdir() if p.is_dir())
    run = lambda c: subprocess.run(c, shell=True, check=True, cwd=root)  # noqa: E731
    run(f"{sys.executable} -m pip install -q -r requirements.txt && {sys.executable} -m pip install -q -e .")
    run(f"{sys.executable} -m flake8 src tests scripts")
    run(f"{sys.executable} -m pytest -q")
    return "ok"


@dsl.component(base_image=BASE_IMAGE)
def train_and_track(repo_zip_url: str, tests_status: str, model: Output[Model], metrics: Output[Metrics]) -> float:
    """EDA + training with MLflow (sqlite store inside the step); exports model + KFP metrics."""
    import io
    import json
    import shutil
    import subprocess
    import sys
    import urllib.request
    import zipfile
    from pathlib import Path

    assert tests_status == "ok"
    data = urllib.request.urlopen(repo_zip_url, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall("/tmp/src")
    root = next(p for p in Path("/tmp/src").iterdir() if p.is_dir())
    run = lambda c: subprocess.run(c, shell=True, check=True, cwd=root)  # noqa: E731
    run(f"{sys.executable} -m pip install -q -r requirements.txt && {sys.executable} -m pip install -q -e .")
    run(f"{sys.executable} scripts/download_data.py --skip-if-present")
    run(f"{sys.executable} -m cardiorisk.eda")
    run(f"{sys.executable} -m cardiorisk.train")

    meta = json.loads((root / "models/model_metadata.json").read_text())
    for k, v in {**meta["cv_metrics"], **meta["test_metrics"]}.items():
        metrics.log_metric(k, float(v))
    metrics.log_metric("decision_threshold", float(meta["decision_threshold"]))

    out = Path(model.path)
    out.mkdir(parents=True, exist_ok=True)
    for f in ["models/model.joblib", "models/model_metadata.json", "reports/metrics.json", "mlflow.db"]:
        shutil.copy(root / f, out / Path(f).name)
    shutil.make_archive(str(out / "mlartifacts"), "zip", root / "mlartifacts")
    model.metadata.update({"framework": "scikit-learn", "family": meta["model_family"],
                           "version": meta["model_version"], "threshold": meta["decision_threshold"]})
    return float(meta["test_metrics"]["test_roc_auc"])


@dsl.component(base_image=BASE_IMAGE)
def quality_gate(test_roc_auc: float, min_roc_auc: float) -> bool:
    print(f"test ROC-AUC {test_roc_auc:.4f} vs gate {min_roc_auc}")
    return test_roc_auc >= min_roc_auc


@dsl.component(base_image=BASE_IMAGE)
def publish_model(model: Input[Model], approved: Output[Model]):
    """Marks the model as approved (copy). In a fuller setup this would push to a registry/bucket."""
    import shutil

    shutil.copytree(model.path, approved.path, dirs_exist_ok=True)
    approved.metadata.update({**model.metadata, "stage": "approved"})
    print("approved model:", approved.metadata)


@dsl.pipeline(name="heart-disease-risk-training",
              description="Test -> train (MLflow) -> quality gate -> publish, for the UCI heart disease classifier")
def heart_disease_pipeline(repo_zip_url: str, min_roc_auc: float = 0.85):
    tests = fetch_and_test(repo_zip_url=repo_zip_url)
    trained = train_and_track(repo_zip_url=repo_zip_url, tests_status=tests.output)
    trained.set_cpu_limit("2").set_memory_limit("4G")
    gate = quality_gate(test_roc_auc=trained.outputs["Output"], min_roc_auc=min_roc_auc)
    with dsl.If(gate.output == True):  # noqa: E712 - KFP condition syntax
        publish_model(model=trained.outputs["model"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--repo-zip-url", default="")
    ap.add_argument("--host", default=None, help="KFP endpoint; default = in-cluster from a Kubeflow notebook")
    args = ap.parse_args()

    out = Path(__file__).with_name("heart_disease_pipeline.yaml")
    compiler.Compiler().compile(heart_disease_pipeline, str(out))
    print("compiled", out)
    if not args.compile_only:
        import kfp

        assert args.repo_zip_url, "--repo-zip-url is required to submit"
        run = kfp.Client(host=args.host).create_run_from_pipeline_package(
            str(out), arguments={"repo_zip_url": args.repo_zip_url}, experiment_name="heart-disease-mlops")
        print("submitted run", run.run_id)


if __name__ == "__main__":
    main()
