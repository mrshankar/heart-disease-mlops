"""API tests using FastAPI's TestClient (no server or Docker needed)."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained_model_dir / "model.joblib"))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(trained_model_dir / "model_metadata.json"))
    from cardiorisk.serving import app as app_module
    from cardiorisk.serving.model_store import ModelStore

    app_module.store.__dict__.update(ModelStore.from_env().__dict__)
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ready_reports_model_version(client):
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_version"] == "test+0000"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.45
    assert body["missing_features"] == []
    assert r.headers["x-request-id"] == body["request_id"]


def test_high_vs_low_risk_ordering(client, sample_patient, healthy_patient):
    hi = client.post("/predict", json=sample_patient).json()
    lo = client.post("/predict", json=healthy_patient).json()
    assert hi["probability"] > lo["probability"]
    assert hi["prediction"] == 1 and lo["prediction"] == 0


def test_optional_fields_can_be_omitted(client):
    r = client.post("/predict", json={"age": 61, "sex": 1, "cp": 4, "thalach": 115, "exang": 1})
    assert r.status_code == 200
    assert set(r.json()["missing_features"]) >= {"ca", "thal", "chol"}


@pytest.mark.parametrize("patch", [
    {"age": 5},              # out of range
    {"cp": 7},               # invalid category
    {"chol": "high"},        # wrong type
    {"unknown_field": 1},    # extra field rejected
])
def test_invalid_input_rejected(client, sample_patient, patch):
    r = client.post("/predict", json={**sample_patient, **patch})
    assert r.status_code == 422
    assert r.json()["detail"]


def test_missing_required_field(client, sample_patient):
    sample_patient.pop("age")
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_batch_predict(client, sample_patient, healthy_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient, healthy_patient]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and [p["prediction"] for p in preds] == [1, 0]


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_family"] == "logistic_regression"
    assert body["decision_threshold"] == 0.45


def test_metrics_endpoint_exposes_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "cardiorisk_predictions_total" in text
    assert "cardiorisk_http_request_duration_seconds_bucket" in text


def test_monitoring_dashboard_and_stats(client, sample_patient):
    client.post("/predict", json=sample_patient)
    assert "live monitor" in client.get("/monitoring").text
    stats = client.get("/monitoring/stats").json()
    assert stats["predictions_total"] >= 1 and stats["recent"]


def test_not_ready_returns_503(client, sample_patient):
    from cardiorisk.serving import app as app_module

    saved = app_module.store.model
    app_module.store.model = None
    try:
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=sample_patient).status_code == 503
    finally:
        app_module.store.model = saved


@pytest.mark.parametrize("env_value, expected", [("", 0.45), ("0.6", 0.6)])
def test_threshold_override_from_env(client, sample_patient, monkeypatch, env_value, expected):
    monkeypatch.setenv("DECISION_THRESHOLD", env_value)
    assert client.post("/predict", json=sample_patient).json()["threshold"] == expected
