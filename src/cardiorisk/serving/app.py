"""FastAPI service for heart disease risk prediction.

Endpoints
    GET  /                 service info
    GET  /health           liveness probe
    GET  /ready            readiness probe (model loaded)
    GET  /model-info       model metadata (version, metrics, threshold)
    POST /predict          one patient  -> prediction + confidence
    POST /predict/batch    up to 500 patients
    GET  /metrics          Prometheus exposition format
    GET  /monitoring       lightweight live dashboard (HTML)
    GET  /monitoring/stats JSON feeding the dashboard

Run locally:  uvicorn cardiorisk.serving.app:app --port 8000
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from . import monitoring as mon
from .model_store import ModelStore
from .schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse

mon.configure_logging(os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("cardiorisk.api")
store = ModelStore.from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        store.load()
        store.warmup()
        mon.MODEL_INFO.info({"version": store.version,
                             "family": str(store.metadata.get("model_family", "unknown")),
                             "threshold": str(store.threshold)})
        mon.log_event(log, "model_loaded", model_version=store.version,
                      model_family=store.metadata.get("model_family"), threshold=store.threshold)
    except Exception:  # noqa: BLE001 - keep the pod alive; /ready reports not-ready
        log.exception("Model failed to load")
    yield


app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan,
              description="Predicts heart disease risk from 13 clinical features (UCI coding).")


@app.middleware("http")
async def observe(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    endpoint = request.url.path
    observed = endpoint not in ("/metrics", "/monitoring/stats")   # don't measure the scrapers
    if observed:
        mon.IN_FLIGHT.inc()
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        if observed:
            mon.IN_FLIGHT.dec()
            mon.REQUESTS.labels(request.method, endpoint, str(status)).inc()
            mon.LATENCY.labels(endpoint).observe(elapsed)
            mon.log_event(log, "request", request_id=request_id, method=request.method, path=endpoint,
                          status=status, latency_ms=round(elapsed * 1000, 2),
                          client=request.client.host if request.client else None)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    mon.VALIDATION_ERRORS.inc()
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "error": e["msg"]} for e in exc.errors()]
    mon.log_event(log, "validation_error", request_id=getattr(request.state, "request_id", None), errors=errors)
    return JSONResponse(status_code=422, content={"detail": errors})


def _require_model() -> None:
    if not store.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")


def _record(preds: list[dict], request_id: str) -> None:
    for p in preds:
        mon.PREDICTIONS.labels(str(p["prediction"]), p["risk_band"]).inc()
        mon.PROBABILITY.observe(p["probability"])
        for f in p["missing_features"]:
            mon.MISSING_INPUTS.labels(f).inc()
        mon.RECENT.append({"t": time.time(), "p": p["probability"], "y": p["prediction"], "id": request_id})


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "version": __version__, "model_version": store.version,
            "docs": "/docs", "endpoints": ["/predict", "/predict/batch", "/health", "/ready",
                                           "/model-info", "/metrics", "/monitoring"]}


@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - mon.STARTED_AT, 1)}


@app.get("/ready")
def ready():
    if not store.ready:
        return JSONResponse(status_code=503, content={"ready": False})
    return {"ready": True, "model_version": store.version}


@app.get("/model-info")
def model_info():
    _require_model()
    keys = ("model_name", "model_version", "registry_version", "model_family", "estimator", "best_params",
            "decision_threshold", "trained_at", "git_commit", "cv_metrics", "test_metrics", "input_features")
    return {k: store.metadata.get(k) for k in keys}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    _require_model()
    rid = request.state.request_id
    pred = store.predict([patient.model_dump()])[0]
    _record([pred], rid)
    mon.log_event(log, "prediction", request_id=rid, prediction=pred["prediction"],
                  probability=pred["probability"], risk_band=pred["risk_band"],
                  missing=pred["missing_features"])
    return {**pred, "request_id": rid, "model_version": store.version}


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    _require_model()
    rid = request.state.request_id
    preds = store.predict([p.model_dump() for p in batch.patients])
    _record(preds, rid)
    mon.log_event(log, "batch_prediction", request_id=rid, size=len(preds),
                  positives=sum(p["prediction"] for p in preds))
    return {"request_id": rid, "model_version": store.version, "predictions": preds}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/stats")
def monitoring_stats():
    def total(counter):
        return sum(s.value for m in counter.collect() for s in m.samples if s.name.endswith("_total"))

    by_status, lat_sum, lat_count = {}, 0.0, 0.0
    for m in mon.REQUESTS.collect():
        for s in m.samples:
            if s.name.endswith("_total"):
                by_status[s.labels["status"]] = by_status.get(s.labels["status"], 0) + s.value
    for m in mon.LATENCY.collect():
        for s in m.samples:
            if s.name.endswith("_sum") and s.labels.get("endpoint") == "/predict":
                lat_sum = s.value
            if s.name.endswith("_count") and s.labels.get("endpoint") == "/predict":
                lat_count = s.value
    recent = list(mon.RECENT)
    return {
        "model_version": store.version,
        "uptime_s": round(time.time() - mon.STARTED_AT, 1),
        "requests_by_status": by_status,
        "predictions_total": total(mon.PREDICTIONS),
        "validation_errors": total(mon.VALIDATION_ERRORS),
        "avg_predict_latency_ms": round(lat_sum / lat_count * 1000, 2) if lat_count else None,
        "positive_rate_recent": round(sum(r["y"] for r in recent) / len(recent), 3) if recent else None,
        "recent": recent[-60:],
    }


@app.get("/monitoring", response_class=HTMLResponse)
def monitoring_page():
    return resources.files("cardiorisk.serving").joinpath("dashboard.html").read_text()
