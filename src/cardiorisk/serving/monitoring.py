"""Prometheus metrics and structured (JSON) request logging."""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS = Counter("cardiorisk_http_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("cardiorisk_http_request_duration_seconds", "Request latency",
                    ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("cardiorisk_predictions_total", "Predictions served", ["outcome", "risk_band"])
PROBABILITY = Histogram("cardiorisk_prediction_probability", "Distribution of predicted disease probability",
                        buckets=[i / 10 for i in range(1, 11)])
MISSING_INPUTS = Counter("cardiorisk_missing_inputs_total", "Optional inputs that had to be imputed",
                         ["feature"])
VALIDATION_ERRORS = Counter("cardiorisk_validation_errors_total", "Rejected requests (HTTP 422)")
IN_FLIGHT = Gauge("cardiorisk_requests_in_flight", "Requests currently being processed")
MODEL_INFO = Info("cardiorisk_model", "Metadata of the loaded model")

# small in-memory ring buffer feeding the /monitoring dashboard
RECENT = deque(maxlen=200)
STARTED_AT = time.time()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = True  # we log requests ourselves


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"extra_fields": fields})
