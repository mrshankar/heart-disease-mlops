"""Loads the packaged model + metadata and turns raw inputs into predictions."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .. import config

log = logging.getLogger(__name__)

OPTIONAL_FIELDS = [f for f in config.INPUT_FEATURES if f not in ("age", "sex", "cp")]


@dataclass
class ModelStore:
    model_path: Path
    metadata_path: Path
    model: object = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ModelStore":
        return cls(Path(os.getenv("MODEL_PATH", config.MODEL_PATH)),
                   Path(os.getenv("MODEL_METADATA_PATH", config.METADATA_PATH)))

    def load(self) -> "ModelStore":
        log.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text())
        else:
            log.warning("No metadata at %s - using defaults", self.metadata_path)
            self.metadata = {"model_version": "unknown", "decision_threshold": 0.5}
        return self

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        override = os.getenv("DECISION_THRESHOLD", "").strip()   # empty = use training threshold
        return float(override) if override else float(self.metadata.get("decision_threshold", 0.5))

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict(self, records: list[dict]) -> list[dict]:
        frame = pd.DataFrame.from_records(records, columns=config.INPUT_FEATURES).astype(float)
        proba = self.model.predict_proba(frame)[:, 1]
        thr = self.threshold
        out = []
        for rec, p in zip(records, proba):
            pred = int(p >= thr)
            out.append({
                "prediction": pred,
                "label": "heart disease risk" if pred else "no heart disease risk",
                "probability": round(float(p), 4),
                "confidence": round(float(p if pred else 1 - p), 4),
                "risk_band": "high" if p >= 0.7 else ("moderate" if p >= thr else "low"),
                "threshold": thr,
                "missing_features": [f for f in OPTIONAL_FIELDS if rec.get(f) is None],
            })
        return out

    def warmup(self) -> None:
        example = {f: np.nan for f in config.INPUT_FEATURES} | {"age": 50, "sex": 1, "cp": 2}
        self.predict([example])
