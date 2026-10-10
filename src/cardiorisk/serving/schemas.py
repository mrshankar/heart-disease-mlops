"""Request/response contracts for the prediction API."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """One patient's clinical measurements (UCI coding).

    `ca`, `thal` and `slope` are often not measured in practice, so they are
    optional - the model pipeline imputes them exactly as during training.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
                    "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2,
                    "ca": 2, "thal": 7}
    })

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: Optional[float] = Field(None, ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: Optional[float] = Field(None, ge=80, le=650, description="Serum cholesterol (mg/dl)")
    fbs: Optional[Literal[0, 1]] = Field(None, description="Fasting blood sugar > 120 mg/dl")
    restecg: Optional[Literal[0, 1, 2]] = Field(None, description="Resting ECG result")
    thalach: Optional[float] = Field(None, ge=50, le=230, description="Maximum heart rate achieved")
    exang: Optional[Literal[0, 1]] = Field(None, description="Exercise-induced angina")
    oldpeak: Optional[float] = Field(None, ge=-3, le=7, description="ST depression induced by exercise")
    slope: Optional[Literal[1, 2, 3]] = Field(None, description="Slope of peak exercise ST segment")
    ca: Optional[Literal[0, 1, 2, 3]] = Field(None, description="Major vessels coloured by fluoroscopy")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed defect, 7 reversible defect")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="1 = heart disease risk present")
    label: str
    probability: float = Field(..., description="Model probability of heart disease")
    confidence: float = Field(..., description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]
    threshold: float
    missing_features: list[str] = Field(default_factory=list,
                                        description="Optional inputs that were not supplied and got imputed")


class PredictionResponse(Prediction):
    request_id: str
    model_version: str


class BatchResponse(BaseModel):
    request_id: str
    model_version: str
    predictions: list[Prediction]
