import logging
import os
import secrets
import time
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import pandas as pd
import joblib

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart_risk_api")

# ── Rate limiting ────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Brute-force auth protection ──────────────────────────────
# Tracks failed API-key attempts per IP. In-memory only — resets on restart.
# For production, swap this dict for Redis.
auth_failure_log = defaultdict(list)
MAX_AUTH_FAILURES = 5       # failed attempts allowed
AUTH_FAILURE_WINDOW = 300   # seconds (5 minutes)

@app.middleware("http")
async def brute_force_auth_limiter(request: Request, call_next):
    client_ip = get_remote_address(request)
    now = time.time()

    # Purge old attempts outside the window
    auth_failure_log[client_ip] = [
        ts for ts in auth_failure_log[client_ip]
        if now - ts < AUTH_FAILURE_WINDOW
    ]

    # If this IP already hit the limit, reject BEFORE ever checking the key
    if len(auth_failure_log[client_ip]) >= MAX_AUTH_FAILURES:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many failed authentication attempts. Please try again later."}
        )

    # Let the request run (auth check happens inside here)
    response = await call_next(request)

    # Record a failed attempt so future requests from this IP get blocked
    if response.status_code == 401:
        auth_failure_log[client_ip].append(now)

    return response


# ── Load trained artifacts ─────────────────────────────────
bundle           = joblib.load("heart_risk_lr_model.pkl")
model            = bundle["model"]
scaler           = bundle["scaler"]
feature_columns  = bundle["feature_columns"]
fill_values      = bundle["fill_values"]
best_threshold   = bundle["best_threshold"]

IMPUTED_COLUMNS = ["totChol", "cigsPerDay", "BPMeds", "BMI", "heartRate", "glucose", "education"]

# ── API key auth ────────────────────────────────────────────
API_KEY = os.environ.get("HEART_RISK_API_KEY")
if not API_KEY:
    raise RuntimeError("HEART_RISK_API_KEY must be set — refusing to start with no auth configured.")

def verify_api_key(x_api_key: str = Header(...)):
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class HeartRiskFeatures(BaseModel):
    male:             int   = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    age:              int   = Field(..., ge=18, le=100)
    education:        Optional[float] = Field(None, ge=1, le=4)
    currentSmoker:    int   = Field(..., ge=0, le=1)
    cigsPerDay:       Optional[float] = Field(None, ge=0, le=100)
    BPMeds:           Optional[float] = Field(None, ge=0, le=1)
    prevalentStroke:  int   = Field(..., ge=0, le=1)
    prevalentHyp:     int   = Field(..., ge=0, le=1)
    diabetes:         int   = Field(..., ge=0, le=1)
    totChol:          Optional[float] = Field(None, gt=50, lt=700)
    sysBP:            float = Field(..., gt=0, lt=300)
    diaBP:            float = Field(..., gt=0, lt=200)
    BMI:              Optional[float] = Field(None, gt=10, lt=80)
    heartRate:        Optional[float] = Field(None, gt=20, lt=250)
    glucose:          Optional[float] = Field(None, gt=20, lt=600)

    @model_validator(mode="after")
    def check_smoking_consistency(self):
        if self.currentSmoker == 0 and self.cigsPerDay not in (None, 0):
            raise ValueError("currentSmoker=0 but cigsPerDay is nonzero — inconsistent input.")
        return self


class PredictionResponse(BaseModel):
    chd_risk_probability: float
    predicted_high_risk:  bool
    threshold_used:        float


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, heart_risk: HeartRiskFeatures, _auth: None = Depends(verify_api_key)):
    input_dict = heart_risk.model_dump()

    input_dict["BMI_was_missing"] = int(input_dict["BMI"] is None)

    for col in IMPUTED_COLUMNS:
        if input_dict[col] is None:
            input_dict[col] = fill_values[col]

    try:
        features = pd.DataFrame([input_dict])
        features = features[feature_columns]

        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = int(probability >= best_threshold)
    except Exception:
        logger.exception("Inference failed for input: %s", input_dict)
        raise HTTPException(status_code=500, detail="Prediction failed — please check your input.")

    logger.info(
        "predict: age=%s sysBP=%s -> probability=%.4f high_risk=%s",
        input_dict["age"], input_dict["sysBP"], probability, bool(prediction)
    )

    return PredictionResponse(
        chd_risk_probability=round(float(probability), 4),
        predicted_high_risk=bool(prediction),
        threshold_used=best_threshold
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Heart Disease Risk Predictor API is running"}