"""
Basic test suite for the Heart Disease Risk Predictor API.

Run with: pytest test_main.py -v

Requires HEART_RISK_API_KEY to be set (loaded via .env, same as main.py)
and heart_risk_lr_model.pkl to exist in the working directory (generated
by running pipeline.py).
"""
import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
API_KEY = os.environ.get("HEART_RISK_API_KEY")
HEADERS = {"x-api-key": API_KEY}

# A known-good, realistic input used as the baseline for several tests
VALID_INPUT = {
    "male": 1,
    "age": 55,
    "education": 2,
    "currentSmoker": 0,
    "cigsPerDay": 0,
    "BPMeds": 0,
    "prevalentStroke": 0,
    "prevalentHyp": 1,
    "diabetes": 0,
    "totChol": 240,
    "sysBP": 140,
    "diaBP": 90,
    "BMI": 27.5,
    "heartRate": 75,
    "glucose": 85,
}


def test_known_input_returns_stable_probability():
    """Regression test — catches silent model/scaler drift.
    If this value ever changes without an intentional retrain, something
    in the saved bundle (model, scaler, or fill_values) has drifted.

    NOTE: the expected value below (0.581) has NOT been verified against
    an actual trained model — it's a placeholder. Before relying on this
    test, run the API once with VALID_INPUT, read the real
    chd_risk_probability from the response, and replace 0.581 with that
    value. Until then this test may fail for reasons that have nothing
    to do with a real bug.
    """
    response = client.post("/predict", json=VALID_INPUT, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert "chd_risk_probability" in body
    assert 0.0 <= body["chd_risk_probability"] <= 1.0
    assert body["chd_risk_probability"] == pytest.approx(0.581, abs=0.001)  # TODO: confirm against a real run


def test_out_of_bounds_age_rejected():
    """Malformed input should return a clean 4xx, never a 500 or a
    confident-looking prediction for an impossible patient."""
    bad_input = {**VALID_INPUT, "age": 0}
    response = client.post("/predict", json=bad_input, headers=HEADERS)
    assert response.status_code == 422


def test_wrong_type_rejected():
    bad_input = {**VALID_INPUT, "age": "fifty"}
    response = client.post("/predict", json=bad_input, headers=HEADERS)
    assert response.status_code == 422


def test_missing_bmi_is_imputed():
    """Confirms the optional-field + server-side imputation path works,
    and that BMI_was_missing is set internally without the caller
    supplying it."""
    input_without_bmi = {k: v for k, v in VALID_INPUT.items() if k != "BMI"}
    response = client.post("/predict", json=input_without_bmi, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["chd_risk_probability"] <= 1.0


def test_smoking_inconsistency_rejected():
    """currentSmoker=0 with nonzero cigsPerDay should be rejected by the
    cross-field validator, not silently passed to the model."""
    bad_input = {**VALID_INPUT, "currentSmoker": 0, "cigsPerDay": 5}
    response = client.post("/predict", json=bad_input, headers=HEADERS)
    assert response.status_code == 422


def test_missing_api_key_rejected():
    response = client.post("/predict", json=VALID_INPUT)  # no headers
    assert response.status_code in (401, 422)  # 422 if FastAPI treats header as missing required param


def test_wrong_api_key_rejected():
    response = client.post("/predict", json=VALID_INPUT, headers={"x-api-key": "wrong-key"})
    assert response.status_code == 401


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"