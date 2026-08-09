# Framingham Heart Disease Risk Prediction

Predicting 10-year coronary heart disease (CHD) risk from patient health records, using a tuned Logistic Regression model benchmarked against Random Forest, XGBoost, LightGBM, and CatBoost.

## Overview

- **Target:** `TenYearCHD` — binary, whether the patient developed CHD within 10 years
- **Data:** [Framingham Heart Study dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) — real, publicly available medical data, ~4,240 patient records
- **Models:** Logistic Regression (tuned via GridSearchCV), Random Forest, XGBoost, LightGBM, CatBoost
- **Best model:** Logistic Regression — Test ROC-AUC = 0.705, PR-AUC = 0.315, F1 = 0.337

## Why this dataset

Unlike a synthetic dataset with an engineered target, this is real epidemiological data with genuine missingness, a genuinely weak/diffuse signal, and a known class imbalance (~15% positive rate) — closer to what a real-world healthcare ML task looks like than a dataset built to be easy.

## Pipeline

1. Stratified train/val/test split (80/10/10), preserving the ~85/15 class ratio in every split
2. **Missingness investigation:** checked whether missing values in each column correlated with the target *before* imputing. Found that `BMI`'s missingness was meaningfully informative (52.6% CHD rate among the 19 rows missing BMI vs. ~15% baseline) — added a `BMI_was_missing` indicator column to preserve that signal ahead of imputation
3. **Imputation:** median for numeric columns (`totChol`, `cigsPerDay`, `BMI`, `heartRate`, `glucose`), mode for categorical/binary columns (`BPMeds`, `education`) — all fit on the training split only
4. **Scaling:** `StandardScaler` applied for Logistic Regression only (fit on train, applied to val/test); tree models use unscaled features
5. **Class imbalance handling:** `class_weight='balanced'` (Logistic Regression, Random Forest), `scale_pos_weight` (XGBoost, LightGBM), `auto_class_weights='Balanced'` (CatBoost)
6. **Hyperparameter tuning:** GridSearchCV over penalty (`l1`/`l2`), solver, and `C` for Logistic Regression, scored on average precision (PR-AUC), 5-fold stratified CV
7. **Threshold tuning:** default 0.5 classification threshold is a poor fit for this imbalance — the optimal F1 threshold for each model was found using validation-set probabilities only, then applied to both validation and test predictions
8. Evaluation via Precision, Recall, F1, ROC-AUC, and PR-AUC (accuracy is not used — with an 85/15 split, a model predicting "no CHD" for everyone would score ~85% accuracy while being useless)

## Handling Class Imbalance

Two separate techniques were combined, addressing two separate problems:

- **Class weighting** (during training) — makes the model's loss function penalize misclassifying the minority class more heavily, so it actually learns the minority class exists
- **Threshold tuning** (after training) — the default 0.5 cutoff on `predict()` is not calibrated for an imbalanced problem. Random Forest's recall was 0 on the test set at the default threshold despite reasonable underlying probability estimates (ROC-AUC 0.634); tuning its threshold down to ~0.18 recovered a usable recall of 0.406. Every model's threshold was tuned individually using validation-set probabilities, never test data.

## Results

| Model | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall |
|---|---|---|---|---|---|
| **Logistic Regression (tuned)** | **0.705** | **0.315** | **0.337** | 0.269 | 0.453 |
| Random Forest | 0.634 | 0.249 | 0.280 | 0.213 | 0.406 |
| LightGBM | 0.595 | 0.209 | 0.268 | 0.220 | 0.344 |
| CatBoost | 0.578 | 0.229 | 0.238 | 0.198 | 0.297 |
| XGBoost | 0.554 | 0.199 | 0.231 | 0.183 | 0.313 |

Tuned Logistic Regression outperforms the tree ensembles tried here — but this isn't an apples-to-apples comparison. Only Logistic Regression went through hyperparameter tuning (GridSearchCV); Random Forest, XGBoost, LightGBM, and CatBoost are all running hand-picked, untuned hyperparameters. The honest framing is "a tuned linear model beat several untuned tree ensembles," not "linear models beat tree ensembles on this problem" — the latter would require tuning all five fairly before it could be claimed. See [Possible Extensions](#possible-extensions).

**Is the gap real, or just this particular split?** With a test set this small (~420 rows, ~60 positive cases), a single point-estimate gap could easily be noise. A bootstrap analysis (1000 resamples of the test set) puts the LR-vs-RF PR-AUC difference at a mean of 0.060, with a 95% CI of **[0.0002, 0.118]**. The interval is technically entirely above zero, but the lower bound sits right at the edge — this is a real but fragile signal, not a strongly confirmed one. A different train/test split, or more data, could plausibly shift this conclusion either way.

Notably, the engineered `BMI_was_missing` indicator ranks 5th among LR's coefficients — ahead of `totChol`, `prevalentHyp`, and `diaBP`. The missingness pattern flagged during EDA (52.6% CHD rate among rows missing BMI vs. ~15% baseline) wasn't just an interesting observation — it carried real predictive signal into the final model.

## Leakage Check

`df.corr(numeric_only=True)['TenYearCHD']` shows a maximum correlation of 0.225 (`age`), with a smooth, gradual decline across the remaining features (down to ~0.02) — no single feature dominates. This is consistent with a genuinely hard, real-world prediction problem rather than a hidden shortcut in the data, and matches published ROC-AUC benchmarks (~0.65–0.75) for CHD prediction on this dataset.

## Why Logistic Regression did well here

CHD risk relationships in this dataset (age, blood pressure, cholesterol) are reasonably linear/additive in the medical literature — a good fit for logistic regression's assumptions. Tree ensembles tend to have an advantage when there are strong nonlinear interactions to exploit; the weak, diffuse correlation structure here didn't give them much of that to capitalize on. That said, since the tree models here are untuned, this is a plausible explanation rather than a confirmed one — tuning them could close some or all of the gap. A useful comparison point either way: a companion project (FIFA player rating prediction) saw tree ensembles clearly outperform a linear baseline, so the right model likely depends on the structure of the data rather than a fixed hierarchy of "more complex = better."

## Serving

The saved model bundle (`heart_risk_lr_model.pkl`) includes everything the API needs to reproduce the training pipeline's preprocessing at inference time:
- The tuned Logistic Regression model
- `scaler` (the `StandardScaler` fit on the training split)
- `fill_values` (training-set medians/modes for the imputed columns)
- `feature_columns` (exact column order the model expects, including the engineered `BMI_was_missing` indicator)
- `best_threshold` (the F1-optimal classification threshold found on the validation set)

The API (`main.py`) includes:
- **Missingness-aware imputation** — optional fields (`totChol`, `cigsPerDay`, `BPMeds`, `BMI`, `heartRate`, `glucose`, `education`) can be omitted; if so, they're filled in using the same train-only `fill_values` the training pipeline persisted. `BMI_was_missing` is derived the same way it was during training.
- **Field-level bounds validation** matching plausible clinical ranges (e.g. `sysBP` 0–300, `BMI` 10–80)
- **Consistency check** — rejects requests where `currentSmoker=0` but `cigsPerDay` is nonzero
- API key authentication (fails at startup if no key is configured)
- **Brute-force lockout** — after 5 failed API-key attempts from an IP within a 5-minute window, that IP is rejected with a `429` before the key is even checked again
- Rate limiting (10 requests/minute per IP)
- Error handling around inference (returns a clean `500` without leaking internals on unexpected failures)
- Logging of each prediction

### Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download the dataset from the Kaggle link above and place `framingham.csv` in this folder
3. Run the training pipeline to generate the saved model:
   ```
   python pipeline.py
   ```
   This writes `heart_risk_lr_model.pkl`
4. Create a `.env` file in this folder with your own API key:
   ```
   HEART_RISK_API_KEY=your-own-secret-here
   ```
5. Run the API:
   ```
   uvicorn main:app --reload
   ```
6. Test it interactively at `http://127.0.0.1:8000/docs`

## Possible Extensions

- SMOTE or other resampling techniques as an alternative/complement to class weighting
- Grid search extended to the tree ensembles (currently only Logistic Regression is tuned)
- Calibration curve analysis, since threshold tuning assumes probabilities are reasonably well-calibrated
- Feature interactions (e.g. `age × sysBP`) given the modest individual correlations

## Files

- `pipeline.py` — full pipeline: missingness investigation, imputation, scaling, class-imbalance handling, hyperparameter tuning, threshold tuning, evaluation
- `main.py` — FastAPI serving layer
- `framingham.csv` — input data, not included in repo. Download from [Kaggle: Framingham Heart Study Dataset](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) and place it in this project's folder (`heart-disease-project/`) before running `pipeline.py`
- `heart_risk_lr_model.pkl` — saved model bundle, not included in repo (generated by running `pipeline.py`)
