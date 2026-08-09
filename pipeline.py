import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)

# ── Config ────────────────────────────────────────────────

config = {
    'random_state':  42,
    'test_size':     0.2,
    'val_test_size': 0.5
}


def main():
    df = pd.read_csv("framingham.csv")

    # X/y defining
    X = df.drop(columns='TenYearCHD')
    y = df["TenYearCHD"]

    # train/test/val split (val_test_size splits the remaining 20% into two 10% halves)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=config["random_state"], test_size=config["test_size"], stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=config["random_state"], test_size=config["val_test_size"], stratify=y_test)

    # ~85/15 class split -> used below for scale_pos_weight in XGBoost and LightGBM
    # (RF uses its own class_weight='balanced'; CatBoost uses auto_class_weights='Balanced' — neither touches this variable)
    counts = y_train.value_counts()
    neg, pos = counts[0], counts[1]
    weight = neg / pos

    # ── Baseline model params ─────────────────────────────────
    # NOTE: no 'penalty'/'l1_ratio' here — both now live in lr_grid below,
    # since sklearn 1.8+ deprecated 'penalty' in favor of 'l1_ratio', and
    # mixing both (as the old version of this file did) throws a
    # "penalty is deprecated, please use l1_ratio only" warning.

    lr_params = {
        'max_iter':     1000,
        'random_state': config["random_state"],
        'class_weight': 'balanced'
    }

    # ── Manual/default model hyperparameters (untuned) ────────

    rf_params = {
        'n_estimators': 300,
        'random_state': config["random_state"],
        'n_jobs':       -1,
        'class_weight': 'balanced'
    }

    xgb_params = {
        'n_estimators':  500,
        'learning_rate': 0.1,
        'max_depth':     6,
        'random_state':  config["random_state"],
        'n_jobs':        -1,
        'scale_pos_weight': weight
    }

    lgbm_params = {
        'n_estimators':  500,
        'learning_rate': 0.05,
        'num_leaves':    31,
        'random_state':  config["random_state"],
        'n_jobs':        -1,
        'verbose':       -1,
        'objective':     'binary',
        'scale_pos_weight': weight
    }

    cat_params = {
        'iterations':    500,
        'learning_rate': 0.1,
        'depth':         6,
        'random_state':  config["random_state"],
        'verbose':       0,
        'auto_class_weights': 'Balanced'
    }

    # missingness indicator — added before imputation so it can still detect the NaNs
    X_train["BMI_was_missing"] = X_train["BMI"].isna().astype(int)
    X_val["BMI_was_missing"]   = X_val["BMI"].isna().astype(int)
    X_test["BMI_was_missing"]  = X_test["BMI"].isna().astype(int)

    # Numeric columns -> median from training set only
    fill_values= {}
    numeric_cols_with_na = ["totChol", "cigsPerDay", "BMI", "heartRate", "glucose"]
    for col in numeric_cols_with_na:
        median_val = X_train[col].median()
        fill_values[col] = median_val
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col]   = X_val[col].fillna(median_val)
        X_test[col]  = X_test[col].fillna(median_val)

    # Categorical/binary columns -> mode from training set only
    mode_cols_with_na = ["BPMeds", "education"]
    for col in mode_cols_with_na:
        mode_val = X_train[col].mode()[0]   # mode() can return multiple values, take the first
        fill_values[col] = mode_val
        X_train[col] = X_train[col].fillna(mode_val)
        X_val[col]   = X_val[col].fillna(mode_val)
        X_test[col]  = X_test[col].fillna(mode_val)

    # scaler — only the logistic regression model needs scaled features; tree models use the originals
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ── Logistic Regression grid search ───────────────────────
    # Runs BEFORE the models dict is built, so the tuned estimator (not an
    # untuned placeholder) is what actually gets fit/predicted/evaluated below.
    #
    # Grid covers two solver/penalty combos:
    #   - lbfgs  → L2 only (default penalty)
    #   - saga   → elasticnet with l1_ratio=1 (pure L1)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["random_state"])

    lr_grid = [
        {'solver': ['lbfgs'], 'C': [0.01, 0.1, 1, 10]},
        {'solver': ['saga'],  'l1_ratio': [1], 'C': [0.01, 0.1, 1, 10]},
    ]

    lr_search = GridSearchCV(
        LogisticRegression(**lr_params),
        param_grid=lr_grid,
        scoring='average_precision',
        cv=cv_strategy,
        n_jobs=-1,
    )
    lr_search.fit(X_train_scaled, y_train)
    print("Best LR params:", lr_search.best_params_)
    print("Best LR CV avg-precision:", lr_search.best_score_)

    # model list — "lr" already fit by GridSearchCV above
    models = {
        "lr": lr_search.best_estimator_,
        "rf": RandomForestClassifier(**rf_params),
        "catboost": CatBoostClassifier(**cat_params),
        "lgbm": LGBMClassifier(**lgbm_params),
        "xgb": XGBClassifier(**xgb_params)
    }

    # model fit — skip "lr", it's already fitted
    for name, model in models.items():
        if name != "lr":
            model.fit(X_train, y_train)

    # val/test predictions (probabilities only, hard predictions come after threshold tuning)
    val_probas = {}
    for name, model in models.items():
        X_eval = X_val_scaled if name == "lr" else X_val
        val_probas[name] = model.predict_proba(X_eval)[:, 1]

    test_probas = {}
    for name, model in models.items():
        X_eval = X_test_scaled if name == "lr" else X_test
        test_probas[name] = model.predict_proba(X_eval)[:, 1]

    # find each model's best threshold using VAL probabilities only
    # (never tune the threshold on test data — that would leak test info into the decision)
    best_thresholds = {}
    for name in models:
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probas[name])
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        best_idx = f1_scores.argmax()
        best_thresholds[name] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        print(f"{name}: best threshold = {best_thresholds[name]:.3f}, val F1 at this threshold = {f1_scores[best_idx]:.3f}")

    # apply each model's own tuned threshold to both val and test
    val_preds  = {name: (val_probas[name]  >= best_thresholds[name]).astype(int) for name in models}
    test_preds = {name: (test_probas[name] >= best_thresholds[name]).astype(int) for name in models}

    # val eval
    val_results = {}
    for name in models:
        y_pred, y_proba = val_preds[name], val_probas[name]
        val_results[name] = {
            "confusion_matrix": confusion_matrix(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall":    recall_score(y_val, y_pred),
            "F1":        f1_score(y_val, y_pred),
            "ROC-AUC":   roc_auc_score(y_val, y_proba),
            "PR-AUC":    average_precision_score(y_val, y_proba)
        }

    val_df = pd.DataFrame(val_results).T.sort_values("PR-AUC", ascending=False)
    print("\n" + "="*50)
    print("VAL RESULTS")
    print("="*50)
    print(val_df)

    # test eval
    test_results = {}
    for name in models:
        y_pred, y_proba = test_preds[name], test_probas[name]
        test_results[name] = {
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
            "ROC-AUC":   roc_auc_score(y_test, y_proba),
            "PR-AUC":    average_precision_score(y_test, y_proba)
        }

    test_df = pd.DataFrame(test_results).T.sort_values("PR-AUC", ascending=False)
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(test_df)

    # ── Bootstrap CI on the LR vs RF PR-AUC gap ───────────────
    # The test set is small (~420 rows, ~60 positives), so a single point
    # estimate of "LR beat RF" could just be noise from this particular
    # split. Resampling the test set with replacement many times and
    # recomputing PR-AUC for both models each time shows how much that
    # gap actually wobbles — if the 95% CI on the difference stays above
    # 0, that's real evidence LR is better here, not just a lucky split.
    
    n_bootstraps = 1000
    rng = np.random.default_rng(config["random_state"])
    n = len(y_test)
    y_test_arr = y_test.values

    lr_boot_scores = []
    rf_boot_scores = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        y_sample = y_test_arr[idx]
        lr_boot_scores.append(average_precision_score(y_sample, test_probas["lr"][idx]))
        rf_boot_scores.append(average_precision_score(y_sample, test_probas["rf"][idx]))

    diffs = np.array(lr_boot_scores) - np.array(rf_boot_scores)
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

    print("\n" + "="*50)
    print("BOOTSTRAP CI: LR PR-AUC minus RF PR-AUC (test set, 1000 resamples)")
    print("="*50)
    print(f"Mean difference: {diffs.mean():.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    if ci_lower > 0:
        print("-> CI entirely above 0: LR's PR-AUC advantage over RF appears robust to resampling.")
    elif ci_upper < 0:
        print("-> CI entirely below 0: RF actually outperforms LR once resampling variability is considered.")
    else:
        print("-> CI straddles 0: the LR-vs-RF gap is not statistically distinguishable from noise at this sample size.")

    # ── Correlation check ──────────────────────────────────────
    # Sanity check for leakage: a low, gradually-declining correlation table
    # (no single dominant feature) is consistent with a genuinely hard,
    # real-world prediction problem rather than a hidden shortcut in the data.
    print("\ncorr:")
    print(df.corr(numeric_only=True)["TenYearCHD"].sort_values(ascending=False).head(20))

    # ── Feature importance check (Logistic Regression coefficients) ──
    print("LR coefficients:")
    coefficients = pd.Series(models["lr"].coef_[0], index=X_train.columns)
    print(coefficients.sort_values(ascending=False)) 

    # saving the lr model
    out_path = "heart_risk_lr_model.pkl"
    joblib.dump({
    "model":           models["lr"],
    "scaler":          scaler,
    "fill_values":     fill_values,
    "best_threshold":  best_thresholds["lr"],  # singular key, float value
    "feature_columns": list(X_train.columns),
}, out_path)

if __name__ == "__main__":
    main()