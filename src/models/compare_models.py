import json
import sys
from pathlib import Path
 
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
 
from src.models.evaluate import (
    compute_metrics, plot_model_comparison, save_metrics
)
from src.utils.logger import get_logger

import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
 
log = get_logger("compare_models")
 
 
def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
 
 
def load_data():
    d = ROOT / "data" / "processed"
    return (
        np.load(d / "X_train.npy"),
        np.load(d / "X_val.npy"),
        np.load(d / "y_train.npy"),
        np.load(d / "y_val.npy"),
    )
 
 
def build_models(params: dict) -> dict:
    mp = params["models"]
    models = {}
 
    # Naive Bayes
    models["naive_bayes"] = ("ComplementNB", ComplementNB(alpha=mp["naive_bayes"]["alpha"]))
 
    # Logistic Regression
    lr_p = mp["logistic_regression"]
    models["logistic_regression"] = (
        "LogisticRegression",
        LogisticRegression(
            C=lr_p["C"], max_iter=lr_p["max_iter"],
            solver=lr_p["solver"], random_state=42
        ),
    )
 
    # Random Forest
    rf_p = mp["random_forest"]
    models["random_forest"] = (
        "RandomForestClassifier",
        RandomForestClassifier(
            n_estimators=rf_p["n_estimators"],
            max_depth=rf_p["max_depth"],
            random_state=rf_p["random_state"],
            n_jobs=-1,
        ),
    )
 
    # Gradient Boosting
    gb_p = mp["gradient_boosting"]
    models["gradient_boosting"] = (
        "GradientBoostingClassifier",
        GradientBoostingClassifier(
            n_estimators=gb_p["n_estimators"],
            learning_rate=gb_p["learning_rate"],
            max_depth=gb_p["max_depth"],
            random_state=gb_p["random_state"],
        ),
    )
 
    # Linear SVM (wrapped in Platt scaling for probabilities)
    svm_p = mp["svm"]
    base_svm = LinearSVC(C=svm_p["C"], max_iter=3000, random_state=42)
    models["svm"] = (
        "LinearSVC+Platt",
        CalibratedClassifierCV(base_svm, cv=3, method="sigmoid"),
    )
 
    return models
 
 
def make_non_negative(X):
    X_min = X.min()
    return X - X_min if X_min < 0 else X
 
 
def main():
    params = load_params()
    mlflow_cfg = params["mlflow"]
 
    mlflow.set_tracking_uri(ROOT / mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
 
    X_train, X_val, y_train, y_val = load_data()
 
    model_defs = build_models(params)
    all_results = {}
 
    for name, (class_name, model) in model_defs.items():
        log.info(f"Training {name} ({class_name}) …")
        Xt = X_train
        Xv = X_val
        if name == "naive_bayes":
            Xt = make_non_negative(X_train)
            Xv = make_non_negative(X_val)
 
        with mlflow.start_run(run_name=f"compare_{name}"):
            model.fit(Xt, y_train)
            y_pred = model.predict(Xv)
            y_prob = model.predict_proba(Xv)[:, 1]
            metrics = compute_metrics(y_val, y_pred, y_prob)
            all_results[name] = metrics
 
            mlflow.log_param("model_class", class_name)
            mlflow.log_params(params["models"].get(name, {}))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            mlflow.set_tag("stage", "comparison")
 
            log.info(
                f"  {name}: acc={metrics['accuracy']:.4f}  "
                f"f1={metrics['f1']:.4f}  auc={metrics.get('roc_auc', 0):.4f}"
            )
 
    # ── Select best model by F1 ───────────────────────────────────────────────
    best_name = max(all_results, key=lambda k: all_results[k]["f1"])
    log.info(f"\nBest model: {best_name} (F1={all_results[best_name]['f1']:.4f})")
 
    comparison = {
        "results": all_results,
        "best_model": best_name,
        "best_metrics": all_results[best_name],
    }
 
    metrics_dir = ROOT / "reports" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(comparison, metrics_dir / "model_comparison.json")
 
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(all_results, fig_dir / "model_comparison.png")
 
    log.info("Model comparison complete ✓")
 
 
if __name__ == "__main__":
    main()