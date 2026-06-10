import json
import sys
from pathlib import Path
 
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
 
from src.models.evaluate import compute_metrics, save_metrics, plot_confusion_matrix
from src.utils.logger import get_logger

import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
 
log = get_logger("train_baseline")
 
 
def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        p = yaml.safe_load(f)
    return {**p["baseline"], **p["mlflow"]}
 
 
def load_data():
    d = ROOT / "data" / "processed"
    X_train = np.load(d / "X_train.npy")
    X_val = np.load(d / "X_val.npy")
    y_train = np.load(d / "y_train.npy")
    y_val = np.load(d / "y_val.npy")
    return X_train, X_val, y_train, y_val
 
 
def make_non_negative(X):
    """Naive Bayes requires non-negative features."""
    X_min = X.min()
    if X_min < 0:
        X = X - X_min
    return X
 
 
def main():
    params = load_params()
    min_accuracy: float = params["min_accuracy"]
 
    mlflow.set_tracking_uri(ROOT / params["tracking_uri"])
    mlflow.set_experiment(params["experiment_name"])
 
    X_train, X_val, y_train, y_val = load_data()
    X_train_nn = make_non_negative(X_train)
    X_val_nn = make_non_negative(X_val)
 
    log.info(f"Training baseline Naive Bayes (alpha={params.get('alpha', 1.0)}) …")
 
    with mlflow.start_run(run_name="baseline_naive_bayes"):
        alpha = params.get("alpha", 1.0)
        model = ComplementNB(alpha=alpha)
        model.fit(X_train_nn, y_train)
 
        y_pred = model.predict(X_val_nn)
        y_prob = model.predict_proba(X_val_nn)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)
 
        mlflow.log_param("model", "ComplementNB")
        mlflow.log_param("alpha", alpha)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
 
        log.info("Baseline metrics:")
        for k, v in metrics.items():
            log.info(f"  {k}: {v:.4f}")
 
        # ── Accuracy gate ──────────────────────────────────────────────────
        if metrics["accuracy"] < min_accuracy:
            msg = (
                f"BASELINE GATE FAILED: accuracy={metrics['accuracy']:.4f} "
                f"< min={min_accuracy}"
            )
            log.error(msg)
            mlflow.set_tag("gate_passed", "false")
            # Don't raise — just warn. Pipeline continues.
        else:
            log.info(f"Baseline gate PASSED ✓ (accuracy={metrics['accuracy']:.4f})")
            mlflow.set_tag("gate_passed", "true")
 
        # ── Persist ────────────────────────────────────────────────────────
        model_dir = ROOT / "models" / "baseline"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.pkl")
 
        metrics_dir = ROOT / "reports" / "metrics"
        save_metrics(
            {"baseline": metrics, "min_accuracy_gate": min_accuracy},
            metrics_dir / "baseline_metrics.json",
        )
 
        fig_dir = ROOT / "reports" / "figures"
        plot_confusion_matrix(y_val, y_pred, fig_dir / "baseline_confusion_matrix.png")
        mlflow.log_artifact(str(fig_dir / "baseline_confusion_matrix.png"))
 
    log.info("Baseline training complete ✓")
 
 
if __name__ == "__main__":
    main()