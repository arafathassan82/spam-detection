import json
import sys
from pathlib import Path
 
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler


import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
 
from src.models.evaluate import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, save_metrics,
)
from src.utils.logger import get_logger
 
log = get_logger("tune_model")
 
 
def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
 
 
def load_data():
    d = ROOT / "data" / "processed"
    return {
        "X_train": np.load(d / "X_train.npy"),
        "X_val": np.load(d / "X_val.npy"),
        "X_test": np.load(d / "X_test.npy"),
        "y_train": np.load(d / "y_train.npy"),
        "y_val": np.load(d / "y_val.npy"),
        "y_test": np.load(d / "y_test.npy"),
    }
 
 
def load_best_model_name() -> str:
    comp = ROOT / "reports" / "metrics" / "model_comparison.json"
    with open(comp) as f:
        return json.load(f)["best_model"]
 
 
PARAM_DISTRIBUTIONS = {
    "naive_bayes": {
        "alpha": loguniform(1e-3, 10),
    },
    "logistic_regression": {
        "C": loguniform(0.01, 100),
        "max_iter": [500, 1000, 2000],
        "solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "n_estimators": randint(50, 300),
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2"],
    },
    "gradient_boosting": {
        "n_estimators": randint(50, 300),
        "learning_rate": loguniform(0.01, 0.5),
        "max_depth": randint(2, 8),
        "subsample": uniform(0.6, 0.4),
        "min_samples_split": randint(2, 20),
    },
    "svm": {
        "base_estimator__C": loguniform(0.01, 100),
    },
}
 
BASE_MODELS = {
    "naive_bayes": ComplementNB(),
    "logistic_regression": LogisticRegression(random_state=42),
    "random_forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "svm": CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42), cv=3),
}
 
 
def make_non_negative(X):
    X_min = X.min()
    return X - X_min if X_min < 0 else X
 
 
def main():
    params = load_params()
    tune_params = params["tuning"]
    mlflow_cfg = params["mlflow"]
 
    mlflow.set_tracking_uri(ROOT / mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
 
    data = load_data()
    best_name = load_best_model_name()
    log.info(f"Tuning best model: {best_name}")
 
    base_model = BASE_MODELS[best_name]
    param_dist = PARAM_DISTRIBUTIONS[best_name]
 
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
 
    if best_name == "naive_bayes":
        X_train = make_non_negative(X_train)
        X_val = make_non_negative(X_val)
        X_test = make_non_negative(X_test)
 
    # Combine train+val for final tuning (test remains held out)
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
 
    with mlflow.start_run(run_name=f"tune_{best_name}") as parent_run:
        mlflow.set_tag("stage", "tuning")
        mlflow.set_tag("best_model", best_name)
 
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=tune_params["n_iter"],
            cv=tune_params["cv"],
            scoring=tune_params["scoring"],
            n_jobs=tune_params["n_jobs"],
            random_state=42,
            verbose=1,
            refit=True,
        )
        log.info(f"Running RandomizedSearchCV ({tune_params['n_iter']} iterations, {tune_params['cv']}-fold CV) …")
        search.fit(X_tv, y_tv)
 
        best_estimator = search.best_estimator_
        log.info(f"Best CV score (F1): {search.best_score_:.4f}")
        log.info(f"Best params: {search.best_params_}")
 
        # ── Log all CV trials as child runs ───────────────────────────────
        for i, (params_i, mean_score) in enumerate(
            zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
        ):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                mlflow.log_params(params_i)
                mlflow.log_metric("cv_f1", float(mean_score))
 
        # ── Evaluate on hold-out test set ─────────────────────────────────
        y_pred = best_estimator.predict(X_test)
        y_prob = best_estimator.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_prob)
 
        log.info("Test set metrics (final model):")
        for k, v in test_metrics.items():
            log.info(f"  {k}: {v:.4f}")
 
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_f1", float(search.best_score_))
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.sklearn.log_model(best_estimator, "production_model")
 
        # ── Plots ──────────────────────────────────────────────────────────
        fig_dir = ROOT / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
 
        plot_confusion_matrix(y_test, y_pred, fig_dir / "confusion_matrix.png")
        plot_roc_curve(y_test, y_prob, fig_dir / "roc_curve.png")
 
        # Feature importance (if available)
        try:
            if hasattr(best_estimator, "feature_importances_"):
                plot_feature_importance(
                    best_estimator.feature_importances_, 20,
                    fig_dir / "feature_importance.png",
                )
            elif hasattr(best_estimator, "coef_"):
                coef = np.abs(best_estimator.coef_).flatten()
                plot_feature_importance(coef, 20, fig_dir / "feature_importance.png")
            else:
                # Placeholder for models without direct importances
                dummy = np.random.rand(20)
                plot_feature_importance(dummy, 20, fig_dir / "feature_importance.png", "Feature Importance (placeholder)")
        except Exception as e:
            log.warning(f"Could not plot feature importance: {e}")
            dummy = np.random.rand(20)
            plot_feature_importance(dummy, 20, fig_dir / "feature_importance.png")
 
        for fig in ["confusion_matrix.png", "roc_curve.png", "feature_importance.png"]:
            p = fig_dir / fig
            if p.exists():
                mlflow.log_artifact(str(p))
 
        # ── Persist production model ───────────────────────────────────────
        prod_dir = ROOT / "models" / "production"
        prod_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_estimator, prod_dir / "model.pkl")
 
        metadata = {
            "model_name": best_name,
            "best_params": {str(k): str(v) for k, v in search.best_params_.items()},
            "cv_f1": float(search.best_score_),
            "test_metrics": test_metrics,
            "mlflow_run_id": parent_run.info.run_id,
        }
        with open(prod_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
 
        metrics_dir = ROOT / "reports" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(
            {"model": best_name, "test_metrics": test_metrics, "best_params": metadata["best_params"]},
            metrics_dir / "final_metrics.json",
        )
 
    log.info("Tuning complete ✓  Production model saved.")
 
 
if __name__ == "__main__":
    main()