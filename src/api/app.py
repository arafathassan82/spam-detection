import json
import sys
from pathlib import Path
from typing import List, Optional
 
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
 
from src.utils.text_utils import clean_text, extract_handcrafted_features
from src.utils.logger import get_logger
 
log = get_logger("api")
 
# ── App setup ─────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title="Spam Detection API",
    description="Production ML pipeline — classify emails as spam or ham.",
    version="1.0.0",
)
 
 
# ── Model loader ──────────────────────────────────────────────────────────────
 
class ModelRegistry:
    """Loads and caches all pipeline artefacts on first access."""
 
    def __init__(self):
        self._model = None
        self._vectorizer = None
        self._selector = None
        self._scaler = None
        self._metadata = None
 
    def _load(self):
        baseline_dir = ROOT / "models" / "baseline"
        prod_dir = ROOT / "models" / "production"
 
        log.info("Loading model artefacts …")
        try:
            self._model = joblib.load(prod_dir / "model.pkl")
            log.info("Loaded production model.")
        except FileNotFoundError:
            self._model = joblib.load(baseline_dir / "model.pkl")
            log.warning("Production model not found — using baseline model.")
 
        self._vectorizer = joblib.load(baseline_dir / "tfidf_vectorizer.pkl")
        self._selector = joblib.load(baseline_dir / "feature_selector.pkl")
        try:
            self._scaler = joblib.load(baseline_dir / "hand_feat_scaler.pkl")
        except FileNotFoundError:
            self._scaler = None
 
        meta_path = prod_dir / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"model_name": "baseline", "note": "no metadata file found"}
 
    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model
 
    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self._load()
        return self._vectorizer
 
    @property
    def selector(self):
        if self._selector is None:
            self._load()
        return self._selector
 
    @property
    def scaler(self):
        if self._scaler is None:
            self._load()
        return self._scaler
 
    @property
    def metadata(self):
        if self._metadata is None:
            self._load()
        return self._metadata
 
 
registry = ModelRegistry()
 
 
# ── Prediction helpers ────────────────────────────────────────────────────────
 
def preprocess_and_predict(texts: List[str]) -> List[dict]:
    cleaned = [clean_text(t) for t in texts]
 
    # TF-IDF
    X_tfidf = registry.vectorizer.transform(cleaned)
 
    # Hand-crafted features
    hand_feats = extract_handcrafted_features(texts)
    if registry.scaler is not None:
        hand_feats = registry.scaler.transform(hand_feats)
 
    # Combine
    from scipy.sparse import hstack, issparse
    X_combined = hstack([X_tfidf, hand_feats])
 
    # Feature selection
    X_sel = registry.selector.transform(X_combined)
    if issparse(X_sel):
        X_sel = X_sel.toarray()
 
    # Handle NB non-negativity
    X_min = X_sel.min()
    if X_min < 0:
        X_sel = X_sel - X_min
 
    y_pred = registry.model.predict(X_sel)
    y_prob = registry.model.predict_proba(X_sel)[:, 1]
 
    return [
        {
            "label": int(pred),
            "label_name": "spam" if pred == 1 else "ham",
            "spam_probability": float(prob),
            "confidence": float(max(prob, 1 - prob)),
        }
        for pred, prob in zip(y_pred, y_prob)
    ]
 
 
# ── Request / Response schemas ────────────────────────────────────────────────
 
class EmailInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Email text (subject + body)")
    subject: Optional[str] = Field(None, description="Optional email subject")
 
    def combined_text(self) -> str:
        if self.subject:
            return f"{self.subject} {self.text}"
        return self.text
 
 
class PredictionResponse(BaseModel):
    label: int
    label_name: str
    spam_probability: float
    confidence: float
 
 
class BatchEmailInput(BaseModel):
    emails: List[EmailInput] = Field(..., min_length=1, max_length=100)
 
 
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
 
 
# ── Endpoints ─────────────────────────────────────────────────────────────────
 
@app.get("/health", tags=["Ops"])
def health():
    """Liveness check. Also warms up model artefacts on first call."""
    try:
        _ = registry.model
        return {"status": "ok", "model": registry.metadata.get("model_name", "unknown")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")
 
 
@app.get("/model/info", tags=["Ops"])
def model_info():
    """Return metadata of the currently loaded model."""
    return registry.metadata
 
 
@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(email: EmailInput):
    """Classify a single email as spam or ham."""
    try:
        results = preprocess_and_predict([email.combined_text()])
        return results[0]
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchEmailInput):
    """Classify multiple emails in a single request (max 100)."""
    try:
        texts = [e.combined_text() for e in batch.emails]
        results = preprocess_and_predict(texts)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        log.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 
 
# ── Dev entry point ───────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)