import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import MaxAbsScaler
 
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
 
from src.utils.logger import get_logger
from src.utils.text_utils import extract_handcrafted_features
 
log = get_logger("build_features")
 
SELECTOR_MAP = {
    "chi2": chi2,
    "mutual_info": mutual_info_classif,
    "anova": f_classif,
}
 
 
def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["features"]
 
 
def build_tfidf(params: dict) -> TfidfVectorizer:
    tp = params["tfidf"]
    return TfidfVectorizer(
        max_features=tp["max_features"],
        ngram_range=tuple(tp["ngram_range"]),
        min_df=tp["min_df"],
        max_df=tp["max_df"],
        sublinear_tf=tp["sublinear_tf"],
        strip_accents="unicode",
        analyzer="word",
    )
 
 
def build_selector(params: dict, score_func) -> SelectKBest:
    sp = params["selection"]
    return SelectKBest(score_func=score_func, k=sp["k_best"])
 
 
def load_split(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "interim" / f"{name}.csv"
    return pd.read_csv(path)
 
 
def combine_features(tfidf_matrix, hand_feats: np.ndarray) -> np.ndarray:
    """Horizontally stack TF-IDF sparse matrix with dense hand features."""
    scaler = MaxAbsScaler()
    hand_scaled = scaler.fit_transform(hand_feats)
    return hstack([tfidf_matrix, hand_scaled])
 
 
def main():
    params = load_params()
 
    log.info("Loading interim splits …")
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
 
    text_col = "text_clean"
    label_col = "label"
 
    y_train = train_df[label_col].values
    y_val = val_df[label_col].values
    y_test = test_df[label_col].values
 
    # ── TF-IDF ────────────────────────────────────────────────────────────────
    log.info("Fitting TF-IDF vectoriser …")
    vectorizer = build_tfidf(params)
    X_tfidf_train = vectorizer.fit_transform(train_df[text_col].fillna(""))
    X_tfidf_val = vectorizer.transform(val_df[text_col].fillna(""))
    X_tfidf_test = vectorizer.transform(test_df[text_col].fillna(""))
    log.info(f"TF-IDF shape (train): {X_tfidf_train.shape}")
 
    # ── Hand-crafted features ────────────────────────────────────────────────
    log.info("Extracting hand-crafted features …")
    hand_train = extract_handcrafted_features(train_df["text"].fillna("").tolist())
    hand_val = extract_handcrafted_features(val_df["text"].fillna("").tolist())
    hand_test = extract_handcrafted_features(test_df["text"].fillna("").tolist())
 
    # ── Combine ───────────────────────────────────────────────────────────────
    scaler = MaxAbsScaler()
    hand_train_scaled = scaler.fit_transform(hand_train)
    hand_val_scaled = scaler.transform(hand_val)
    hand_test_scaled = scaler.transform(hand_test)
 
    X_train_full = hstack([X_tfidf_train, hand_train_scaled])
    X_val_full = hstack([X_tfidf_val, hand_val_scaled])
    X_test_full = hstack([X_tfidf_test, hand_test_scaled])
    log.info(f"Combined feature shape (train): {X_train_full.shape}")
 
    # ── Feature selection ─────────────────────────────────────────────────────
    sel_params = params["selection"]
    k = sel_params["k_best"]
    method = sel_params["method"]
    score_func = SELECTOR_MAP.get(method, chi2)
 
    # chi2 requires non-negative → convert to csr and ensure >=0
    X_train_nn = X_train_full.tocsr()
    X_train_nn.data = np.abs(X_train_nn.data)
 
    log.info(f"Selecting top {k} features via {method} …")
    selector = SelectKBest(score_func=score_func, k=min(k, X_train_full.shape[1]))
    X_train_sel = selector.fit_transform(X_train_nn, y_train)
    X_val_sel = selector.transform(X_val_full.tocsr())
    X_test_sel = selector.transform(X_test_full.tocsr())
    log.info(f"Selected feature shape (train): {X_train_sel.shape}")
 
    # ── Persist ───────────────────────────────────────────────────────────────
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
 
    # Save as dense numpy arrays (smaller models do fine with dense)
    def _save(mat, path):
        arr = mat.toarray() if issparse(mat) else mat
        np.save(path, arr)
 
    _save(X_train_sel, processed_dir / "X_train.npy")
    _save(X_val_sel, processed_dir / "X_val.npy")
    _save(X_test_sel, processed_dir / "X_test.npy")
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "y_test.npy", y_test)
 
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(selector, model_dir / "feature_selector.pkl")
    joblib.dump(scaler, model_dir / "hand_feat_scaler.pkl")
    log.info("Feature arrays and transformers saved ✓")
 
 
if __name__ == "__main__":
    main()