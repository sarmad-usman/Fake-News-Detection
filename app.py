# =============================================================================
# app.py
# Fake News Detection • Pro — Revamped with extensive diagnostics & polished UI
# Author: Muhammad Sarmad Usman
# Notes:
#  - Place your model (lr_model.pkl) and vectorizer (tfidf_vectorizer.pkl)
#    in the same folder as this app, or upload them via the sidebar.
#  - This file focuses on UI, diagnostics, explainability, and professional output.
#  - No model training is performed here; keep training in Jupyter/Notebook.
# =============================================================================

from __future__ import annotations

import os
import io
import re
import sys
import time
import json
import base64
import textwrap
from typing import List, Tuple, Dict, Optional, Any
from html import escape as html_escape

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML metrics for diagnostics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)

# optional libs
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

try:
    from newspaper import Article  # type: ignore
    NEWSPAPER_AVAILABLE = True
except Exception:
    NEWSPAPER_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detection • Pro", page_icon="📰", layout="wide")

# -----------------------------------------------------------------------------
# Constants & defaults
# -----------------------------------------------------------------------------
MODEL_PATH = "lr_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DATA_TRUE = os.path.join("data", "True.csv")
DATA_FAKE = os.path.join("data", "Fake.csv")

DEFAULT_THRESHOLD = 0.50
PRED_LABELS = {0: "Fake", 1: "Real"}

SAMPLE_NEWS = [
    {
        "title": "Government announces policy update",
        "text": "Breaking: The government has passed new legislation aiming to reduce emissions by 30% by 2030. Officials say the policy will include incentives for clean energy."
    },
    {
        "title": "Celebrity endorses miracle cure",
        "text": "A viral post claims a celebrity's 'miracle drink' cures all diseases overnight. Experts warn there's no scientific evidence supporting these claims."
    },
    {
        "title": "Elections: turnout reaches record high",
        "text": "Local reports confirm voter turnout reached a record high this year, with queues forming outside polling stations before dawn."
    },
    {
        "title": "Conspiracy theory resurfaces online",
        "text": "A fringe blog alleges an elaborate conspiracy. No credible sources have verified any part of this claim."
    },
]

# -----------------------------------------------------------------------------
# Custom CSS for improved styling (keeps background unchanged)
# -----------------------------------------------------------------------------
CUSTOM_CSS = r"""
<style>
/* Keep app background as default; restyle cards and sidebar */
[data-testid="stSidebar"] .css-1d391kg { padding: 18px 18px 30px 18px; }
[data-testid="stSidebar"] .css-1d391kg { border-radius: 12px; box-shadow: rgba(0,0,0,0.06) 0 6px 18px; }

/* Header */
.header-title { font-size:28px; font-weight:800; margin:0; }
.header-sub { opacity:0.8; margin-top:4px; }

/* Card */
.card {
  padding:14px;
  border-radius:12px;
  background: rgba(255,255,255,0.02);
  border: 1px solid rgba(120,120,120,0.06);
  margin-bottom:8px;
}

/* Badge */
.badge-real { background:#10B981; color:white; padding:6px 10px; border-radius:999px; font-weight:600; }
.badge-missing { background:#EF4444; color:white; padding:6px 10px; border-radius:999px; font-weight:600; }

/* Small diagnostics box */
.diag-box { background: rgba(0,0,0,0.03); padding:10px; border-radius:8px; font-family:monospace; font-size:13px; }

/* Buttons */
.stButton>button { padding:8px 12px; border-radius:8px; }

/* Table aesthetics */
[data-testid="stDataFrame"] table { border-collapse: collapse; }

/* Make metrics visually consistent */
.metric-box { display:flex; align-items:center; justify-content:space-between; padding:12px; border-radius:10px; background: rgba(255,255,255,0.015); border:1px solid rgba(100,100,100,0.04); }

/* Fix footer position */
.footer { position: relative; bottom: 0; width: 100%; margin-top: 2rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def safe_get_vocab(vectorizer) -> List[str]:
    """Return vocabulary list from vectorizer across scikit-learn versions."""
    if vectorizer is None:
        return []
    try:
        names = vectorizer.get_feature_names_out()
        return list(names)
    except Exception:
        try:
            names = vectorizer.get_feature_names()
            return list(names)
        except Exception:
            try:
                vocab_map = getattr(vectorizer, "vocabulary_", None)
                if isinstance(vocab_map, dict):
                    max_idx = max(vocab_map.values())
                    arr = [""] * (max_idx + 1)
                    for token, idx in vocab_map.items():
                        if 0 <= idx < len(arr):
                            arr[idx] = token
                    return [t for t in arr if t]
            except Exception:
                pass
    return []

@st.cache_resource(show_spinner=False)
def load_model(path: str = MODEL_PATH) -> Optional[Any]:
    """Load model with joblib; return None and warn if missing/fails."""
    if not os.path.exists(path):
        # don't crash — return None so app can run and warn user
        return None
    try:
        m = joblib.load(path)
        return m
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_vectorizer(path: str = VECTORIZER_PATH) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    try:
        v = joblib.load(path)
        return v
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_dataset(true_path: str = DATA_TRUE, fake_path: str = DATA_FAKE) -> Optional[pd.DataFrame]:
    """Load combined True/Fake dataset if both files exist; return None otherwise."""
    try:
        if os.path.exists(true_path) and os.path.exists(fake_path):
            true_df = pd.read_csv(true_path)
            fake_df = pd.read_csv(fake_path)
            def combine(df):
                if "title" in df.columns and "text" in df.columns:
                    return (df["title"].astype(str) + ". " + df["text"].astype(str)).str.strip()
                elif "text" in df.columns:
                    return df["text"].astype(str)
                elif "title" in df.columns:
                    return df["title"].astype(str)
                else:
                    return df.iloc[:, 0].astype(str)
            true_df["text"] = combine(true_df)
            fake_df["text"] = combine(fake_df)
            true_df["label"] = 1
            fake_df["label"] = 0
            df = pd.concat([true_df[["text","label"]], fake_df[["text","label"]]], ignore_index=True)
            df = df.dropna(subset=["text"]).reset_index(drop=True)
            return df
    except Exception:
        return None
    return None

# Load resources
MODEL = load_model()
VECTORIZER = load_vectorizer()
DATASET = load_dataset()
VOCAB = safe_get_vocab(VECTORIZER)

# -----------------------------------------------------------------------------
# Prediction wrappers and explainability
# -----------------------------------------------------------------------------
def vectorize_texts(texts: List[str]):
    if VECTORIZER is None:
        raise RuntimeError("Vectorizer not loaded.")
    return VECTORIZER.transform(texts)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def predict_proba(texts: List[str]) -> np.ndarray:
    """Return Nx2 array of [prob_fake, prob_real]. Robust to missing model/vectorizer."""
    if VECTORIZER is None or MODEL is None:
        n = len(texts)
        return np.tile(np.array([[0.5, 0.5]]), (n, 1))
    X = vectorize_texts(texts)
    # try predict_proba
    try:
        probs = MODEL.predict_proba(X)
        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T
        # Ensure shape Nx2 for binary (in case probabilities are [n,2] already)
        if probs.shape[1] == 2:
            # We want [fake, real] ordering consistent with labels 0->Fake,1->Real
            return probs
        # if multiclass, reduce to 2 columns by combining last class as 'real' roughly (fallback)
        if probs.shape[1] > 2:
            # take column 1 as positive if model used 0/1 classes — this is approximate
            pos = probs[:, -1]
            return np.vstack([1 - pos, pos]).T
    except Exception:
        pass
    # decision function fallback
    try:
        df = MODEL.decision_function(X)
        if isinstance(df, np.ndarray):
            if df.ndim == 1:
                pos = sigmoid(df)
                return np.vstack([1 - pos, pos]).T
            else:
                # softmax across last dim
                e = np.exp(df - np.max(df, axis=1, keepdims=True))
                p = e / e.sum(axis=1, keepdims=True)
                if p.shape[1] >= 2:
                    return p[:, :2]
                else:
                    pos = p[:, -1]
                    return np.vstack([1 - pos, pos]).T
    except Exception:
        pass
    # predict labels fallback
    try:
        preds = np.asarray(MODEL.predict(X))
        if preds.ndim == 1:
            return np.vstack([1 - preds, preds]).T
        return preds
    except Exception:
        n = len(texts)
        return np.tile(np.array([[0.5, 0.5]]), (n, 1))

def predict_label(texts: List[str], threshold: float = DEFAULT_THRESHOLD) -> Tuple[np.ndarray, np.ndarray]:
    probs = predict_proba(texts)  # shape (n, 2)
    pos = probs[:, 1]
    labels = (pos >= threshold).astype(int)
    return labels, probs

def get_top_coeff_tokens(n: int = 30) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Return top tokens (token, coef) pushing towards Real (positive) and towards Fake (negative).
    Requires MODEL.coef_ and VOCAB to align; will attempt to build mapping otherwise.
    """
    try:
        if MODEL is None or VECTORIZER is None:
            return [], []
        if not hasattr(MODEL, "coef_"):
            return [], []
        coefs = np.asarray(MODEL.coef_)
        if coefs.ndim > 1:
            coefs = coefs[0]
        vocab = VOCAB
        if len(vocab) != len(coefs):
            # try to reorder using VECTORIZER.vocabulary_
            mapping = getattr(VECTORIZER, "vocabulary_", None)
            if isinstance(mapping, dict):
                max_idx = max(mapping.values())
                arr = [""] * (max_idx + 1)
                coef_arr = np.zeros(max_idx + 1)
                for token, idx in mapping.items():
                    if 0 <= idx <= max_idx:
                        arr[idx] = token
                        if idx < len(coefs):
                            coef_arr[idx] = float(coefs[idx])
                vocab = [t for t in arr if t]
                coefs = coef_arr[: len(vocab)]
            else:
                # abort
                return [], []
        idx_sorted = np.argsort(coefs)
        top_neg_idx = idx_sorted[:n]
        top_pos_idx = idx_sorted[-n:][::-1]
        top_pos = [(vocab[i], float(coefs[i])) for i in top_pos_idx if i < len(vocab)]
        top_neg = [(vocab[i], float(coefs[i])) for i in top_neg_idx if i < len(vocab)]
        return top_pos, top_neg
    except Exception:
        return [], []

def highlight_contributions(text: str, max_terms: int = 12) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Lightweight explainability:
      - Tokenize via vectorizer analyzer if available
      - Contribution = tfidf_value * coef
      - Return HTML-highlighted text and list of (token, contribution)
    """
    if VECTORIZER is None or MODEL is None:
        return html_escape(text), []
    try:
        analyzer = VECTORIZER.build_analyzer()
    except Exception:
        analyzer = lambda s: re.findall(r"[A-Za-z]{2,}", s.lower())
    tokens = analyzer(text) if text else []
    if not tokens:
        return html_escape(text), []
    try:
        X = vectorize_texts([text]).tocoo()
    except Exception:
        return html_escape(text), []

    contribs: Dict[str, float] = {}
    coefs = None
    try:
        if hasattr(MODEL, "coef_"):
            c = np.asarray(MODEL.coef_)
            coefs = c[0] if c.ndim > 1 else c.ravel()
    except Exception:
        coefs = None

    feature_names = VOCAB
    for idx, val in zip(X.col, X.data):
        token = feature_names[idx] if idx < len(feature_names) else None
        if not token:
            continue
        weight = float(val) * (float(coefs[idx]) if (coefs is not None and idx < len(coefs)) else 0.0)
        contribs[token] = contribs.get(token, 0.0) + weight

    top_items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_terms]

    def color_for(score: float) -> str:
        opacity = min(0.45, 0.12 + min(1.0, abs(score)) / 5.0)
        if score > 0:
            return f"rgba(16,185,129,{opacity})"  # greenish for Real
        else:
            return f"rgba(239,68,68,{opacity})"  # redish for Fake

    token_set = set(t for t, _ in top_items)
    colored_tokens = []
    top_map = dict(top_items)
    for tok in tokens:
        esc = html_escape(tok)
        if tok in token_set:
            score = top_map[tok]
            bg = color_for(score)
            colored_tokens.append(f'<span style="background:{bg}; padding:2px 8px; border-radius:6px; margin:2px; display:inline-block;">{esc}</span>')
        else:
            colored_tokens.append(esc)
    html = " ".join(colored_tokens)
    return html, top_items

# -----------------------------------------------------------------------------
# URL scraping with diagnostics
# -----------------------------------------------------------------------------
def scrape_url_to_text(url: str) -> Tuple[str, str, Dict[str, str]]:
    """
    Try multiple strategies to extract article text:
     - newspaper3k (best)
     - requests + BeautifulSoup (fallback)
    Returns (title, text, diagnostic)
    Diagnostic will include status/reason/len/status_code/content-type
    """
    diag: Dict[str, str] = {"status": "unknown", "reason": ""}
    url = (url or "").strip()
    if not url:
        diag["reason"] = "empty_url"
        return "", "", diag

    # Try newspaper
    if NEWSPAPER_AVAILABLE:
        try:
            art = Article(url)
            art.download()
            art.parse()
            title = art.title or ""
            text = art.text or ""
            if text.strip():
                diag["status"] = "ok"
                diag["reason"] = "newspaper3k"
                diag["len"] = str(len(text))
                return title, text, diag
            else:
                diag["reason"] = "newspaper3k_no_text"
        except Exception as e:
            diag["reason"] = f"newspaper3k_error: {e}"

    # Try requests + BeautifulSoup
    if REQUESTS_AVAILABLE:
        try:
            resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0 (compatible)"})
            diag["status_code"] = str(resp.status_code)
            diag["content_type"] = resp.headers.get("content-type", "")
            if resp.status_code != 200:
                diag["reason"] = f"non_200_{resp.status_code}"
                return "", "", diag
            soup = BeautifulSoup(resp.text, "html.parser")
            paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
            text = " ".join(paras)
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""
            if text.strip():
                diag["status"] = "ok"
                diag["reason"] = "requests_bs4"
                diag["len"] = str(len(text))
                return title, text, diag
            else:
                diag["reason"] = "no_paragraphs_extracted_possible_js_or_paywall"
                return "", "", diag
        except Exception as e:
            diag["reason"] = f"requests_error: {e}"
            return "", "", diag

    diag["reason"] = "no_extraction_libraries"
    return "", "", diag

# -----------------------------------------------------------------------------
# Model & vectorizer sanity checks
# -----------------------------------------------------------------------------
def model_vectorizer_sanity() -> Dict[str, Any]:
    """
    Return diagnostics about model/vectorizer compatibility and potential causes of
    low accuracy. This does not retrain, just inspects artifacts.
    """
    diag: Dict[str, Any] = {}
    diag["model_loaded"] = MODEL is not None
    diag["vectorizer_loaded"] = VECTORIZER is not None
    diag["vocab_size"] = len(VOCAB)
    # coef length
    if MODEL is None:
        diag["model_coef_len"] = None
        diag["coef_vocab_mismatch"] = None
    else:
        try:
            if hasattr(MODEL, "coef_"):
                c = np.asarray(MODEL.coef_)
                coef_len = int(c.shape[-1])
                diag["model_coef_len"] = coef_len
                diag["coef_vocab_mismatch"] = coef_len != len(VOCAB)
            else:
                diag["model_coef_len"] = None
                diag["coef_vocab_mismatch"] = None
        except Exception as e:
            diag["model_coef_len"] = None
            diag["coef_vocab_mismatch"] = None
            diag["coef_error"] = str(e)
    # Basic suggestion rules
    suggestions: List[str] = []
    if not diag["model_loaded"]:
        suggestions.append("Model not loaded. Place lr_model.pkl in app folder.")
    if not diag["vectorizer_loaded"]:
        suggestions.append("Vectorizer not loaded. Place tfidf_vectorizer.pkl in app folder.")
    if diag.get("coef_vocab_mismatch"):
        suggestions.append("Model coefficients length does not match vectorizer vocabulary. Rebuild/save model with same vectorizer.")
    if diag["vocab_size"] == 0:
        suggestions.append("Vectorizer vocabulary appears empty. Check how the vectorizer was saved.")
    if DATASET is not None:
        # Check class balance
        try:
            counts = DATASET["label"].value_counts().to_dict()
            diag["dataset_class_counts"] = counts
            if min(counts.values()) / max(counts.values()) < 0.01:
                suggestions.append("Severe class imbalance in dataset — consider balancing during training or stratified sampling.")
        except Exception:
            pass
    diag["suggestions"] = suggestions
    return diag

# -----------------------------------------------------------------------------
# Evaluation utilities (for dataset tab)
# -----------------------------------------------------------------------------
def evaluate_on_dataset(threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Evaluate current model on dataset (if available). Returns metrics and test split used.
    """
    result: Dict[str, Any] = {"available": False}
    if DATASET is None or VECTORIZER is None or MODEL is None:
        result["available"] = False
        return result
    try:
        X = VECTORIZER.transform(DATASET["text"].astype(str))
        y = DATASET["label"].astype(int).to_numpy()
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        # try predict_proba
        try:
            y_prob = MODEL.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback: decision function + sigmoid, or predict
            try:
                df = MODEL.decision_function(X_test)
                if df.ndim == 1:
                    y_prob = sigmoid(df)
                else:
                    e = np.exp(df - np.max(df, axis=1, keepdims=True))
                    p = e / e.sum(axis=1, keepdims=True)
                    # take last col as pos (approx)
                    y_prob = p[:, -1]
            except Exception:
                y_prob = np.asarray(MODEL.predict(X_test)).astype(float)
        y_pred = (y_prob >= threshold).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
            "brier": float(brier_score_loss(y_test, y_prob)),
        }
        result["available"] = True
        result["metrics"] = metrics
        result["y_test"] = y_test
        result["y_prob"] = y_prob
        result["y_pred"] = y_pred
        return result
    except Exception as e:
        result["available"] = False
        result["error"] = str(e)
        return result

# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def make_download_button_bytes(bytes_data: bytes, filename: str, label: str):
    st.download_button(label=label, data=bytes_data, file_name=filename, mime="text/csv")

def text_to_downloadable_file(text: str, filename: str = "extracted.txt"):
    b = text.encode("utf-8")
    st.download_button(label=f"⬇️ Download {filename}", data=b, file_name=filename, mime="text/plain")

def create_pdf_report_single(text: str, prediction: str, prob_real: float, prob_fake: float, tokens: List[Tuple[str, float]], filename: str = "report.pdf"):
    """
    Create a simple PDF report if reportlab is available.
    Returns bytes for download button.
    """
    if not REPORTLAB_AVAILABLE:
        return None
    try:
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        margin = 60
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, height - margin, "Fake News Detection • Report")
        c.setFont("Helvetica", 10)
        c.drawString(margin, height - margin - 20, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Prediction
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, height - margin - 60, "Prediction:")
        c.setFont("Helvetica", 12)
        c.drawString(margin + 80, height - margin - 60, f"{prediction} (Real: {prob_real:.3f}, Fake: {prob_fake:.3f})")
        # Text (shortened)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, height - margin - 100, "Excerpt:")
        c.setFont("Helvetica", 10)
        wrapped = textwrap.wrap(text[:2000], 120)
        y = height - margin - 120
        for line in wrapped:
            c.drawString(margin, y, line)
            y -= 12
            if y < 80:
                c.showPage()
                y = height - margin
        # Tokens
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y - 10, "Top contributing tokens:")
        c.setFont("Helvetica", 10)
        y -= 30
        for t, s in tokens[:40]:
            c.drawString(margin, y, f"{t}: {s:.4f}")
            y -= 10
            if y < 80:
                c.showPage()
                y = height - margin
        c.save()
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Session state initialization & isolation (prevent overrides)
# -----------------------------------------------------------------------------
def init_session_state():
    # Use distinct keys per tab/action to avoid cross-tab override issues.
    if "single_input" not in st.session_state:
        st.session_state["single_input"] = ""
    if "single_result" not in st.session_state:
        st.session_state["single_result"] = {}
    if "url_input" not in st.session_state:
        st.session_state["url_input"] = ""
    if "url_extracted_text" not in st.session_state:
        st.session_state["url_extracted_text"] = ""
    if "batch_last" not in st.session_state:
        st.session_state["batch_last"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "model_diag_cached" not in st.session_state:
        st.session_state["model_diag_cached"] = model_vectorizer_sanity()
    if "eval_cached" not in st.session_state:
        st.session_state["eval_cached"] = evaluate_on_dataset(DEFAULT_THRESHOLD)

init_session_state()

# -----------------------------------------------------------------------------
# Sidebar UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='font-size:18px;font-weight:800'>Fake News • Pro</div><div style='opacity:0.8'>📰</div></div>", unsafe_allow_html=True)
    st.caption("TF-IDF + LogisticRegression • Lightweight — Add model files in the folder or upload below.")
    st.markdown("---")
    with st.expander("⚙️ Settings", expanded=True):
        threshold = st.slider("Decision threshold (Real if probability ≥ threshold)", min_value=0.05, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.01, key="sidebar_threshold")
        show_explain = st.checkbox("Show token-level explainability", value=True, key="sidebar_explain")
        save_history = st.checkbox("Save prediction history (session)", value=True, key="sidebar_history")
        show_debug = st.checkbox("Show debug diagnostics", value=False, key="sidebar_debug")
    with st.expander("Model / Vectorizer", expanded=False):
        st.write("Model loaded:" , "✅" if MODEL is not None else "❌")
        st.write("Vectorizer loaded:", "✅" if VECTORIZER is not None else "❌")
        st.write(f"Vocabulary size: {len(VOCAB)}")
        st.markdown("---")
        st.write("Upload new artifacts (optional)")
        uploaded_model = st.file_uploader("Upload model (.pkl)", type=["pkl"], key="upload_model")
        uploaded_vec = st.file_uploader("Upload vectorizer (.pkl)", type=["pkl"], key="upload_vectorizer")
        if uploaded_model is not None:
            try:
                # Save uploaded file to disk to replace model
                bytes_data = uploaded_model.read()
                with open(MODEL_PATH, "wb") as f:
                    f.write(bytes_data)
                st.success("Model saved to disk. Restart the app to reload model.")
            except Exception as e:
                st.error(f"Could not save model: {e}")
        if uploaded_vec is not None:
            try:
                bytes_data = uploaded_vec.read()
                with open(VECTORIZER_PATH, "wb") as f:
                    f.write(bytes_data)
                st.success("Vectorizer saved to disk. Restart the app to reload vectorizer.")
            except Exception as e:
                st.error(f"Could not save vectorizer: {e}")
    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Clear session history"):
        st.session_state["history"] = []
        st.success("Session history cleared.")
    st.markdown("---")
    st.caption("Tips: Use longer excerpts (2–4 sentences) for more reliable TF-IDF signals. If URL extraction fails, copy-paste article text manually into Single Check.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
left, right = st.columns([0.8, 0.2])
with left:
    st.markdown("<div class='header-title'>Fake News Detection</div><div class='header-sub'>Professional screening tool — results are signals, not final verification.</div>", unsafe_allow_html=True)
with right:
    if MODEL is not None:
        st.markdown("<div style='text-align:right'><span class='badge-real'>Model: ready</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:right'><span class='badge-missing'>Model: missing</span></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs main UI
# -----------------------------------------------------------------------------
tab_single, tab_url, tab_batch, tab_model, tab_insights, tab_help = st.tabs(
    ["🔎 Single Check", "🔗 URL Check", "📂 Batch Check", "📈 Model & Dataset", "🧠 Insights", "❓ Help"]
)

# --------------------------------------
# TAB: Single Check
# --------------------------------------
with tab_single:
    st.subheader("Single article check — fast, explained")
    colA, colB = st.columns([0.72, 0.28], gap="large")
    with colB:
        st.markdown("**Samples**")
        sample_idx = st.selectbox("Pick a sample article", options=list(range(len(SAMPLE_NEWS))), format_func=lambda i: SAMPLE_NEWS[i]["title"])
        if st.button("Load sample text", key="load_sample"):
            st.session_state["single_input"] = SAMPLE_NEWS[sample_idx]["text"]
        st.markdown("---")
        st.markdown("**Quick tips**")
        st.markdown("- Paste at least 2–3 sentences (TF-IDF requires a bit of context).")
        st.markdown("- Use the Threshold slider in the sidebar to be conservative.")
        st.markdown("- Toggle explainability to see token-level contributions.")
    with colA:
        text_area_key = "single_text_area"
        user_text = st.text_area("Paste or type news article text here", value=st.session_state.get("single_input", ""), height=260, key=text_area_key)
        analyze_btn = st.button("Analyze article", key="analyze_single")

        if analyze_btn:
            txt = (st.session_state.get(text_area_key) or "").strip()
            if not txt:
                st.warning("Please provide some article text to analyze.")
            else:
                # perform prediction
                labels, probs = predict_label([txt], threshold=threshold)
                prob_fake, prob_real = float(probs[0, 0]), float(probs[0, 1])
                label_text = PRED_LABELS[int(labels[0])]
                confidence = max(prob_fake, prob_real)

                # Save results to session (isolated)
                res = {
                    "text": txt,
                    "label": label_text,
                    "prob_real": prob_real,
                    "prob_fake": prob_fake,
                    "threshold": threshold,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state["single_result"] = res
                if save_history:
                    st.session_state["history"].append(res)

                # Display results
                r1, r2 = st.columns([0.55, 0.45])
                with r1:
                    st.markdown("<div class='card'><div style='opacity:0.8'>Prediction</div>"
                                f"<div style='font-size:20px;font-weight:800;margin-top:6px'>{html_escape(label_text)}</div>"
                                f"<div style='margin-top:8px'>Confidence: <strong>{confidence:.1%}</strong></div></div>", unsafe_allow_html=True)
                    # suggestion helpfulness
                    if label_text == "Real" and prob_real < 0.6:
                        st.info("Prediction near threshold. Consider increasing threshold for stricter Real classification.")
                    if label_text == "Fake" and prob_fake < 0.6:
                        st.info("Prediction leans Fake but not strongly. Check tokens below for context.")
                with r2:
                    # bar visualization
                    fig, ax = plt.subplots(figsize=(5.2, 1.3))
                    ax.barh([0, 1], [prob_fake, prob_real])
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(["Fake", "Real"])
                    ax.set_xlim(0, 1)
                    ax.xaxis.set_visible(False)
                    for i, v in enumerate([prob_fake, prob_real]):
                        ax.text(v + 0.02, i, f"{v:.1%}", va="center")
                    st.pyplot(fig)

                # full probability display with helpful metric cards
                metrics_row = st.columns(3)
                with metrics_row[0]:
                    st.markdown(f"<div class='metric-box'><div>Real</div><div style='font-weight:700'>{prob_real:.3f}</div></div>", unsafe_allow_html=True)
                with metrics_row[1]:
                    st.markdown(f"<div class='metric-box'><div>Fake</div><div style='font-weight:700'>{prob_fake:.3f}</div></div>", unsafe_allow_html=True)
                with metrics_row[2]:
                    st.markdown(f"<div class='metric-box'><div>Threshold</div><div style='font-weight:700'>{threshold:.2f}</div></div>", unsafe_allow_html=True)

                st.markdown("---")
                # Explainability
                if show_explain:
                    st.markdown("#### Top contributing tokens (local explainability)")
                    html, top_items = highlight_contributions(txt, max_terms=30)
                    if top_items:
                        st.markdown(html, unsafe_allow_html=True)
                        df_expl = pd.DataFrame([{"token": t, "contribution": s, "direction": ("Real" if s > 0 else "Fake")} for t, s in top_items])
                        st.dataframe(df_expl, use_container_width=True)
                    else:
                        st.info("Not enough signal to compute token contributions for this text.")

                # Offer exports
                st.markdown("---")
                col_dl1, col_dl2 = st.columns([0.5, 0.5])
                with col_dl1:
                    df_single = pd.DataFrame([res])
                    make_download_button_bytes(df_to_csv_bytes(df_single), "single_prediction.csv", "⬇️ Download CSV")
                with col_dl2:
                    if REPORTLAB_AVAILABLE:
                        pdf_bytes = create_pdf_report_single(txt, label_text, prob_real, prob_fake, top_items)
                        if pdf_bytes:
                            st.download_button("⬇️ Download PDF report", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
                        else:
                            st.info("PDF generation failed or no reportlab available.")
                    else:
                        st.info("Install reportlab to enable PDF reports.")

# --------------------------------------
# TAB: URL Check
# --------------------------------------
with tab_url:
    st.subheader("URL-based article extraction & analysis")
    st.markdown("Paste an article URL and the app will attempt to extract the text. If extraction fails, we'll show diagnostics and what you can try.")
    col_u1, col_u2 = st.columns([0.7, 0.3])
    with col_u1:
        st.text_input("Article URL", value=st.session_state.get("url_input", ""), key="url_input_field")
    with col_u2:
        if st.button("Fetch & analyze", key="fetch_url"):
            st.session_state["url_input"] = st.session_state.get("url_input_field", "").strip()
            url = st.session_state["url_input"]
            if not url:
                st.warning("Please paste a URL.")
            else:
                title, text, diag = scrape_url_to_text(url)
                st.session_state["url_extraction_diag"] = diag
                st.session_state["url_extracted_text"] = text
                st.session_state["url_title"] = title
    # show diagnostics
    if "url_extraction_diag" in st.session_state:
        diag = st.session_state["url_extraction_diag"]
        st.markdown("**Extraction diagnostics**")
        st.code(json.dumps(diag, indent=2))
        if diag.get("status") != "ok" or not st.session_state.get("url_extracted_text"):
            st.error("Could not extract article text automatically.")
            st.markdown("**Troubleshooting / suggestions**")
            st.markdown("- Open the article in your browser and copy-paste the article body into Single Check.")
            st.markdown("- If you see a non-200 status code, the site may block scrapers or be geo-restricted.")
            st.markdown("- Paywalled, JavaScript-heavy or CDN-loaded pages often require manual copy-paste.")
            st.markdown("- Retry with a different news site; some sites use anti-bot protections.")
        else:
            st.success("Article extracted — review or edit before analyzing.")
            if st.session_state.get("url_title"):
                st.markdown(f"**Title:** {html_escape(st.session_state.get('url_title'))}")
            st.text_area("Extracted text (editable)", value=st.session_state.get("url_extracted_text", ""), height=260, key="url_text_area")
            if st.button("Analyze extracted text", key="analyze_extracted"):
                ex = st.session_state.get("url_text_area", "").strip()
                if not ex:
                    st.warning("No text to analyze.")
                else:
                    labels, probs = predict_label([ex], threshold=threshold)
                    prob_fake, prob_real = float(probs[0, 0]), float(probs[0, 1])
                    label_text = PRED_LABELS[int(labels[0])]
                    st.success(f"Prediction: **{label_text}** — Confidence: **{max(prob_fake, prob_real):.1%}**")
                    # show tokens if requested
                    if show_explain:
                        html, top_items = highlight_contributions(ex, max_terms=30)
                        if top_items:
                            st.markdown(html, unsafe_allow_html=True)
                            df_expl = pd.DataFrame([{"token": t, "contribution": s, "direction": ("Real" if s > 0 else "Fake")} for t, s in top_items])
                            st.dataframe(df_expl, use_container_width=True)
                        else:
                            st.info("Not enough signal to compute token contributions for this text.")

# --------------------------------------
# TAB: Batch Check
# --------------------------------------
with tab_batch:
    st.subheader("Batch classification (CSV)")
    st.markdown("Upload a CSV with a column containing article text. The app will classify each row and let you download the labeled CSV.")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="batch_uploader")
    chosen_col = None
    if uploaded is not None:
        try:
            sample = pd.read_csv(uploaded, nrows=5)
            cols = list(sample.columns)
            st.write("Detected columns:", cols)
            default_idx = cols.index("text") if "text" in cols else 0
            chosen_col = st.selectbox("Select column containing text", options=cols, index=default_idx)
            if st.button("Run batch classification", key="run_batch"):
                df = pd.read_csv(uploaded)
                if chosen_col not in df.columns:
                    st.error(f"Column {chosen_col} not found.")
                else:
                    texts = df[chosen_col].astype(str).fillna("")
                    labels, probs = predict_label(texts.tolist(), threshold=threshold)
                    df["prediction"] = [PRED_LABELS[int(l)] for l in labels]
                    df["prob_fake"] = probs[:, 0]
                    df["prob_real"] = probs[:, 1]
                    st.session_state["batch_last"] = df
                    st.success(f"Processed {len(df)} rows.")
                    st.dataframe(df.head(50), use_container_width=True)
                    make_download_button_bytes(df_to_csv_bytes(df), "batch_predictions.csv", "⬇️ Download labeled CSV")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

# --------------------------------------
# TAB: Model & Dataset Diagnostics
# --------------------------------------
with tab_model:
    st.subheader("Model & dataset diagnostics — check if model is working correctly")
    st.markdown("This section helps diagnose why accuracy might be low and checks the consistency between your saved model and vectorizer.")

    diag = st.session_state.get("model_diag_cached", model_vectorizer_sanity())
    st.markdown("**Artifact status**")
    st.write(f"- Model loaded: {'✅' if diag.get('model_loaded') else '❌'}")
    st.write(f"- Vectorizer loaded: {'✅' if diag.get('vectorizer_loaded') else '❌'}")
    st.write(f"- Vocabulary size: {diag.get('vocab_size')}")

    if diag.get("model_coef_len") is not None:
        st.write(f"- Model coefficients length: {diag.get('model_coef_len')}")
    if diag.get("coef_vocab_mismatch"):
        st.warning("Model coefficients length does not match vectorizer vocabulary. This causes incorrect mapping of coefficients to tokens and will break interpretability and degrade performance. Retrain or re-save the model with the same vectorizer.")

    if diag.get("suggestions"):
        with st.expander("Automated suggestions"):
            for s in diag["suggestions"]:
                st.write("- " + s)

    st.markdown("---")
    st.write("**Quick sanity checks**")
    # show top coefficients if available
    pos, neg = get_top_coeff_tokens(n=40)
    if pos or neg:
        colp, coln = st.columns(2)
        with colp:
            st.write("Top tokens toward Real")
            if pos:
                st.dataframe(pd.DataFrame(pos, columns=["token", "coef"]), use_container_width=True)
        with coln:
            st.write("Top tokens toward Fake")
            if neg:
                st.dataframe(pd.DataFrame(neg, columns=["token", "coef"]), use_container_width=True)
    else:
        st.info("Top coefficients unavailable (maybe model lacks coef_ or vocab mismatch).")

    st.markdown("---")
    st.write("**Evaluate on dataset (if available)**")
    eval_res = st.session_state.get("eval_cached", evaluate_on_dataset(threshold))
    if not eval_res.get("available"):
        st.info("Dataset or model/vectorizer missing — place data/True.csv and data/Fake.csv for evaluation.")
        if eval_res.get("error"):
            st.error(f"Evaluation error: {eval_res.get('error')}")
    else:
        metrics = eval_res["metrics"]
        # Fix for the IndexError: Create exactly 5 columns for the 6 metrics
        cols = st.columns(6)
        metric_items = list(metrics.items())
        
        for i, (k, v) in enumerate(metric_items):
            if i < len(cols):  # Safety check to prevent index errors
                with cols[i]:
                    st.markdown(f"<div class='card'><div style='opacity:0.8'>{html_escape(k.capitalize())}</div><div style='font-size:18px;font-weight:700;margin-top:6px'>{v:.4f}</div></div>", unsafe_allow_html=True)

        # ROC and PR curves
        try:
            y_test = eval_res["y_test"]
            y_prob = eval_res["y_prob"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = metrics.get("roc_auc", 0.0)
                fig = plt.figure(figsize=(5.2, 4.0))
                plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.legend(loc="lower right")
                st.pyplot(fig)
            with c2:
                st.markdown("Precision–Recall Curve")
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                fig2 = plt.figure(figsize=(5.2, 4.0))
                plt.plot(rec, prec)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Could not draw diagnostic curves: {e}")

        st.markdown("---")
        st.write("Confusion matrix (on test split)")
        try:
            y_test = eval_res["y_test"]
            y_prob = eval_res["y_prob"]
            y_pred = eval_res["y_pred"]
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Fake", "Real"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Fake", "Real"])
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, str(v), ha="center", va="center", color="white" if v > (cm.max() / 2.0) else "black")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not build confusion matrix: {e}")

        st.markdown("---")
        st.write("Calibration / Reliability checks")
        try:
            # reliability diagram (binned predicted prob vs observed freq)
            y_test = eval_res["y_test"]
            y_prob = eval_res["y_prob"]
            bins = np.linrange(0.0, 1.0, 11)
            binids = np.digitize(y_prob, bins) - 1
            bin_true_rate = []
            bin_centers = []
            for i in range(len(bins) - 1):
                mask = binids == i
                if mask.sum() == 0:
                    bin_true_rate.append(np.nan)
                else:
                    bin_true_rate.append(y_test[mask].mean())
                bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
            fig = plt.figure(figsize=(6, 3.2))
            plt.plot(bin_centers, bin_true_rate, "o-", label="Observed")
            plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed frequency")
            plt.title("Reliability diagram (binned)")
            plt.legend()
            st.pyplot(fig)
            st.write(f"Brier score (lower better): {metrics.get('brier'):.4f}")
        except Exception as e:
            st.warning(f"Could not compute calibration: {e}")

        st.markdown("---")
        st.write("If accuracy is unexpectedly low:")
        st.markdown("- Check `model_vectorizer_sanity` diagnostics above (vocab/coefs mismatch is a common cause).")
        st.markdown("- Verify dataset format and labels (`text`, `label` with 0/1).")
        st.markdown("- Check class balance and consider stratified training or class weighting.")
        st.markdown("- Re-train the model using the same TF-IDF vectorizer you save (don't re-fit a new vectorizer post-training).")

# --------------------------------------
# TAB: Insights
# --------------------------------------
with tab_insights:
    st.subheader("Insights — vocabulary, wordclouds, and sample inspection")
    if DATASET is None:
        st.info("Add data/True.csv and data/Fake.csv to enable dataset insights.")
    else:
        st.write(f"Dataset loaded with {len(DATASET)} rows")
        if WORDCLOUD_AVAILABLE:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("WordCloud — Fake")
                fake_text = " ".join(DATASET.query("label==0")["text"].astype(str).values)
                wc = WordCloud(width=780, height=400).generate(fake_text)
                fig = plt.figure(figsize=(6.0, 3.4)); plt.imshow(wc); plt.axis("off"); st.pyplot(fig)
            with c2:
                st.markdown("WordCloud — Real")
                real_text = " ".join(DATASET.query("label==1")["text"].astype(str).values)
                wc2 = WordCloud(width=780, height=400).generate(real_text)
                fig2 = plt.figure(figsize=(6.0, 3.4)); plt.imshow(wc2); plt.axis("off"); st.pyplot(fig2)
        else:
            st.warning("Install `wordcloud` to enable word clouds.")

    st.markdown("---")
    st.write("Vectorizer vocabulary summary")
    try:
        st.write(f"Vocabulary size: **{len(VOCAB)}**")
        if len(VOCAB) > 0:
            sample_terms = np.random.choice(VOCAB, size=min(60, len(VOCAB)), replace=False)
            st.write("Sample terms:", ", ".join(sorted(sample_terms)))
    except Exception:
        st.write("Vocabulary not available for this vectorizer.")

    st.markdown("---")
    st.write("Manual sample inspection")
    sample_idx = st.number_input("Pick a dataset sample index", min_value=0, max_value=(len(DATASET)-1 if DATASET is not None else 0), value=0, step=1)
    if DATASET is not None:
        sample_row = DATASET.iloc[int(sample_idx)]
        st.markdown("**Sample text**")
        st.write(sample_row["text"][:2000])
        if st.button("Analyze this sample", key="analyze_sample"):
            labels, probs = predict_label([sample_row["text"]], threshold=threshold)
            prob_fake, prob_real = float(probs[0,0]), float(probs[0,1])
            st.write("Prediction:", PRED_LABELS[int(labels[0])], f"• Real: {prob_real:.3f} • Fake: {prob_fake:.3f}")
            if show_explain:
                html, top_items = highlight_contributions(sample_row["text"], max_terms=30)
                if top_items:
                    st.markdown(html, unsafe_allow_html=True)
                    df_expl = pd.DataFrame([{"token": t, "contribution": s, "direction": ("Real" if s>0 else "Fake")} for t,s in top_items])
                    st.dataframe(df_expl, use_container_width=True)
                else:
                    st.info("Not enough signal to compute token contributions for this sample.")

# --------------------------------------
# TAB: Help / FAQ
# --------------------------------------
with tab_help:
    st.subheader("FAQ & guidance")
    st.markdown("""
    **Is the model 100% accurate?**  
    No — this tool is for screening. Use the prediction **as one signal** and always verify with credible sources.

    **Why is accuracy low?**  
    Common causes:
    - model and vectorizer mismatch (vocab/coefs not aligned)
    - dataset class imbalance
    - insufficient or noisy training data
    - using a different preprocessing pipeline for production vs training

    **What to do when URL extraction fails?**  
    - Many news sites are JS-driven or paywalled — copy-paste the article manually into Single Check.
    - Try another news site or use a backend with headless browser (outside scope of this app).

    **How to improve the model?**
    - Re-train using the same vectorizer you save (don't fit a new vectorizer after training)
    - Use more labeled training data and cross-validation
    - Apply class weighting or resampling for class imbalance
    - Consider more advanced models (fine-tuned transformer), but these require more resources.

    **Extra features built-in**
    - Token-level explainability (tfidf * coef)
    - PDF & CSV exports for single and batch results
    - Model / vectorizer compatibility checks
    - Calibration & ROC/PR diagnostics (requires dataset)
    """)

    st.markdown("---")
    st.write("Session history (latest 200):")
    hist_df = pd.DataFrame(st.session_state.get("history", [])[-200:])
    st.dataframe(hist_df, use_container_width=True)
    if not hist_df.empty:
        make_download_button_bytes(df_to_csv_bytes(hist_df), "session_history.csv", "⬇️ Download history (CSV)")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.caption("© 2025 • Fake News Detection • Developed by Muhammad Sarmad Usman using Streamlit. Results are for guidance only; always verify with trusted sources.")
st.markdown('</div>', unsafe_allow_html=True)