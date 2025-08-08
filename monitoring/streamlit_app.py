# monitoring/streamlit_app.py
# Minimal monitoring dashboard with submit box in sidebar

from pathlib import Path
import json
import os
from typing import List
import re
import altair as alt

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt


# Optional: sklearn metrics if available
try:
    from sklearn.metrics import accuracy_score, precision_score, classification_report
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Sentiment Monitoring", layout="wide")

# Fixed locations inside the container
LOG_FILE = Path("/app/logs/prediction_logs.json")
IMDB_CSV = Path("/app/monitoring/IMDB Dataset.csv")

# URL to FastAPI service. In Docker, use the container name on the same network.
API_URL = os.getenv("API_URL", "http://sentiment_api:8000/predict")

# -----------------------
# Helpers
# -----------------------
WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

def token_len_series(text_series: pd.Series) -> pd.Series:
    # Robust token count: counts word-like tokens (letters/numbers/underscore)
    # Avoids empty/HTML edge cases better than simple .split()
    return (
        text_series.fillna("")
        .astype(str)
        .map(lambda s: len(WORD_RE.findall(s)))
        .astype(int)
    )

@st.cache_data(show_spinner=False)
def load_logs(ndjson_path: Path) -> pd.DataFrame:
    if not ndjson_path.exists():
        return pd.DataFrame(columns=["timestamp","request_text","predicted_sentiment","true_label"])

    rows: List[dict] = []
    with ndjson_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(rows)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ["request_text", "predicted_sentiment", "true_label"]:
        if col in df:
            df[col] = df[col].astype("string")
    return df

@st.cache_data(show_spinner=False)
def load_imdb(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["review","sentiment"])
    return pd.read_csv(csv_path)

def sentence_lengths(text_series: pd.Series) -> pd.Series:
    return text_series.fillna("").astype(str).str.split().map(len)

def safe_precision(true_labels, pred_labels) -> float:
    if SKLEARN_AVAILABLE:
        return float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
    classes = sorted(set(true_labels) | set(pred_labels))
    vals = []
    for c in classes:
        tp = sum((pred_labels == c) & (true_labels == c))
        fp = sum((pred_labels == c) & (true_labels != c))
        vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0

def safe_accuracy(true_labels, pred_labels) -> float:
    if SKLEARN_AVAILABLE:
        return float(accuracy_score(true_labels, pred_labels))
    return float(np.mean((true_labels == pred_labels).astype(float))) if len(true_labels) else 0.0

# -----------------------
# Sidebar: Submit new review
# -----------------------
st.sidebar.header("Submit a Review")

default_text = st.session_state.get("last_text", "")
new_text = st.sidebar.text_area("Review text", value=default_text, height=140, key="sidebar_text")

label_options = ["positive", "negative"]
default_label = st.session_state.get("last_true_label", "positive")
true_label = st.sidebar.selectbox("True label", options=label_options, index=label_options.index(default_label))

if st.sidebar.button("Submit"):
    payload = {"text": new_text, "true_label": true_label}
    try:
        r = requests.post(API_URL, json=payload, timeout=10)
        r.raise_for_status()
        pred = r.json().get("sentiment", "")
        correct = (pred == true_label)
        st.session_state["last_prediction"] = pred
        st.session_state["last_correct"] = bool(correct)
        st.session_state["last_text"] = new_text
        st.session_state["last_true_label"] = true_label
        # refresh logs cache and rerun to update charts
        load_logs.clear()
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Submission failed: {e}")

# Show latest outcome in the sidebar (if available)
if "last_prediction" in st.session_state:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Prediction:** {st.session_state['last_prediction']}")
    if st.session_state.get("last_correct"):
        st.sidebar.success("Correct")
    else:
        st.sidebar.error("Incorrect")


# -----------------------
# Main area
# -----------------------
st.title("Sentiment Monitoring Dashboard")

logs_df = load_logs(LOG_FILE)
imdb_df = load_imdb(IMDB_CSV)

if logs_df.empty:
    st.warning("No logs found yet. Submit a review on the left to generate logs.")
else:
    st.success(f"Loaded {len(logs_df):,} log entries.")

if imdb_df.empty:
    st.warning("IMDB dataset not found. Data drift vs training will be limited.")
else:
    st.info(f"Loaded IMDB dataset with {len(imdb_df):,} rows.")

# -----------------------
# Data Drift: sentence length distributions (single overlaid density chart)
# -----------------------
st.header("Data Drift: Sentence Length Distribution")

IMDB_TEXT_COL = "review"

if (
    not imdb_df.empty and IMDB_TEXT_COL in imdb_df.columns
    and not logs_df.empty and "request_text" in logs_df
):
    imdb_lengths = token_len_series(imdb_df[IMDB_TEXT_COL]).rename("length").to_frame()
    imdb_lengths["source"] = "Training (IMDB)"

    live_lengths = token_len_series(logs_df["request_text"]).rename("length").to_frame()
    live_lengths["source"] = "Live Inference"

    both = pd.concat([imdb_lengths, live_lengths], ignore_index=True)

    # Cap display at the shared 99th percentile to avoid long-tail squashing
    cap = int(np.nanmax([
        imdb_lengths["length"].quantile(0.99),
        live_lengths["length"].quantile(0.99),
        1
    ]))

    base = alt.Chart(both)

    density = (
        base.transform_density(
            density="length",
            groupby=["source"],
            as_=["length", "density"],
            extent=[0, cap],
            steps=120,
        )
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X(
                "length:Q",
                title="Sentence length (tokens)",
                scale=alt.Scale(domain=(0, cap)),
                axis=alt.Axis(titlePadding=16, labelPadding=6)
            ),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("source:N", legend=alt.Legend(title="")),
            tooltip=["source:N", alt.Tooltip("length:Q", format=".0f"), alt.Tooltip("density:Q", format=".3f")],
        )
        # add explicit bottom padding so the x-axis title never clips
        .properties(height=320, padding={"left": 5, "right": 5, "top": 5, "bottom": 50})
        .configure_axis(labelLimit=1000)  # avoid truncating tick labels
        .configure_view(stroke=None)      # optional: remove outer border
    )

    st.altair_chart(density, use_container_width=True)
else:
    st.info("Not enough data for sentence length comparison.")

# -----------------------
# Target Drift: Label Distribution (side-by-side bars)
# -----------------------
st.header("Target Drift: Label Distribution")

IMDB_LABEL_COL = "sentiment"

if (
    not imdb_df.empty and IMDB_LABEL_COL in imdb_df.columns
    and not logs_df.empty and "predicted_sentiment" in logs_df
):
    train = (
        imdb_df[IMDB_LABEL_COL]
        .astype("string").str.lower().str.strip()
        .value_counts(normalize=True)
        .rename_axis("label").reset_index(name="proportion")
    )
    train["source"] = "Training"

    live = (
        logs_df["predicted_sentiment"]
        .astype("string").str.lower().str.strip()
        .value_counts(normalize=True)
        .rename_axis("label").reset_index(name="proportion")
    )
    live["source"] = "Live"

    drift_df = pd.concat([train, live], ignore_index=True)

    # Build label order from the union actually present; keep a sensible ordering
    pref = ["negative", "positive"]
    present = [l for l in pref if l in set(drift_df["label"])]
    if not present:  # safety
        present = sorted(drift_df["label"].unique())

    # Ensure both sources have all labels (fill missing with 0)
    for lab in present:
        for src in ("Training", "Live"):
            if not ((drift_df["label"] == lab) & (drift_df["source"] == src)).any():
                drift_df = pd.concat(
                    [drift_df, pd.DataFrame([{"label": lab, "proportion": 0.0, "source": src}])],
                    ignore_index=True,
                )

    drift_df["label"] = pd.Categorical(drift_df["label"], categories=present, ordered=True)

    chart = (
        alt.Chart(drift_df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="", sort=present, axis=alt.Axis(labelAngle=0, labelPadding=8)),
            xOffset=alt.XOffset("source:N"),
            y=alt.Y("proportion:Q", title="Proportion", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("source:N", legend=alt.Legend(title="")),
            tooltip=["source:N", "label:N", alt.Tooltip("proportion:Q", format=".2f")],
        )
        .properties(height=320)
        .configure_axis(labelLimit=1000)  # prevent truncation
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Not enough data for target drift comparison.")

# -----------------------
# Accuracy & Precision from feedback
# -----------------------
st.header("Model Accuracy & User Feedback")

if not logs_df.empty and {"predicted_sentiment", "true_label"}.issubset(set(logs_df.columns)):
    labeled = logs_df.dropna(subset=["true_label", "predicted_sentiment"]).copy()
    labeled["true_label"] = labeled["true_label"].astype("string")
    labeled["predicted_sentiment"] = labeled["predicted_sentiment"].astype("string")

    if len(labeled) == 0:
        st.info("No user feedback available in logs yet.")
    else:
        acc = safe_accuracy(labeled["true_label"], labeled["predicted_sentiment"])
        prec = safe_precision(labeled["true_label"], labeled["predicted_sentiment"])

        if acc < 0.80:
            st.error(f"Warning: Accuracy below threshold — {acc:.2%}")
        else:
            st.success(f"Accuracy: {acc:.2%}")

        st.metric("Precision (macro)", f"{prec:.2%}")
        st.caption(f"Evaluated on {len(labeled)} log entries with feedback.")

        if SKLEARN_AVAILABLE:
            st.text("Classification Report:")
            st.code(classification_report(labeled["true_label"], labeled["predicted_sentiment"], zero_division=0))
else:
    st.info("Waiting for logged feedback to compute metrics.")