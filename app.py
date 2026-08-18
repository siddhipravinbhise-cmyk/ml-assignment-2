import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Credit Card Default Predictor | BITS WILP",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; color: #0D9488; font-weight: 700; margin-bottom: 0px; }
    .sub-title { font-size: 1rem; color: #64748B; margin-bottom: 20px; }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Credit Card Default Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">M.Tech AIML / DSE Machine Learning Assignment 2 | BITS Student ID: 2025ac05921</div>', unsafe_allow_html=True)

st.sidebar.header("Configuration & Input")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"], help="Upload test_data.csv")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "k-NN",
    "Naive Bayes",
    "Random Forest",
    "Gradient Boosting",
]
selected_model_name = st.sidebar.selectbox("Select ML Model", model_options)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Student Information**\n"
    "- **Name:** Siddhi Pravin Bhise\n"
    "- **ID:** 2025ac05921\n"
    "- **Dataset:** UCI Credit Card Default (ID: 350)"
)

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.joblib")
    models = {}
    for name in model_options:
        key = name.lower().replace(" ", "_").replace("-", "")
        models[name] = joblib.load(f"model/{key}.joblib")
    return scaler, models

try:
    scaler, models = load_artifacts()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure models are trained.")
    st.stop()

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    if os.path.exists("test_data.csv"):
        data = pd.read_csv("test_data.csv")
        st.info("Loaded default `test_data.csv` from repository root.")
    else:
        st.warning("Please upload a CSV file to evaluate.")
        st.stop()

if "target" in data.columns:
    X_test = data.drop(columns=["target"])
    y_test = data["target"]
else:
    X_test = data
    y_test = None

needs_scaling = selected_model_name in ["Logistic Regression", "k-NN", "Naive Bayes"]
X_proc = scaler.transform(X_test) if needs_scaling else X_test

clf = models[selected_model_name]
y_pred = clf.predict(X_proc)
y_proba = clf.predict_proba(X_proc)[:, 1] if hasattr(clf, "predict_proba") else y_pred

tab1, tab2, tab3 = st.tabs(["Model Performance", "Leaderboard Comparison", "Dataset Sample"])

with tab1:
    st.subheader(f"Evaluation Metrics: {selected_model_name}")
    if y_test is not None:
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC Score", f"{auc:.4f}")
        c3.metric("Precision", f"{prec:.4f}")
        c4.metric("Recall", f"{rec:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")
        st.subheader("Confusion Matrix")
        col_cm, col_info = st.columns([1, 1])

        with col_cm:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Non-Default", "Default"],
                yticklabels=["Non-Default", "Default"],
            )
            plt.ylabel("Actual Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            st.pyplot(fig)

        with col_info:
            st.markdown(
                f"""
            **Prediction Insights for {selected_model_name}:**
            - **True Negatives:** {cm[0][0]}
            - **False Positives:** {cm[0][1]}
            - **False Negatives:** {cm[1][0]}
            - **True Positives:** {cm[1][1]}
            
            The **MCC (Matthews Correlation Coefficient)** of **{mcc:.4f}** balances false positives and false negatives for imbalanced default prediction.
            """
            )

with tab2:
    st.subheader("All Models Leaderboard")
    if os.path.exists("model/metrics.json"):
        with open("model/metrics.json", "r") as f:
            leaderboard_data = json.load(f)
        df_lb = pd.DataFrame(leaderboard_data).T
        st.dataframe(df_lb.style.highlight_max(axis=0, color="#2DD4BF"))

with tab3:
    st.subheader("Uploaded Dataset Head")
    st.dataframe(data.head(10))
