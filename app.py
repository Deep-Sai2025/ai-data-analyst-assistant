"""Phase 4: The App — a Streamlit chat UI around the Phase 3 AI Brain.

WHAT IT DOES:
  1. You upload any CSV (or use the built-in bank data by default).
  2. You ask a question in plain English.
  3. Gemini writes pandas code, the sandbox runs it, and you see
     the code, the answer, and the chart — all in your browser.
  4. The "Why?" button explains ONE prediction with SHAP: which
     features pushed the model toward "will subscribe" and which
     pushed it away.

RUN IT (from the project folder, same terminal as Phase 3):
  pip install streamlit shap scikit-learn
  streamlit run app.py

Your browser opens at http://localhost:8501 automatically.
The Gemini key still comes from the GEMINI_API_KEY environment variable,
so set it first (PowerShell: $env:GEMINI_API_KEY = "your-key-here").
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")  # charts save to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Let this file import the Phase 3 brain (ask_gemini + run_in_sandbox)
# no matter where Streamlit runs it from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.phase3_gemini_analyst import ask_gemini, run_in_sandbox  # noqa: E402

# On Streamlit Cloud the key lives in st.secrets, not in the environment.
# Copy it into the environment once so the Phase 3 brain finds it.
# Local runs skip this and use the GEMINI_API_KEY environment variable.
if "GEMINI_API_KEY" not in os.environ:
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

DEFAULT_DATA = os.path.join("data", "bank_data", "bank-full.csv")


def load_data(uploaded_file):
    """Load the uploaded CSV, or the built-in bank data if nothing uploaded.

    sep=None with the python engine auto-detects the separator, so both
    comma CSVs and the semicolon bank file just work.
    """
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    return pd.read_csv(DEFAULT_DATA, sep=";")


def explain_one_prediction(df: pd.DataFrame, row_index: int):
    """Train a simple model and explain one row's prediction with SHAP.

    Why logistic regression? SHAP has an exact, fast explainer for linear
    models, and Phase 1 already showed logreg works on this data
    (AUC 0.77). The explanation is per-row: each feature gets a number
    saying how much it pushed this row toward "yes" or toward "no".
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    import shap

    # Drop the target and the leaky column (duration is only known AFTER
    # the call ends, so it must never explain a subscription).
    drop = [c for c in ["y", "duration"] if c in df.columns]
    X = pd.get_dummies(df.drop(columns=drop), drop_first=True)
    y = (df["y"] == "yes").astype(int)

    # Train on a sample so it stays fast even on 45,000 rows.
    sample = df.sample(n=min(5_000, len(df)), random_state=42).index
    X_train, y_train = X.loc[sample], y.loc[sample]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y_train)

    # Explain the chosen row (scaled the same way as the training data).
    row = X.iloc[[row_index]]
    row_scaled = scaler.transform(row)
    explainer = shap.LinearExplainer(model, X_scaled)
    shap_values = explainer.shap_values(row_scaled)[0]

    prediction = model.predict(row_scaled)[0]
    proba = model.predict_proba(row_scaled)[0][1]

    # Top 10 features by absolute impact, most important first.
    order = np.argsort(np.abs(shap_values))[::-1][:10]
    feats = X.columns[order]
    vals = shap_values[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["green" if v > 0 else "red" for v in vals]
    ax.barh(feats[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value  (green = pushes toward YES, red = pushes toward NO)")
    ax.set_title(f"Why this prediction? Row {row_index}")
    plt.tight_layout()
    return prediction, proba, fig


# ---------------------------------------------------------------------------
# THE APP
# ---------------------------------------------------------------------------
st.title("AI Data Analyst Assistant")
st.write("Upload any CSV, ask a question in plain English, and the AI "
         "writes the pandas code, runs it safely, and answers — with charts.")

uploaded = st.file_uploader("Upload a CSV file (optional — bank data is used by default)",
                            type=["csv"])
try:
    df = load_data(uploaded)
except Exception as e:  # noqa: BLE001 - show a friendly error, not a crash
    st.error(f"Could not load the data: {e}")
    st.stop()

st.write(f"Loaded: **{len(df):,} rows, {len(df.columns)} columns**")
st.dataframe(df.head(10))

# --- Ask a question -------------------------------------------------------
st.header("Ask a question")
question = st.text_input("Your question", placeholder="e.g. which job has the highest subscription rate?")

if st.button("Ask"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Gemini is writing code..."):
            try:
                code = ask_gemini(question, df)
            except Exception as e:  # noqa: BLE001
                st.error(f"Gemini error: {e}")
                st.stop()
        st.subheader("Code Gemini wrote")
        st.code(code, language="python")

        ok, result = run_in_sandbox(code, df)
        if ok:
            st.subheader("Answer")
            st.write(result)
            if os.path.exists("chart.png"):
                st.image("chart.png")
                os.remove("chart.png")  # fresh chart per question
        else:
            st.error(result)

# --- Why? (SHAP explanation) ----------------------------------------------
st.header("Why? — explain one prediction")
if "y" not in df.columns:
    st.info("The 'Why?' button needs a target column called `y` "
            "(like the bank data has). Upload a CSV with a `y` column to use it.")
else:
    row_index = st.number_input("Row number to explain", min_value=0,
                                max_value=len(df) - 1, value=0, step=1)
    if st.button("Explain this row"):
        with st.spinner("Training a model and computing SHAP values..."):
            try:
                prediction, proba, fig = explain_one_prediction(df, int(row_index))
            except ImportError:
                st.error("SHAP or scikit-learn is not installed. Run: "
                         "pip install shap scikit-learn")
                st.stop()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not explain this row: {e}")
                st.stop()
        label = "YES (will subscribe)" if prediction == 1 else "NO (will not subscribe)"
        st.write(f"**Model prediction: {label}** — probability of subscribing: {proba:.2%}")
        st.pyplot(fig)
        st.write("Green bars pushed this row toward subscribing; "
                 "red bars pushed it away. Longer bar = bigger influence.")
