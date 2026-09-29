"""Phase 1: EDA + baselines on Bank Marketing data.
Run from ~/workspace/ai-analyst-assistant with the xai venv:
~/workspace/xai-readmission/.venv/bin/python src/phase1_eda_baseline.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

RANDOM_STATE = 42
df = pd.read_csv("data/bank_data/bank-full.csv", sep=";")
print("shape:", df.shape)

# ---- EDA: answer analyst questions ----
print("\n--- subscription rate by job ---")
print((df.assign(sub=(df["y"] == "yes").astype(int))
         .groupby("job")["sub"].agg(["mean", "count"])
         .sort_values("mean", ascending=False).round(3)))
print("\n--- subscription rate by education ---")
print((df.assign(sub=(df["y"] == "yes").astype(int))
         .groupby("education")["sub"].mean().round(3)))
print("\n--- avg balance: subscribers vs not ---")
print(df.assign(sub=(df["y"] == "yes")).groupby("sub")["balance"].mean().round(0))
print("\n--- subscription rate by month ---")
print((df.assign(sub=(df["y"] == "yes").astype(int))
         .groupby("month")["sub"].mean().sort_values(ascending=False).round(3)))

# ---- modeling: predict subscription ----
# NOTE: 'duration' is dropped -> leakage (only known after the call ends)
df["target"] = (df["y"] == "yes").astype(int)
print("\npositive rate:", df["target"].mean().round(4))
X = df.drop(columns=["y", "target", "duration"])
y = df["target"]

num_cols = X.select_dtypes(exclude="object").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore",
                                           sparse_output=False))]), cat_cols),
])

for name, clf in {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=RANDOM_STATE),
        "hgb": HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                              class_weight="balanced")}.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f"{name}: AUC={roc_auc_score(y_test, p):.4f} "
          f"AP={average_precision_score(y_test, p):.4f} "
          f"F1={f1_score(y_test, pred):.4f}")
print("\nDone. Next: re-run WITHOUT dropping 'duration' and compare. "
      "That gap = leakage.")
