import json
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 8421
np.random.seed(SEED)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
LOCAL_XLS = "default_credit_card.xls"

if not os.path.exists(LOCAL_XLS):
    print("Downloading Credit Card Default Dataset...")
    urllib.request.urlretrieve(DATA_URL, LOCAL_XLS)

df = pd.read_excel(LOCAL_XLS, header=1)
df.rename(
    columns={"PAY_0": "PAY_1", "default payment next month": "target"}, inplace=True
)
if "ID" in df.columns:
    df.drop(columns=["ID"], inplace=True)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Saved test_data.csv (6000 rows)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.joblib", compress=3)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        True,
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=6, random_state=SEED),
        False,
    ),
    "k-NN": (KNeighborsClassifier(n_neighbors=15), True),
    "Naive Bayes": (GaussianNB(), True),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=SEED, n_jobs=-1
        ),
        False,
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=SEED
        ),
        False,
    ),
}

metrics_results = {}

for name, (clf, needs_scaling) in models.items():
    X_tr = X_train_scaled if needs_scaling else X_train
    X_te = X_test_scaled if needs_scaling else X_test

    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    y_proba = (
        clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_pred
    )

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics_results[name] = {
        "Accuracy": round(float(acc), 4),
        "AUC": round(float(auc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
    }

    clean_filename = name.lower().replace(" ", "_").replace("-", "")
    joblib.dump(clf, f"model/{clean_filename}.joblib", compress=3)

with open("model/metrics.json", "w") as f:
    json.dump(metrics_results, f, indent=2)

print("\nTraining Complete! Evaluated Results:")
print(pd.DataFrame(metrics_results).T)
