"""
Credit Card Fraud Detection
===========================
Spot fraudulent transactions in a highly imbalanced dataset.

Dataset: Kaggle Credit Card Fraud (ULB / Worldline), 284,807 transactions
from September 2013, only 492 fraudulent (0.17%). Features V1-V28 are
PCA-transformed for privacy. Downloaded from a public GitHub mirror of
the Kaggle file (no login needed). The CSV is about 100 MB and is
downloaded at run time, not committed to the repo.

Method: stratified train/test split, standardise Amount and Time,
logistic regression with class_weight="balanced", evaluated with
ROC AUC, average precision and a confusion matrix. Accuracy is
meaningless here (guessing "not fraud" every time scores 99.83%),
so the imbalanced metrics are what matter.
"""

import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")
URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Using cached data: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading credit card data (~100 MB) from {URL} ...")
    urllib.request.urlretrieve(URL, DATA_PATH)
    return pd.read_csv(DATA_PATH)


def main():
    df = load_data()
    print(f"Dataset: {len(df):,} transactions")
    print(f"Fraudulent: {int(df['Class'].sum()):,} ({df['Class'].mean() * 100:.3f}%)")
    print(f"Legitimate: {int((df['Class'] == 0).sum()):,}")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\nLogistic Regression (class_weight=balanced)")
    print(f"  ROC AUC           : {roc_auc_score(y_test, probs):.4f}")
    print(f"  Average precision : {average_precision_score(y_test, probs):.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))
    print(
        "Note: accuracy is not reported on purpose. Predicting 'not fraud' "
        "for every transaction would score 99.83% accuracy and catch zero fraud."
    )


if __name__ == "__main__":
    main()
