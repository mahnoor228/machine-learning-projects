# Credit Card Fraud Detection

Spot fraudulent transactions in a dataset where fraud is extremely rare.

## What it does

Trains a classifier to flag fraudulent credit card transactions, and shows
why accuracy is the wrong metric when only 0.17 percent of transactions
are fraud.

## Data

Kaggle Credit Card Fraud dataset (ULB / Worldline): 284,807 transactions
from two days in September 2013, of which only 492 (0.173 percent) are
fraudulent. Features V1 to V28 are PCA transformed for privacy. The CSV
(about 100 MB) is downloaded automatically from a public GitHub mirror of
the Kaggle file, no login needed. It is not committed to this repo.

## Model

- Stratified 80/20 train/test split so fraud appears in both sets.
- Standardise Time and Amount.
- Logistic Regression with `class_weight="balanced"`.
- Evaluated with ROC AUC, average precision and a confusion matrix.

## Results

| Metric | Value |
| --- | --- |
| ROC AUC | **0.9722** |
| Average precision | **0.7159** |
| Fraud recall | **91.8 percent** (90 of 98 fraud cases in the test set caught) |
| Fraud precision | 6.1 percent (1,394 transactions flagged, 90 of them real fraud) |

The low precision is the honest trade off with this much imbalance: to
catch almost all fraud, the model flags many legitimate transactions too.
Accuracy is deliberately not reported, since predicting "not fraud" for
every transaction would score 99.83 percent accuracy and catch zero fraud.

## How to run

```bash
pip install -r requirements.txt
python main.py
```
