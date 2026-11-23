"""
House Price Prediction
========================
Predict California house prices (median value per block) from features
like median income, house age, rooms and location, using regression.

Dataset: California Housing (scikit-learn, fetched from a public mirror).
Models compared: Linear Regression and Random Forest Regressor.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]:,} houses, {X.shape[1]} features")
    print("Features:", ", ".join(data.feature_names))
    print("Target: median house value (in $100,000s)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    for name, model in [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = (model, preds)
        print(f"\n{name}")
        print(f"  R2 score : {r2_score(y_test, preds):.4f}")
        print(f"  RMSE     : {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
        print(f"  MAE      : {mean_absolute_error(y_test, preds):.4f}")

    best_name, (best_model, best_preds) = max(
        results.items(), key=lambda kv: r2_score(y_test, kv[1][1])
    )
    print(f"\nBest model: {best_name}")

    importances = best_model.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\nFeature importances (Random Forest):")
    for i in order:
        print(f"  {data.feature_names[i]:<12} {importances[i]:.4f}")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_preds, s=8, alpha=0.4)
    plt.plot([0, 5], [0, 5], "r--")
    plt.xlabel("Actual price ($100k)")
    plt.ylabel("Predicted price ($100k)")
    plt.title(f"Actual vs Predicted House Prices ({best_name})")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120)
    print("\nSaved plot: actual_vs_predicted.png")


if __name__ == "__main__":
    main()
