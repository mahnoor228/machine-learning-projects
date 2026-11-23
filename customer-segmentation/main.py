"""
Customer Segmentation
=====================
Group mall customers into segments using K-Means clustering on
annual income and spending score.

Dataset: Mall Customers (200 shoppers, public mirror on GitHub).
Method: elbow method to pick k, then K-Means with k=5.
"""

import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_URLS = [
    "https://raw.githubusercontent.com/rithwik00/Mall-Customer-Dataset/main/Mall_Customers.csv",
    "https://raw.githubusercontent.com/abcom-mltutorials/mall/main/Mall_Customers.csv",
    "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/Mall_Customers.csv",
]
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")


def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Using cached data: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = None
        for url in DATA_URLS:
            try:
                print(f"Downloading data from {url} ...")
                urllib.request.urlretrieve(url, DATA_PATH)
                df = pd.read_csv(DATA_PATH)
                print(f"Downloaded {len(df)} rows.")
                break
            except Exception as e:
                print(f"  failed: {e}")
        if df is None:
            raise RuntimeError("Could not download Mall_Customers.csv from any mirror.")
    df.columns = df.columns.str.strip()
    return df


def main():
    df = load_data()
    print(f"Dataset: {df.shape[0]} customers, {df.shape[1]} columns")
    print(df.head().to_string(index=False))
    print("\nSummary:")
    print(df.describe().round(2).to_string())

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    X_scaled = StandardScaler().fit_transform(X)

    # Elbow method
    wcss = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    print("\nElbow method (WCSS):")
    for k, w in enumerate(wcss, start=1):
        print(f"  k={k:<2} WCSS={w:8.1f}")

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, 11), wcss, "o-")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.savefig("elbow.png", dpi=120)

    # K-Means with k=5 (the elbow)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = labels

    print("\nK-Means with k=5")
    print(f"  Silhouette score: {silhouette_score(X_scaled, labels):.4f}")
    print("\nCluster sizes and centres (income k$, spending score):")
    for c in range(5):
        members = df[df["Cluster"] == c]
        centre = kmeans.cluster_centers_[c]
        centre_orig = StandardScaler().fit(X).inverse_transform([centre])[0]
        print(
            f"  Cluster {c}: {len(members):>3} customers | "
            f"centre income ${centre_orig[0]:.0f}k, spending {centre_orig[1]:.0f}"
        )

    plt.figure(figsize=(8, 6))
    for c in range(5):
        members = df[df["Cluster"] == c]
        plt.scatter(
            members["Annual Income (k$)"],
            members["Spending Score (1-100)"],
            s=40,
            label=f"Cluster {c}",
        )
    centres = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centres[:, 0], centres[:, 1], s=200, marker="X", c="black", label="Centroids")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments (K-Means, k=5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("clusters.png", dpi=120)
    print("\nSaved plots: elbow.png, clusters.png")


if __name__ == "__main__":
    main()
