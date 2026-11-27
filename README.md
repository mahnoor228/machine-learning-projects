# Machine Learning Projects

Five beginner friendly machine learning projects, each one complete and
runnable end to end: real datasets, real models, real results. Built to
learn the core ideas of ML by doing them, one project at a time.

## Projects

| # | Project | What it does | Tech | Headline result |
| --- | --- | --- | --- | --- |
| 1 | [House Price Prediction](house-price-prediction/) | Predict California house prices from income, location and house features | Python, scikit-learn | Random Forest R2 of 0.806 |
| 2 | [Customer Segmentation](customer-segmentation/) | Group mall shoppers into 5 segments by income and spending | Python, scikit-learn | Silhouette score of 0.555 |
| 3 | [Movie Recommendation System](movie-recommendation/) | Recommend movies you will like, based on MovieLens ratings | Python, pandas, scikit-learn | RMSE of 0.91 on held out ratings |
| 4 | [Plant Disease Detection](plant-disease-detection/) | Classify tomato leaf photos as healthy or diseased | Python, TensorFlow | 79.6% validation accuracy |
| 5 | [Credit Card Fraud Detection](credit-card-fraud-detection/) | Spot fraud in 284,807 transactions where only 0.17% are fraud | Python, scikit-learn | ROC AUC of 0.972, 91.8% fraud recall |

## How to run

Each project is self contained. Pick one, install its dependencies and run it:

```bash
cd house-price-prediction
pip install -r requirements.txt
python main.py
```

Or install everything once from the root:

```bash
pip install -r requirements.txt
```

Every project downloads its own data automatically on the first run, no
accounts or API keys needed. Datasets are never committed to the repo;
they are fetched at run time and ignored by git.

## What you will learn

- **Regression** with house prices: linear models vs random forests, and
  which features actually matter.
- **Clustering** with customer segmentation: the elbow method, K-Means,
  and reading a silhouette score.
- **Recommender systems** with MovieLens: user-item matrices, cosine
  similarity, and evaluating with RMSE.
- **Computer vision** with plant leaves: CNNs, data augmentation, and
  stratified train/validation splits.
- **Imbalanced classification** with fraud detection: why accuracy lies,
  and the metrics that matter instead (ROC AUC, average precision, recall).

## Repo layout

```
machine-learning-projects/
├── house-price-prediction/       # regression, California Housing
├── customer-segmentation/        # K-Means, Mall Customers
├── movie-recommendation/         # collaborative filtering, MovieLens
├── plant-disease-detection/      # CNN, PlantVillage subset
├── credit-card-fraud-detection/  # imbalanced classification, Kaggle data
└── requirements.txt              # everything, in one place
```

## Data sources

- California Housing (scikit-learn)
- Mall Customers (public GitHub mirror)
- MovieLens ml-latest-small (GroupLens)
- PlantVillage (CC0, Hughes and Salathe 2015)
- Credit Card Fraud Detection (Kaggle, ULB / Worldline)
