# Customer Segmentation

Group mall shoppers into distinct segments with K-Means clustering.

## What it does

Clusters 200 mall customers by annual income and spending score, so a
business can see who its big spenders, careful savers and average shoppers are.

## Data

Mall Customers dataset (200 rows, 5 columns: CustomerID, Gender, Age,
Annual Income, Spending Score). Downloaded automatically from a public
GitHub mirror, no login needed.

## Model

- Features are standardised, then the elbow method runs K-Means for k = 1 to 10.
- The elbow lands at k = 5, so the final model is K-Means with 5 clusters.

## Results

Silhouette score: **0.5547** (a solid, well separated clustering).

| Cluster | Customers | Centre: income | Centre: spending score | Who they are |
| --- | --- | --- | --- | --- |
| 0 | 81 | $55k | 50 | Average shoppers |
| 1 | 39 | $87k | 82 | High income, big spenders |
| 2 | 22 | $26k | 79 | Low income, big spenders |
| 3 | 35 | $88k | 17 | High income, careful spenders |
| 4 | 23 | $26k | 21 | Low income, careful spenders |

The plots `elbow.png` and `clusters.png` show the elbow curve and the five
segments with their centroids.

## How to run

```bash
pip install -r requirements.txt
python main.py
```
