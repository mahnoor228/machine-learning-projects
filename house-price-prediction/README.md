# House Price Prediction

Predict California house prices from block level features using regression.

## What it does

Trains two regression models on the California Housing dataset (20,640 blocks)
and compares them. The target is the median house value in $100,000s.

## Data

California Housing, loaded with `sklearn.datasets.fetch_california_housing`.
8 features: median income, house age, average rooms, average bedrooms,
population, average occupancy, latitude and longitude. No login or manual
download needed, scikit-learn fetches it from its public mirror.

## Model

- Linear Regression (baseline)
- Random Forest Regressor (200 trees)

The data is split 80/20 into train and test sets.

## Results

| Model | R2 | RMSE | MAE |
| --- | --- | --- | --- |
| Linear Regression | 0.5758 | 0.7456 | 0.5332 |
| Random Forest | **0.8059** | **0.5043** | **0.3268** |

Random Forest wins clearly. Median income is by far the most important
feature (importance 0.53), followed by average occupancy (0.14) and
latitude/longitude (0.09 each). The plot `actual_vs_predicted.png` shows
predictions tracking actual prices closely.

## How to run

```bash
pip install -r requirements.txt
python main.py
```
