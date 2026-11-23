"""
Movie Recommendation System
===========================
Item-based collaborative filtering on the MovieLens dataset.

Dataset: MovieLens ml-latest-small (GroupLens) - 100,836 ratings,
9,742 movies, 610 users. Downloaded as a zip, no login needed.

Method:
  * Build a user-item rating matrix.
  * Compute cosine similarity between movies.
  * Recommend movies similar to one you liked.
  * Evaluate on held-out ratings: predict a rating as the similarity
    weighted average of the user's other ratings, report RMSE.
"""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def load_data():
    if not os.path.exists(os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv")):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Downloading MovieLens ml-latest-small from {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        print("Downloaded and extracted.")
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ml-latest-small", "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "ml-latest-small", "movies.csv"))
    return ratings, movies


def recommend(movie_title, movies, item_sim, top_n=10):
    """Return the top_n most similar movies to the given title."""
    matches = movies[movies["title"].str.lower().str.contains(movie_title.lower())]
    if matches.empty:
        raise ValueError(f"No movie found matching '{movie_title}'")
    idx = matches.index[0]
    movie_id = movies.loc[idx, "movieId"]
    col = list(item_sim.index).index(movie_id)
    sims = item_sim.iloc[col].sort_values(ascending=False)
    sims = sims[sims.index != movie_id][:top_n]
    out = movies.set_index("movieId").loc[sims.index, "title"]
    return list(zip(out.values, sims.values))


def main():
    ratings, movies = load_data()
    print(f"Ratings: {len(ratings):,} | Movies: {len(movies):,} | Users: {ratings['userId'].nunique()}")

    # User-item matrix (users x movies), missing ratings = 0
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    print(f"User-item matrix: {matrix.shape[0]} users x {matrix.shape[1]} movies")

    # Item-item cosine similarity
    item_sim = pd.DataFrame(
        cosine_similarity(matrix.T),
        index=matrix.columns,
        columns=matrix.columns,
    )

    print("\nIf you liked Toy Story (1995), you may also like:")
    for title, sim in recommend("Toy Story", movies, item_sim, top_n=10):
        print(f"  {title:<55} similarity {sim:.3f}")

    print("\nIf you liked The Dark Knight (2008), you may also like:")
    for title, sim in recommend("Dark Knight", movies, item_sim, top_n=10):
        print(f"  {title:<55} similarity {sim:.3f}")

    # Evaluation: hold out 20% of ratings, predict them with item-based CF
    rng = np.random.default_rng(42)
    mask = rng.random(len(ratings)) < 0.2
    train_r = ratings[~mask]
    test_r = ratings[mask]
    train_matrix = train_r.pivot_table(index="userId", columns="movieId", values="rating")
    train_matrix = train_matrix.reindex(columns=matrix.columns).fillna(0)
    train_sim = cosine_similarity(train_matrix.T)

    movie_ids = list(matrix.columns)
    id_to_col = {mid: i for i, mid in enumerate(movie_ids)}
    user_ids = list(matrix.index)
    user_to_row = {u: i for i, u in enumerate(user_ids)}
    train_arr = train_matrix.values

    preds, actuals = [], []
    for _, row in test_r.iterrows():
        u = user_to_row.get(row["userId"])
        c = id_to_col.get(row["movieId"])
        if u is None or c is None:
            continue
        user_ratings = train_arr[u]
        rated = user_ratings > 0
        rated[c] = False  # do not use the held-out rating itself
        sims = train_sim[c][rated]
        vals = user_ratings[rated]
        if sims.sum() == 0:
            continue
        pred = float(np.dot(sims, vals) / sims.sum())
        preds.append(pred)
        actuals.append(row["rating"])

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    print(f"\nEvaluation on {len(actuals):,} held-out ratings")
    print(f"  Item-based CF RMSE: {rmse:.4f} (ratings are 0.5 to 5.0 stars)")
    print("  (A naive always-predict-the-mean baseline scores about 1.05 RMSE.)")


if __name__ == "__main__":
    main()
