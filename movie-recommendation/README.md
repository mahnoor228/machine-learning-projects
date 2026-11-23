# Movie Recommendation System

An item based collaborative filtering recommender built on MovieLens.

## What it does

Given a movie you liked, it suggests similar movies based on how hundreds
of other users rated them. It also predicts ratings you have not given yet.

## Data

MovieLens ml-latest-small (GroupLens): 100,836 ratings, 9,742 movies,
610 users. The zip is downloaded automatically from GroupLens, no login
needed.

## Model

- Build a user-item rating matrix (610 users x 9,724 movies).
- Compute cosine similarity between every pair of movies.
- Recommend the most similar movies to a given title.
- To evaluate, 20 percent of ratings are held out and predicted as the
  similarity weighted average of the user's other ratings.

## Results

- **RMSE on 19,233 held-out ratings: 0.9125** (ratings run 0.5 to 5.0 stars).
  A naive always-predict-the-mean baseline scores about 1.05, so the
  recommender beats it clearly.
- Liked Toy Story (1995)? Top picks are Toy Story 2 (similarity 0.573),
  Jurassic Park (0.566) and Independence Day (0.564).
- Liked The Dark Knight (2008)? Top picks are Inception (0.727),
  Iron Man (0.671) and The Dark Knight Rises (0.666).

## How to run

```bash
pip install -r requirements.txt
python main.py
```
