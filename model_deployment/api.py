
from fastapi import FastAPI
import polars as pl
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = FastAPI(title="Movie Recommendation API")

TOP_K = 5

# Load artifacts once at startup
rows_df = pl.read_parquet("/workspaces/netflix-movie-rec/models/artifacts/tfidf_rows.parquet")

with open("/workspaces/netflix-movie-rec/models/artifacts/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

titles = rows_df["title"].to_list()
genres = rows_df["listed_in"].to_list()
descriptions = rows_df["description"].to_list()
content_ratings = rows_df["content_rating"].to_list()

@app.get("/recommend")
def recommend(title: str, k: int = TOP_K):
    if title not in titles:
        return {"error": "Movie not found"}

    idx = titles.index(title)

    sims = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_indices = sims.argsort()[::-1][1 : k + 1]

    results = []
    for i in top_indices:
        results.append({
            "title": titles[i],
            "genre": genres[i],
            "description": descriptions[i],
            "content_rating": content_ratings[i],
            "similarity": float(sims[i])
        })

    return {"results": results}
