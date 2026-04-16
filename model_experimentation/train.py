import mlflow
import mlflow.sklearn
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
import numpy as np
from itertools import combinations
from datetime import datetime

# Config
DATA_PATH = "data/transformed/t_movie_features.parquet"
MODEL_DIR = Path("models/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "movie-similarity"

# Load data
df = pl.read_parquet(DATA_PATH)

# Combine text features into a single field
df = df.with_columns(
    (
        pl.col("title").fill_null("") + " "
        + pl.col("description").fill_null("") + " "
        + pl.col("type").fill_null("") + " "
        + pl.col("actors").fill_null("")
        + pl.col("listed_in").fill_null("") + " "
        + pl.col("movie_info").fill_null("") + " "
        + pl.col("critics_consensus").fill_null("") + " "
        + pl.col("content_rating").fill_null("") + " "
        + pl.col("directors").fill_null("")
    ).alias("text_features")
)

pdf = df.to_pandas()

# After building the final df used for TF-IDF
final_df = df.select(["movie_id", "title", "listed_in","text_features", "description", "content_rating"])

# Save aligned metadata
final_df.write_parquet("models/artifacts/tfidf_rows.parquet")

# Build matrix from final_df["text_features"]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20_000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(final_df["text_features"].to_list())

with open("models/artifacts/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

# MLflow setup
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name=f"tfidf-cosine-v1_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):

    # Compute full cosine similarity matrix (can be large but OK for ~2k items)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Remove self-similarity
    np.fill_diagonal(similarity_matrix, np.nan)

    # Similarity distribution metrics
    mean_similarity = np.nanmean(similarity_matrix)
    median_similarity = np.nanmedian(similarity_matrix)
    p95_similarity = np.nanpercentile(similarity_matrix, 95)

    # Top-K evaluation
    TOP_K = 5
    genres = final_df["listed_in"].to_list()

    topk_indices = np.argsort(similarity_matrix, axis=1)[:, -TOP_K:]

    # Genre overlap
    genre_overlaps = []

    # Intra-list similarity
    intra_list_sims = []

    for i, recs in enumerate(topk_indices):
        base_genres = set(genres[i].split(", "))
        rec_genres = [
            set(genres[j].split(", ")) for j in recs
        ]

        overlaps = [
            len(base_genres & g) / max(len(base_genres), 1)
            for g in rec_genres
        ]
        genre_overlaps.append(np.mean(overlaps))

        # Diversity (ILS)
        pair_sims = [
            similarity_matrix[a, b]
            for a, b in combinations(recs, 2)
        ]
        if pair_sims:
            intra_list_sims.append(np.mean(pair_sims))

    avg_genre_overlap = float(np.mean(genre_overlaps))
    avg_intra_list_similarity = float(np.mean(intra_list_sims))


    # Coverage
    recommended_items = set(topk_indices.flatten())
    item_coverage = len(recommended_items) / tfidf_matrix.shape[0]

    num_movies = tfidf_matrix.shape[0]
    num_features = tfidf_matrix.shape[1]
    matrix_density = tfidf_matrix.nnz / (num_movies * num_features)


    # Save artifacts
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    matrix_path = MODEL_DIR / "tfidf_matrix.pkl"

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # Log to MLflow (same indentation level as file saves)

    mlflow.log_param("model_type", "tfidf")
    mlflow.log_param("top_k", TOP_K)
    mlflow.log_param("num_movies", num_movies)
    mlflow.log_param(
        "text_fields",
        "title,description,type,actors,listed_in,"
        "movie_info,critics_consensus,content_rating,directors"
    )

    # Similarity metrics
    mlflow.log_metric("mean_similarity", mean_similarity)
    mlflow.log_metric("median_similarity", median_similarity)
    mlflow.log_metric("p95_similarity", p95_similarity)

    # Relevance & diversity
    mlflow.log_metric("avg_genre_overlap", avg_genre_overlap)
    mlflow.log_metric("avg_intra_list_similarity", avg_intra_list_similarity)

    # Coverage
    mlflow.log_metric("item_coverage", item_coverage)

    # System metrics
    mlflow.log_metric("num_features", num_features)
    mlflow.log_metric("matrix_density", matrix_density)

    # Artifacts
    mlflow.log_artifact(matrix_path)
    mlflow.log_artifact("models/artifacts/tfidf_rows.parquet")
    mlflow.log_artifact(vectorizer_path)

print("Model training complete")
