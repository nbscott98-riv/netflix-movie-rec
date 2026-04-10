
import pickle
import polars as pl
import numpy as np
import mlflow
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Config
DATA_PATH = "data/modeling/movie_features.parquet"
VECTORIZER_PATH = "models/artifacts/tfidf_vectorizer.pkl"
MATRIX_PATH = "models/artifacts/tfidf_matrix.pkl"

TOP_K = 5
EXPERIMENT_NAME = "movie-similarity-evaluation"

ARTIFACT_DIR = Path("tmp_eval_artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# Load data
rows_df = pl.read_parquet("models/artifacts/tfidf_rows.parquet")
titles = rows_df["title"].to_list()

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

with open(MATRIX_PATH, "rb") as f:
    tfidf_matrix = pickle.load(f)

assert tfidf_matrix.shape[0] == len(titles), \
    "TF-IDF matrix and metadata are misaligned"

# Helper functions
def find_similar(idx, k=TOP_K):
    sims = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_idx = sims.argsort()[-(k + 1):][::-1][1:]
    return [(titles[i], sims[i]) for i in top_idx]

def jaccard(a, b):
    a = set(a.split(", "))
    b = set(b.split(", "))
    return len(a & b) / len(a | b) if a | b else 0

# MLflow setup
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="tfidf-eval"):

    # -------- params --------
    mlflow.log_param("model_type", "tfidf")
    mlflow.log_param("top_k", TOP_K)
    mlflow.log_param("num_movies", len(titles))
    mlflow.log_param(
        "text_fields",
        "title,description,cast,actors,listed_in"
    )

    # Evaluation 1: sanity checks
    sample_indices = np.random.choice(
        len(titles), size=5, replace=False
    )

    nn_lines = []
    for idx in sample_indices:
        nn_lines.append(f"\nMovie: {titles[idx]}")
        for title, score in find_similar(idx):
            nn_lines.append(f"  → {title} (score={score:.3f})")

    nn_text = "\n".join(nn_lines)
    nn_path = ARTIFACT_DIR / "nearest_neighbors.txt"
    nn_path.write_text(nn_text)

    mlflow.log_artifact(nn_path)

    # Evaluation 2: similarity distribution
    all_sims = cosine_similarity(tfidf_matrix, tfidf_matrix)
    upper = all_sims[np.triu_indices_from(all_sims, k=1)]

    mean_sim = float(upper.mean())
    median_sim = float(np.median(upper))
    p95_sim = float(np.percentile(upper, 95))

    mlflow.log_metric("mean_similarity", mean_sim)
    mlflow.log_metric("median_similarity", median_sim)
    mlflow.log_metric("p95_similarity", p95_sim)

    # Evaluation 3: genre overlap proxy
    #genre_scores = []
    #for idx in sample_indices:
    #    base_genres = df["listed_in"][idx]
    #    for title, _ in find_similar(idx):
    #        sim_idx = titles.index(title)
    #        sim_genres = df["listed_in"][sim_idx]
    #        genre_scores.append(jaccard(base_genres, sim_genres))

    #if genre_scores:
    #    avg_overlap = float(np.mean(genre_scores))
    #    mlflow.log_metric("avg_genre_overlap", avg_overlap)

print("Evaluation logged to MLflow")