import mlflow
import mlflow.sklearn
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

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

with mlflow.start_run(run_name="tfidf-cosine-v1"):

    # Vectorization
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=20_000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(pdf["text_features"])

    # Save artifacts
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    matrix_path = MODEL_DIR / "tfidf_matrix.pkl"

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(matrix_path, "wb") as f:
        pickle.dump(tfidf_matrix, f)

    # Log to MLflow
    mlflow.log_param("model_type", "TFIDF + Cosine Similarity")
    mlflow.log_param("max_features", 20_000)
    mlflow.log_param("ngram_range", "(1,2)")

    mlflow.log_artifact(vectorizer_path)
    mlflow.log_artifact(matrix_path)

    mlflow.log_metric("num_movies", pdf.shape[0])

print("Model training complete")
