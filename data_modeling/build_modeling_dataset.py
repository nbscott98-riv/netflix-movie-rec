
import polars as pl
from pathlib import Path
import re


# Paths
TRANSFORMED_DIR = Path("data/transformed")
MODELING_DIR = Path("data/modeling")

MODELING_DIR.mkdir(parents=True, exist_ok=True)

NETFLIX_PATH = TRANSFORMED_DIR / "t_netflix.parquet"
RT_SUMMARY_PATH = TRANSFORMED_DIR / "t_rotten_tomatoes_summary.parquet"

OUTPUT_PATH = MODELING_DIR / "movie_features.parquet"

# Helper functions
def normalize_title(title: str) -> str:
    """
    Normalize movie titles for joining:
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    if title is None:
        return None

    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title

# Load transformed datasets
print("Loading transformed datasets...")

netflix = pl.read_parquet(NETFLIX_PATH)
rt_summary = pl.read_parquet(RT_SUMMARY_PATH)

# Normalize join keys
print(" Normalizing titles for joins...")

netflix = netflix.with_columns(
    pl.col("title")
    .map_elements(normalize_title)
    .alias("normalized_title")
)

rt_summary = rt_summary.with_columns(
    pl.col("movie_title")
    .map_elements(normalize_title)
    .alias("normalized_title")
)

# Select modeling columns
print("Selecting modeling-relevant columns...")

netflix_model = netflix.select(
    [
        pl.col("show_id").alias("movie_id"),
        "title",
        "normalized_title",
        "description",
        "cast",
        "listed_in",      # genres
        "release_year",
        "type",           # Movie / TV Show
    ]
)

rt_summary_model = rt_summary.select(
    [
        "normalized_title",
        "runtime",
        "actors",
    ]
)

# Join datasets
print("Joining Netflix and Rotten Tomatoes data...")

movies = netflix_model.join(
    rt_summary_model,
    on="normalized_title",
    how="left",
)


# Final light cleanup
print("Final cleanup...")

movies = movies.with_columns(
    [
        pl.col("description").fill_null(""),
        pl.col("cast").fill_null(""),
        pl.col("actors").fill_null(""),
        pl.col("listed_in").fill_null(""),
    ]
)

# Save unified modeling dataset
print(f"Writing unified modeling dataset to {OUTPUT_PATH}")

movies.write_parquet(OUTPUT_PATH)

print(f"Done. Modeling dataset contains {movies.height} movies.")
print(movies.head(10))
