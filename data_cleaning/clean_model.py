import pandas as pd
import unicodedata

from pathlib import Path
from data_validation.validate_schema import validate_dataset

TRANSFORMED_DIR = Path("data/transformed")
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

#Call stored data
cleaned_data_mf = pd.read_parquet("data/modeling/movie_features.parquet")
before_rows = len(cleaned_data_mf)


# Remove rows with missing data
cleaned_data_mf = cleaned_data_mf.dropna(
    subset=['normalized_title','title', 'release_year', 'release_year', 'runtime', 'actors', 'description']
)

# Remove duplicate rows
cleaned_data_mf = cleaned_data_mf.drop_duplicates(subset=['normalized_title'])

cleaned_data_mf.head()
after_rows = len(cleaned_data_mf)

print(cleaned_data_mf.head())

print(f"Rows before cleaning: {before_rows}")
print(f"Rows after cleaning:  {after_rows}")
print(f"Rows removed:         {before_rows - after_rows}")

# Call validation

validate_dataset(
    cleaned_data=cleaned_data_mf,
    dataset_name="Movie Features Model",
    required_columns=[
        "title",
        "normalized_title",
        "description",
        "cast",
        "actors",
        "listed_in",
        "release_year",
        "type",
    ],
    type_expectations={
        "title": "object",
        "normalized_title": "object",
        "description": "object",
        "cast": "object",
        "actors": "object",
        "listed_in": "object",
        "release_year": "int64",
        "type": "object",
        "runtime": "float64",
        "movie_info": "object",
        "critics_consensus": "object",
        "content_rating": "object",
        "directors": "object",
    },
    critical_columns=[
        "title",
        "normalized_title",
    ],
)

# Save transformed datasets
cleaned_data_mf.to_parquet(
    TRANSFORMED_DIR / "t_movie_features.parquet",
    index=False
)