import pandas as pd
import unicodedata

from pathlib import Path
from data_validation.validate_schema import validate_dataset

TRANSFORMED_DIR = Path("data/transformed")
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

#Call stored data
cleaned_data_nt = pd.read_parquet("data/processed/netflix.parquet")
cleaned_data_rt_s = pd.read_parquet("data/processed/rotten_tomatoes_summary.parquet")
before_rows_n = len(cleaned_data_nt)
before_rows_r = len(cleaned_data_rt_s)


# Remove rows with missing data
cleaned_data_nt = cleaned_data_nt.dropna(
    subset=['show_id','title', 'release_year', 'date_added']
)
cleaned_data_rt_s = cleaned_data_rt_s.dropna(
    subset=['rotten_tomatoes_link', 'movie_title', 'runtime', 'actors', 'genres']
)

# Remove duplicate rows
cleaned_data_nt = cleaned_data_nt.drop_duplicates(subset=['show_id'])
cleaned_data_rt_s = cleaned_data_rt_s.drop_duplicates(subset=['rotten_tomatoes_link'])

# Remove any inconsistent data types
for col in ['release_year']:
    cleaned_data_nt[col] = pd.to_numeric(cleaned_data_nt[col], errors='coerce')

# Remove any observations that have invalid datetime values
cleaned_data_nt['date_added'] = pd.to_datetime(cleaned_data_nt['date_added'], errors='coerce')
cleaned_data_nt = cleaned_data_nt.dropna(subset=['date_added'])

# Remove any observations where the text length is less than 3 standard deviations
# from the mean text length
cleaned_data_nt['cast_length'] = cleaned_data_nt['cast'].apply(lambda x: len(x) if pd.notnull(x) else 0)
mean_cast_length = cleaned_data_nt['cast_length'].mean()
std_cast_length = cleaned_data_nt['cast_length'].std()
cleaned_data_nt = cleaned_data_nt[cleaned_data_nt['cast_length'] >= (mean_cast_length - 3 * std_cast_length)]
cleaned_data_nt['description_length'] = cleaned_data_nt['description'].apply(lambda x: len(x) if pd.notnull(x) else 0)
mean_description_length = cleaned_data_nt['description_length'].mean()
std_description_length = cleaned_data_nt['description_length'].std()
cleaned_data_nt = cleaned_data_nt[cleaned_data_nt['description_length'] >= (mean_description_length - 3 * std_description_length)]
cleaned_data_rt_s['critics_consensus'] = cleaned_data_rt_s['critics_consensus'].apply(lambda x: len(x) if pd.notnull(x) else 0)
mean_critics_consensus_length = cleaned_data_rt_s['critics_consensus'].mean()
std_critics_consensus_length = cleaned_data_rt_s['critics_consensus'].std()
cleaned_data_rt_s = cleaned_data_rt_s[cleaned_data_rt_s['critics_consensus'] >= (mean_critics_consensus_length - 3 * std_critics_consensus_length)]
cleaned_data_rt_s['movie_info'] = cleaned_data_rt_s['movie_info'].apply(lambda x: len(x) if pd.notnull(x) else 0)
mean_movie_info_length = cleaned_data_rt_s['movie_info'].mean()
std_movie_info_length = cleaned_data_rt_s['movie_info'].std()
cleaned_data_rt_s = cleaned_data_rt_s[cleaned_data_rt_s['movie_info'] >= (mean_movie_info_length - 3 * std_movie_info_length)]

# Remove/clean the large text columns for non-character string values
# (i.e. unicode characters)
def clean_text(text):
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return text

cleaned_data_nt['title'] = cleaned_data_nt['title'].apply(clean_text)
cleaned_data_nt['director'] = cleaned_data_nt['director'].apply(clean_text)
cleaned_data_nt['cast'] = cleaned_data_nt['cast'].apply(clean_text)
cleaned_data_nt['country'] = cleaned_data_nt['country'].apply(clean_text)
cleaned_data_nt['duration'] = cleaned_data_nt['duration'].apply(clean_text)
cleaned_data_nt['listed_in'] = cleaned_data_nt['listed_in'].apply(clean_text)
cleaned_data_nt['description'] = cleaned_data_nt['description'].apply(clean_text)
cleaned_data_rt_s['movie_info'] = cleaned_data_rt_s['movie_info'].apply(clean_text)
cleaned_data_rt_s['critics_consensus'] = cleaned_data_rt_s['critics_consensus'].apply(clean_text)
cleaned_data_rt_s['actors'] = cleaned_data_rt_s['actors'].apply(clean_text)


cleaned_data_nt.head()
after_rows_n = len(cleaned_data_nt)
after_rows_r = len(cleaned_data_rt_s)


print(cleaned_data_nt.head())

print(f"Netflix")
print(f"Rows before cleaning: {before_rows_n}")
print(f"Rows after cleaning:  {after_rows_n}")
print(f"Rows removed:         {before_rows_n - after_rows_n}")
print(f"Rotten Tomatoes")
print(f"Rows before cleaning: {before_rows_r}")
print(f"Rows after cleaning:  {after_rows_r}")
print(f"Rows removed:         {before_rows_r - after_rows_r}")

# Call validation

validate_dataset(
    cleaned_data=cleaned_data_nt,
    dataset_name="Netflix",
    required_columns=[
        "show_id",
        "title",
        "release_year",
        "date_added",
    ],
    type_expectations={
        "show_id": "object",
        "title": "object",
        "release_year": "int64",
        "date_added": "datetime64[ns]",
    },
    critical_columns=[
        "show_id",
        "title",
    ],
)


validate_dataset(
    cleaned_data=cleaned_data_rt_s,
    dataset_name="Rotten Tomatoes Summary",
    required_columns=[
        "rotten_tomatoes_link",
        "movie_title",
        "runtime",
        "actors",
        "genres",
    ],
    type_expectations={
        "rotten_tomatoes_link": "object",
        "movie_title": "object",
        "runtime": "float64",
        "actors": "object",
        "genres": "object",
    },
    critical_columns=[
        "rotten_tomatoes_link",
    ],
)

# Save transformed datasets
cleaned_data_nt.to_parquet(
    TRANSFORMED_DIR / "t_netflix.parquet",
    index=False
)

cleaned_data_rt_s.to_parquet(
    TRANSFORMED_DIR / "t_rotten_tomatoes_summary.parquet",
    index=False
)