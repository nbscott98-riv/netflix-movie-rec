import pandas as pd
import unicodedata

from pathlib import Path

TRANSFORMED_DIR = Path("data/transformed")
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

#Call stored data
cleaned_data_nt = pd.read_parquet("data/processed/netflix.parquet")
cleaned_data_rt_s = pd.read_parquet("data/processed/rotten_tomatoes_summary.parquet")
cleaned_data_rt_r = pd.read_parquet("data/processed/rotten_tomatoes_review.parquet")
before_rows = len(cleaned_data_nt)


# Remove rows with missing data
cleaned_data_nt = cleaned_data_nt.dropna(
    subset=['show_id','title', 'release_year', 'date_added']
)
cleaned_data_rt_s = cleaned_data_rt_s.dropna(
    subset=['rotten_tomatoes_link', 'movie_title', 'runtime', 'actors']
)
cleaned_data_rt_r = cleaned_data_rt_r.dropna(
    subset=['rotten_tomatoes_link', 'critic_name']
)

# Remove duplicate rows
cleaned_data_nt = cleaned_data_nt.drop_duplicates(subset=['show_id'])
cleaned_data_rt_s = cleaned_data_rt_s.drop_duplicates(subset=['rotten_tomatoes_link'])
cleaned_data_rt_r = cleaned_data_rt_r.drop_duplicates(subset=['rotten_tomatoes_link'])

# Remove any inconsistent data types
for col in ['release_year']:
    cleaned_data_nt[col] = pd.to_numeric(cleaned_data_nt[col], errors='coerce')

# Remove any observations that have invalid datetime values
cleaned_data_nt['date_added'] = pd.to_datetime(cleaned_data_nt['date_added'], errors='coerce')
cleaned_data_nt = cleaned_data_nt.dropna(subset=['date_added'])

# Remove any observations where the views value is less than 3 standard deviations
# from the mean
#mean_views = cleaned_data['views'].mean()
#std_views = cleaned_data['views'].std()
#cleaned_data = cleaned_data[cleaned_data['views'] >= (mean_views - 3 * std_views)]

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

cleaned_data_nt.head()
after_rows = len(cleaned_data_nt)


# Save transformed datasets
cleaned_data_nt.to_parquet(
    TRANSFORMED_DIR / "t_netflix.parquet",
    index=False
)

cleaned_data_rt_s.to_parquet(
    TRANSFORMED_DIR / "t_rotten_tomatoes_summary.parquet",
    index=False
)

cleaned_data_rt_r.to_parquet(
    TRANSFORMED_DIR / "t_rotten_tomatoes_review.parquet",
    index=False
)

print(cleaned_data_nt.head())

print(f"Rows before cleaning: {before_rows}")
print(f"Rows after cleaning:  {after_rows}")
print(f"Rows removed:         {before_rows - after_rows}")