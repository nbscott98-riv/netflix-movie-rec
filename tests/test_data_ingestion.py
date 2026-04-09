netflix = pl.read_parquet("data/processed/netflix.parquet")
rotten_sum = pl.read_parquet("data/processed/rotten_tomatoes_summary.parquet")
rotten_rev = pl.read_parquet("data/processed/rotten_tomatoes_review.parquet")

print(netflix.select("title").head())
print(rotten_sum.select("movie_title").head())
print(rotten_rev.select("movie_title").head())