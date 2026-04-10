import polars as pl

df = pl.read_parquet("data/transformed/t_movie_features.parquet")
print(df.sample(5))
