
import polars as pl

df = pl.read_parquet("models/artifacts/tfidf_rows.parquet")

print(df.schema)
