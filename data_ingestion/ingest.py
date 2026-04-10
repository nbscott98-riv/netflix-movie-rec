import polars as pl
import os
import numpy as np
import pandas as pd
import unicodedata
import warnings
import kagglehub
from kagglehub import KaggleDatasetAdapter

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_from_kaggle(dataset: str, file_path: str) -> pl.DataFrame:
    """
    Generic Kaggle dataset loader using Polars.
    """
    lf = kagglehub.load_dataset(
        KaggleDatasetAdapter.POLARS,
        dataset,
        file_path
    )
    return lf.collect()


def save_data(df: pl.DataFrame, name: str) -> None:
    """
    Save raw CSV and processed Parquet versions.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df.write_csv(f"{RAW_DIR}/{name}.csv")
    df.write_parquet(f"{PROCESSED_DIR}/{name}.parquet")


if __name__ == "__main__":
    # ----------------------------
    # Netflix dataset
    # ----------------------------
    print("Loading Netflix data...")
    netflix_df = load_from_kaggle(
        dataset="shivamb/netflix-shows",
        file_path="netflix_titles.csv"
    )

    print("Saving Netflix data...")
    save_data(netflix_df, "netflix")

    print(netflix_df.head())

    # ----------------------------
    # Rotten Tomatoes summary dataset
    # ----------------------------
    print("\nLoading Rotten Tomatoes summary data...")
    rotten_s_df = load_from_kaggle(
        dataset="stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset",
        file_path="rotten_tomatoes_movies.csv"
    )

    print("Saving Rotten Tomatoes summary data...")
    save_data(rotten_s_df, "rotten_tomatoes_summary")

    print(rotten_s_df.head())