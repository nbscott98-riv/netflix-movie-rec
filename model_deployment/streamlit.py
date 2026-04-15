# app/streamlit_app.py

import streamlit as st
import polars as pl
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# App config
st.set_page_config(
    page_title="🎬 Movie Similarity Explorer",
    layout="centered"
)

st.title("🎬 Movie Similarity Explorer")
st.write(
    "Search for a movie and discover similar titles "
    "based on content similarity."
)

TOP_K = 5

# Load artifacts (cached)
@st.cache_data
def load_metadata():
    return pl.read_parquet("models/artifacts/tfidf_rows.parquet")

@st.cache_resource
def load_model():
    with open("models/artifacts/tfidf_matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    return matrix

rows_df = load_metadata()
tfidf_matrix = load_model()

titles = rows_df["title"].to_list()
listed_in = rows_df["listed_in"].to_list()
description = rows_df["description"].to_list()
content_rating = rows_df["content_rating"].to_list()

# Safety guard
assert tfidf_matrix.shape[0] == len(titles), (
    "TF‑IDF matrix and row metadata are misaligned."
)

# Helper function
def find_similar_movies(idx, k=TOP_K):
    sims = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    top_idx = sims.argsort()[-(k + 1):][::-1][1:]
    return [
        {
            "title": titles[i],
            "similarity": float(sims[i]),
            "genre": listed_in[i],
            "description": description[i],
            "content_rating": content_rating[i],
        }
        for i in top_idx
    ]

# UI: movie selection
selected_title = st.selectbox(
    "🎥 Select a movie",
    options=sorted(titles),
)


if selected_title:
    response = requests.get(
        "http://localhost:8000/recommend",
        params={"title": selected_title}
    )

    data = response.json()

    if "results" in data:
        for rank, movie in enumerate(data["results"], start=1):
            st.markdown(
                f"""
**{rank}. {movie['title']}**  
*Genre:* `{movie['genre']}`  
*Rating* `{movie['content_rating']}`  
*Description:* `{movie['description']}`  
"""
            )


# Footer
st.markdown("---")
st.caption(
    "Model create for test use for BANA 7075 Final Project. (Group 9)"
)