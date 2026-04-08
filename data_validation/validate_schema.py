import great_expectations as gx
import polars as pl

context = gx.get_context()


def validate_dataset(df_pd, suite_name, required_columns, critical_columns):
    # Create datasource
    try:
        data_source = context.data_sources.get("pandas")
    except Exception:
        data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name=suite_name)

    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df_pd}
    )

    # Create suite
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    # Column existence
    for col in required_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnToExist(column=col)
        )

    # Null checks (only critical)
    for col in critical_columns:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column=col)
        )

    # Validate
    results = batch.validate(suite)
    print(f"{suite_name} validation success:", results.success)

    return results.success

netflix_df = pl.read_parquet("data/transformed/t_netflix.parquet").to_pandas()

validate_dataset(
    netflix_df,
    "netflix_suite",
    required_columns=[
        "show_id", "title", "release_year", "date_added"
    ],
    critical_columns=[
        "show_id", "title"
    ]
)

reviews_df = pl.read_parquet(
    "data/transformed/t_rotten_tomatoes_review.parquet"
).to_pandas()

validate_dataset(
    reviews_df,
    "rotten_reviews_suite",
    required_columns=[
        "rotten_tomatoes_link", "critic_name"
    ],
    critical_columns=[
       "rotten_tomatoes_link"
    ]
)

summary_df = pl.read_parquet(
    "data/transformed/t_rotten_tomatoes_summary.parquet"
).to_pandas()

validate_dataset(
    summary_df,
    "rotten_summary_suite",
    required_columns=[
        "rotten_tomatoes_link", "movie_title", "runtime", "actors"
    ],
    critical_columns=[
        "rotten_tomatoes_link"
    ]
)