import great_expectations as gx
from great_expectations.expectations import (
    ExpectColumnToExist,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToNotBeNull,
)


def validate_dataset(
    cleaned_data,
    dataset_name,
    required_columns,
    type_expectations,
    critical_columns,
):
    """
    Generic Great Expectations validation for cleaned datasets.
    Raises an exception if validation fails.
    """

    # Create Data Context
    context = gx.get_context()

    # Create Data Source, Data Asset, Batch Definition, and Batch
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(
        name=f"{dataset_name} cleaned data"
    )
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "batch definition"
    )
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": cleaned_data}
    )

    # Create Expectation Suite
    suite = gx.ExpectationSuite(
        name=f"{dataset_name} expectations"
    )

    # Column Existence Validation
    for col in required_columns:
        suite.add_expectation(
            ExpectColumnToExist(column=col)
        )

    # Data Type Validation
    for col, dtype in type_expectations.items():
        suite.add_expectation(
            ExpectColumnValuesToBeOfType(
                column=col,
                type_=dtype
            )
        )

    # Null Value Validation (Critical Columns)
    for col in critical_columns:
        suite.add_expectation(
            ExpectColumnValuesToNotBeNull(column=col)
        )

    # Add the Expectation Suite to the Data Context
    suite = context.suites.add(suite)

    # Execute Validation
    results = batch.validate(suite)

    print(
        f"{dataset_name} validation success:",
        results.success
    )

    # Stop Pipeline on Failure
    if not results.success:
        print(results)
        raise ValueError(
            f"{dataset_name} failed validation."
        )
