# pipelines/validate_data.py

import great_expectations as gx
import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame) -> Dict:
    """
    Validate the online shoppers dataset using Great Expectations.
    Returns dict (orient='split') for XCom/Redis.
    """
    if df is None or df.empty:
        raise ValueError("❌ Empty DataFrame received for validation")

    log.info("Starting validation with Great Expectations.")

    context = gx.get_context(context_root_dir=None)

    context.add_datasource(
        name="default_pandas_datasource",
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"]
            }
        }
    )

    suite_name = "online_shoppers_dynamic_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    batch_request = gx.core.batch.RuntimeBatchRequest(
        datasource_name="default_pandas_datasource",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="online_shoppers_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_identifier"}
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # Example expectations
    numeric_columns = [
        "Administrative","Administrative_Duration","Informational","Informational_Duration",
        "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues","SpecialDay"
    ]
    for col in numeric_columns:
        if col in df.columns:
            validator.expect_column_values_to_be_between(
                column=col, min_value=df[col].min(), max_value=df[col].max()
            )

    categorical_columns = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend","Revenue"]
    for col in categorical_columns:
        if col in df.columns:
            validator.expect_column_values_to_be_in_set(column=col, value_set=df[col].dropna().unique().tolist())

    validator.save_expectation_suite(discard_failed_expectations=False)
    validation_result = validator.validate()

    if validation_result.success:
        log.info("✅ Data validation passed!")
        return df.to_dict(orient="split")
    else:
        log.error("❌ Data validation failed.")
        raise Exception("Data validation failed.")
