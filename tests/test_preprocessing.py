import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor, clean_data, engineer_features
from src.utils import CATEGORICAL_FEATURES, ID_COLUMN, NUMERIC_FEATURES, TARGET


def test_clean_data_drops_exact_duplicates(raw_loan_dataframe):
    assert raw_loan_dataframe.duplicated().any()
    cleaned = clean_data(raw_loan_dataframe)
    assert not cleaned.duplicated().any()


def test_clean_data_drops_rows_with_missing_target(raw_loan_dataframe):
    assert raw_loan_dataframe[TARGET].isna().any()
    cleaned = clean_data(raw_loan_dataframe)
    assert cleaned[TARGET].isna().sum() == 0


def test_clean_data_drops_id_column(raw_loan_dataframe):
    cleaned = clean_data(raw_loan_dataframe)
    assert ID_COLUMN not in cleaned.columns


def test_clean_data_converts_3plus_dependents_to_numeric_three(raw_loan_dataframe):
    cleaned = clean_data(raw_loan_dataframe)
    assert pd.api.types.is_numeric_dtype(cleaned["dependents"])
    assert (cleaned["dependents"] == 3).sum() == 1


def test_clean_data_target_is_integer(raw_loan_dataframe):
    cleaned = clean_data(raw_loan_dataframe)
    assert pd.api.types.is_integer_dtype(cleaned[TARGET])


def test_clean_data_preserves_missing_numeric_values(raw_loan_dataframe):
    # The row with a missing loan_amount/credit_history has a valid target and
    # should survive cleaning with those values still missing, not imputed here.
    cleaned = clean_data(raw_loan_dataframe)
    assert cleaned["loan_amount"].isna().any()
    assert cleaned["credit_history"].isna().any()


def test_engineer_features_total_income_is_sum_of_incomes(synthetic_dataset):
    base = synthetic_dataset.drop(columns=["total_income", "loan_income_ratio"])
    engineered = engineer_features(base)
    expected = base["applicant_income"] + base["coapplicant_income"]
    assert np.allclose(engineered["total_income"], expected)


def test_engineer_features_loan_income_ratio_matches_definition(synthetic_dataset):
    base = synthetic_dataset.drop(columns=["total_income", "loan_income_ratio"])
    engineered = engineer_features(base)
    expected = base["loan_amount"] / (base["applicant_income"] + base["coapplicant_income"])
    assert np.allclose(engineered["loan_income_ratio"], expected)


def test_engineer_features_handles_zero_income_without_error(synthetic_dataset):
    base = synthetic_dataset.drop(columns=["total_income", "loan_income_ratio"]).copy()
    base.loc[0, "applicant_income"] = 0
    base.loc[0, "coapplicant_income"] = 0

    engineered = engineer_features(base)

    assert np.isfinite(engineered.loc[0, "loan_income_ratio"])


def test_build_preprocessor_output_has_no_missing_values(synthetic_dataset):
    X = synthetic_dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X.loc[0, "credit_history"] = np.nan
    X.loc[1, "gender"] = np.nan

    transformed = build_preprocessor().fit_transform(X)

    assert not np.isnan(transformed).any()


def test_build_preprocessor_splits_numeric_and_categorical_outputs(synthetic_dataset):
    X = synthetic_dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    feature_names = preprocessor.get_feature_names_out()

    assert any(name.startswith("numeric__") for name in feature_names)
    assert any(name.startswith("categorical__") for name in feature_names)
    assert len(feature_names) > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_build_preprocessor_ignores_unseen_categories_at_transform_time(synthetic_dataset):
    X = synthetic_dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["property_area"] = "Overseas"

    # Should not raise, thanks to OneHotEncoder(handle_unknown="ignore").
    preprocessor.transform(unseen)
