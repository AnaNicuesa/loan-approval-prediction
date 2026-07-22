"""Data cleaning, feature engineering, and the sklearn preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, ID_COLUMN


def load_raw_data(path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_dependents(value):
    """Converts the raw '3+' dependents category to numeric 3.

    Accepts either a scalar (e.g. a Streamlit form value) or a pandas Series
    (a full dataframe column), so the same rule can be reused everywhere the
    raw dependents value is parsed.
    """
    if isinstance(value, pd.Series):
        return pd.to_numeric(
            value.astype(str).str.replace("3+", "3", regex=False), errors="coerce"
        )
    return 3 if str(value) == "3+" else int(value)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes unusable rows and normalizes raw values before feature engineering."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    df["dependents"] = parse_dependents(df["dependents"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived columns that capture household affordability."""
    df = df.copy()
    df["total_income"] = df["applicant_income"].fillna(0) + df["coapplicant_income"].fillna(0)
    df["loan_income_ratio"] = df["loan_amount"] / df["total_income"].replace(0, 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Column-wise imputation, scaling, and encoding, wired for use inside a model Pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
