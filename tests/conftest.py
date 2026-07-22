"""Shared fixtures for the test suite.

Tests never touch the real dataset or the committed model artifacts (except
where a test explicitly wants to sanity-check the committed `best_model.joblib`
and skips itself if that file isn't present). Everything else runs against a
small, fast, synthetic dataset built here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.train import build_pipeline
from src.utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


@pytest.fixture
def raw_loan_dataframe() -> pd.DataFrame:
    """A hand-built stand-in for loans_modified.csv: an exact duplicate row, a
    missing target, a missing loan_id, a '3+' dependents value, and scattered
    missing values -- the same issues `01_Exploratory_Data_Analysis.ipynb` found
    in the real file, compressed into four rows."""
    rows = [
        dict(
            loan_id="LP001", gender="Male", married="Yes", dependents="0", education="Graduate",
            self_employed="No", applicant_income=5000, coapplicant_income=0, loan_amount=150,
            loan_amount_term=360, credit_history=1.0, property_area="Urban", loan_status=1.0,
        ),
        dict(
            loan_id="LP002", gender="Female", married="No", dependents="3+", education="Not Graduate",
            self_employed="Yes", applicant_income=2000, coapplicant_income=500, loan_amount=100,
            loan_amount_term=360, credit_history=0.0, property_area="Rural", loan_status=0.0,
        ),
        dict(
            loan_id="LP003", gender="Male", married="Yes", dependents="1", education="Graduate",
            self_employed=None, applicant_income=3000, coapplicant_income=1000, loan_amount=120,
            loan_amount_term=360, credit_history=1.0, property_area="Semiurban", loan_status=None,
        ),
        dict(
            loan_id=None, gender="Male", married="No", dependents="2", education="Graduate",
            self_employed="No", applicant_income=4000, coapplicant_income=0, loan_amount=None,
            loan_amount_term=360, credit_history=None, property_area="Urban", loan_status=1.0,
        ),
    ]
    df = pd.DataFrame(rows)
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def synthetic_dataset() -> pd.DataFrame:
    """A larger synthetic dataset with the processed-data schema. `credit_history`
    is deliberately made predictive of the target -- mirroring the real dataset --
    so models fit on it in tests aren't degenerate."""
    rng = np.random.RandomState(42)
    n = 120

    credit_history = rng.binomial(1, 0.7, size=n).astype(float)
    flip = rng.rand(n) < 0.15
    loan_status = np.where(flip, 1 - credit_history, credit_history).astype(int)

    df = pd.DataFrame(
        {
            "applicant_income": rng.normal(5000, 2000, size=n).clip(min=500),
            "coapplicant_income": rng.normal(1500, 1000, size=n).clip(min=0),
            "loan_amount": rng.normal(150, 50, size=n).clip(min=10),
            "loan_amount_term": rng.choice([360, 180, 120], size=n).astype(float),
            "credit_history": credit_history,
            "dependents": rng.choice([0, 1, 2, 3], size=n).astype(float),
            "gender": rng.choice(["Male", "Female"], size=n),
            "married": rng.choice(["Yes", "No"], size=n),
            "education": rng.choice(["Graduate", "Not Graduate"], size=n),
            "self_employed": rng.choice(["Yes", "No"], size=n),
            "property_area": rng.choice(["Urban", "Semiurban", "Rural"], size=n),
        }
    )
    df["total_income"] = df["applicant_income"] + df["coapplicant_income"]
    df["loan_income_ratio"] = df["loan_amount"] / df["total_income"]
    df[TARGET] = loan_status
    return df


@pytest.fixture
def fitted_pipeline(synthetic_dataset):
    """A small, fast Random Forest pipeline fit on the synthetic dataset -- enough
    real signal for SHAP and metrics to behave meaningfully, small enough to fit
    in a fraction of a second."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = synthetic_dataset[feature_columns]
    y = synthetic_dataset[TARGET]

    pipeline = build_pipeline(RandomForestClassifier(n_estimators=25, max_depth=4, random_state=42))
    pipeline.fit(X, y)
    return pipeline, X, y
