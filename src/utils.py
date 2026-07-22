"""Shared constants and small helpers used across the pipeline."""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

RANDOM_STATE = 42

# 0.5 is the standard default. A real deployment should likely raise this
# (e.g. to ~0.6) since a false-positive approval is more costly than a
# false-negative rejection -- see 04_Model_Evaluation.ipynb's threshold
# sweep and business trade-off discussion.
DECISION_THRESHOLD = 0.5

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "loans_modified.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"

TARGET = "loan_status"
ID_COLUMN = "loan_id"

NUMERIC_FEATURES = [
    "applicant_income",
    "coapplicant_income",
    "loan_amount",
    "loan_amount_term",
    "credit_history",
    "dependents",
    "total_income",
    "loan_income_ratio",
]

CATEGORICAL_FEATURES = [
    "gender",
    "married",
    "education",
    "self_employed",
    "property_area",
]

def ensure_directories() -> None:
    """Creates the output directories this pipeline writes to, if missing.

    Call this explicitly at the start of any script/notebook that saves
    figures, reports, models, or processed data -- importing this module
    must not have side effects.
    """
    for directory in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

# Three representative applicants used by both the SHAP local-explanation
# section in 04_Model_Evaluation.ipynb and the sanity checks in
# 05_Streamlit_Demo.ipynb, so both notebooks reason about the same cases.
EXAMPLE_APPLICANTS = {
    "strong_applicant": dict(
        gender="Male", married="Yes", dependents="0", education="Graduate",
        self_employed="No", applicant_income=6000, coapplicant_income=2000,
        loan_amount=120, loan_amount_term=360, credit_history=1.0, property_area="Semiurban",
    ),
    "weak_applicant": dict(
        gender="Female", married="No", dependents="3+", education="Not Graduate",
        self_employed="Yes", applicant_income=1800, coapplicant_income=0,
        loan_amount=280, loan_amount_term=360, credit_history=0.0, property_area="Rural",
    ),
    "borderline_applicant": dict(
        gender="Male", married="Yes", dependents="1", education="Graduate",
        self_employed="No", applicant_income=3200, coapplicant_income=1500,
        loan_amount=110, loan_amount_term=360, credit_history=1.0, property_area="Urban",
    ),
}


@contextmanager
def timer() -> Iterator[dict]:
    """Measures wall-clock time of the wrapped block, in seconds.

    Usage:
        with timer() as t:
            do_work()
        elapsed = t["seconds"]
    """
    state = {"seconds": None}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - start
