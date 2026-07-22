"""Single-record inference used by the Streamlit app and the demo notebook."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing import engineer_features, parse_dependents
from src.utils import DECISION_THRESHOLD, MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"


def build_input_frame(
    gender: str,
    married: str,
    dependents: str,
    education: str,
    self_employed: str,
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_amount_term: float,
    credit_history: float,
    property_area: str,
) -> pd.DataFrame:
    """Turns raw form inputs into a one-row dataframe matching the training schema."""
    dependents_numeric = parse_dependents(dependents)

    raw = pd.DataFrame(
        [
            {
                "gender": gender,
                "married": married,
                "dependents": dependents_numeric,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount,
                "loan_amount_term": loan_amount_term,
                "credit_history": credit_history,
                "property_area": property_area,
            }
        ]
    )
    return engineer_features(raw)


def predict_loan_status(model: Any, input_frame: pd.DataFrame) -> dict[str, Any]:
    """Returns the predicted label alongside the model's confidence."""
    probability_approved = float(model.predict_proba(input_frame)[0, 1])
    approved = probability_approved >= DECISION_THRESHOLD

    return {
        "approved": approved,
        "label": "Approved" if approved else "Rejected",
        "probability_approved": probability_approved,
        "confidence": probability_approved if approved else 1 - probability_approved,
    }
