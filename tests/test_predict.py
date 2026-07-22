import joblib
import pytest

from src.predict import DEFAULT_MODEL_PATH, build_input_frame, predict_loan_status
from src.utils import EXAMPLE_APPLICANTS


def test_build_input_frame_converts_3plus_dependents_to_three():
    frame = build_input_frame(
        gender="Male", married="Yes", dependents="3+", education="Graduate",
        self_employed="No", applicant_income=5000, coapplicant_income=0,
        loan_amount=150, loan_amount_term=360, credit_history=1.0, property_area="Urban",
    )
    assert frame.loc[0, "dependents"] == 3


def test_build_input_frame_computes_total_income():
    frame = build_input_frame(
        gender="Male", married="Yes", dependents="0", education="Graduate",
        self_employed="No", applicant_income=4000, coapplicant_income=1000,
        loan_amount=150, loan_amount_term=360, credit_history=1.0, property_area="Urban",
    )
    assert frame.loc[0, "total_income"] == 5000


def test_build_input_frame_returns_single_row():
    frame = build_input_frame(
        gender="Male", married="Yes", dependents="1", education="Graduate",
        self_employed="No", applicant_income=4000, coapplicant_income=1000,
        loan_amount=150, loan_amount_term=360, credit_history=1.0, property_area="Urban",
    )
    assert len(frame) == 1


def test_predict_loan_status_label_matches_probability_threshold(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    result = predict_loan_status(pipeline, X.iloc[[0]])

    proba = result["probability_approved"]
    assert result["approved"] == (proba >= 0.5)
    assert result["label"] == ("Approved" if proba >= 0.5 else "Rejected")


def test_predict_loan_status_confidence_is_at_least_fifty_percent(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    result = predict_loan_status(pipeline, X.iloc[[0]])

    expected_confidence = (
        result["probability_approved"] if result["approved"] else 1 - result["probability_approved"]
    )
    assert result["confidence"] == pytest.approx(expected_confidence)
    assert 0.5 <= result["confidence"] <= 1.0


@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="models/best_model.joblib not built yet -- run the notebooks first",
)
def test_committed_best_model_produces_valid_probabilities_for_example_applicants():
    model = joblib.load(DEFAULT_MODEL_PATH)
    for name, applicant in EXAMPLE_APPLICANTS.items():
        frame = build_input_frame(**applicant)
        result = predict_loan_status(model, frame)
        assert 0.0 <= result["probability_approved"] <= 1.0, name
