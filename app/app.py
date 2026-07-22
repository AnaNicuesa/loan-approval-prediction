"""Streamlit demo for the loan approval model.

Run from the repository root with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.explain import get_local_factors, plot_waterfall
from src.predict import DEFAULT_MODEL_PATH, build_input_frame, predict_loan_status
from src.utils import REPORTS_DIR, ensure_directories

ensure_directories()

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon=None,
    layout="centered",
)


@st.cache_resource
def get_model():
    import joblib

    return joblib.load(DEFAULT_MODEL_PATH)


@st.cache_data
def get_model_metrics() -> pd.DataFrame | None:
    report_path = REPORTS_DIR / "model_results.csv"
    if not report_path.exists():
        return None
    return pd.read_csv(report_path).sort_values("roc_auc", ascending=False)


def render_sidebar(metrics: pd.DataFrame | None) -> None:
    st.sidebar.title("About this project")
    st.sidebar.write(
        "Predicts whether a loan application would be approved, based on "
        "applicant demographics, income, and credit history. Trained on a "
        "historical loan dataset with Logistic Regression, Random Forest, "
        "and XGBoost, selected by cross-validated ROC AUC."
    )

    if metrics is not None and not metrics.empty:
        best = metrics.iloc[0]
        st.sidebar.subheader("Model in production")
        st.sidebar.markdown(f"**{best['model'].replace('_', ' ').title()}**")
        st.sidebar.metric("ROC AUC", f"{best['roc_auc']:.3f}")
        st.sidebar.metric("F1 score", f"{best['f1']:.3f}")
        st.sidebar.metric("Recall", f"{best['recall']:.3f}")

    st.sidebar.divider()
    st.sidebar.caption(
        "This tool is a portfolio demonstration and does not represent a "
        "real lending decision system."
    )


def render_form() -> dict:
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    with col2:
        applicant_income = st.number_input("Applicant income (monthly)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Coapplicant income (monthly)", min_value=0, value=0, step=100)
        loan_amount = st.number_input("Loan amount (in thousands)", min_value=0, value=150, step=10)
        loan_amount_term = st.selectbox("Loan term (days)", [360, 180, 120, 240, 60, 300, 480, 84, 36, 12], index=0)
        credit_history = st.selectbox("Credit history meets guidelines", ["Yes", "No"])

    return {
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "loan_amount": float(loan_amount),
        "loan_amount_term": float(loan_amount_term),
        "credit_history": 1.0 if credit_history == "Yes" else 0.0,
        "property_area": property_area,
    }


def render_result(result: dict) -> None:
    st.divider()

    if result["approved"]:
        st.success(f"Loan Approved — confidence {result['confidence']:.1%}")
    else:
        st.error(f"Loan Rejected — confidence {result['confidence']:.1%}")

    st.write("Predicted probability of approval:")
    st.progress(result["probability_approved"])
    st.caption(f"{result['probability_approved']:.1%} approved vs. {1 - result['probability_approved']:.1%} rejected")


def render_explanation(model, input_frame: pd.DataFrame) -> None:
    st.subheader("Why this prediction?")
    st.write(
        "Based on SHAP (SHapley Additive exPlanations), the industry-standard "
        "method for attributing a model's prediction to its input features."
    )

    increasing, decreasing = get_local_factors(model, input_frame, top_n=4)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factors that increased approval probability**")
        if increasing:
            for feature, value, shap_value in increasing:
                st.markdown(f"- {feature.replace('_', ' ')} = {value}  (+{shap_value:.3f})")
        else:
            st.caption("No feature pushed the prediction toward approval.")

    with col2:
        st.markdown("**Factors that decreased approval probability**")
        if decreasing:
            for feature, value, shap_value in decreasing:
                st.markdown(f"- {feature.replace('_', ' ')} = {value}  ({shap_value:.3f})")
        else:
            st.caption("No feature pushed the prediction toward rejection.")

    with st.expander("See the full contribution breakdown"):
        fig = plot_waterfall(model, input_frame)
        st.pyplot(fig, clear_figure=True)

    st.caption(
        "This explains how the model reached its prediction, not a causal claim "
        "about the applicant. It reflects patterns learned from historical data "
        "and should not be read as evidence that changing one factor would "
        "change a real outcome."
    )


def main() -> None:
    metrics = get_model_metrics()
    render_sidebar(metrics)

    st.title("Loan Approval Predictor")
    st.write(
        "Fill in the applicant details below and press **Predict** to see "
        "whether the model would approve this loan application."
    )

    inputs = render_form()

    if st.button("Predict", type="primary"):
        model = get_model()
        input_frame = build_input_frame(**inputs)
        result = predict_loan_status(model, input_frame)
        render_result(result)
        render_explanation(model, input_frame)


if __name__ == "__main__":
    main()
