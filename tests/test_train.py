from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src import train as train_module
from src.train import (
    build_credit_history_only_pipeline,
    build_majority_baseline,
    build_pipeline,
    get_model_specs,
    load_model,
    save_model,
)
from src.utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def test_get_model_specs_returns_expected_models_in_order():
    specs = get_model_specs()
    assert [spec.name for spec in specs] == ["logistic_regression", "random_forest", "xgboost"]


def test_get_model_specs_uses_grid_search_only_for_logistic_regression():
    specs = {spec.name: spec for spec in get_model_specs()}
    assert specs["logistic_regression"].search_type == "grid"
    assert specs["random_forest"].search_type == "random"
    assert specs["xgboost"].search_type == "random"


def test_build_pipeline_has_preprocessor_before_classifier():
    pipeline = build_pipeline(LogisticRegression())
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]


def test_majority_baseline_always_predicts_the_same_class(synthetic_dataset):
    X = synthetic_dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = synthetic_dataset[TARGET]

    baseline = build_majority_baseline()
    baseline.fit(X, y)
    predictions = baseline.predict(X)

    assert len(set(predictions)) == 1
    assert predictions[0] == y.mode().iloc[0]


def test_credit_history_only_pipeline_is_blind_to_other_features(synthetic_dataset):
    X = synthetic_dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = synthetic_dataset[TARGET]

    pipeline = build_credit_history_only_pipeline()
    pipeline.fit(X, y)

    perturbed = X.copy()
    perturbed["applicant_income"] = perturbed["applicant_income"] * 100 + 50_000
    perturbed["property_area"] = "Rural"

    original_proba = pipeline.predict_proba(X)[:, 1]
    perturbed_proba = pipeline.predict_proba(perturbed)[:, 1]

    assert np.allclose(original_proba, perturbed_proba)


def test_save_and_load_model_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path)

    model = LogisticRegression()
    path = save_model(model, "unit_test_model")

    assert Path(path).exists()
    loaded = load_model("unit_test_model")
    assert isinstance(loaded, LogisticRegression)
