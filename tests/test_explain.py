import numpy as np

from src.explain import (
    compare_with_impurity_importance,
    explain_instance,
    get_local_factors,
    mean_shap_by_value,
)


def test_explain_instance_values_are_additive(fitted_pipeline):
    # The core SHAP guarantee: base value + sum of contributions == the model's
    # actual predicted probability for that instance.
    pipeline, X, _ = fitted_pipeline
    input_frame = X.iloc[[0]]

    explanation = explain_instance(pipeline, input_frame)
    predicted_proba = pipeline.predict_proba(input_frame)[0, 1]

    reconstructed = explanation.base_values + explanation.values.sum()
    assert np.isclose(reconstructed, predicted_proba, atol=1e-6)


def test_explain_instance_feature_names_have_no_transformer_prefixes(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    explanation = explain_instance(pipeline, X.iloc[[0]])
    assert all("__" not in name for name in explanation.feature_names)


def test_explain_instance_one_hot_categorical_collapses_to_single_contribution(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    explanation = explain_instance(pipeline, X.iloc[[0]])
    # property_area has 3 one-hot dummy columns after encoding; collapsed back
    # to original features there should be exactly one contribution for it.
    assert list(explanation.feature_names).count("property_area") == 1


def test_get_local_factors_increasing_and_decreasing_signs(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    increasing, decreasing = get_local_factors(pipeline, X.iloc[[0]], top_n=20)

    assert all(shap_value > 0 for _, _, shap_value in increasing)
    assert all(shap_value < 0 for _, _, shap_value in decreasing)


def test_get_local_factors_respects_top_n(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    increasing, decreasing = get_local_factors(pipeline, X.iloc[[0]], top_n=2)

    assert len(increasing) <= 2
    assert len(decreasing) <= 2


def test_get_local_factors_sorted_by_magnitude(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    increasing, _ = get_local_factors(pipeline, X.iloc[[0]], top_n=20)

    shap_values = [shap_value for _, _, shap_value in increasing]
    assert shap_values == sorted(shap_values, reverse=True)


def test_mean_shap_by_value_covers_every_observed_value(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    grouped = mean_shap_by_value(pipeline, X, "credit_history")
    assert set(grouped.index) == set(X["credit_history"].unique())


def test_compare_with_impurity_importance_ranks_are_within_range(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    comparison = compare_with_impurity_importance(pipeline, X)
    n_features = len(comparison)

    assert comparison["shap_rank"].between(1, n_features).all()
    assert comparison["impurity_rank"].between(1, n_features).all()


def test_compare_with_impurity_importance_flags_credit_history_as_top_feature(fitted_pipeline):
    # credit_history was constructed to dominate the synthetic target, so both
    # attribution methods should agree it's the single most important feature.
    pipeline, X, _ = fitted_pipeline
    comparison = compare_with_impurity_importance(pipeline, X)

    assert comparison["mean_abs_shap"].idxmax() == "credit_history"
