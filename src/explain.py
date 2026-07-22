"""SHAP-based explanations for the tree-based models in this project.

Contributions are computed in the transformed (scaled, one-hot encoded)
feature space the classifier actually sees, then collapsed back onto the
original columns so a waterfall plot reads "Rural = property_area" instead
of "1 = categorical__property_area_Rural".
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import CATEGORICAL_FEATURES, FIGURES_DIR

POSITIVE_CLASS_INDEX = 1


def _original_column(transformed_name: str) -> str:
    """Maps a ColumnTransformer output name (e.g. 'categorical__property_area_Rural')
    back to its source column ('property_area'); numeric columns pass through
    unchanged since they were never one-hot encoded."""
    _, column = transformed_name.split("__", 1)
    categorical_match = next(
        (cat for cat in CATEGORICAL_FEATURES if column.startswith(f"{cat}_")),
        None,
    )
    return categorical_match or column


def _transform(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(preprocessor.transform(X), columns=feature_names, index=X.index)


def _collapse_to_original_features(
    shap_row: np.ndarray,
    transformed_feature_names: list[str],
    raw_row: pd.Series,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sums one-hot dummy contributions back onto their source column and pairs
    each with the applicant's actual (pre-encoding) value for display."""
    contributions: dict[str, float] = {}
    display_values: dict[str, object] = {}

    for shap_value, transformed_name in zip(shap_row, transformed_feature_names):
        source_column = _original_column(transformed_name)
        contributions[source_column] = contributions.get(source_column, 0.0) + shap_value
        display_values[source_column] = raw_row[source_column]

    columns = list(contributions.keys())
    values = np.array([contributions[c] for c in columns])
    data = np.array([display_values[c] for c in columns], dtype=object)
    return columns, values, data


def _aggregate_shap_matrix(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Runs TreeExplainer once over X and returns a (n_samples, n_original_features)
    matrix of collapsed SHAP values, shared by every plot and metric in this module."""
    transformed = _transform(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    raw_explanation = explainer(transformed)
    shap_values = raw_explanation.values[:, :, POSITIVE_CLASS_INDEX]

    columns = None
    rows = []
    for row_idx in range(shap_values.shape[0]):
        columns, values, _ = _collapse_to_original_features(
            shap_values[row_idx], list(transformed.columns), X.iloc[row_idx]
        )
        rows.append(values)

    return np.array(rows), columns


def explain_instance(pipeline, input_frame: pd.DataFrame) -> shap.Explanation:
    """Builds a single-instance SHAP explanation for the approved-class probability."""
    transformed = _transform(pipeline, input_frame)
    explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
    raw_explanation = explainer(transformed)

    shap_row = raw_explanation.values[0, :, POSITIVE_CLASS_INDEX]
    base_value = raw_explanation.base_values[0, POSITIVE_CLASS_INDEX]

    columns, values, display_values = _collapse_to_original_features(
        shap_row, list(transformed.columns), input_frame.iloc[0]
    )

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=display_values,
        feature_names=columns,
    )


def get_local_factors(
    pipeline, input_frame: pd.DataFrame, top_n: int = 5
) -> tuple[list[tuple[str, object, float]], list[tuple[str, object, float]]]:
    """Splits one applicant's SHAP contributions into what pushed the prediction up
    and what pulled it down, each as (feature, raw_value, shap_value) sorted by size."""
    explanation = explain_instance(pipeline, input_frame)
    triples = list(zip(explanation.feature_names, explanation.data, explanation.values))

    increasing = sorted((t for t in triples if t[2] > 0), key=lambda t: t[2], reverse=True)[:top_n]
    decreasing = sorted((t for t in triples if t[2] < 0), key=lambda t: t[2])[:top_n]
    return increasing, decreasing


def plot_waterfall(
    pipeline,
    input_frame: pd.DataFrame,
    max_display: int = 10,
    title: str = "SHAP Explanation for This Applicant",
) -> plt.Figure:
    explanation = explain_instance(pipeline, input_frame)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def plot_global_importance(pipeline, X: pd.DataFrame, max_display: int = 15, save: bool = True) -> plt.Figure:
    """Mean absolute SHAP value per original feature — how much each feature matters,
    regardless of direction."""
    matrix, columns = _aggregate_shap_matrix(pipeline, X)
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[-max_display:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(columns)[order], mean_abs[order], color="#008BFB")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature impact (SHAP)")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_global_importance.png")
    return fig


def plot_summary_beeswarm(pipeline, X: pd.DataFrame, max_display: int = 15, save: bool = True) -> plt.Figure:
    """SHAP's standard beeswarm plot: one dot per applicant per feature, colored by
    that applicant's feature value, so both magnitude and direction of effect are visible
    at once. Categorical columns are recoded to integers purely for the color scale."""
    matrix, columns = _aggregate_shap_matrix(pipeline, X)

    color_values = X[columns].copy()
    for column in CATEGORICAL_FEATURES:
        if column in color_values.columns:
            color_values[column] = pd.factorize(color_values[column])[0]

    shap.summary_plot(
        matrix,
        color_values,
        feature_names=columns,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.gca().set_title("SHAP Summary — Feature Impact and Direction", pad=15)
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "shap_summary_beeswarm.png")
    return fig


def mean_shap_by_value(pipeline, X: pd.DataFrame, feature: str) -> pd.Series:
    """Average SHAP contribution of one feature, grouped by its raw value.

    A single sample-wide average SHAP value is misleading for a feature like
    `credit_history`, where the effect is large but points in opposite
    directions depending on the value — this grouped view is unambiguous
    regardless of how skewed or bimodal the feature is.
    """
    matrix, columns = _aggregate_shap_matrix(pipeline, X)
    column_index = columns.index(feature)
    return (
        pd.Series(matrix[:, column_index], index=X[feature].values)
        .groupby(level=0)
        .mean()
        .sort_values(ascending=False)
    )


def compare_with_impurity_importance(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Ranks features by mean |SHAP| alongside the classifier's built-in impurity-based
    importance, to make the two methods directly comparable side by side."""
    matrix, columns = _aggregate_shap_matrix(pipeline, X)
    mean_abs_shap = np.abs(matrix).mean(axis=0)

    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = list(preprocessor.get_feature_names_out())
    impurity_by_transformed = dict(zip(transformed_names, classifier.feature_importances_))

    impurity_by_original: dict[str, float] = {}
    for transformed_name, importance in impurity_by_transformed.items():
        source_column = _original_column(transformed_name)
        impurity_by_original[source_column] = impurity_by_original.get(source_column, 0.0) + importance

    comparison = pd.DataFrame(
        {
            "mean_abs_shap": mean_abs_shap,
            "impurity_importance": [impurity_by_original[c] for c in columns],
        },
        index=columns,
    )
    comparison["shap_rank"] = comparison["mean_abs_shap"].rank(ascending=False).astype(int)
    comparison["impurity_rank"] = comparison["impurity_importance"].rank(ascending=False).astype(int)
    return comparison.sort_values("mean_abs_shap", ascending=False)
