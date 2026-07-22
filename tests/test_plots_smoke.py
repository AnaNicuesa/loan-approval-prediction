"""Smoke tests for the plotting functions: do they run and return a Figure
without crashing. All calls use save=False (or take no save option) so the
real project figures under outputs/figures/ are never touched by the test
suite."""

import matplotlib.pyplot as plt

from src.evaluate import plot_confusion_matrix, plot_feature_importance, plot_precision_recall
from src.explain import plot_global_importance, plot_summary_beeswarm, plot_waterfall


def test_plot_confusion_matrix_returns_figure(fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    y_pred = pipeline.predict(X)

    fig = plot_confusion_matrix(y, y_pred, "unit_test_model", save=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_precision_recall_returns_figure(fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    y_proba = pipeline.predict_proba(X)[:, 1]

    fig = plot_precision_recall(y, y_proba, "unit_test_model", save=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_returns_figure_for_tree_model(fitted_pipeline):
    pipeline, _, _ = fitted_pipeline

    fig = plot_feature_importance(pipeline, "unit_test_model", save=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_waterfall_returns_figure(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline

    fig = plot_waterfall(pipeline, X.iloc[[0]])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_global_importance_returns_figure(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline

    fig = plot_global_importance(pipeline, X, save=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_summary_beeswarm_returns_figure(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline

    fig = plot_summary_beeswarm(pipeline, X, save=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
