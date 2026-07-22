"""Metrics, plots, and the model comparison report used in the evaluation notebook."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import FIGURES_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def evaluate_thresholds(y_true, y_proba, thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Precision, recall, F1, and the confusion matrix counts at each candidate
    decision threshold. ROC AUC is threshold-independent, so it isn't repeated here --
    that's the point being illustrated: ranking quality and decision quality are
    two different questions."""
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png")
    return fig


def plot_precision_recall(y_true, y_proba, model_name: str, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"precision_recall_{model_name}.png")
    return fig


def get_feature_names(preprocessor) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def plot_feature_importance(
    pipeline,
    model_name: str,
    top_n: int = 15,
    save: bool = True,
) -> plt.Figure | None:
    """Plots impurity/coefficient-based importance for tree models and logistic regression."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return None

    order = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order], importances[order], color="#4C72B0")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"feature_importance_{model_name}.png")
    return fig


def build_comparison_report(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Assembles the per-model metrics into the report saved to outputs/reports."""
    return pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
