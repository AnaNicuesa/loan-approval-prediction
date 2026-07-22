import numpy as np

from src.evaluate import build_comparison_report, compute_metrics, evaluate_thresholds


def test_compute_metrics_on_perfect_predictions():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.8, 0.2, 0.95]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_with_no_positive_predictions_does_not_raise():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 0, 0]
    y_proba = [0.1, 0.2, 0.3, 0.05]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_thresholds_returns_one_row_per_threshold():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_proba = np.array([0.2, 0.9, 0.6, 0.4, 0.8, 0.1, 0.55, 0.7])

    result = evaluate_thresholds(y_true, y_proba, thresholds=(0.3, 0.5, 0.7))

    assert len(result) == 3
    assert list(result["threshold"]) == [0.3, 0.5, 0.7]


def test_evaluate_thresholds_confusion_counts_sum_to_total_rows():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_proba = np.array([0.2, 0.9, 0.6, 0.4, 0.8, 0.1, 0.55, 0.7])

    result = evaluate_thresholds(y_true, y_proba)
    totals = result[["true_positives", "false_positives", "false_negatives", "true_negatives"]].sum(axis=1)

    assert (totals == len(y_true)).all()


def test_evaluate_thresholds_recall_never_increases_with_higher_threshold():
    # A higher decision threshold can only approve a subset of what a lower one
    # approved, so recall on the positive class is guaranteed non-increasing --
    # true for any y_true/y_proba, not just this particular sample.
    rng = np.random.RandomState(0)
    y_true = rng.binomial(1, 0.5, size=30)
    y_proba = rng.rand(30)

    result = evaluate_thresholds(y_true, y_proba, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9))
    recalls = result["recall"].to_numpy()

    assert all(recalls[i] >= recalls[i + 1] - 1e-12 for i in range(len(recalls) - 1))


def test_build_comparison_report_sorts_by_roc_auc_descending():
    results = [
        {"model": "a", "roc_auc": 0.7},
        {"model": "b", "roc_auc": 0.9},
        {"model": "c", "roc_auc": 0.8},
    ]

    report = build_comparison_report(results)

    assert list(report["model"]) == ["b", "c", "a"]
