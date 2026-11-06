"""Evaluation helpers on hand-made labels and probabilities with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import comparison_table, compute_metrics, top_confident_errors

Y = np.array([0, 0, 0, 1, 1, 1])
P = np.array([0.1, 0.4, 0.8, 0.9, 0.6, 0.3])  # 1 FP (0.8), 1 FN (0.3)


def test_compute_metrics_known_values():
    m = compute_metrics(Y, P)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(6 / 9)  # 3+2+1 of the 9 (pos, neg) pairs ranked correctly


def test_perfect_model():
    assert compute_metrics([0, 1], [0.2, 0.9]) == {"accuracy": 1, "precision": 1, "recall": 1, "f1": 1, "roc_auc": 1}


def test_top_confident_errors_ranking():
    df = pd.DataFrame({"path": [f"{i}.jpg" for i in range(6)], "label": np.where(Y, "dog", "cat")})
    err = top_confident_errors(df, P, k=5)
    assert list(err["path"]) == ["2.jpg", "5.jpg"]  # cat at 0.8 (80% conf), then dog at 0.3 (70%)
    assert err["confidence"].tolist() == pytest.approx([0.8, 0.7])


def test_comparison_table():
    t = comparison_table({"a": compute_metrics(Y, P)},
                         {"a": {"params": 10, "img_size": 128, "epochs_run": 3, "train_seconds": 120}})
    assert t.loc["a", "train_min"] == 2.0 and t.loc["a", "input"] == "128×128"
