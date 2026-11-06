"""Test-set evaluation: metrics, confusion matrices, ROC curves, comparison table, error analysis.

Only used in Phase 3. Everything reads the saved test split in a fixed order (no shuffling), so
results are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data import CLASS_NAMES
from src.viz import save_figure


def predict_split(model, ds) -> np.ndarray:
    """P(dog) for every image of an *unshuffled* dataset, in dataset order."""
    return model.predict(ds, verbose=0).ravel()


def compute_metrics(y_true, p_dog, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1 (dog = positive class) and ROC-AUC."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p_dog) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, p_dog),
    }


def plot_confusion_matrix(y_true, p_dog, title: str, save_to=None):
    """Counts with row-normalized percentages."""
    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(p_dog) >= 0.5).astype(int))
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    if save_to:
        save_figure(fig, save_to)
    return fig


def plot_roc_curves(y_true, probs: dict[str, np.ndarray], save_to=None):
    """One ROC curve per model on shared axes."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test set)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_to:
        save_figure(fig, save_to)
    return fig


def comparison_table(results: dict[str, dict], metas: dict[str, dict]) -> pd.DataFrame:
    """One row per model: params, test metrics, training time."""
    rows = []
    for name, m in results.items():
        meta = metas[name]
        rows.append({
            "model": name,
            "params": meta["params"],
            "trainable_params": meta.get("trainable_params", meta["params"]),
            "input": f"{meta['img_size']}×{meta['img_size']}",
            **{k: round(v, 4) for k, v in m.items()},
            "epochs": meta["epochs_run"],
            "train_min": round(meta["train_seconds"] / 60, 1),
            # GPU runs used Keras' PyTorch backend (TensorFlow has no Windows GPU support).
            "trained_on": "GPU (RTX 5060)" if meta.get("keras_backend") == "torch" else "CPU",
        })
    return pd.DataFrame(rows).set_index("model")


def top_confident_errors(df: pd.DataFrame, p_dog, k: int = 16) -> pd.DataFrame:
    """The k misclassified images the model was most confident about (highest wrong-class probability)."""
    out = df.assign(p_dog=np.asarray(p_dog))
    out["pred_label"] = np.where(out["p_dog"] >= 0.5, "dog", "cat")
    out["confidence"] = np.where(out["p_dog"] >= 0.5, out["p_dog"], 1 - out["p_dog"])
    wrong = out[out["pred_label"] != out["label"]]
    return wrong.sort_values("confidence", ascending=False).head(k)


def plot_error_grid(errors: pd.DataFrame, image_dir, title: str, cols: int = 4, save_to=None):
    """Grid of misclassified images, each titled with the index, true label, and predicted label with confidence."""
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (idx, r) in zip(np.atleast_1d(axes).ravel(), errors.iterrows()):
        ax.imshow(Image.open(Path(image_dir) / r["path"]).convert("RGB"))
        ax.set_title(f"#{idx} true {r['label']}\npred {r['pred_label']} {r['confidence']:.0%}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    if save_to:
        save_figure(fig, save_to, dpi=110)
    return fig
