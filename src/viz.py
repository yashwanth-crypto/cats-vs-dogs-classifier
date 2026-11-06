"""Plotting helpers. Every function returns the matplotlib Figure and saves it when `save_to` is set."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

FIG_DIR = Path("reports/figures")
CLASS_COLORS = {"cat": "#E07A5F", "dog": "#3D7EA6"}


def save_figure(fig, path, dpi: int = 130) -> None:
    """fig.savefig with retries: on Windows, antivirus or the search indexer can briefly lock a
    just-written file, which makes the next overwrite fail with OSError (Errno 22)."""
    import time

    for attempt in range(5):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1.0 + attempt)


def _save(fig, save_to):
    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, save_to)
    return fig


def plot_class_balance(splits: dict[str, pd.DataFrame], save_to=None):
    """Grouped bar chart of cat/dog counts per split."""
    counts = pd.DataFrame({k: v["label"].value_counts() for k, v in splits.items()}).T[["cat", "dog"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts.plot.bar(ax=ax, color=[CLASS_COLORS["cat"], CLASS_COLORS["dog"]], rot=0, width=0.7)
    for c in ax.containers:
        ax.bar_label(c, fmt="{:,.0f}", fontsize=8)
    ax.set(title="Class balance per split", ylabel="images")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, save_to)


def plot_sample_grid(df: pd.DataFrame, image_dir, n_per_class: int = 6, seed: int = 42, save_to=None):
    """Random examples: one row of cats, one row of dogs."""
    fig, axes = plt.subplots(2, n_per_class, figsize=(2 * n_per_class, 4.4))
    for row, label in enumerate(["cat", "dog"]):
        sample = df[df["label"] == label].sample(n_per_class, random_state=seed)
        for ax, rel in zip(axes[row], sample["path"]):
            ax.imshow(Image.open(Path(image_dir) / rel).convert("RGB"))
            ax.set_title(label, fontsize=9, color=CLASS_COLORS[label])
            ax.axis("off")
    fig.suptitle("Sample training images")
    fig.tight_layout()
    return _save(fig, save_to)


def image_sizes(df: pd.DataFrame, image_dir) -> pd.DataFrame:
    """Width and height of every image (reads only the file header, so it's fast)."""
    sizes = []
    for rel in df["path"]:
        with Image.open(Path(image_dir) / rel) as im:
            sizes.append(im.size)
    return df.assign(width=[s[0] for s in sizes], height=[s[1] for s in sizes])


def plot_image_sizes(sized: pd.DataFrame, save_to=None):
    """Width vs height scatter, coloured by class, with the median marked."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in ["cat", "dog"]:
        part = sized[sized["label"] == label]
        ax.scatter(part["width"], part["height"], s=4, alpha=0.25, color=CLASS_COLORS[label], label=label)
    mw, mh = sized["width"].median(), sized["height"].median()
    ax.scatter([mw], [mh], marker="x", s=80, color="black", label=f"median {mw:.0f}×{mh:.0f}")
    ax.set(title="Original image sizes", xlabel="width (px)", ylabel="height (px)")
    leg = ax.legend()
    for handle in leg.legend_handles[:2]:  # make the tiny class dots readable in the legend
        handle.set_sizes([30])
        handle.set_alpha(1)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, save_to)


def plot_augmented(image: np.ndarray, augment, n: int = 8, save_to=None):
    """One image (float32, [0,255]) next to `n` random augmentations of it."""
    batch = np.repeat(image[np.newaxis], n, axis=0)
    out = np.asarray(augment(batch, training=True))
    fig, axes = plt.subplots(1, n + 1, figsize=(2 * (n + 1), 2.3))
    for ax, img, title in zip(axes, [image, *out], ["original"] + [f"aug {i + 1}" for i in range(n)]):
        ax.imshow(np.clip(img, 0, 255).astype("uint8"))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Training-time augmentation (flip, rotation, zoom, brightness)")
    fig.tight_layout()
    return _save(fig, save_to)


def plot_history(history: dict, title: str, save_to=None, stage_boundary: int | None = None):
    """Train vs val accuracy and loss per epoch; the best epoch (lowest val_loss) is marked.

    stage_boundary: first epoch of a second training stage (e.g. fine-tuning), drawn as a shaded region.
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(epochs, history[metric], "o-", ms=3, label="train")
        ax.plot(epochs, history[f"val_{metric}"], "o-", ms=3, label="val")
        ax.axvline(best, color="grey", ls="--", lw=1, label=f"best epoch ({best})")
        if stage_boundary:
            ax.axvspan(stage_boundary - 0.5, epochs[-1] + 0.5, color="gold", alpha=0.15, label="fine-tuning")
        ax.set(title=metric, xlabel="epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_to)


def plot_gradcam_grid(rows: pd.DataFrame, image_dir, model, meta: dict, title: str, save_to=None):
    """For each row (path, label, pred_label, confidence): the photo and its Grad-CAM overlay side by side."""
    from src.gradcam import make_gradcam_heatmap, overlay
    from src.predict import preprocess

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.9 * n))
    axes = np.atleast_2d(axes)
    for (ax_img, ax_cam), (_, r) in zip(axes, rows.iterrows()):
        img = Image.open(Path(image_dir) / r["path"]).convert("RGB")
        heat = make_gradcam_heatmap(model, preprocess(img, meta["img_size"]), meta["gradcam_layer"])
        ax_img.imshow(img)
        ax_img.set_title(f"true {r['label']}", fontsize=9)
        ax_cam.imshow(overlay(img, heat))
        ok = r["label"] == r["pred_label"]
        ax_cam.set_title(f"pred {r['pred_label']} {r['confidence']:.0%}", fontsize=9,
                         color="green" if ok else "red")
        ax_img.axis("off")
        ax_cam.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, save_to)
