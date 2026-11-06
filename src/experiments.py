"""The three experiments, defined once and run from either the notebook or the command line.

    .venv/Scripts/python -m src.experiments baseline_cnn
    .venv/Scripts/python -m src.experiments improved_cnn
    .venv/Scripts/python -m src.experiments mobilenetv2

Uses only the train and val splits. The test split is reserved for evaluation (Phase 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.data import SEED, build_augmentation, find_pet_images_dir, load_split, make_dataset
from src.models import build_scratch_cnn, build_transfer_model, unfreeze_top
from src.train import set_seed, train

# name -> settings. "build" returns a compiled model.
EXPERIMENTS = {
    # From scratch, no augmentation: shows what happens without regularization (overfitting).
    "baseline_cnn": dict(img_size=128, augment=False, epochs=30, patience=5,
                         build=lambda: build_scratch_cnn(128, improved=False)),
    # Same depth, plus BatchNorm, Dropout and augmentation: shows the effect of regularization.
    # Augmented validation curves are noisy, so it gets more patience before LR cuts / stopping.
    "improved_cnn": dict(img_size=128, augment=True, epochs=60, patience=12, lr_patience=4,
                         build=lambda: build_scratch_cnn(128, improved=True)),
    # ImageNet MobileNetV2: stage 1 trains only the new head on a frozen base; stage 2 fine-tunes
    # the top of the base at a low learning rate. Same augmentation as the improved CNN.
    "mobilenetv2": dict(img_size=160, augment=True, epochs=10, patience=3,
                        build=lambda: build_transfer_model(160),
                        fine_tune=dict(n_layers=30, learning_rate=1e-5, extra_epochs=10, patience=4)),
}


def run_experiment(name: str, image_dir: str | Path, models_dir: str | Path = "models",
                   batch_size: int = 32) -> dict:
    """Build, train and save one experiment. Returns its history dict."""
    cfg = EXPERIMENTS[name]
    set_seed(SEED)
    train_df, val_df = load_split("train"), load_split("val")
    size = cfg["img_size"]
    train_ds = make_dataset(train_df, image_dir, size, batch_size, shuffle=True, cache=True,
                            augment=build_augmentation() if cfg["augment"] else None)
    val_ds = make_dataset(val_df, image_dir, size, batch_size, cache=True)
    model = cfg["build"]()
    common = dict(models_dir=models_dir, reference_image=Path(image_dir) / val_df["path"].iloc[0],
                  extra_meta={"augmentation": cfg["augment"], "batch_size": batch_size})
    history = train(model, train_ds, val_ds, name=name, epochs=cfg["epochs"],
                    patience=cfg["patience"], lr_patience=cfg.get("lr_patience", 2), **common)

    ft = cfg.get("fine_tune")
    if ft:  # stage 2: continue from the best stage-1 weights with the top of the base unfrozen
        import json

        stage1_epochs = len(history["loss"])
        stage1_seconds = json.loads((Path(models_dir) / f"{name}_meta.json").read_text())["train_seconds"]
        unfreeze_top(model, ft["n_layers"], ft["learning_rate"])
        common["extra_meta"] |= {"fine_tune_from_epoch": stage1_epochs + 1, **{f"fine_tune_{k}": v for k, v in ft.items()}}
        history = train(model, train_ds, val_ds, name=name, epochs=stage1_epochs + ft["extra_epochs"],
                        patience=ft["patience"], prior_history=history, prior_seconds=stage1_seconds,
                        **common)
    return history


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        sys.exit(f"usage: python -m src.experiments {{{','.join(EXPERIMENTS)}}}")
    run_experiment(sys.argv[1], find_pet_images_dir("data"))
