"""Shared training helper: callbacks, timing, and saving model + history + meta.

For a model called `name`, `train()` writes three files to `models_dir`:

    <name>.keras          best weights by val_loss (ModelCheckpoint)
    <name>_history.json   per-epoch metrics, e.g. {"loss": [...], "val_accuracy": [...], ...}
    <name>_meta.json      input size, params, training time, versions, Grad-CAM layer and a
                          reference prediction used to check that the model loads identically
                          on another machine (Colab -> local)

Check a model copied from Colab:
    .venv/Scripts/python -m src.train verify models/smoke_baseline.keras
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.data import CLASS_NAMES, SEED


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow."""
    keras.utils.set_random_seed(seed)


def _model_input_size(model: keras.Model) -> int:
    return int(model.input_shape[1])


def reference_prediction(model: keras.Model, image_path: str | Path) -> float:
    """P(dog) for one image file, using exactly the training pipeline's decode + resize."""
    size = _model_input_size(model)
    img = tf.io.decode_jpeg(tf.io.read_file(str(image_path)), channels=3)
    img = tf.image.resize(img, (size, size))[tf.newaxis]
    return float(model.predict(img, verbose=0)[0, 0])


def train(
    model: keras.Model,
    train_ds,
    val_ds,
    name: str,
    epochs: int = 30,
    models_dir: str | Path = "models",
    patience: int = 5,
    lr_patience: int = 2,
    reference_image: str | Path | None = None,
    extra_meta: dict | None = None,
    prior_history: dict | None = None,
    prior_seconds: float = 0.0,
) -> dict:
    """Fit `model` with EarlyStopping, ModelCheckpoint and ReduceLROnPlateau (all on val_loss).

    The in-memory model ends up holding the best weights, matching the saved .keras file.
    Returns the history dict.

    Two-stage training (e.g. transfer learning then fine-tuning): pass the first stage's history
    and seconds as `prior_history` / `prior_seconds`. Epoch numbering continues, the saved
    history and time cover both stages, and the checkpoint is only replaced when the second stage
    beats the first stage's best val_loss. `epochs` is the total, counting both stages.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True,
            initial_value_threshold=min(prior_history["val_loss"]) if prior_history else None,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, min_lr=1e-6),
    ]

    start = time.perf_counter()
    hist = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2, shuffle=False,
        initial_epoch=len(prior_history["loss"]) if prior_history else 0,
    )  # shuffling is done by the tf.data pipeline
    train_seconds = time.perf_counter() - start + prior_seconds

    # restore_best_weights only applies when training stops early; reload the checkpoint so the
    # in-memory model always equals the best saved one.
    model.set_weights(keras.models.load_model(model_path).get_weights())

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    if prior_history:
        history = {k: list(prior_history.get(k, [])) + history[k] for k in history}
    (models_dir / f"{name}_history.json").write_text(json.dumps(history, indent=1))

    meta = {
        "name": name,
        "img_size": _model_input_size(model),
        "class_names": CLASS_NAMES,
        "params": int(model.count_params()),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "train_seconds": round(train_seconds, 1),
        "epochs_run": len(history["loss"]),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss": min(history["val_loss"]),
        "best_val_accuracy": history["val_accuracy"][int(np.argmin(history["val_loss"]))],
        "gradcam_layer": getattr(model, "gradcam_layer", None),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "keras_backend": keras.backend.backend(),
    }
    if reference_image is not None:
        meta["reference"] = {
            "image": Path(reference_image).as_posix().split("PetImages/")[-1],
            "p_dog": reference_prediction(model, reference_image),
        }
    meta.update(extra_meta or {})
    (models_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=1))
    return history


def load_meta(model_path: str | Path) -> dict:
    """Read the meta JSON that sits next to a .keras file."""
    model_path = Path(model_path)
    return json.loads(model_path.with_name(f"{model_path.stem}_meta.json").read_text())


def verify(model_path: str | Path, image_dir: str | Path, tol: float = 1e-4) -> bool:
    """Reload a saved model and check it reproduces the reference prediction stored in its meta."""
    meta = load_meta(model_path)
    model = keras.models.load_model(model_path)
    ref = meta["reference"]
    p = reference_prediction(model, Path(image_dir) / ref["image"])
    if meta.get("keras_backend", "tensorflow") != keras.backend.backend():
        tol = max(tol, 1e-3)  # GPU/PyTorch vs CPU/TensorFlow float differences (amplified by BatchNorm)
    ok = abs(p - ref["p_dog"]) <= tol
    print(
        f"{Path(model_path).name}: trained with TF {meta['tensorflow']} / Keras {meta['keras']}, "
        f"loaded with TF {tf.__version__} / Keras {keras.__version__}\n"
        f"  reference {ref['image']}: saved p_dog={ref['p_dog']:.6f}, now p_dog={p:.6f} -> "
        f"{'MATCH' if ok else 'MISMATCH'}"
    )
    return ok


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        from src.data import find_pet_images_dir

        sys.exit(0 if verify(sys.argv[2], find_pet_images_dir("data")) else 1)
    print(__doc__)


def strip_optimizer(model_path: str | Path) -> None:
    """Re-save a model without its optimizer state (about 3x smaller; fine for inference/deployment)."""
    model = keras.models.load_model(model_path, compile=False)
    model.save(model_path)
