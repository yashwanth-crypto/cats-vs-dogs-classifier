"""Single-image inference used by the Streamlit app.

    label, confidence, overlay = predict_image(pil_image, "mobilenetv2", with_gradcam=True)
    looks_like_pet, top_guess = is_cat_or_dog(pil_image)

Every saved model takes raw 0-255 RGB at its own input size (stored in <name>_meta.json), so
the only preprocessing here is: fix EXIF rotation -> RGB -> resize. The resize uses the same
function as the training pipeline (tf.image.resize, bilinear).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def available_models(models_dir: str | Path = MODELS_DIR) -> list[str]:
    """Names of models that have both a .keras file and a meta JSON, smoke-test models excluded."""
    models_dir = Path(models_dir)
    return sorted(
        p.stem for p in models_dir.glob("*.keras")
        if (models_dir / f"{p.stem}_meta.json").exists() and not p.stem.startswith("smoke")
    )


@lru_cache(maxsize=8)
def load_model(name: str, models_dir: str | Path = MODELS_DIR):
    """Load (and cache) a model and its meta. Cached per process, so the app loads each model once."""
    import keras

    models_dir = Path(models_dir)
    model = keras.models.load_model(models_dir / f"{name}.keras", compile=False)  # inference only
    meta = json.loads((models_dir / f"{name}_meta.json").read_text())
    return model, meta


def preprocess(pil_image: Image.Image, img_size: int) -> np.ndarray:
    """PIL image (any mode or size) -> float32 array of shape (1, img_size, img_size, 3) in [0, 255]."""
    import tensorflow as tf

    img = ImageOps.exif_transpose(pil_image)  # phone photos are often stored rotated
    img = img.convert("RGB")                  # handles RGBA, grayscale (L), palette (P), CMYK
    arr = tf.image.resize(np.asarray(img), (img_size, img_size))  # same resize as training
    return arr.numpy()[np.newaxis].astype("float32")


def predict_proba(pil_image: Image.Image, name: str, models_dir: str | Path = MODELS_DIR) -> float:
    """P(dog) in [0, 1]."""
    model, meta = load_model(name, models_dir)
    x = preprocess(pil_image, meta["img_size"])
    return float(model(x, training=False).numpy()[0, 0])


def predict_image(
    pil_image: Image.Image, name: str, with_gradcam: bool = False, models_dir: str | Path = MODELS_DIR
) -> tuple[str, float, Image.Image | None]:
    """Return (label, confidence, overlay).

    label is 'cat' or 'dog', confidence is P(predicted label), and overlay is a Grad-CAM image at
    the original resolution (None unless with_gradcam=True).
    """
    model, meta = load_model(name, models_dir)
    x = preprocess(pil_image, meta["img_size"])
    p_dog = float(model(x, training=False).numpy()[0, 0])
    label, confidence = ("dog", p_dog) if p_dog >= 0.5 else ("cat", 1.0 - p_dog)

    heat_overlay = None
    if with_gradcam:
        from src.gradcam import make_gradcam_heatmap, overlay

        heatmap = make_gradcam_heatmap(model, x, meta["gradcam_layer"])
        heat_overlay = overlay(ImageOps.exif_transpose(pil_image), heatmap)
    return label, confidence, heat_overlay


# --------------------------------------------------------------------------- #
# "Is this even a cat or a dog?" guard
# --------------------------------------------------------------------------- #
# A binary model has to answer cat or dog, even for a photo of a car, and is often very confident
# about it. As a guard, we ask a stock ImageNet MobileNetV2 (1,000 classes) for its top-5 guesses.
# ImageNet has 118 dog breeds (classes 151-268) and 5 domestic cats (281-285).
DOG_CLASSES = set(range(151, 269))
CAT_CLASSES = set(range(281, 286))
PET_CLASSES = DOG_CLASSES | CAT_CLASSES


@lru_cache(maxsize=1)
def _imagenet_model():
    import keras

    return keras.applications.MobileNetV2(weights="imagenet")  # 224x224, downloads 14 MB once


def pet_check_from_probs(probs: np.ndarray, top_k: int = 5) -> tuple[bool, int]:
    """Given 1,000 ImageNet probabilities, return (any top-k class is a cat/dog, top-1 class id)."""
    top = np.argsort(probs)[::-1][:top_k]
    return bool(PET_CLASSES & set(top.tolist())), int(top[0])


def is_cat_or_dog(pil_image: Image.Image, top_k: int = 5) -> tuple[bool, str]:
    """Return (looks like a cat or dog, ImageNet's top-1 guess as readable text)."""
    import keras

    x = keras.applications.mobilenet_v2.preprocess_input(preprocess(pil_image, 224))
    probs = _imagenet_model()(x, training=False).numpy()[0]
    is_pet, top1 = pet_check_from_probs(probs, top_k)
    top_name = keras.applications.mobilenet_v2.decode_predictions(probs[np.newaxis], top=1)[0][0][1]
    return is_pet, top_name.replace("_", " ")
