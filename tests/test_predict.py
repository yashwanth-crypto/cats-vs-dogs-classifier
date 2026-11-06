"""Tests for src/predict.py using a tiny untrained model saved to a temp folder."""

import json

import numpy as np
import pytest
from PIL import Image

from src.models import build_scratch_cnn
from src.predict import (
    available_models, load_model, pet_check_from_probs, predict_image, predict_proba, preprocess,
)


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    model = build_scratch_cnn(img_size=32)
    model.save(d / "tiny_cnn.keras")
    (d / "tiny_cnn_meta.json").write_text(json.dumps({"name": "tiny_cnn", "img_size": 32, "gradcam_layer": "relu3"}))
    model.save(d / "smoke_x.keras")  # smoke models are hidden from the app
    (d / "smoke_x_meta.json").write_text(json.dumps({"img_size": 32}))
    (d / "orphan.keras").write_bytes(b"")  # no meta -> not listed
    return d


def _img(mode, size=(57, 83)):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (size[1], size[0], 4), dtype=np.uint8)
    return Image.fromarray(arr, "RGBA").convert(mode)


def test_available_models(models_dir):
    assert available_models(models_dir) == ["tiny_cnn"]


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "CMYK"])
def test_preprocess_any_mode(mode):
    x = preprocess(_img(mode), 32)
    assert x.shape == (1, 32, 32, 3) and x.dtype == np.float32
    assert 0 <= x.min() and x.max() <= 255


def test_predict_image_label_and_confidence(models_dir):
    label, conf, heat = predict_image(_img("RGB", (300, 200)), "tiny_cnn", models_dir=models_dir)
    p_dog = predict_proba(_img("RGB", (300, 200)), "tiny_cnn", models_dir)
    assert label in {"cat", "dog"} and 0.5 <= conf <= 1.0
    assert conf == pytest.approx(p_dog if label == "dog" else 1 - p_dog, abs=1e-6)
    assert heat is None  # Grad-CAM only when asked


def test_predict_image_with_gradcam(models_dir):
    img = _img("RGBA", (300, 200))
    _, _, heat = predict_image(img, "tiny_cnn", with_gradcam=True, models_dir=models_dir)
    assert heat.size == (300, 200) and heat.mode == "RGB"


def _probs(top_ids):
    p = np.full(1000, 1e-4)
    for rank, i in enumerate(top_ids):
        p[i] = 0.5 / (rank + 1)
    return p


@pytest.mark.parametrize("top_ids,expected", [
    ([207, 1, 2, 3, 4], True),       # golden retriever first
    ([817, 511, 281, 4, 5], True),   # sports car, convertible, then tabby cat (3rd) -> still a pet
    ([817, 511, 436, 751, 656], False),  # all vehicles
    ([817, 511, 436, 751, 656, 207], False),  # dog only at rank 6 -> outside top-5
])
def test_pet_check_from_probs(top_ids, expected):
    is_pet, top1 = pet_check_from_probs(_probs(top_ids))
    assert is_pet is expected and top1 == top_ids[0]


def test_model_is_cached(models_dir):
    assert load_model("tiny_cnn", models_dir)[0] is load_model("tiny_cnn", models_dir)[0]
