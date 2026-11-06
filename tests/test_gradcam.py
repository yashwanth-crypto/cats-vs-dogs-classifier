"""Grad-CAM on both architectures (small untrained models, no downloads)."""

import numpy as np
import pytest
from PIL import Image

from src.gradcam import make_gradcam_heatmap, overlay
from src.models import build_scratch_cnn, build_transfer_model


@pytest.mark.parametrize("builder,size", [
    (lambda: build_scratch_cnn(64), 64),
    (lambda: build_scratch_cnn(64, improved=True), 64),
    (lambda: build_transfer_model(96, weights=None), 96),
])
def test_heatmap_shape_and_range(builder, size):
    model = builder()
    x = np.random.default_rng(0).uniform(0, 255, (1, size, size, 3)).astype("float32")
    heat = make_gradcam_heatmap(model, x, model.gradcam_layer)
    assert heat.ndim == 2 and heat.shape[0] == heat.shape[1] < size
    assert np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1


def test_overlay_keeps_original_size():
    img = Image.new("RGBA", (120, 80), (200, 100, 50, 255))
    heat = np.linspace(0, 1, 16, dtype="float32").reshape(4, 4)
    out = overlay(img, heat)
    assert out.size == (120, 80) and out.mode == "RGB"
