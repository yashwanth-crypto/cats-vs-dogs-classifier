"""Grad-CAM: which parts of the image pushed the model toward its prediction.

    heatmap = make_gradcam_heatmap(model, x, layer_name)   # x: (1, H, W, 3) in [0, 255]
    overlay_img = overlay(pil_image, heatmap)

Every model in this project has a flat graph (MobileNetV2 is built via `input_tensor`), so the
conv layer named in the model's meta (`gradcam_layer`) is a top-level layer.

For a sigmoid P(dog) output, gradients of P(dog) highlight "dog evidence". When the model
predicts *cat*, we use the gradient of P(cat) = 1 - P(dog) instead, so the map always shows
evidence for the predicted class.
"""

from __future__ import annotations

import keras
import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image


def make_gradcam_heatmap(model: keras.Model, x: np.ndarray, layer_name: str) -> np.ndarray:
    """Return a 2-D heatmap in [0, 1] at the conv layer's spatial resolution."""
    grad_model = keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        p_dog = preds[:, 0]
        score = tf.where(p_dog >= 0.5, p_dog, 1.0 - p_dog)  # score of the predicted class
    grads = tape.gradient(score, conv_out)                  # (1, h, w, c)
    weights = tf.reduce_mean(grads, axis=(1, 2))            # channel importance (1, c)
    cam = tf.einsum("bhwc,bc->bhw", conv_out, weights)[0]   # weighted sum of feature maps
    cam = tf.nn.relu(cam).numpy()                           # keep positive evidence only
    peak = cam.max()
    return (cam / peak if peak > 0 else cam).astype("float32")


def overlay(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap onto the original image (at the original resolution)."""
    img = pil_image.convert("RGB")
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(img.size, Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(img) / 255.0 + alpha * colored
    return Image.fromarray(np.uint8(255 * np.clip(blended, 0, 1)))
