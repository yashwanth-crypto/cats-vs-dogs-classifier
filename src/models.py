"""Model builders. Every model:

* takes raw RGB pixels in [0, 255] at its own square input size (preprocessing is a layer inside
  the model, so training, evaluation and the app all feed images the same way);
* ends in a single sigmoid unit: P(dog). Class ids are cat = 0, dog = 1;
* is compiled with Adam + binary cross-entropy, tracking accuracy and ROC-AUC.

The name of the last convolutional layer is stored on the model as `gradcam_layer` so that
`train.py` can record it in the meta JSON for Grad-CAM.
"""

from __future__ import annotations

import keras
from keras import layers


def compile_binary(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    """Compile a binary classifier with the project's standard optimizer, loss and metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def build_scratch_cnn(img_size: int = 128, improved: bool = False, block_dropout: float = 0.1,
                      bn_momentum: float = 0.9, learning_rate: float | None = None) -> keras.Model:
    """CNN trained from scratch: 3 blocks of Conv2D -> MaxPooling, then Flatten -> Dense -> Dropout -> sigmoid.

    improved=False: the plain baseline (expected to overfit; trained without augmentation), Adam 1e-3.
    improved=True:  same depth and filter counts, plus BatchNorm after every conv *and* after the
                    Dense layer, light Dropout after every block, and Adam 3e-4. Only regularization
                    and optimization settings change, so the comparison is fair.

    Why these improved-model settings (found by diagnosis; see tasks/plan.md):
      * BatchNorm on the conv blocks alone made the flattened features (32,768 values) large enough
        to blow up the 4.2M-weight Dense layer at lr 1e-3: its ReLUs died and the model predicted
        one class forever (50% accuracy), under both TensorFlow and PyTorch backends.
        BatchNorm after the Dense layer plus lr 3e-4 keeps it stable.
      * momentum 0.9 (Keras default 0.99) lets BatchNorm's running statistics, which are used at
        prediction time, keep up with the fast-changing weights early in training.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)

    for i, filters in enumerate((32, 64, 128), start=1):
        # BatchNorm makes the conv bias redundant, so drop it in the improved model.
        x = layers.Conv2D(filters, 3, padding="same", use_bias=not improved, name=f"conv{i}")(x)
        if improved:
            x = layers.BatchNormalization(momentum=bn_momentum, name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
        if improved and block_dropout > 0:
            x = layers.Dropout(block_dropout, name=f"drop{i}")(x)

    x = layers.Flatten(name="flatten")(x)
    if improved:
        x = layers.Dense(128, use_bias=False, name="dense")(x)
        x = layers.BatchNormalization(momentum=bn_momentum, name="dense_bn")(x)
        x = layers.Activation("relu", name="dense_relu")(x)
    else:
        x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_dog")(x)

    name = "improved_cnn" if improved else "baseline_cnn"
    model = keras.Model(inputs, outputs, name=name)
    model.gradcam_layer = "relu3"  # last conv block activation (pre-pooling)
    if learning_rate is None:
        learning_rate = 3e-4 if improved else 1e-3
    return compile_binary(model, learning_rate)


# --------------------------------------------------------------------------- #
# Transfer learning
# --------------------------------------------------------------------------- #
def build_transfer_model(img_size: int = 160, weights: str | None = "imagenet",
                         dropout: float = 0.2) -> keras.Model:
    """MobileNetV2 (ImageNet weights) with a frozen base and a new head: GAP -> Dropout -> sigmoid.

    The base is built on top of a preprocessing layer via `input_tensor`, so its layers sit
    directly in this model's graph instead of inside a nested sub-model. That makes the last conv
    activation ('out_relu') directly reachable for Grad-CAM.

    Preprocessing: Rescaling(1/127.5, offset=-1) maps [0, 255] -> [-1, 1], exactly what
    keras.applications.mobilenet_v2.preprocess_input does, but as a layer that saves cleanly.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(inputs)
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), input_tensor=x,
                                          include_top=False, weights=weights)
    for layer in base.layers:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_dog")(x)

    model = keras.Model(inputs, outputs, name="mobilenetv2")
    model.gradcam_layer = "out_relu"
    return compile_binary(model, learning_rate=1e-3)


def unfreeze_top(model: keras.Model, n_layers: int = 30, learning_rate: float = 1e-5) -> keras.Model:
    """Fine-tuning stage: unfreeze the last `n_layers` base layers (BatchNorm stays frozen) and
    recompile with a low learning rate so the pretrained features are only nudged."""
    head = {"gap", "head_dropout", "p_dog"}
    base_layers = [l for l in model.layers if l.name not in head and l.name not in {"image", "preprocess"}]
    for layer in base_layers[-n_layers:]:
        # Frozen BatchNorm keeps its ImageNet statistics; updating them on small batches hurts.
        layer.trainable = not isinstance(layer, layers.BatchNormalization)
    return compile_binary(model, learning_rate=learning_rate)
