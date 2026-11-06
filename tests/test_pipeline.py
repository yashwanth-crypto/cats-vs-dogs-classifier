"""Tests for the tf.data pipeline, model builders and training helper (fixture data, tiny runs)."""

import json
from pathlib import Path

import keras
import numpy as np
import pytest

from src.data import find_pet_images_dir, make_dataset, scan_and_clean
from src.models import build_scratch_cnn
from src.train import load_meta, set_seed, train, verify

IMAGE_DIR = find_pet_images_dir(Path(__file__).parent / "fixtures")


@pytest.fixture(scope="module")
def df():
    kept, _ = scan_and_clean(IMAGE_DIR, tf_check=False)
    return kept


def test_dataset_shapes_and_ranges(df):
    ds = make_dataset(df, IMAGE_DIR, img_size=64, batch_size=8)
    x, y = next(iter(ds))
    assert x.shape == (8, 64, 64, 3) and x.dtype.name == "float32"
    assert y.shape == (8,)
    assert 0.0 <= float(np.min(x)) and float(np.max(x)) <= 255.0
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert sum(int(b[0].shape[0]) for b in ds) == len(df)  # every image, once


def test_grayscale_jpeg_becomes_rgb(df):
    gray = df[df["path"] == "Cat/9.jpg"]  # saved as grayscale by make_fixtures.py
    x, _ = next(iter(make_dataset(gray, IMAGE_DIR, img_size=32)))
    assert x.shape[-1] == 3


@pytest.mark.parametrize("improved", [False, True])
def test_scratch_cnn_forward(improved):
    model = build_scratch_cnn(img_size=64, improved=improved)
    out = model.predict(np.random.uniform(0, 255, (4, 64, 64, 3)).astype("float32"), verbose=0)
    assert out.shape == (4, 1) and np.all((out >= 0) & (out <= 1))
    assert model.get_layer(model.gradcam_layer)  # Grad-CAM layer exists
    has_bn = any(isinstance(l, keras.layers.BatchNormalization) for l in model.layers)
    assert has_bn == improved


def test_train_saves_reloadable_model(df, tmp_path):
    set_seed()
    model = build_scratch_cnn(img_size=32)
    ds = make_dataset(df, IMAGE_DIR, img_size=32, batch_size=8)
    history = train(
        model, ds, ds, name="tiny", epochs=1, models_dir=tmp_path,
        reference_image=IMAGE_DIR / df["path"].iloc[0],
    )
    for suffix in (".keras", "_history.json", "_meta.json"):
        assert (tmp_path / f"tiny{suffix}").exists()
    assert set(history) >= {"loss", "val_loss", "accuracy", "val_accuracy"}
    assert json.loads((tmp_path / "tiny_history.json").read_text()) == history

    meta = load_meta(tmp_path / "tiny.keras")
    assert meta["img_size"] == 32 and meta["gradcam_layer"] == "relu3"
    assert meta["params"] == model.count_params() and meta["train_seconds"] > 0
    assert meta["reference"]["image"] == df["path"].iloc[0]
    assert verify(tmp_path / "tiny.keras", IMAGE_DIR)  # reload reproduces the prediction


def test_cache_gives_same_images(df):
    plain = next(iter(make_dataset(df, IMAGE_DIR, 32, batch_size=20)))[0].numpy()
    cached_ds = make_dataset(df, IMAGE_DIR, 32, batch_size=20, cache=True)
    for _ in range(2):  # first pass fills the cache, second pass reads from it
        cached = next(iter(cached_ds))[0].numpy()
        assert cached.dtype == np.float32
        assert np.abs(cached - plain).max() <= 0.5  # only uint8 rounding differs


def test_augmentation_train_only_and_shape(df):
    from src.data import build_augmentation

    aug = make_dataset(df, IMAGE_DIR, 32, batch_size=20, augment=build_augmentation())
    plain = make_dataset(df, IMAGE_DIR, 32, batch_size=20)
    xa, xp = next(iter(aug))[0].numpy(), next(iter(plain))[0].numpy()
    assert xa.shape == xp.shape and not np.allclose(xa, xp)
    assert np.array_equal(xp, next(iter(plain))[0].numpy())  # no augmentation -> deterministic


def test_transfer_model_frozen_and_flat():
    from src.models import build_transfer_model, unfreeze_top

    model = build_transfer_model(img_size=96, weights=None)  # no download in tests
    out = model.predict(np.random.uniform(0, 255, (2, 96, 96, 3)).astype("float32"), verbose=0)
    assert out.shape == (2, 1)
    trainable = {l.name for l in model.layers if l.trainable and l.weights}
    assert trainable == {"p_dog"}  # only the new head trains in stage 1
    assert model.get_layer("out_relu")  # Grad-CAM layer reachable without nesting

    unfreeze_top(model, n_layers=10)
    newly = [l for l in model.layers[-13:-3] if l.trainable and l.weights]
    assert newly and not any(isinstance(l, keras.layers.BatchNormalization) and l.trainable for l in model.layers)


def test_two_stage_training_merges_history(df, tmp_path):
    set_seed()
    model = build_scratch_cnn(img_size=32)
    ds = make_dataset(df, IMAGE_DIR, img_size=32, batch_size=8)
    h1 = train(model, ds, ds, name="two", epochs=1, models_dir=tmp_path)
    h2 = train(model, ds, ds, name="two", epochs=3, models_dir=tmp_path,
               prior_history=h1, prior_seconds=100.0)
    assert len(h2["loss"]) == 3 and h2["loss"][0] == h1["loss"][0]
    meta = load_meta(tmp_path / "two.keras")
    assert meta["epochs_run"] == 3 and meta["train_seconds"] > 100
