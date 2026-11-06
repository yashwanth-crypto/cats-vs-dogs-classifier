"""Data loading, cleaning and splitting for the Microsoft PetImages (Cats vs Dogs) dataset.

Pipeline:
    find_pet_images_dir()  -> locate the PetImages folder under data/
    scan_and_clean()       -> every readable JPEG with its label; junk files dropped and listed
    make_splits()          -> stratified 70/15/15 train/val/test split, saved as CSV

Paths in the CSVs are stored *relative to the PetImages folder* with forward slashes
(e.g. "Cat/123.jpg"), so the same split files work on Windows and on Colab.

Run the whole step from the project root:
    .venv/Scripts/python -m src.data
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ["cat", "dog"]  # index = label id: cat -> 0, dog -> 1
SPLITS_DIR = Path("data/splits")

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Locating and cleaning
# --------------------------------------------------------------------------- #
def find_pet_images_dir(root: str | Path = "data") -> Path:
    """Return the first folder named 'PetImages' under `root` (it may be nested after unzipping)."""
    root = Path(root)
    if root.name == "PetImages" and root.is_dir():
        return root
    for p in sorted(root.rglob("PetImages")):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No 'PetImages' folder found under {root.resolve()}")


def _check_image(path: Path, tf_check: bool) -> str | None:
    """Return None if `path` is a fully decodable JPEG, otherwise a short reason string.

    Two checks:
      1. Pillow: header verify + full pixel load (catches truncated files) and format must be JPEG
         (the dataset contains PNG/BMP/GIF files with a .jpg extension).
      2. TensorFlow decode_jpeg (optional): the exact decoder the training pipeline uses, so
         anything that would crash training mid-epoch is removed here instead.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            fmt = im.format
            im.load()
    except Exception as e:  # noqa: BLE001 - any failure means "unusable"
        return f"unreadable ({type(e).__name__})"
    if fmt != "JPEG":
        return f"not a JPEG ({fmt})"
    if tf_check:
        import tensorflow as tf  # local import: keeps cleaning usable without TF

        try:
            tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=3)
        except Exception as e:  # noqa: BLE001
            return f"TensorFlow cannot decode ({type(e).__name__})"
    return None


def scan_and_clean(
    image_dir: str | Path, tf_check: bool = True, workers: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan `image_dir/Cat` and `image_dir/Dog`, keeping only fully decodable JPEGs.

    Returns:
        kept:    DataFrame[path, label, label_id] sorted by path (path relative to image_dir)
        dropped: DataFrame[path, reason] for every file that was removed
    """
    image_dir = Path(image_dir)
    candidates: list[tuple[Path, str]] = []
    for folder in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        label = folder.name.lower()
        if label not in CLASS_NAMES:
            log.warning("Skipping unexpected folder %s", folder)
            continue
        candidates += [(f, label) for f in sorted(folder.iterdir()) if f.is_file()]

    # Pillow checks are I/O bound, so a thread pool speeds up the 25k-file scan.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        reasons = list(pool.map(lambda c: _check_image(c[0], tf_check=False), candidates))
    if tf_check:  # TF decode in the main thread (TF manages its own threads)
        reasons = [r or _check_image(f, tf_check=True) for (f, _), r in zip(candidates, reasons)]

    kept, dropped = [], []
    for (f, label), reason in zip(candidates, reasons):
        rel = f.relative_to(image_dir).as_posix()
        if reason is None:
            kept.append({"path": rel, "label": label, "label_id": CLASS_NAMES.index(label)})
        else:
            dropped.append({"path": rel, "reason": reason})

    kept_df = pd.DataFrame(kept, columns=["path", "label", "label_id"]).sort_values("path")
    dropped_df = pd.DataFrame(dropped, columns=["path", "reason"]).sort_values("path")
    log.info("Scanned %d files: kept %d, dropped %d", len(candidates), len(kept_df), len(dropped_df))
    return kept_df.reset_index(drop=True), dropped_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(
    df: pd.DataFrame,
    out_dir: str | Path | None = SPLITS_DIR,
    seed: int = SEED,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split (default 70/15/15). Writes {train,val,test}.csv if out_dir is set."""
    # Sort first: train_test_split depends on row order, and the split must be reproducible.
    df = df.sort_values("path").reset_index(drop=True)
    # Integer sizes avoid float round-off (e.g. 0.15/0.85*340 = 60.0000001 -> 61).
    n_test, n_val = round(test_frac * len(df)), round(val_frac * len(df))
    train_val, test = train_test_split(
        df, test_size=n_test, stratify=df["label_id"], random_state=seed
    )
    train, val = train_test_split(
        train_val,
        test_size=n_val,
        stratify=train_val["label_id"],
        random_state=seed,
    )
    splits = {
        name: part.sort_values("path").reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")
    return splits


def load_split(name: str, splits_dir: str | Path = SPLITS_DIR) -> pd.DataFrame:
    """Read one saved split ('train', 'val' or 'test')."""
    return pd.read_csv(Path(splits_dir) / f"{name}.csv")


# --------------------------------------------------------------------------- #
# tf.data input pipeline
# --------------------------------------------------------------------------- #
def make_dataset(
    df: pd.DataFrame,
    image_dir: str | Path,
    img_size: int,
    batch_size: int = 32,
    shuffle: bool = False,
    augment=None,
    seed: int = SEED,
    cache: bool = False,
):
    """Build a batched, prefetched tf.data.Dataset of (images, labels) from a split DataFrame.

    Images are decoded, resized to (img_size, img_size) and returned as float32 in [0, 255].
    Scaling or model-specific preprocessing happens *inside each model*, so the same pipeline
    (and the app) can feed any of the models.

    Args:
        df:        split DataFrame with 'path' (relative to image_dir) and 'label_id' columns.
        image_dir: the PetImages folder.
        shuffle:   shuffle every epoch (use for training only).
        augment:   optional Keras layer/model applied to training batches only (see build_augmentation).
        cache:     keep resized images in RAM as uint8 after the first pass, so later epochs skip
                   JPEG decoding (17k images at 128px ~ 0.85 GB, at 160px ~ 1.3 GB).
    """
    import tensorflow as tf

    image_dir = Path(image_dir)
    paths = [str(image_dir / p) for p in df["path"]]
    labels = df["label_id"].astype("float32").to_numpy()

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (img_size, img_size))  # bilinear, returns float32
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        # Store compactly as uint8, convert back to float32 after reading from the cache.
        ds = ds.map(lambda x, y: (tf.cast(tf.round(x), tf.uint8), y)).cache()
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_augmentation(seed: int = SEED):
    """Random flip, rotation, zoom and brightness, applied to training batches only.

    Works on float32 images in [0, 255] (value_range matches the pipeline output). Mild settings:
    pets are photographed upright, so only small rotations and horizontal flips.
    """
    from keras import layers

    return Augmentation([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.08, fill_mode="reflect", seed=seed),  # about ±29 degrees
        layers.RandomZoom(0.15, fill_mode="reflect", seed=seed),
        layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed),
    ])


class Augmentation:
    """Applies Keras preprocessing layers one after another.

    Used instead of keras.Sequential because, inside a tf.data map, each preprocessing layer
    switches itself to TensorFlow ops, which works under every Keras backend. Sequential routes
    through the active backend and fails when that's PyTorch (GPU training on Windows).
    """

    def __init__(self, aug_layers):
        self.layers = aug_layers

    def __call__(self, x, training: bool = True):
        for layer in self.layers:
            x = layer(x, training=training)
        return x


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    image_dir = find_pet_images_dir("data")
    log.info("Using dataset at %s", image_dir)
    kept, dropped = scan_and_clean(image_dir)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    dropped.to_csv(SPLITS_DIR / "dropped.csv", index=False, lineterminator="\n")
    log.info("Dropped files by reason:\n%s", dropped["reason"].value_counts().to_string())
    splits = make_splits(kept)
    for name, part in splits.items():
        counts = part["label"].value_counts().to_dict()
        log.info("%-5s %6d images  %s", name, len(part), counts)


if __name__ == "__main__":
    main()
