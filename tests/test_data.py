"""Tests for src/data.py: cleaning and stratified splitting, on the tiny fixture dataset."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import find_pet_images_dir, make_splits, scan_and_clean

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cleaned():
    return scan_and_clean(find_pet_images_dir(FIXTURES))


def test_find_pet_images_dir_nested():
    assert find_pet_images_dir(FIXTURES).name == "PetImages"


def test_find_pet_images_dir_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_pet_images_dir(tmp_path)


def test_keeps_only_valid_jpegs(cleaned):
    kept, _ = cleaned
    assert len(kept) == 20
    assert kept["label"].value_counts().to_dict() == {"cat": 10, "dog": 10}
    assert set(kept["label_id"]) == {0, 1}
    assert (kept.loc[kept.label == "cat", "label_id"] == 0).all()


def test_drops_every_junk_file(cleaned):
    _, dropped = cleaned
    assert set(dropped["path"]) == {
        "Cat/corrupt.jpg",
        "Cat/truncated.jpg",
        "Cat/Thumbs.db",
        "Dog/fake.jpg",
        "Dog/notes.txt",
    }
    assert "not a JPEG" in dropped.set_index("path").loc["Dog/fake.jpg", "reason"]


def test_paths_are_relative_posix(cleaned):
    kept, _ = cleaned
    assert all(p.startswith(("Cat/", "Dog/")) and "\\" not in p for p in kept["path"])


def _big_df(n_per_class=200):
    rows = [
        {"path": f"{c.capitalize()}/{i}.jpg", "label": c, "label_id": k}
        for k, c in enumerate(["cat", "dog"])
        for i in range(n_per_class)
    ]
    return pd.DataFrame(rows)


def test_split_sizes_disjoint_and_stratified():
    df = _big_df()
    s = make_splits(df, out_dir=None)
    assert [len(s[k]) for k in ("train", "val", "test")] == [280, 60, 60]
    paths = [set(s[k]["path"]) for k in ("train", "val", "test")]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert set().union(*paths) == set(df["path"])
    for part in s.values():
        assert abs(part["label_id"].mean() - 0.5) < 0.01


def test_split_is_deterministic(tmp_path):
    df = _big_df()
    make_splits(df, out_dir=tmp_path / "a")
    make_splits(df.sample(frac=1, random_state=1), out_dir=tmp_path / "b")  # row order must not matter
    for name in ("train", "val", "test"):
        assert (tmp_path / "a" / f"{name}.csv").read_bytes() == (tmp_path / "b" / f"{name}.csv").read_bytes()
