"""Generate the tiny PetImages-style fixture folder used by the tests.

Run once (outputs are committed):  .venv/Scripts/python tests/make_fixtures.py

Layout mirrors the real dataset (PetImages/Cat, PetImages/Dog) and includes the
kinds of junk the real dataset contains, so the cleaning step can be tested:
  - 10 valid JPEGs per class (random sizes, RGB and grayscale)
  - Cat/corrupt.jpg       random bytes with a .jpg name
  - Cat/truncated.jpg     a JPEG cut off halfway
  - Dog/fake.jpg          a PNG saved with a .jpg name
  - Dog/notes.txt         not an image at all
  - Cat/Thumbs.db         Windows thumbnail cache
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent / "fixtures" / "PetImages"


def main() -> None:
    rng = np.random.default_rng(42)
    for cls in ("Cat", "Dog"):
        d = ROOT / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(10):
            w, h = rng.integers(40, 90, size=2)
            arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            if i == 9:  # one grayscale image per class
                img = img.convert("L")
            img.save(d / f"{i}.jpg", format="JPEG")

    # Junk files
    (ROOT / "Cat" / "corrupt.jpg").write_bytes(rng.bytes(500))
    good = (ROOT / "Cat" / "0.jpg").read_bytes()
    (ROOT / "Cat" / "truncated.jpg").write_bytes(good[: len(good) // 2])
    Image.fromarray(rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)).save(
        ROOT / "Dog" / "fake.jpg", format="PNG"
    )
    (ROOT / "Dog" / "notes.txt").write_text("not an image")
    (ROOT / "Cat" / "Thumbs.db").write_bytes(b"\x00" * 64)
    print(f"Fixtures written to {ROOT}")


if __name__ == "__main__":
    main()
