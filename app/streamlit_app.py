"""Cats vs Dogs: upload a photo, get a prediction, and see what the model looked at.

Run from the project root:
    .venv/Scripts/streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `src` importable
from src.predict import _imagenet_model, available_models, is_cat_or_dog, load_model, predict_image  # noqa: E402

LOW_CONFIDENCE = 0.60
NICE_NAMES = {
    "mobilenetv2": "MobileNetV2 (transfer learning)",
    "improved_cnn": "Improved CNN (from scratch)",
    "baseline_cnn": "Baseline CNN (from scratch)",
}
EMOJI = {"cat": "🐱", "dog": "🐶"}

st.set_page_config(page_title="Cat or Dog?", page_icon="🐾", layout="centered")
st.title("🐾 Cat or Dog?")
st.caption("Upload a photo of a cat or a dog. A CNN trained on about 17k photos will classify it, "
           "and Grad-CAM shows which parts of the image drove the decision.")

# ----------------------------------------------------------------------------- sidebar
models = available_models()
if not models:
    st.error("No trained models found in `models/`. Train one first: "
             "`python -m src.experiments mobilenetv2`.")
    st.stop()

# The app offers the improved CNN and MobileNetV2 (the baseline exists only for the comparison).
choices = [m for m in ("mobilenetv2", "improved_cnn") if m in models] or models
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", choices, format_func=lambda m: NICE_NAMES.get(m, m))
    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
    use_guard = st.toggle("Check it's a cat or dog first", value=True,
                          help="Runs a stock ImageNet model first. A cat/dog classifier always "
                               "answers cat or dog, even for a photo of a car.")

    _, meta = load_model(model_name)
    results_csv = ROOT / "reports" / "comparison_table.csv"
    if results_csv.exists():
        row = pd.read_csv(results_csv, index_col="model").loc[model_name]
        st.metric("Test accuracy", f"{row['accuracy']:.1%}")
    else:
        st.metric("Validation accuracy", f"{meta['best_val_accuracy']:.1%}")
    st.caption(f"Input {meta['img_size']}×{meta['img_size']} · {meta['params']:,} parameters")


@st.cache_resource(show_spinner="Loading model…")
def warm_up(name: str, guard: bool) -> None:
    """Load the models once per server (src.predict caches them for the process)."""
    load_model(name)
    if guard:
        _imagenet_model()


warm_up(model_name, use_guard)

# ----------------------------------------------------------------------------- main
EXAMPLES = {"Cat": "cat.jpg", "Dog": "dog.jpg", "Not a pet (mislabelled in the dataset)": "not_a_pet.jpg"}
uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
# "…or try an example": also settable via the URL, e.g. ?example=dog (used for README screenshots).
url_example = {"cat": "Cat", "dog": "Dog", "not_a_pet": "Not a pet (mislabelled in the dataset)"}.get(
    st.query_params.get("example", ""))
options = ["—", *EXAMPLES]
example = st.selectbox("…or try an example", options,
                       index=options.index(url_example) if url_example else 0, disabled=uploaded is not None)

if uploaded is not None:
    source = uploaded
elif example != "—":
    source = ROOT / "app" / "examples" / EXAMPLES[example]
else:
    st.stop()

try:
    image = Image.open(source)
    image.load()
except (UnidentifiedImageError, OSError):
    st.error("Couldn't read that file as an image. Try a different JPG or PNG.")
    st.stop()

with st.spinner("Thinking…"):
    label, confidence, heat = predict_image(image, model_name, with_gradcam=show_gradcam)
    is_pet, imagenet_guess = is_cat_or_dog(image) if use_guard else (True, "")

if not is_pet:
    st.warning(f"**This doesn't look like a cat or a dog.** A general-purpose ImageNet model "
               f"thinks it's: *{imagenet_guess}*.")
    st.caption(f"Forced to choose, the cat/dog model would say {label} ({confidence:.0%}). "
               "Treat that with suspicion: it was only ever trained on cats and dogs.")
elif confidence < LOW_CONFIDENCE:
    st.info(f"🤔 **I'm not sure.** Leaning {label} at only {confidence:.0%}. "
            "Try a clearer photo with one animal, facing the camera.")
else:
    st.success(f"## {EMOJI[label]} {label.capitalize()}")
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

if show_gradcam and heat is not None:
    left, right = st.columns(2)
    left.image(image, caption="Your photo", width="stretch")
    right.image(heat, caption=f"Grad-CAM: evidence for '{label}'", width="stretch")
    st.caption("Red areas influenced the prediction most. A good model focuses on the animal "
               "(face, ears, fur) rather than the background.")
else:
    st.image(image, caption="Your photo", width="stretch")
