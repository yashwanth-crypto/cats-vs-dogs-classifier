# Cats vs Dogs Image Classifier: project rules

**Goal:** classify photos as Cat or Dog. Compare a baseline CNN, an improved CNN and MobileNetV2 transfer learning, explain predictions with Grad-CAM, and serve them in a Streamlit app.

**Working plan:** `tasks/plan.md` (decisions, risks) and `tasks/todo.md` (tasks with acceptance criteria). `PLAN.md` is the original brief.

## Structure
- `src/`: reusable code (`data.py`, `models.py`, `train.py`, `evaluate.py`, `gradcam.py`, `predict.py`)
- `notebooks/image_classification.ipynb`: main deliverable. Imports `src/`. Models train locally on CPU (`python -m src.experiments <name>`); Colab is supported too.
- `app/streamlit_app.py`: upload a photo and get a prediction
- `models/`: `<name>.keras` + `<name>_history.json` + `<name>_meta.json`
- `reports/figures/`: every saved plot
- `report/`: 2-3 page PDF report
- `data/`: Microsoft PetImages (gitignored, except `data/splits/*.csv`)

## Rules
- Modular, commented code. Keep logic in `src/`. The notebook only orchestrates and explains.
- Seed everything with 42 (Python, NumPy, TensorFlow, splits).
- Never touch the test split before Phase 3 (evaluation). Callbacks monitor `val_*` only.
- Preprocessing lives inside each model. Every model takes raw 0-255 RGB at its own input size, stored in its meta JSON.
- Python 3.11 venv (`.venv`). TensorFlow is pinned in `requirements.txt`, and Colab installs the same pin.

## Commands
- Train (GPU, RTX 5060 via PyTorch backend): `KERAS_BACKEND=torch .venv/Scripts/python -m src.experiments {baseline_cnn,improved_cnn,mobilenetv2}` (logs in `logs/`). Drop `KERAS_BACKEND=torch` for CPU. Everything else (tests, notebook, app, Grad-CAM) uses the default TensorFlow backend.
- Notebook: `.venv/Scripts/python notebooks/build_notebook.py --execute`
- Tests: `.venv/Scripts/python -m pytest -q`
- App: `.venv/Scripts/streamlit run app/streamlit_app.py` (`?example=dog|cat|not_a_pet` preselects an example)
- Report PDF: `.venv/Scripts/python report/make_report.py` (reads saved results; needs reportlab, pypdf)
