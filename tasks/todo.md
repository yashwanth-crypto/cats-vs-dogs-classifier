# Task List: Cats vs Dogs Image Classifier

See `tasks/plan.md` for decisions, the dependency graph, and risks.

**Standard commands** (from the project root, with the venv active):
- Tests: `python -m pytest -q`
- Lint/import check: `python -m compileall -q src app`
- App: `streamlit run app/streamlit_app.py`

---

## Progress log
- 2026-11-06 (final): T16 done. Pushed to https://github.com/yashwanth-crypto/cats-vs-dogs-classifie as a single commit authored only by the user (no co-author trailers). Notebook is 1.0 MB, email draft is in `report/submission_email.txt`. **Left for the user:** send the email, and optionally try the app with their own photos.
- 2026-11-06 (later): T12-T15 done. README with app screenshots; `report/report.pdf` is 2 pages and built by `report/make_report.py` from the saved results. App has a "try an example" picker. Still to do: T16 (push, notebook size, email draft), plus a real-photo app check and a fresh-clone check.
- 2026-11-06: Tasks 1-11 done. **Test set:** baseline 84.6%, improved CNN 85.7%, MobileNetV2 98.9% accuracy (see `reports/comparison_table.md`). The improved CNN needed 3 runs (diagnosis in plan.md). Models were stripped of optimizer state (17/17/9.6 MB) with identical test metrics. Tasks 12-13 are built and checked (guard catches wallpapers that both models call "dog" at 87-92%, and an app upload works end to end). Still to do: a real-photo upload check in the app, then T14-T16.

## Phase 1: Foundation and end-to-end path

### Task 1: Environment, repo skeleton, CLAUDE.md
**Description:** Create a Python 3.11 venv, pin dependencies to Colab's TensorFlow version, init git, and lay out the folder structure so every later task has a home.

**Acceptance criteria:**
- [x] `.venv` built with Python 3.11. `requirements.txt` pins `tensorflow==2.21.0` (+ `keras==3.15.1`) and lists numpy, pandas, matplotlib, seaborn, scikit-learn, Pillow, streamlit, pytest.
- [x] Folders exist: `src/` (with `__init__.py`), `app/`, `models/`, `reports/figures/`, `report/`, `notebooks/`, `tests/fixtures/`, `data/`. `.gitignore` covers `data/`, `.venv/`, `__pycache__/`, `*.keras` over 50MB (or Git LFS).
- [x] `CLAUDE.md` gives the goal, structure, "modular, commented code", "seed everything with 42", "never touch the test set before Phase 3", and the test command.

**Verification:**
- [x] `python -c "import tensorflow as tf, PIL, streamlit; print(tf.__version__)"` prints the pinned version
- [x] A sample image from `data/` opens and displays with PIL/matplotlib
- [x] `git status` is clean after the first commit

**Dependencies:** None (you download the Kaggle data into `data/` in parallel)
**Files:** `requirements.txt`, `.gitignore`, `CLAUDE.md`, `README.md` (stub), `src/__init__.py`
**Scope:** S

---

### Task 2: Clean dataset and make the stratified split
**Description:** `src/data.py` scans `data/PetImages/`, fully decodes every file, drops corrupt or non-image files (logging how many), parses `cat`/`dog` labels from the `Cat/` and `Dog/` folder names, and writes a stratified 70/15/15 split (seed 42) to CSV.

**Acceptance criteria:**
- [x] `scan_and_clean(dir) -> (DataFrame[path, label], n_dropped)`. Corrupt files are dropped, not raised.
- [x] `make_splits(df, seed=42)` writes `data/splits/{train,val,test}.csv`. Class ratios match within 1% across splits and no file appears in two splits.
- [x] Running it twice gives identical CSVs.

**Verification:**
- [x] `python -m pytest tests/test_data.py -q` passes, using `tests/fixtures/` (`Cat/` and `Dog/` subfolders: about 10 tiny JPEGs + 1 corrupt file + 1 `.txt`)
- [x] On the real data: 25,002 files scanned, **24,769 kept, 233 dropped** (173 BMP, 51 GIF, 4 PNG, 1 PSD saved as .jpg; 4 unreadable incl. Thumbs.db). Split **17,339 / 3,715 / 3,715**. Dropped list in `data/splits/dropped.csv`

**Dependencies:** Task 1
**Files:** `src/data.py`, `tests/test_data.py`, `tests/fixtures/*`
**Scope:** S-M

---

### Task 3: tf.data pipeline, baseline CNN, training helper (Colab smoke run)
**Description:** Add `make_dataset(csv, img_size, batch, augment=None, shuffle)` to `data.py`, `build_scratch_cnn(img_size, improved=False)` to `models.py` (the `Rescaling(1/255)` layer goes inside the model), and `train(model, train_ds, val_ds, name, epochs)` to `train.py`. `train()` wires EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau, then saves the `.keras` file, the history JSON, and a meta JSON (img size, params, train seconds, Keras version). Prove the Colab → local round-trip with a 1-epoch run on a 1k-image subset.

**Acceptance criteria:**
- [x] The pipeline yields `(batch, H, W, 3)` uint8/float images and `(batch,)` labels in {0,1}, with prefetch enabled.
- [x] The baseline model has 3 Conv-MaxPool blocks → Flatten → Dense → Dropout → sigmoid, and compiles with Adam + binary cross-entropy.
- [ ] `models/smoke_baseline.keras`, trained on Colab, loads locally with `keras.models.load_model` and gives the same prediction on one image on both machines (within 1e-4).

**Verification:**
- [x] `python -m pytest tests/test_pipeline.py -q`: batch shapes and dtypes on fixtures, and a model forward pass gives `(N,1)` in [0,1]
- [ ] Colab notebook cell runs `train()` for 1 epoch without error
- [x] Local load-and-predict script matches the Colab output

**Status note (2026-11-06):** Local round-trip verified. 1 epoch / 1k images takes **7 s on this PC's CPU** (AVX-512), so a full epoch is about 2 min. `python -m src.train verify <model>` checks a reloaded model against the reference prediction in its meta. The Colab criteria are only needed if we train on Colab (see plan.md, "Training location").

**Dependencies:** Task 2
**Files:** `src/data.py`, `src/models.py`, `src/train.py`, `tests/test_pipeline.py`, `notebooks/image_classification.ipynb` (setup + smoke cells)
**Scope:** M

---

### Task 4: `predict_image` v0 and a bare Streamlit page
**Description:** `src/predict.py` loads a model plus its meta JSON, resizes a PIL image to the model's input size, and returns `(label, confidence)`. `app/streamlit_app.py` is an uploader, a preview, and a label. No styling yet.

**Acceptance criteria:**
- [x] `predict_image(pil_image, model_name) -> (label: str, confidence: float)`. Handles RGBA, grayscale, and any image size.
- [x] The model is cached (`st.cache_resource`), not reloaded on every upload.
- [ ] Uploading a jpg or png shows the image and a label.

**Verification:**
- [x] `python -m pytest tests/test_predict.py -q` (uses the smoke model or a tiny untrained model saved in a fixture)
- [ ] Manual: `streamlit run app/streamlit_app.py`, upload one cat and one dog photo, and see a label (accuracy doesn't matter yet)

**Dependencies:** Task 3
**Files:** `src/predict.py`, `app/streamlit_app.py`, `tests/test_predict.py`
**Scope:** S

### ✅ Checkpoint A: one path works end to end
- [ ] `python -m pytest -q` all green
- [ ] Colab-trained model → `models/` → app shows a prediction
- [ ] Commit, then review with the human before full training

---

## Phase 2: Data and models

### Task 5: EDA figures and the train-only augmentation layer
**Description:** Notebook EDA section that saves figures. Add a `build_augmentation()` Keras layer stack (RandomFlip, RandomRotation, RandomZoom, RandomBrightness) to `data.py`, applied to the train dataset only.

**Acceptance criteria:**
- [x] Saved to `reports/figures/`: `class_balance.png`, `sample_grid.png` (cats and dogs), `image_sizes.png` (width vs height scatter), `augmented_samples.png` (1 image × 8 augmentations).
- [x] `make_dataset(..., augment=...)` applies augmentation only when asked. Val and test pipelines never augment.

**Verification:**
- [x] `python -m pytest tests/test_pipeline.py -q`: augmented batch shape is unchanged, and with augmentation off two passes give identical output
- [x] Manual: open the 4 PNGs and check they look right

**Dependencies:** Checkpoint A
**Files:** `src/data.py`, `notebooks/image_classification.ipynb`, `tests/test_pipeline.py`
**Scope:** S

---

### Task 6: Baseline CNN full training (no augmentation)
**Description:** Train the Task 3 baseline on the full train split on Colab, 128×128, no augmentation, to show overfitting. Save curves.

**Acceptance criteria:**
- [x] `models/baseline_cnn.keras` plus `_history.json` and `_meta.json` exist locally.
- [x] `reports/figures/curves_baseline_cnn.png` shows train vs val accuracy and loss. A note in the notebook describes the train/val gap.
- [x] Only train and val are used.

**Verification:**
- [x] The model loads locally and the app (Task 4) can select it
- [x] Manual: the curves PNG shows a visible train/val gap or plateau

**Dependencies:** Checkpoint A
**Files:** `notebooks/image_classification.ipynb`, add `plot_history()` to `src/train.py`
**Scope:** S (mostly Colab wall-clock)

---

### Task 7: Improved CNN (BatchNorm, Dropout, augmentation)
**Description:** `build_scratch_cnn(improved=True)`: same depth, plus BatchNorm and Dropout, trained with the Task 5 augmentation.

**Acceptance criteria:**
- [x] Same number of conv blocks as the baseline, so the comparison is fair. Only the regularization changes.
- [x] `models/improved_cnn.keras` + JSONs and `reports/figures/curves_improved_cnn.png` are saved.
- [x] The notebook notes how the train/val gap changed compared with the baseline.

**Verification:**
- [x] `python -m pytest tests/test_pipeline.py -q` includes an `improved=True` forward-pass test
- [x] The model loads locally and predicts through `predict_image`

**Dependencies:** Tasks 5, 6
**Files:** `src/models.py`, `notebooks/image_classification.ipynb`, `tests/test_pipeline.py`
**Scope:** S

---

### Task 8: MobileNetV2 transfer learning (frozen head, then fine-tune)
**Description:** `build_transfer_model(img_size=160, fine_tune_at=None)`: ImageNet MobileNetV2 with `preprocess_input` inside the model, base frozen, new head (GAP → Dropout → Dense sigmoid). Stage 1 trains the head. Stage 2 (optional) unfreezes the top ~30 layers at LR 1e-5.

**Acceptance criteria:**
- [x] The model takes raw 0-255 160×160 input, and the base layer name is recorded in meta for Grad-CAM.
- [x] `models/mobilenetv2.keras` + JSONs and `reports/figures/curves_mobilenetv2.png` are saved, with both stages marked on the curves if fine-tuned.
- [x] Training time covers both stages.

**Verification:**
- [x] `python -m pytest tests/test_pipeline.py -q`: forward pass works and the frozen-layer count is correct (use `weights=None` in tests to avoid the download)
- [x] The model loads locally and predicts through `predict_image`

**Dependencies:** Task 5 (pipeline at 160px), Checkpoint A
**Files:** `src/models.py`, `notebooks/image_classification.ipynb`, `tests/test_pipeline.py`
**Scope:** M

### ✅ Checkpoint B: three models saved with curves
- [ ] All 3 `.keras` files and their JSONs are in `models/`, and all 3 curve PNGs are saved
- [ ] `python -m pytest -q` all green
- [ ] Human reviews the val numbers before opening the test set

---

## Phase 3: Evaluation and explainability (test set first used here)

### Task 9: Test-set metrics, confusion matrices, comparison table
**Description:** `src/evaluate.py`: `evaluate_model(model, test_ds) -> dict` with accuracy, precision, recall, F1, ROC-AUC, and per-sample probabilities, plus confusion-matrix plotting and a comparison table (model, params, test acc, F1, AUC, train time).

**Acceptance criteria:**
- [x] `reports/figures/cm_{model}.png` for each model, plus `roc_curves.png` (all 3 on one plot).
- [x] `reports/comparison_table.csv` and a markdown version, with real numbers.
- [x] Metrics come from sklearn on the saved test CSV, with no shuffling, so they're reproducible.

**Verification:**
- [x] `python -m pytest tests/test_evaluate.py -q`: on hand-made y_true/y_prob, metrics match known values
- [x] Manual: the table numbers match the confusion-matrix counts

**Dependencies:** Checkpoint B
**Files:** `src/evaluate.py`, `tests/test_evaluate.py`, `notebooks/image_classification.ipynb`
**Scope:** M

---

### Task 10: Most-confidently-wrong error grid with commentary
**Description:** Using the Task 9 probabilities, plot the 16 most confidently wrong test images for the best model (or each model) with true label, predicted label, and confidence, plus written notes on patterns (odd angle, both animals, occlusion, humans, text).

**Acceptance criteria:**
- [x] `reports/figures/errors_{model}.png` is saved.
- [x] A notebook markdown cell has at least 3 observed failure patterns, each with example indices.

**Verification:**
- [x] `python -m pytest tests/test_evaluate.py -q` includes a `top_confident_errors()` ranking test
- [x] Manual: the grid images really are misclassified

**Dependencies:** Task 9
**Files:** `src/evaluate.py`, `notebooks/image_classification.ipynb`
**Scope:** S

---

### Task 11: Grad-CAM helper and overlay figures
**Description:** `src/gradcam.py`: `make_gradcam_heatmap(model, img_array, layer_name)` and `overlay(pil_img, heatmap)`. Works for the flat scratch CNN and the nested MobileNetV2 (the layer name comes from meta).

**Acceptance criteria:**
- [x] The heatmap is 2D, values are in [0,1], and there are no NaNs on either model type.
- [x] `reports/figures/gradcam_correct.png` and `gradcam_incorrect.png` (at least 4 each) are saved.

**Verification:**
- [x] `python -m pytest tests/test_gradcam.py -q`: shape, range, and both architectures (small untrained models)
- [x] Manual: the heatmaps highlight the animal on most correct examples

**Dependencies:** Checkpoint B (runs in parallel with T9/T10)
**Files:** `src/gradcam.py`, `tests/test_gradcam.py`, `notebooks/image_classification.ipynb`
**Scope:** S-M

### ✅ Checkpoint C: all figures in `reports/figures/`
- [ ] Comparison table, 3 CMs, ROC, error grid, and Grad-CAM figures are all present
- [ ] `python -m pytest -q` all green
- [ ] Commit

---

## Phase 4: App

### Task 12: Full `predict.py` and the full Streamlit UI
**Description:** `predict_image(pil_image, model_name, with_gradcam=False) -> (label, confidence, overlay|None)`. The UI has the uploader (jpg/png), preview, label with a confidence bar, a sidebar model selector (improved CNN / MobileNetV2), a Grad-CAM toggle, and a friendly message when confidence is below 60%.

**Acceptance criteria:**
- [x] Switching models in the sidebar changes the prediction without restarting, and both models are cached.
- [x] The Grad-CAM toggle shows the overlay next to the original.
- [x] Below 60% confidence: "I'm not sure, this may not be a clear cat or dog photo" instead of a bare label. Bad uploads (corrupt file) show an error, not a stack trace.

**Verification:**
- [x] `python -m pytest tests/test_predict.py -q`: returns a 3-tuple, overlay only when asked, and handles RGBA and grayscale
- [ ] Manual (with your own photos): 2 cats, 2 dogs, and 1 corrupt file through both models with the toggle on and off

**Dependencies:** Task 11
**Files:** `src/predict.py`, `app/streamlit_app.py`, `tests/test_predict.py`
**Scope:** M

---

### Task 13: ImageNet "not a cat or dog" guard
**Description:** Before the binary model runs, an ImageNet MobileNetV2 (from `keras.applications`, cached) checks whether any top-5 class is a cat (ImageNet 281-285) or dog (151-268) class. If none is, the app says "This doesn't look like a cat or dog" and still shows the binary result, greyed out.

**Acceptance criteria:**
- [x] `is_cat_or_dog(pil_image) -> (bool, top_label)` in `predict.py`.
- [x] The app shows the guard message for non-pet photos and skips it for pets.

**Verification:**
- [x] `python -m pytest tests/test_predict.py -q` (guard logic tested with a mocked top-5)
- [x] Manual: a car, a person, and a landscape photo trigger the message; the 4 pet photos don't

**Dependencies:** Task 12
**Files:** `src/predict.py`, `app/streamlit_app.py`, `tests/test_predict.py`
**Scope:** S

### ✅ Checkpoint D: app done
- [ ] `streamlit run app/streamlit_app.py` works end to end with cat, dog, and non-pet photos
- [ ] Screenshots saved to `reports/figures/app_*.png` for the README and report
- [ ] Commit

---

## Phase 5: Write-up and submission

### Task 14: README with setup, results table, screenshots
**Description:** Setup (Python 3.11 venv, requirements, Kaggle download steps), how to train (Colab), how to run tests and the app, the real results table, key figures, app screenshots, and limitations.

**Acceptance criteria:**
- [x] Someone can follow it from a clean clone to a running app.
- [x] The results table matches `reports/comparison_table.csv` exactly.

**Verification:**
- [x] Manual: follow the README in a fresh venv. Install, `pytest`, and `streamlit run` all work. (Checked from a fresh GitHub clone: 102 files byte-identical, 37 tests pass, models reload, app runs. Reused the existing venv rather than reinstalling.)

**Dependencies:** Checkpoints C, D
**Files:** `README.md`
**Scope:** S

---

### Task 15: 2-3 page report PDF
**Description:** Six sections, as in `PLAN.md`: problem and use cases, dataset and cleaning, methods, results (table, curves, CM, Grad-CAM), limitations, future work. Draft in Markdown/Docx and export to PDF.

**Acceptance criteria:**
- [x] `report/report.pdf` is ≤3 pages, has all 6 sections, and every number matches the comparison table.
- [x] Includes the Kaggle link, the count of dropped corrupt files, and the split sizes.

**Verification:**
- [x] Manual: page count ≤3, figures readable, numbers cross-checked against the CSV

**Dependencies:** Checkpoints C, D (can run in parallel with T14)
**Files:** `report/report.md` (or `.docx`), `report/report.pdf`
**Scope:** S

---

### Task 16: Clean notebook run and submission package
**Description:** Restart & Run All on the notebook (the eval sections load saved models, with no retraining needed), check every section has a markdown heading, push to GitHub, and draft the submission email for you to send.

**Acceptance criteria:**
- [x] The notebook runs top to bottom without errors, with outputs visible and headings per phase.
- [x] The GitHub repo is pushed with the README rendering correctly. No `data/` folder or oversized files are committed.
- [x] The email draft is written to `report/submission_email.txt`: subject `Final Project Submission – Yashwanth – Computer Vision Image Classification`, with the Kaggle and GitHub links in the body. Attachments are the `.ipynb` and the report PDF, and their total size is noted.

**Verification:**
- [x] Manual: Restart & Run All completes, the repo page renders, and the attachment sizes are reasonable
- [ ] **You** send the email from your LMS email address before the deadline

**Dependencies:** Tasks 14, 15
**Files:** `notebooks/image_classification.ipynb`, `report/submission_email.txt`
**Scope:** S

### ✅ Final checkpoint
- [ ] Every rubric row in `PLAN.md` is covered
- [ ] `python -m pytest -q` all green, notebook clean, app works
- [ ] Submitted

---

## Stretch goals (only after the final checkpoint)
- [ ] Fashion-MNIST run with a 10-class confusion matrix
- [ ] TFLite export of the best model: size and latency compared with `.keras`
- [ ] Test-time-augmentation toggle in the app
