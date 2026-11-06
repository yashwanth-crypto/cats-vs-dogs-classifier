# Implementation Plan: Cats vs Dogs Image Classifier

> Replanned from `PLAN.md` (2026-11-06). `PLAN.md` stays as the original brief; this file plus `tasks/todo.md` are now the working plan. Detailed tasks live in `tasks/todo.md`.

## Overview
Train three Keras CNNs on Kaggle Dogs vs Cats (baseline from scratch, regularized CNN, MobileNetV2 transfer), evaluate them on a held-out test set, explain predictions with Grad-CAM, and ship a Streamlit app where a user uploads a photo and gets a prediction. Deliverables: notebook, `src/` package, saved models, figures, Streamlit app, README, 2-3 page PDF report, submission email.

The main change from `PLAN.md` is the order. The original went phase by phase (all data work, then all models, then the app). This plan builds a **thin end-to-end path first**: a small data subset, a baseline model trained for one epoch on Colab, loaded locally, and served by a bare Streamlit page. That surfaces the riskiest problems (TF install, Colab-to-local model compatibility, path handling) in the first few hours, not in week 3. Each later task then fills in one part of that working path.

## What changed from PLAN.md, and why
| Change | Reason |
|---|---|
| Project root is `C:\Image classifier\` (not `cv-project/`) | That's the folder that exists. |
| **Python 3.11 venv**, not the default Python 3.14 | Python 3.14 is the machine default and TensorFlow doesn't support it. 3.11 is already installed (`py -V:Astral/CPython3.11.15`). |
| **Pin `tensorflow` to the version Colab runs** | A `.keras` file saved by a newer Keras on Colab may fail to load under an older local Keras. Matching versions removes this risk. |
| Splits saved to `data/splits/{train,val,test}.csv` | Colab and local machine use the exact same test set, and the split is reproducible without re-running the cleaning step. |
| Preprocessing goes **inside each model** (`Rescaling` layer, or `mobilenet_v2.preprocess_input` wrapped in a layer) | `predict.py` and the app then feed raw 0-255 pixels to any model. That removes a whole class of train/serve mismatch bugs. |
| New `src/train.py` (shared callbacks, timing, history to JSON) | Each of the 3 models is then a few lines in the notebook. Training time is recorded the same way for the comparison table. |
| New `src/evaluate.py` | Metrics, confusion matrix, and the error grid as reusable functions, not loose notebook cells. |
| `pytest` tests on a tiny fixture folder (including one corrupt file) | Quick proof that cleaning, splitting, shapes, Grad-CAM and predict work, without needing the 25k images. |
| "Not a cat/dog" check uses an **ImageNet MobileNetV2 guard**, not just the 60% threshold | A binary sigmoid is often >90% confident on a car photo, so the threshold alone won't catch most non-pets. ImageNet already has cat and dog classes, so a quick top-5 check works well. This is Task 13. |
| Email is drafted, **sent by you** | It goes from your LMS email address, and Claude doesn't send email for you. |

## Architecture decisions
- **Stack:** Python 3.11, TensorFlow/Keras (pinned to Colab's version), numpy, pandas, matplotlib, seaborn, scikit-learn, Pillow, streamlit, pytest.
- **Data:** Microsoft PetImages (`PetImages/Cat/*.jpg`, `PetImages/Dog/*.jpg`), labels taken from folder names. The data stays in `data/` (gitignored). Colab downloads its own copy with the same command.
- **Split:** stratified 70/15/15, seed 42, saved to CSV. The test set is used only in Phase 4.
- **Input sizes:** 128×128 for the scratch CNNs, 160×160 for MobileNetV2.
- **Training:** on Colab GPU. The notebook clones or mounts the repo and imports `src/`. Models are saved to Drive and copied into `models/`.
- **Model files:** `models/{baseline_cnn,improved_cnn,mobilenetv2}.keras` plus `models/{name}_history.json` and `models/{name}_meta.json` (input size, train time, params, Keras version).
- **App inference** runs on CPU locally, which is fine for single images.

## Dependency graph
```
T1 env + skeleton
 └─ T2 clean + split  ─────────────────────────────┐
     └─ T3 pipeline + baseline + train helper (Colab smoke)  │
         └─ T4 predict.py v0 + bare Streamlit  ◄── CHECKPOINT A (end-to-end works)
             │
             ├─ T5 EDA + augmentation ──┐
             │                          ├─ T7 improved CNN
             ├─ T6 baseline full train  │
             │                          └─ T8 MobileNetV2 transfer
             │                                   ◄── CHECKPOINT B (3 models saved)
             ├─ T9 evaluate: metrics + CM + comparison table (needs T6-T8)
             ├─ T10 error-analysis grid (needs T9)
             └─ T11 gradcam.py + figures (needs T6-T8)
                                                 ◄── CHECKPOINT C (all figures)
                 T12 predict.py full + app full (needs T11)
                 T13 ImageNet "neither" guard (needs T12)
                                                 ◄── CHECKPOINT D (app done)
                 T14 README   T15 report PDF   (need C, D)
                 T16 notebook clean run + submission pack
                                                 ◄── FINAL
```

## Task list (index; full detail in `tasks/todo.md`)

### Phase 1: Foundation and end-to-end path
- [x] Task 1: Environment, repo skeleton, CLAUDE.md
- [x] Task 2: Clean dataset and make the stratified split
- [x] Task 3: tf.data pipeline, baseline CNN, training helper (Colab smoke run)
- [x] Task 4: `predict_image` v0 and a bare Streamlit page

### Checkpoint A: one path works end to end
- [ ] A model trained on Colab loads locally and the app labels an uploaded photo

### Phase 2: Data and models
- [x] Task 5: EDA figures and the train-only augmentation layer
- [x] Task 6: Baseline CNN full training (no augmentation)
- [x] Task 7: Improved CNN (BatchNorm, Dropout, augmentation)
- [x] Task 8: MobileNetV2 transfer learning (frozen head, then fine-tune)

### Checkpoint B: three models saved with curves

### Phase 3: Evaluation and explainability
- [x] Task 9: Test-set metrics, confusion matrices, comparison table
- [x] Task 10: Most-confidently-wrong error grid with commentary
- [x] Task 11: Grad-CAM helper and overlay figures

### Checkpoint C: all figures in `reports/figures/`

### Phase 4: App
- [x] Task 12: Full `predict.py` and the full Streamlit UI
- [x] Task 13: ImageNet "not a cat or dog" guard

### Checkpoint D: app works with cat, dog, and non-pet photos

### Phase 5: Write-up and submission
- [x] Task 14: README with setup, results table, screenshots
- [x] Task 15: 2-3 page report PDF
- [ ] Task 16: Clean notebook run and submission package

### Checkpoint: Final

## Parallelization
- After Checkpoint A: **T5** (EDA) and **T6** (baseline full train) are independent.
- After Checkpoint B: **T9→T10** and **T11** are independent (different files).
- **T14** and **T15** can be written in parallel once Checkpoints C and D pass.
- Colab training (T6-T8) is wall-clock heavy, so draft the notebook markdown and the report skeleton while models train.

## Risks and mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| TensorFlow won't install on Python 3.14 (machine default) | High | Use a Python 3.11 venv from day one (T1). Fallback: `tensorflow-cpu` or PyTorch, as `PLAN.md` allows. |
| A `.keras` file from Colab doesn't load locally (Keras version skew) | High | Pin the local TF to Colab's version. T3 proves the round-trip before any long training run. |
| Corrupt or non-JPEG files crash the pipeline mid-epoch | Med | T2 fully decodes each file with PIL (`verify()` + `load()`) and drops failures. A test fixture includes a corrupt file. |
| Colab session timeouts lose training progress | Med | ModelCheckpoint to Drive, EarlyStopping, and one model per Colab session. |
| Grad-CAM on MobileNetV2 fails because the base model is nested | Med | `gradcam.py` takes an explicit layer name and handles nested models. T11 tests it on both model types. |
| A non-pet photo gets a confident "Dog" | Med | T13 adds an ImageNet guard. The README and report state the limitation either way. |
| Data leakage (test images seen during tuning) | High | The test CSV is used only in T9-T11. Callbacks monitor `val_*`. |
| Report runs past 3 pages | Low | Fixed 6-section outline with a word budget per section (T15). |

## Training location (updated 2026-11-06)
**GPU:** RTX 5060 Laptop GPU (8 GB) through **Keras 3 on the PyTorch backend** (`KERAS_BACKEND=torch`, torch 2.14.1+cu130). TensorFlow has no native-Windows GPU support and WSL isn't installed. The model code is unchanged, and models saved from the GPU run reload under TensorFlow with identical predictions (checked with `src.train verify`). The improved CNN is about 4× faster than CPU, and MobileNetV2 is limited by the CPU data pipeline. The baseline CNN was trained on CPU before the switch, which makes no difference to the results.

### Earlier note (Task 3, CPU)
TensorFlow has no GPU support on native Windows, but this PC's CPU is fast: 1 epoch of 1,000 images at 128px takes 7 s, so a full 17k-image epoch is about 2 min. The scratch CNNs (about 15-20 epochs each) can train locally in about 30-45 min each. MobileNetV2 with a frozen base is also feasible on CPU. Colab stays supported: the notebook clones the repo, installs the pinned versions, downloads the data and copies models to Drive.

## Improved CNN diagnosis (2026-11-06)
The first improved-CNN run (BatchNorm after each conv + Dropout 0.2 per block + head Dropout 0.5 + augmentation, Adam 1e-3) reached only **83.3% val, below the baseline's 85.2%**. Training accuracy stayed at about 66%. Archived in `models/failed_runs/` and `logs/improved_cnn_v1_underfit.log`.

What the diagnosis found:
1. Augmentation is correct under both backends (every sample in a batch differs, and batches differ).
2. Removing the per-block dropout made it **worse**: train and val stuck at exactly 50.1% (one class forever). BatchNorm-scaled conv features (32,768 values after Flatten) blow up the 4.2M-weight Dense layer at lr 1e-3, and its ReLUs die. Reproduced under the **TensorFlow backend too**, so it's not a PyTorch-backend bug.
3. Adding BatchNorm after the Dense layer let it train, but inference-mode accuracy lagged training-mode accuracy. BatchNorm's running statistics (momentum 0.99) were too slow to follow the weights.

**Fix:** BatchNorm after the Dense layer as well, BN momentum 0.9, Adam 3e-4, block Dropout 0.1, and patience 8 / max 50 epochs. The depth and filter counts are unchanged. Worth a sentence in the report: "regularization" helps only when the optimization stays stable.

## Resolved questions (2026-11-06)
1. **Deadline:** as soon as possible. Build straight through, and leave stretch goals until the end.
2. **Dataset:** **Microsoft PetImages** (Kaggle mirror: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset). Labels are the `Cat/` and `Dog/` folders, about 25k images, and the set has known corrupt files, which gives the cleaning step real work to do. It downloads straight from Microsoft without a Kaggle login, so both this PC and Colab can fetch it with one command. T2 parses labels from folder names.
3. **GitHub:** public repo. The `gh` CLI isn't installed, so in T16 you either create the empty repo on github.com or install `gh` and log in. After that I push.
4. **Task 13:** in scope, no longer optional.
