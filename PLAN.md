# Final Project Plan: Computer Vision, Cats vs Dogs Image Classification

Build brief for Claude Code. Work phase by phase, and finish each phase's "Done when" check before moving on.

## Goal
Train CNNs to classify images as **Cat or Dog**, compare a from-scratch CNN against a transfer-learning model, explain predictions with Grad-CAM, and ship a Streamlit app where a user uploads an image and gets a prediction.

## Decisions (change here if needed)
- **Dataset:** Kaggle Dogs vs Cats (25k labelled training images, balanced). Download manually from Kaggle into `data/`. Claude Code can't log in for you.
  - Known issue: a few corrupt or non-image files exist, so the preprocessing step must detect and drop them.
  - Why not Fashion-MNIST as the main set: 28x28 grayscale items don't transfer to real uploaded photos, so the app demo would look weak. It can be a stretch comparison.
- **Split:** 70 / 15 / 15 (train / val / test), stratified, seed 42. Never touch the test set until final evaluation.
- **Input size:** 128x128 for the scratch CNN, 160x160 for MobileNetV2, pixels scaled to [0, 1] (or the model's own preprocess function).
- **Stack:** Python 3.10+, TensorFlow/Keras, numpy, pandas, matplotlib, seaborn, scikit-learn, Pillow, streamlit (PyTorch is an acceptable fallback if TF won't install).
- **Hardware:** use Colab (free GPU) for training. Save models to Drive, then copy them into `models/` locally.

## Project structure
```
cv-project/
├── CLAUDE.md                  # short project rules for Claude Code
├── README.md                  # setup, run, results, screenshots
├── requirements.txt
├── data/                      # Kaggle images (gitignored)
├── notebooks/image_classification.ipynb   # main deliverable
├── src/
│   ├── data.py                # load, clean, split, augmentation pipeline
│   ├── models.py              # build_scratch_cnn(), build_transfer_model()
│   ├── gradcam.py             # Grad-CAM heatmap helper
│   └── predict.py             # load model, predict_image() -> label, confidence
├── models/                    # saved .keras files
├── reports/figures/           # all saved plots
├── app/streamlit_app.py
└── report/report.pdf          # 2-3 pages
```

## Phase 0: Setup
- Create the folder, virtual env, `requirements.txt`, git repo, and `.gitignore` (ignore `data/`, `.venv`, large model files if needed).
- Download the Kaggle dataset into `data/`.
- Write a short `CLAUDE.md`: goal, structure, "modular and commented code", "seed everything with 42".
- **Done when:** dependencies install and a sample image loads and displays.

## Phase 1: Data prep and EDA
- Scan all files, **drop corrupt or unreadable images**, and log how many were removed.
- Parse labels from file names, build a DataFrame, and make the stratified 70/15/15 split.
- EDA plots: class balance, sample grid per class, image size distribution (width vs height scatter).
- Build `tf.data` pipelines: resize, normalize, batch, prefetch. Augmentation (flip, rotation, zoom, brightness) on the train set only.
- **Done when:** the three splits exist, an augmented-sample figure is saved, and the pipeline yields correct batch shapes.

## Phase 2: Models
1. **Baseline CNN (from scratch):** 3 blocks of Conv2D, MaxPooling, then Flatten, Dense, Dropout, sigmoid. No augmentation, to show overfitting.
2. **Improved CNN:** BatchNorm, Dropout and data augmentation, with the same depth. This shows the effect of regularization.
3. **Transfer learning (MobileNetV2, ImageNet weights):** freeze the base, train a new head, then optionally fine-tune the top layers at a low learning rate.
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau. Adam optimizer, binary cross-entropy.
- Save all three models to `models/`.
- **Done when:** all three train, the checkpoints are saved, and the train/val curves are plotted for each.

## Phase 3: Evaluation (test set only)
- Accuracy, precision, recall, F1, ROC-AUC, and a **confusion matrix** per model.
- **One comparison table:** model, params, test accuracy, F1, training time.
- **Error analysis:** a grid of the most confidently wrong predictions, with comments (odd angles, both animals in frame, occlusion).
- **Grad-CAM** overlays on correct and incorrect examples to show what the model looks at.
- Expected ballpark: baseline about 70 to 80%, improved CNN about 85 to 90%, MobileNetV2 above 95%. Report real numbers, not these.
- **Done when:** the comparison table, confusion matrices, error grid and Grad-CAM figures are saved to `reports/figures/`.

## Phase 4: Streamlit app (bonus)
- `src/predict.py`: `predict_image(pil_image)` returns label, confidence, and the Grad-CAM overlay.
- `app/streamlit_app.py`:
  - File uploader (jpg/png), image preview, predicted label with a confidence bar.
  - Sidebar model selector (improved CNN or MobileNetV2).
  - Toggle to show the Grad-CAM heatmap.
  - Friendly message for low confidence (below 60%) or non-cat/dog images.
- **Done when:** `streamlit run app/streamlit_app.py` works end to end with a few test photos, including a non-cat/dog image.

## Phase 5: Report (2 to 3 pages, PDF)
1. Problem statement and real-world use (retail tagging, healthcare imaging, security).
2. Dataset: Kaggle link, size, classes, cleaning steps.
3. Methods: preprocessing, augmentation, the three models.
4. Results: comparison table, training curves, confusion matrix, Grad-CAM example.
5. Limitations: binary task only, dataset bias (pose, background), non-cat/dog inputs get forced into a class.
6. Future work: multi-class breeds, a "neither" class, ONNX/TFLite deployment.
- Draft in Markdown or Docx, export to PDF, and keep it to 3 pages max.

## Phase 6: Package and submit
- **Notebook:** Restart & Run All cleanly, outputs visible, markdown headings per phase.
- **GitHub repo** with README (setup, run, screenshots, results table).
- **Email** from the same ID used for the LMS credentials:
  - **Subject:** `Final Project Submission – Yashwanth – Computer Vision Image Classification`
  - **Attachments:** `.ipynb`, report PDF (keep size small; no dataset since it's a Kaggle one)
  - **Body:** Kaggle dataset link and GitHub repo link
- Check the deadline now. Late submissions need prior permission.

## Stretch goals (only after the core works)
- Run the same pipeline on **Fashion-MNIST** (10 classes) for a multi-class confusion matrix.
- Export the best model to TFLite and compare size and speed.
- Add a test-time-augmentation option in the app.

## Rubric checklist
| Criterion | Where it's covered |
|---|---|
| Code quality | modular `src/`, comments, clean notebook |
| Accuracy | 3 models compared, test-set metrics |
| Visualisation | curves, confusion matrices, Grad-CAM, error grid |
| Report clarity | fixed 6-section structure |
| Innovation | transfer learning, Grad-CAM, augmentation study, app extras |

## Suggested prompts for Claude Code
- "Read PLAN.md. Complete Phase 0 and Phase 1 only, then stop and summarise."
- "Do Phase 2 in the notebook. Keep reusable code in `src/`. I'll train on Colab."
- "Do Phase 3: evaluation, comparison table, error analysis and Grad-CAM."
- "Build the Streamlit app from Phase 4 using the saved models."
- "Write the README and a 3-page report draft from the notebook results."
