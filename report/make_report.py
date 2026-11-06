"""Build report/report.pdf (2-3 pages) from the saved results.

    .venv/Scripts/python report/make_report.py

Numbers are read from reports/comparison_table.csv, models/*_meta.json and models/*_history.json,
so the report always matches the trained models. Report-sized figures go to report/figures/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

FIG = ROOT / "report" / "figures"
NAMES = {"baseline_cnn": "Baseline CNN", "improved_cnn": "Improved CNN", "mobilenetv2": "MobileNetV2"}
KAGGLE = "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"


# --------------------------------------------------------------------------- figures
def fig_curves(hist: dict, metas: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)
    for ax, name in zip(axes, NAMES):
        h = hist[name]
        ep = np.arange(1, len(h["accuracy"]) + 1)
        ax.plot(ep, h["accuracy"], label="train", lw=1.6)
        ax.plot(ep, h["val_accuracy"], label="val", lw=1.6)
        ax.axvline(metas[name]["best_epoch"], color="grey", ls="--", lw=0.8)
        if metas[name].get("fine_tune_from_epoch"):
            ax.axvspan(metas[name]["fine_tune_from_epoch"] - 0.5, ep[-1] + 0.5, color="gold", alpha=0.2,
                       label="fine-tune")
        ax.set_title(NAMES[name], fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("accuracy", fontsize=8)
    axes[0].set_ylim(0.5, 1.0)
    fig.tight_layout()
    out = FIG / "curves.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_cm_roc(y, probs: dict) -> Path:
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    cm_ = confusion_matrix(y, (probs["mobilenetv2"] >= 0.5).astype(int))
    pct = cm_ / cm_.sum(axis=1, keepdims=True)
    axes[0].imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{cm_[i, j]:,}\n{pct[i, j]:.1%}", ha="center", va="center", fontsize=8,
                         color="white" if pct[i, j] > 0.5 else "black")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
    axes[0].set_xlabel("predicted", fontsize=8)
    axes[0].set_ylabel("true", fontsize=8)
    axes[0].set_title("MobileNetV2 confusion matrix", fontsize=9)
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, lw=1.5, label=f"{NAMES[name]} ({roc_auc_score(y, p):.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[1].set_title("ROC curves (AUC)", fontsize=9)
    axes[1].set_xlabel("false positive rate", fontsize=8)
    axes[1].set_ylabel("true positive rate", fontsize=8)
    axes[1].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.tick_params(labelsize=7)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG / "cm_roc.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_gradcam(image_dir, test_df, probs, models, metas) -> Path:
    """Same photo through MobileNetV2 and the improved CNN, plus two MobileNetV2 mistakes."""
    from src.gradcam import make_gradcam_heatmap, overlay
    from src.predict import preprocess

    # The same "correct" cat the notebook's Grad-CAM grid starts with (same selection rule).
    scored = test_df.assign(p=probs["mobilenetv2"])
    correct_cat = scored[(scored.label == "cat") & (scored.p < 0.5)].sample(3, random_state=42).iloc[0]
    panels = [
        (correct_cat["path"], "mobilenetv2", "MobileNetV2: on the cat"),
        (correct_cat["path"], "improved_cnn", "Improved CNN: on the cage"),
        (test_df.loc[277, "path"], "mobilenetv2", "#277 labelled cat: finds the dog"),
        (test_df.loc[2669, "path"], "mobilenetv2", "#2669 curled puppies -> 'cat'"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.3))
    for ax, (path, name, title) in zip(axes, panels):
        img = Image.open(Path(image_dir) / path).convert("RGB")
        heat = make_gradcam_heatmap(models[name], preprocess(img, metas[name]["img_size"]), metas[name]["gradcam_layer"])
        ax.imshow(overlay(img, heat))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out = FIG / "gradcam.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- document
def build_pdf(table: pd.DataFrame, metas: dict, figs: dict, n_dropped: dict, split_sizes: dict) -> Path:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica", fontSize=9.3, leading=12.2,
                          spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                        spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1f3b57"))
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=16, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
    cap = ParagraphStyle("cap", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444444"),
                         alignment=TA_CENTER, spaceAfter=6)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, bulletIndent=0, spaceAfter=2)

    def P(text, style=body):
        return Paragraph(text, style)

    def B(text):
        return Paragraph(text, bullet, bulletText="•")

    def figure(path, width_cm, caption):
        w, h = Image.open(path).size
        return KeepTogether([RLImage(str(path), width=width_cm * cm, height=width_cm * cm * h / w), P(caption, cap)])

    t = table
    mb, imp, base = t.loc["mobilenetv2"], t.loc["improved_cnn"], t.loc["baseline_cnn"]
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    dropped_txt = ", ".join(f"{v} {k}" for k, v in n_dropped.items())

    s = []
    s.append(P("Cats vs Dogs: Image Classification with CNNs", title))
    s.append(P("Computer Vision Final Project &nbsp;·&nbsp; Yashwanth &nbsp;·&nbsp; November 2026", sub))
    s.append(Spacer(1, 4))

    s.append(P("1. Problem statement and real-world use", h1))
    s.append(P(
        "The task is to decide whether a photo shows a <b>cat</b> or a <b>dog</b>. It's a small binary problem, but "
        "it contains the core difficulties of real image classification: large variation in pose, lighting, "
        "background and image quality, occlusion, and noisy labels. The same pipeline (clean data, a CNN or "
        "transfer-learned backbone, honest held-out evaluation, explanation) underlies <b>retail product "
        "tagging</b> (auto-labelling catalogue photos), <b>healthcare imaging</b> (flagging scans for review, "
        "where explainability matters as much as accuracy), and <b>security</b> (detecting people or vehicles in "
        "camera feeds). We compare a CNN trained from scratch, a regularized CNN and MobileNetV2 transfer "
        "learning, explain predictions with Grad-CAM, and deploy the best model in a Streamlit app."))

    s.append(P("2. Dataset", h1))
    s.append(P(
        f"Microsoft PetImages, the Kaggle Cats vs Dogs set (<link href='{KAGGLE}' color='blue'>{KAGGLE}</link>, "
        "CDLA-Permissive-2.0): 25,000 photos, 12,500 per class, in <i>Cat/</i> and <i>Dog/</i> folders. "
        f"<b>Cleaning:</b> every file was fully decoded with Pillow and with TensorFlow's JPEG decoder (the one "
        f"used in training). <b>233 of 25,002 files were dropped</b>: {dropped_txt}. Most were other image formats saved "
        "with a .jpg name, and two were Windows Thumbs.db files. Images vary from 50 px to 500 px (median 448×375), so all "
        f"are resized to a square input. <b>Split:</b> stratified 70/15/15 with seed 42, giving "
        f"{split_sizes['train']:,} / {split_sizes['val']:,} / {split_sizes['test']:,} images, balanced within 0.2%. "
        "The test split was used only once, for the final evaluation."))

    s.append(P("3. Methods", h1))
    s.append(P(
        "<b>Preprocessing</b> is a layer inside each model: rescaling to [0, 1] for the CNNs and [-1, 1] for "
        "MobileNetV2. Training, evaluation and the app therefore all feed raw RGB pixels. <b>Augmentation</b> "
        "(training batches only): random horizontal flip, rotation (±29°), zoom (±15%) and brightness (±15%). "
        "All models use Adam, binary cross-entropy, and EarlyStopping / ModelCheckpoint / ReduceLROnPlateau "
        "on validation loss."))
    arch = [
        ["", "Baseline CNN", "Improved CNN", "MobileNetV2"],
        ["Input", "128×128", "128×128", "160×160"],
        ["Body", "3×[Conv-ReLU-MaxPool]\n32/64/128 filters", "same + BatchNorm,\nDropout 0.1 per block",
         "ImageNet MobileNetV2\n(frozen, then top 30 fine-tuned)"],
        ["Head", "Flatten-Dense128-\nDropout0.5-sigmoid", "same + BatchNorm\non Dense", "GAP-Dropout0.2-sigmoid"],
        ["Augmentation", "no", "yes", "yes"],
        ["Learning rate", "1e-3", "3e-4", "1e-3, then 1e-5"],
    ]
    ta = Table(arch, colWidths=[2.6 * cm, 4.4 * cm, 4.4 * cm, 5.4 * cm])
    ta.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 1), (0, -1), "Helvetica-Bold", 7.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    s.append(ta)
    s.append(Spacer(1, 4))
    s.append(P(
        "<b>Getting the improved CNN right took three runs.</b> Run 1 (BatchNorm + Dropout 0.2 per block, lr 1e-3) "
        "<i>underfit</i> and scored below the baseline. Diagnosis showed the BatchNorm-scaled 32,768 flattened "
        "features blew up the 4.2M-weight Dense layer (its ReLUs died and the model stuck at 50%), on both Keras "
        "backends. Adding BatchNorm after the Dense layer, BN momentum 0.9 and lr 3e-4 fixed it. Run 3 then "
        "relaxed the LR scheduler, which in run 2 had halved the rate five times on the noisy validation curve. "
        "<b>Hardware:</b> TensorFlow has no native-Windows GPU support, so the GPU models trained with Keras 3 "
        "on the PyTorch backend (RTX 5060 Laptop GPU). The baseline trained on CPU. The saved models reload "
        "under TensorFlow with identical test metrics. <b>Deployment:</b> a Streamlit app shows the label, the "
        "confidence and a Grad-CAM overlay. Because a binary model has to say cat or dog even for a car, a "
        "stock ImageNet MobileNetV2 first checks that one of its top-5 classes is a cat or dog breed."))

    s.append(P("4. Results", h1))
    rows = [["Model", "Test acc.", "Precision", "Recall", "F1", "ROC-AUC", "Params", "Epochs", "Train time"]]
    for name in NAMES:
        r = t.loc[name]
        rows.append([NAMES[name], pct(r.accuracy), f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}",
                     f"{r.roc_auc:.3f}", f"{r.params / 1e6:.2f}M", str(r.epochs),
                     f"{r.train_min:.1f} min {'GPU' if 'GPU' in r.trained_on else 'CPU'}"])
    tr = Table(rows, colWidths=[2.9 * cm, 1.7 * cm, 1.7 * cm, 1.5 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 2.6 * cm])
    tr.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 3), (-1, 3), "Helvetica-Bold", 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#eaf6ea")), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    s.append(KeepTogether([tr, P(f"Table 1. Test-set results ({split_sizes['test']:,} images; dog = positive class).", cap)]))
    s.append(P(
        f"<b>MobileNetV2 is clearly best</b> with {pct(mb.accuracy)} accuracy and AUC {mb.roc_auc:.3f}, using half the "
        "parameters of the scratch CNNs. Its ImageNet features reach about 98% validation accuracy after a single epoch, "
        f"and fine-tuning adds the rest. The <b>baseline overfits</b>: training accuracy rises to "
        f"{pct(max(hist['baseline_cnn']['accuracy']))} while validation stalls at about 85% (Fig. 1). The "
        f"<b>improved CNN</b> keeps train and validation within a few points and has lower validation loss "
        f"({metas['improved_cnn']['best_val_loss']:.3f} vs {metas['baseline_cnn']['best_val_loss']:.3f}). "
        f"Regularization fixed the overfitting, but test accuracy rose only from {pct(base.accuracy)} to "
        f"{pct(imp.accuracy)}."))
    s.append(figure(figs["curves"], 17, "Figure 1. Train vs validation accuracy (dashed line: best epoch, kept by checkpointing; "
                                        "shaded: MobileNetV2 fine-tuning)."))
    s.append(figure(figs["cm_roc"], 12.5, "Figure 2. MobileNetV2 confusion matrix (row %) and ROC curves for all three models."))
    s.append(P(
        "<b>Error analysis.</b> The most confidently wrong test images fall into clear groups. (1) <i>Both species "
        "or label noise:</i> #277 (labelled cat) shows a dog next to a cat, and #2082 (labelled dog) is a town logo "
        "with no animal at all. (2) <i>Puppies and curled-up poses</i> look cat-like. (3) <i>Cage bars and "
        "occlusion.</i> (4) <i>Black fur in harsh light, face hidden.</i> (5) <i>A small animal in a busy scene with "
        "people.</i> <b>Grad-CAM</b> (Fig. 3) shows MobileNetV2 attending to the animal, even on its mistakes. "
        "On #277 it highlights the dog. The scratch CNN often focuses on <b>background shortcuts</b> such as cage "
        "bars, picture frames and carpet, which explains its brittleness and is the strongest argument for "
        "transfer learning on a dataset of this size."))
    s.append(figure(figs["gradcam"], 17, "Figure 3. Grad-CAM (red = most influential). Same cat photo through both "
                                         "models, then two MobileNetV2 mistakes."))

    s.append(P("5. Limitations", h1))
    for txt in [
        "<b>Binary task only.</b> Any input is forced into cat or dog. The ImageNet guard catches obvious non-pets "
        "(on three wallpaper photos both models gave 61-92% confident answers, and the guard flagged all three), but "
        "it depends on ImageNet's classes.",
        "<b>Dataset bias.</b> Mostly indoor pet photos from one adoption website. Poses, cages and backgrounds "
        "correlate with labels, and the scratch CNNs exploit this. A few images are mislabelled, so the "
        "measured error is slightly pessimistic.",
        "<b>Single split, single seed.</b> No cross-validation or confidence intervals, so small gaps such as "
        f"baseline vs improved CNN (+{100 * (imp.accuracy - base.accuracy):.1f} points) are not conclusive.",
    ]:
        s.append(B(txt))

    s.append(P("6. Future work", h1))
    for txt in [
        "Multi-class <b>breed</b> classification (Oxford-IIIT Pet, 37 classes) and a learned <b>\"neither\"</b> class "
        "trained on real negative images instead of an external guard.",
        "<b>TFLite / ONNX export</b> of MobileNetV2 for on-device inference, comparing size and latency.",
        "Clean the mislabelled images, add test-time augmentation, and report multi-seed confidence intervals.",
    ]:
        s.append(B(txt))

    out = ROOT / "report" / "report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.4 * cm, title="Cats vs Dogs: Image Classification",
                            author="Yashwanth")
    doc.build(s)
    return out


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    import keras

    from src.data import find_pet_images_dir, load_split, make_dataset
    from src.evaluate import predict_split

    FIG.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv("reports/comparison_table.csv", index_col="model")
    metas = {n: json.loads(Path(f"models/{n}_meta.json").read_text()) for n in NAMES}
    hist = {n: json.loads(Path(f"models/{n}_history.json").read_text()) for n in NAMES}
    image_dir = find_pet_images_dir("data")
    test_df = load_split("test")
    models = {n: keras.models.load_model(f"models/{n}.keras", compile=False) for n in NAMES}
    probs = {n: predict_split(models[n], make_dataset(test_df, image_dir, metas[n]["img_size"], batch_size=64))
             for n in NAMES}
    dropped = pd.read_csv("data/splits/dropped.csv")["reason"].str.extract(r"\((\w+)\)")[0]
    n_dropped = {"BMP": 0, "GIF": 0, "PNG": 0, "PSD": 0}
    n_dropped.update(dropped[dropped.isin(["BMP", "GIF", "PNG", "PSD"])].value_counts().to_dict())
    n_dropped = {k: int(v) for k, v in n_dropped.items()}
    n_dropped["unreadable"] = int((~dropped.isin(["BMP", "GIF", "PNG", "PSD"])).sum())
    split_sizes = {k: len(load_split(k)) for k in ("train", "val", "test")}

    # Exact metrics from the predictions (the CSV is rounded to 4 decimals; re-rounding it gives off-by-one digits).
    from src.evaluate import compute_metrics
    for n in NAMES:
        for k, v in compute_metrics(test_df["label_id"].to_numpy(), probs[n]).items():
            table.loc[n, k] = v
    figs = {"curves": fig_curves(hist, metas), "cm_roc": fig_cm_roc(test_df["label_id"].to_numpy(), probs),
            "gradcam": fig_gradcam(image_dir, test_df, probs, models, metas)}
    out = build_pdf(table, metas, figs, n_dropped, split_sizes)

    from pypdf import PdfReader

    print(f"Wrote {out}: {len(PdfReader(out).pages)} pages, {out.stat().st_size / 1e6:.2f} MB")
