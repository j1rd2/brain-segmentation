import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "prepared_dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAME = "threshold_baseline"
MODEL_DIR = MODELS_DIR / MODEL_NAME
MODEL_RESULTS_DIR = RESULTS_DIR / MODEL_NAME

N_FOLDS = 5
THRESHOLD = 0.55  # threshold applied on normalized FLAIR channel

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def is_mask_file(path: Path) -> bool:
    return path.stem.endswith("_mask")


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()

    if max_val - min_val == 0:
        return np.zeros_like(img, dtype=np.float32)

    return (img - min_val) / (max_val - min_val)


def load_image_and_mask(img_path: Path, mask_path: Path):
    image = tiff.imread(str(img_path))
    mask = tiff.imread(str(mask_path))

    # Convert mask from 0/255 to 0/1
    mask = (mask > 0).astype(np.uint8)

    return image, mask


def threshold_segmentation(image: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    """
    Uses the FLAIR channel (channel index 1) and fixed thresholding.
    """
    flair = image[:, :, 1]
    flair = normalize_minmax(flair)
    pred_mask = (flair >= threshold).astype(np.uint8)
    return pred_mask


def confusion_from_masks(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def compute_metrics(tp: int, fp: int, tn: int, fn: int):
    eps = 1e-8

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Pixel Accuracy": pixel_accuracy,
    }


def collect_pairs(split_dir: Path):
    """
    Collect image/mask pairs recursively from a split directory.
    Example:
        data/prepared_dataset/fold_1/val/<patient_id>/xxx.tif
        data/prepared_dataset/fold_1/val/<patient_id>/xxx_mask.tif
    """
    all_tifs = sorted(split_dir.rglob("*.tif"))
    image_files = [p for p in all_tifs if not is_mask_file(p)]

    pairs = []
    for img_path in image_files:
        mask_path = img_path.with_name(img_path.stem + "_mask.tif")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"Warning: mask not found for {img_path}")

    return pairs


def save_threshold_model():
    model_info = {
        "model_name": MODEL_NAME,
        "method": "fixed_threshold_segmentation",
        "input_channel": 1,
        "input_channel_name": "FLAIR",
        "normalization": "minmax_per_image",
        "threshold": THRESHOLD,
        "output_classes": {
            "0": "background",
            "1": "lesion"
        },
        "notes": (
            "This is a non-trainable classical baseline. "
            "No learned weights are generated. This file stores the baseline configuration."
        )
    }

    with open(MODEL_DIR / f"{MODEL_NAME}_config.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4)


def save_text_summary(summary_df: pd.DataFrame, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{MODEL_NAME.upper()} SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")


# =========================================================
# FOLD EVALUATION
# =========================================================
def evaluate_fold(fold_num: int, threshold: float):
    fold_dir = DATASET_DIR / f"fold_{fold_num}" / "val"

    if not fold_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {fold_dir}")

    pairs = collect_pairs(fold_dir)
    print(f"Fold {fold_num}: found {len(pairs)} validation pairs")

    per_image_rows = []

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for img_path, mask_path in pairs:
        image, y_true = load_image_and_mask(img_path, mask_path)
        y_pred = threshold_segmentation(image, threshold=threshold)

        tp, fp, tn, fn = confusion_from_masks(y_true, y_pred)
        metrics = compute_metrics(tp, fp, tn, fn)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        per_image_rows.append({
            "fold": fold_num,
            "patient": img_path.parent.name,
            "image_name": img_path.name,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "IoU": metrics["IoU"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "Pixel Accuracy": metrics["Pixel Accuracy"],
        })

    fold_metrics = compute_metrics(total_tp, total_fp, total_tn, total_fn)

    fold_summary = {
        "fold": fold_num,
        "TP": total_tp,
        "FP": total_fp,
        "TN": total_tn,
        "FN": total_fn,
        "IoU": fold_metrics["IoU"],
        "Precision": fold_metrics["Precision"],
        "Recall": fold_metrics["Recall"],
        "Pixel Accuracy": fold_metrics["Pixel Accuracy"],
    }

    per_image_df = pd.DataFrame(per_image_rows)
    return per_image_df, fold_summary


def evaluate_final_test(threshold: float):
    test_dir = DATASET_DIR / "final_test"

    if not test_dir.exists():
        print("Final test directory not found. Skipping final test evaluation.")
        return None, None

    pairs = collect_pairs(test_dir)
    print(f"Final test: found {len(pairs)} image-mask pairs")

    per_image_rows = []

    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for img_path, mask_path in pairs:
        image, y_true = load_image_and_mask(img_path, mask_path)
        y_pred = threshold_segmentation(image, threshold=threshold)

        tp, fp, tn, fn = confusion_from_masks(y_true, y_pred)
        metrics = compute_metrics(tp, fp, tn, fn)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        per_image_rows.append({
            "split": "final_test",
            "patient": img_path.parent.name,
            "image_name": img_path.name,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "IoU": metrics["IoU"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "Pixel Accuracy": metrics["Pixel Accuracy"],
        })

    test_metrics = compute_metrics(total_tp, total_fp, total_tn, total_fn)

    test_summary = {
        "split": "final_test",
        "TP": total_tp,
        "FP": total_fp,
        "TN": total_tn,
        "FN": total_fn,
        "IoU": test_metrics["IoU"],
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "Pixel Accuracy": test_metrics["Pixel Accuracy"],
    }

    per_image_df = pd.DataFrame(per_image_rows)
    return per_image_df, test_summary


# =========================================================
# MAIN
# =========================================================
def main():
    save_threshold_model()

    all_fold_summaries = []
    all_per_image = []

    # -------------------------
    # Cross-validation folds
    # -------------------------
    for fold_num in range(1, N_FOLDS + 1):
        per_image_df, fold_summary = evaluate_fold(fold_num, threshold=THRESHOLD)

        all_per_image.append(per_image_df)
        all_fold_summaries.append(fold_summary)

        per_image_df.to_csv(
            MODEL_RESULTS_DIR / f"fold_{fold_num}_per_image_metrics.csv",
            index=False
        )

    summary_df = pd.DataFrame(all_fold_summaries)

    # Global CV totals
    global_tp = int(summary_df["TP"].sum())
    global_fp = int(summary_df["FP"].sum())
    global_tn = int(summary_df["TN"].sum())
    global_fn = int(summary_df["FN"].sum())

    global_metrics = compute_metrics(global_tp, global_fp, global_tn, global_fn)

    global_row = pd.DataFrame([{
        "fold": "global_cv",
        "TP": global_tp,
        "FP": global_fp,
        "TN": global_tn,
        "FN": global_fn,
        "IoU": global_metrics["IoU"],
        "Precision": global_metrics["Precision"],
        "Recall": global_metrics["Recall"],
        "Pixel Accuracy": global_metrics["Pixel Accuracy"],
    }])

    summary_with_global = pd.concat([summary_df, global_row], ignore_index=True)

    # Save fold summary CSV
    summary_with_global.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_cv_summary.csv",
        index=False
    )

    # Save all fold per-image CSV
    all_per_image_df = pd.concat(all_per_image, ignore_index=True)
    all_per_image_df.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_all_per_image_cv.csv",
        index=False
    )

    # -------------------------
    # Final test evaluation
    # -------------------------
    final_test_per_image_df, final_test_summary = evaluate_final_test(threshold=THRESHOLD)

    if final_test_per_image_df is not None and final_test_summary is not None:
        final_test_per_image_df.to_csv(
            MODEL_RESULTS_DIR / f"{MODEL_NAME}_final_test_per_image.csv",
            index=False
        )

        final_test_summary_df = pd.DataFrame([final_test_summary])
        final_test_summary_df.to_csv(
            MODEL_RESULTS_DIR / f"{MODEL_NAME}_final_test_summary.csv",
            index=False
        )
    else:
        final_test_summary_df = pd.DataFrame()

    # -------------------------
    # Combined summary table
    # -------------------------
    combined_summary_df = summary_with_global.copy()

    if not final_test_summary_df.empty:
        final_test_row = pd.DataFrame([{
            "fold": "final_test",
            "TP": int(final_test_summary["TP"]),
            "FP": int(final_test_summary["FP"]),
            "TN": int(final_test_summary["TN"]),
            "FN": int(final_test_summary["FN"]),
            "IoU": final_test_summary["IoU"],
            "Precision": final_test_summary["Precision"],
            "Recall": final_test_summary["Recall"],
            "Pixel Accuracy": final_test_summary["Pixel Accuracy"],
        }])
        combined_summary_df = pd.concat([combined_summary_df, final_test_row], ignore_index=True)

    combined_summary_df.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_summary_table.csv",
        index=False
    )

    save_text_summary(
        combined_summary_df,
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_summary_table.txt"
    )

    # -------------------------
    # Console output
    # -------------------------
    print("\n=== Threshold Baseline Summary Table ===")
    print(combined_summary_df.to_string(index=False))

    print(f"\nModel config saved in: {MODEL_DIR / f'{MODEL_NAME}_config.json'}")
    print(f"Results saved in: {MODEL_RESULTS_DIR}")


if __name__ == "__main__":
    main()