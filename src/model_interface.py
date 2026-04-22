from pathlib import Path
import argparse
import json

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import tensorflow as tf


# =========================================================
# OPTIONAL CUSTOM OBJECTS
# =========================================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


CUSTOM_OBJECTS = {
    "dice_coefficient": dice_coefficient,
    "dice_loss": dice_loss,
    "bce_dice_loss": bce_dice_loss,
}


# =========================================================
# PREPROCESSING
# =========================================================
def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    out = np.zeros_like(image, dtype=np.float32)

    for c in range(image.shape[-1]):
        ch = image[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()

        if ch_max - ch_min == 0:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = (ch - ch_min) / (ch_max - ch_min)

    return out


def ensure_image_shape(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C) or (H, W). Got: {image.shape}")

    return image


# =========================================================
# METRICS
# =========================================================
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
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "Pixel Accuracy": float(pixel_accuracy),
    }


# =========================================================
# MODEL LOADING
# =========================================================
def load_keras_model(model_path: Path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False
        )


# =========================================================
# PREDICTION
# =========================================================
def predict_mask(model, image: np.ndarray, threshold: float = 0.5):
    image = ensure_image_shape(image)
    image_norm = normalize_image(image)
    x = np.expand_dims(image_norm, axis=0)

    pred = model.predict(x, verbose=0)[0]

    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    pred_mask = (pred >= threshold).astype(np.uint8)

    return pred, pred_mask


# =========================================================
# SAVING
# =========================================================
def save_outputs(
    image: np.ndarray,
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    output_dir: Path,
    base_name: str,
    overlay_channel: int = 1,
    save_npy: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Binary mask
    tiff.imwrite(str(output_dir / f"{base_name}_pred_mask.tif"), (pred_mask * 255).astype(np.uint8))
    plt.imsave(output_dir / f"{base_name}_pred_mask.png", pred_mask, cmap="gray")

    # Probability map
    plt.imsave(output_dir / f"{base_name}_probability_map.png", pred_prob, cmap="gray")

    if save_npy:
        np.save(output_dir / f"{base_name}_probability_map.npy", pred_prob)

    # Overlay
    if image.ndim == 2:
        base_img = image
    else:
        if overlay_channel < 0 or overlay_channel >= image.shape[-1]:
            raise ValueError(
                f"overlay_channel={overlay_channel} is invalid for image with shape {image.shape}"
            )
        base_img = image[:, :, overlay_channel]

    plt.figure(figsize=(6, 6))
    plt.imshow(base_img, cmap="gray")
    plt.imshow(pred_mask, cmap="Reds", alpha=0.35)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_overlay.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_metrics(metrics: dict, output_dir: Path, base_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"{base_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # also save simple CSV row
    import csv
    csv_path = output_dir / f"{base_name}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Predict segmentation mask for one MRI .tif image using a Keras .h5 model."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 model")
    parser.add_argument("--image", type=str, required=True, help="Path to input .tif image")
    parser.add_argument("--output", type=str, default="results/inference", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")
    parser.add_argument(
        "--overlay-channel",
        type=int,
        default=1,
        help="Channel index to use for overlay background image (default: 1, FLAIR)"
    )
    parser.add_argument(
        "--true-mask",
        type=str,
        default=None,
        help="Optional path to ground-truth mask .tif for metric computation"
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the raw probability map as .npy"
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Loading model from: {model_path}")
    model = load_keras_model(model_path)

    print(f"Reading image: {image_path}")
    image = tiff.imread(str(image_path))
    image = ensure_image_shape(image)

    print(f"Image shape: {image.shape}")
    print(f"Using threshold: {args.threshold}")

    pred_prob, pred_mask = predict_mask(model, image, threshold=args.threshold)

    save_outputs(
        image=image,
        pred_prob=pred_prob,
        pred_mask=pred_mask,
        output_dir=output_dir,
        base_name=image_path.stem,
        overlay_channel=args.overlay_channel,
        save_npy=args.save_npy,
    )

    print("Prediction completed.")
    print(f"Saved outputs in: {output_dir}")

    if args.true_mask is not None:
        true_mask_path = Path(args.true_mask)
        if not true_mask_path.exists():
            raise FileNotFoundError(f"True mask not found: {true_mask_path}")

        true_mask = tiff.imread(str(true_mask_path))
        true_mask = (true_mask > 0).astype(np.uint8)

        if true_mask.shape != pred_mask.shape:
            raise ValueError(
                f"Shape mismatch between true mask {true_mask.shape} and prediction {pred_mask.shape}"
            )

        tp, fp, tn, fn = confusion_from_masks(true_mask, pred_mask)
        metrics = compute_metrics(tp, fp, tn, fn)

        metrics["image_name"] = image_path.name
        metrics["model_path"] = str(model_path)
        metrics["threshold"] = args.threshold

        save_metrics(metrics, output_dir, image_path.stem)

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()