from pathlib import Path
import argparse

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import tensorflow as tf


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
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


def predict_mask(model, image: np.ndarray, threshold: float = 0.5):
    image_norm = normalize_image(image)
    x = np.expand_dims(image_norm, axis=0)  # (1, H, W, C)

    pred = model.predict(x, verbose=0)[0]   # (H, W, 1)
    pred_mask = (pred >= threshold).astype(np.uint8).squeeze()  # (H, W)

    return pred.squeeze(), pred_mask


def save_outputs(image: np.ndarray, pred_prob: np.ndarray, pred_mask: np.ndarray, output_dir: Path, base_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save binary mask as tif/png
    tiff.imwrite(str(output_dir / f"{base_name}_pred_mask.tif"), (pred_mask * 255).astype(np.uint8))
    plt.imsave(output_dir / f"{base_name}_pred_mask.png", pred_mask, cmap="gray")

    # Save probability map
    plt.imsave(output_dir / f"{base_name}_probability_map.png", pred_prob, cmap="gray")

    # Save overlay using FLAIR channel
    flair = image[:, :, 1]

    plt.figure(figsize=(6, 6))
    plt.imshow(flair, cmap="gray")
    plt.imshow(pred_mask, cmap="Reds", alpha=0.35)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_overlay.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict segmentation mask for one MRI .tif image using a .h5 Keras model.")
    parser.add_argument("--model", type=str, required=True, help="Path to .h5 model")
    parser.add_argument("--image", type=str, required=True, help="Path to input .tif image")
    parser.add_argument("--output", type=str, default="results/inference", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary mask")

    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_dir = Path(args.output)

    model = tf.keras.models.load_model(model_path)
    image = tiff.imread(str(image_path))

    pred_prob, pred_mask = predict_mask(model, image, threshold=args.threshold)
    save_outputs(image, pred_prob, pred_mask, output_dir, image_path.stem)

    print(f"Prediction completed.")
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()